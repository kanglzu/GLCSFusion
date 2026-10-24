

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim
from models.wavelet import wavelet_hf_loss

_msssim_impl = getattr(pytorch_msssim, "msssim", None) or getattr(pytorch_msssim, "ms_ssim")

def _msssim(x, y):
    try:
        return _msssim_impl(x, y, normalize=True)
    except TypeError:
        return _msssim_impl(x, y, data_range=1.0)


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.FloatTensor([[-1,0,1],[-2,0,2],[-1,0,1]]).unsqueeze(0).unsqueeze(0)
        ky = torch.FloatTensor([[1,2,1],[0,0,0],[-1,-2,-1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kx)
        self.register_buffer('weighty', ky)

    def forward(self, x):
        sx, sy = self.components(x)
        return torch.abs(sx) + torch.abs(sy)

    def components(self, x):
        sx = F.conv2d(x, self.weightx, padding=1)
        sy = F.conv2d(x, self.weighty, padding=1)
        return sx, sy


class Fusionloss(nn.Module):
    def __init__(self, modality: str = 'CT-MRI', profile: str = 'default',
                 stat_weight: float = 8.0):
        super().__init__()
        self.modality = modality
        self.profile = profile
        self.stat_weight = float(stat_weight)
        self.sobel = Sobelxy()
        lap = torch.FloatTensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('lap_kernel', lap)
        self._weights = {
            'CT-MRI':    (5, 12, 10),
            'PET-MRI':   (8, 16, 3),
            'SPECT-MRI': (6, 16, 4),
            'default':   (10, 10, 2),
        }

    def local_contrast(self, x, k=7):
        p = k // 2
        pool = nn.AvgPool2d(k, stride=1, padding=p)
        mu = pool(x)
        return (pool(x*x) - mu*mu).clamp(min=1e-6).sqrt()

    def corr_loss(self, x, y, eps=1e-6):
        x = x - x.mean(dim=(2, 3), keepdim=True)
        y = y - y.mean(dim=(2, 3), keepdim=True)
        num = (x * y).mean(dim=(2, 3))
        den = x.pow(2).mean(dim=(2, 3)).sqrt() * y.pow(2).mean(dim=(2, 3)).sqrt() + eps
        return (1.0 - num / den).mean()

    def laplacian(self, x):
        return torch.abs(F.conv2d(x, self.lap_kernel, padding=1))

    def intensity_stat_loss(self, fused, src1, src2, w_src1=0.55, eps=1e-6):
        
        
        dims = (2, 3)
        w_src2 = 1.0 - w_src1
        mu_f = fused.mean(dim=dims)
        mu_t = w_src1 * src1.mean(dim=dims) + w_src2 * src2.mean(dim=dims)
        std_f = fused.var(dim=dims, unbiased=False).add(eps).sqrt()
        std_1 = src1.var(dim=dims, unbiased=False).add(eps).sqrt()
        std_2 = src2.var(dim=dims, unbiased=False).add(eps).sqrt()
        std_t = torch.max(std_1, std_2)
        return F.smooth_l1_loss(mu_f, mu_t, beta=0.02) + F.smooth_l1_loss(std_f, std_t, beta=0.02)

    def edge_weighted_l1(self, pred, target, edge, strength=4.0):
        mean = edge.mean(dim=(2, 3), keepdim=True)
        std = edge.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        mask = torch.sigmoid((edge - (mean + 0.20 * std)) * 10.0).detach()
        return ((pred - target).abs() * (1.0 + strength * mask)).mean()

    def multiscale_grad_loss(self, fused, src1, src2, strength=2.0):
        loss = 0.0
        weights = (1.0, 0.55, 0.30)
        for scale_idx, w in enumerate(weights):
            if scale_idx == 0:
                f, a, b = fused, src1, src2
            else:
                k = 2 ** scale_idx
                f = F.avg_pool2d(fused, kernel_size=k, stride=k)
                a = F.avg_pool2d(src1, kernel_size=k, stride=k)
                b = F.avg_pool2d(src2, kernel_size=k, stride=k)
            gf = self.sobel(f)
            edge = torch.max(self.sobel(a), self.sobel(b))
            loss = loss + w * self.edge_weighted_l1(gf, edge, edge, strength=strength)
        return loss

    def vector_grad_loss(self, fused, src1, src2):
        loss = fused.new_zeros(())
        for scale, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
            if scale == 1:
                f, a, b = fused, src1, src2
            else:
                f = F.avg_pool2d(fused, scale, scale)
                a = F.avg_pool2d(src1, scale, scale)
                b = F.avg_pool2d(src2, scale, scale)

            gfx, gfy = self.sobel.components(f)
            gax, gay = self.sobel.components(a)
            gbx, gby = self.sobel.components(b)
            activity_a = torch.sqrt(gax.square() + gay.square() + 1e-6)
            activity_b = torch.sqrt(gbx.square() + gby.square() + 1e-6)
            select_a = torch.sigmoid(8.0 * (activity_a - activity_b)).detach()
            target_x = select_a * gax + (1.0 - select_a) * gbx
            target_y = select_a * gay + (1.0 - select_a) * gby
            loss = loss + weight * (
                F.smooth_l1_loss(gfx, target_x, beta=0.02) +
                F.smooth_l1_loss(gfy, target_y, beta=0.02)
            )
        return 0.5 * loss

    def forward(self, image_vis, image_ir, labels, generate_img, i):
        image_y = image_vis[:, :1, :, :]

        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)

        s1 = 1.0 - _msssim(generate_img, image_y)
        s2 = 1.0 - _msssim(generate_img, image_ir)
        w_mri = 0.6 if self.modality in ('PET-MRI', 'SPECT-MRI') else 0.5
        loss_ssim = w_mri * s1 + (1 - w_mri) * s2

        g_y  = self.sobel(image_y)
        g_ir = self.sobel(image_ir)
        g_f  = self.sobel(generate_img)
        loss_grad = F.l1_loss(g_f, torch.max(g_y, g_ir))
        use_vector_edge = 'vector' in self.profile

        
        lc_f  = self.local_contrast(generate_img)
        lc_y  = self.local_contrast(image_y)
        lc_ir = self.local_contrast(image_ir)
        loss_lc = F.l1_loss(lc_f, torch.max(lc_y, lc_ir))

        
        
        ablation_terms = {
            'abl_base':      (1, 1, 0, 0, 0),
            'abl_plus_grad': (1, 1, 1, 0, 0),
            'abl_plus_lc':   (1, 1, 0, 1, 0),
            'abl_plus_freq': (1, 1, 0, 0, 1),
            'abl_full':      (1, 1, 1, 1, 1),
            'abl_wo_grad':   (1, 1, 0, 1, 1),
            'abl_wo_lc':     (1, 1, 1, 0, 1),
            'abl_wo_freq':   (1, 1, 1, 1, 0),
        }
        if self.profile in ablation_terms:
            use_int, use_ssim, use_grad, use_lc, use_freq = ablation_terms[self.profile]
            loss_freq = wavelet_hf_loss(generate_img, image_y, image_ir)
            loss_total = (16.0 * use_int * loss_in +
                          6.0 * use_ssim * loss_ssim +
                          6.0 * use_grad * loss_grad +
                          6.0 * use_lc * loss_lc +
                          6.0 * use_freq * loss_freq)
            return loss_total, loss_in, loss_ssim, loss_grad

        if self.modality == 'CT-MRI':
            
            
            loss_in = F.l1_loss(generate_img, x_in_max)
            loss_ssim = 0.50 * s1 + 0.50 * s2
            loss_grad = F.l1_loss(g_f, torch.max(g_y, g_ir))
            if use_vector_edge:
                loss_grad = self.vector_grad_loss(generate_img, image_y, image_ir)
            loss_lc = F.l1_loss(lc_f, torch.max(lc_y, lc_ir))

            if self.profile == 'ct_qy_scd':
                avg_src = 0.50 * image_y + 0.50 * image_ir
                low_f = F.avg_pool2d(generate_img, kernel_size=9, stride=1, padding=4)
                low_src = F.avg_pool2d(avg_src, kernel_size=9, stride=1, padding=4)
                loss_low = F.l1_loss(low_f, low_src)
                loss_corr = 0.50 * self.corr_loss(generate_img, image_y) + 0.50 * self.corr_loss(generate_img, image_ir)
                loss_ms = self.multiscale_grad_loss(
                    generate_img, image_y, image_ir, strength=0.75)
                
                
                
                loss_total = (7.0 * loss_ssim + 15.0 * loss_in +
                              7.0 * loss_grad + 8.0 * loss_lc +
                              1.2 * loss_ms + 2.0 * loss_low +
                              1.0 * loss_corr)
            elif (self.profile in ('ct_edge', 'ct_edge_strong', 'ct_edge_mild', 'ct_wavelet')
                    or self.profile.startswith('ct_wavelet_vector')):
                
                
                if self.profile == 'ct_edge_strong':
                    ms_strength, w_grad, w_lc, w_ms = 3.0, 13.0, 11.0, 6.0
                elif self.profile in ('ct_edge_mild', 'ct_wavelet'):
                    ms_strength, w_grad, w_lc, w_ms = 1.0, 8.0, 8.0, 2.0
                else:
                    ms_strength, w_grad, w_lc, w_ms = 2.0, 10.0, 10.0, 4.0
                loss_ms = self.multiscale_grad_loss(
                    generate_img, image_y, image_ir, strength=ms_strength)
                loss_total = (6.0 * loss_ssim + 14.0 * loss_in +
                              w_grad * loss_grad + w_lc * loss_lc +
                              w_ms * loss_ms)
            else:
                loss_total = (6.0 * loss_ssim + 16.0 * loss_in +
                              6.0 * loss_grad + 6.0 * loss_lc)
        elif self.modality == 'PET-MRI':
            
            
            
            
            avg_src = 0.55 * image_y + 0.45 * image_ir
            intensity_target = 0.94 * x_in_max + 0.06 * avg_src
            loss_in = F.l1_loss(generate_img, intensity_target)
            loss_ssim = 0.55 * s1 + 0.45 * s2
            edge_target = torch.max(g_y, g_ir)
            loss_grad = F.l1_loss(g_f, edge_target)
            if use_vector_edge:
                loss_grad = self.vector_grad_loss(generate_img, image_y, image_ir)
            loss_lc = F.l1_loss(lc_f, torch.max(lc_y, lc_ir))

            low_f = F.avg_pool2d(generate_img, kernel_size=9, stride=1, padding=4)
            low_src = F.avg_pool2d(avg_src, kernel_size=9, stride=1, padding=4)
            loss_low = F.l1_loss(low_f, low_src)

            pet_mean = image_ir.mean(dim=(2, 3), keepdim=True)
            pet_std = image_ir.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            pet_mask = torch.sigmoid((image_ir - (pet_mean + 0.25 * pet_std)) * 12.0).detach()
            loss_pet_hot = F.l1_loss(generate_img * pet_mask, image_ir * pet_mask)
            loss_corr = 0.55 * self.corr_loss(generate_img, image_y) + 0.45 * self.corr_loss(generate_img, image_ir)

            loss_total = (10.0 * loss_ssim + 12.5 * loss_in +
                          28.0 * loss_grad + 11.0 * loss_lc +
                          1.5 * loss_low +
                          3.0 * loss_pet_hot + 2.0 * loss_corr)
        elif self.modality == 'SPECT-MRI':
            
            
            
            
            
            avg_src = 0.55 * image_y + 0.45 * image_ir
            edge_target = torch.max(g_y, g_ir)
            loss_grad = F.l1_loss(g_f, edge_target)
            if use_vector_edge:
                loss_grad = self.vector_grad_loss(generate_img, image_y, image_ir)
            loss_lc = F.l1_loss(lc_f, torch.max(lc_y, lc_ir))

            low_f = F.avg_pool2d(generate_img, kernel_size=9, stride=1, padding=4)
            low_src = F.avg_pool2d(avg_src, kernel_size=9, stride=1, padding=4)
            loss_low = F.l1_loss(low_f, low_src)

            high_f = generate_img - low_f
            high_src = x_in_max - F.avg_pool2d(x_in_max, kernel_size=9, stride=1, padding=4)
            loss_high = F.l1_loss(high_f, high_src.detach())

            loss_corr = 0.55 * self.corr_loss(generate_img, image_y) + 0.45 * self.corr_loss(generate_img, image_ir)
            loss_total = (8.0 * loss_ssim + 12.0 * loss_in +
                          18.0 * loss_grad + 10.0 * loss_lc +
                          3.0 * loss_low + 4.0 * loss_high +
                          1.5 * loss_corr)
        else:
            ws, wi, wg = self._weights.get(self.modality, self._weights['default'])
            wlc = 6.0
            loss_total = ws*loss_ssim + wi*loss_in + wg*loss_grad + wlc*loss_lc

        
        if 'wavelet' in self.profile:
            loss_total = loss_total + 6.0 * wavelet_hf_loss(
                generate_img, image_y, image_ir)

        
        
        
        if 'stat' in self.profile:
            stat_w_src1 = 0.55 if self.modality in ('PET-MRI', 'SPECT-MRI') else 0.50
            loss_total = loss_total + self.stat_weight * self.intensity_stat_loss(
                generate_img, image_y, image_ir, w_src1=stat_w_src1)

        return loss_total, loss_in, loss_ssim, loss_grad
