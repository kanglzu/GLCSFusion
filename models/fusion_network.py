
import math
import importlib.util
from functools import partial
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

def icnr_(weight: torch.Tensor, upscale_factor: int,
          initializer=nn.init.kaiming_normal_):
    out_ch, in_ch, kh, kw = weight.shape
    sub = out_ch // (upscale_factor ** 2)
    k = initializer(torch.zeros([sub, in_ch, kh, kw])).transpose(0, 1)
    k = k.contiguous().view(in_ch, sub, -1)
    k = k.repeat(1, 1, upscale_factor ** 2)
    k = k.contiguous().view([in_ch, out_ch, kh, kw]).transpose(0, 1)
    with torch.no_grad():
        weight.copy_(k)



from models.wavelet import WaveletHFFusion
from models.new_cross import (
    SGLMBlock,          
    CSIF,               
    FigureAlignedTrueCrossCSIF,
    CrossMambaBlock,    
    VSSBlock_new,       
    VSSBlock_Cross_new, 
    LDC,
    ECA,
)

_FUSIONMAMBA_CROSS = None

def _load_fusionmamba_cross():
    global _FUSIONMAMBA_CROSS
    if _FUSIONMAMBA_CROSS is None:
        source = Path(__file__).resolve().parents[2] / 'FusionMamba-main' / 'models' / 'cross.py'
        if not source.exists():
            raise FileNotFoundError(f'FusionMamba reference implementation not found: {source}')
        spec = importlib.util.spec_from_file_location('fusionmamba_standard_cross', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FUSIONMAMBA_CROSS = module
    return _FUSIONMAMBA_CROSS

class AddFusion(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


class ConcatProjectionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(2 * channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x1, x2):
        return self.norm(self.proj(torch.cat([x1, x2], dim=-1)))


class SourceGuidedMultiScaleLDC(nn.Module):
    def __init__(self, channels=1, hidden=16):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ldc3 = LDC(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.dw_dil = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.GELU(),
        )
        self.eca = ECA(hidden)
        self.gate = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(hidden, channels, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _grad(x):
        gx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx + gy).clamp(0., 1.)

    def forward(self, out, x1, x2):
        x1g = x1.mean(dim=1, keepdim=True) if x1.shape[1] != 1 else x1
        x2g = x2.mean(dim=1, keepdim=True) if x2.shape[1] != 1 else x2
        if x1g.shape[-2:] != out.shape[-2:]:
            x1g = F.interpolate(x1g, size=out.shape[-2:], mode='bilinear', align_corners=False)
        if x2g.shape[-2:] != out.shape[-2:]:
            x2g = F.interpolate(x2g, size=out.shape[-2:], mode='bilinear', align_corners=False)
        src_max = torch.max(x1g, x2g)
        src_diff = torch.abs(x1g - x2g)
        src_grad = torch.max(self._grad(x1g), self._grad(x2g))
        guide = torch.cat([x1g, x2g, src_max, src_diff, src_grad], dim=1)

        feat = self.in_proj(out)
        d3 = self.ldc3(feat)
        d5 = self.dw5(feat) - F.avg_pool2d(feat, 5, stride=1, padding=2)
        dd = self.dw_dil(feat) - F.avg_pool2d(feat, 7, stride=1, padding=3)
        detail = self.mix(torch.cat([d3, d5, dd], dim=1))
        detail = self.eca(detail) * self.gate(guide)
        return out + torch.tanh(self.scale) * self.out(detail)


class SourceGuidedReconstructionHead(nn.Module):
    def __init__(self, channels: int = 1, hidden: int = 24,
                 residual_scale: float = 0.25):
        super().__init__()
        in_channels = channels * 5  
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, hidden // 8), hidden),
            nn.GELU(),
        )
        self.gate = nn.Conv2d(hidden, channels, 1, bias=True)
        self.correction = nn.Conv2d(hidden, channels, 3, padding=1, bias=True)
        residual_scale = min(max(float(residual_scale), 1e-4), 0.999)
        self.residual_scale = nn.Parameter(
            torch.tensor(math.atanh(residual_scale), dtype=torch.float32))

    def reset_safe_start(self):
        
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    @staticmethod
    def _match_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
        if x.shape[1] == channels:
            return x
        return x.mean(dim=1, keepdim=True).expand(-1, channels, -1, -1)

    def forward(self, decoder: torch.Tensor, x1: torch.Tensor,
                x2: torch.Tensor) -> torch.Tensor:
        channels = decoder.shape[1]
        x1 = self._match_channels(x1, channels)
        x2 = self._match_channels(x2, channels)
        if x1.shape[-2:] != decoder.shape[-2:]:
            x1 = F.interpolate(x1, size=decoder.shape[-2:], mode='bilinear',
                               align_corners=False)
        if x2.shape[-2:] != decoder.shape[-2:]:
            x2 = F.interpolate(x2, size=decoder.shape[-2:], mode='bilinear',
                               align_corners=False)
        features = self.features(torch.cat(
            [decoder, x1, x2, torch.max(x1, x2), torch.abs(x1 - x2)], dim=1))
        gate = torch.sigmoid(self.gate(features))
        selected = gate * x1 + (1.0 - gate) * x2
        residual = self.correction(features)
        return (selected + torch.tanh(self.residual_scale) * residual).clamp(0., 1.)






class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96,
                 norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.proj(x).permute(0, 2, 3, 1)   
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H2, W2 = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)   
        x = self.norm(x)
        x = self.reduction(x)                       
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2,
                 norm_layer=nn.LayerNorm, mode: str = 'bilinear'):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.mode = mode
        if mode == 'pixelshuffle':
            
            
            self.expand_ps = nn.Conv2d(dim, dim * dim_scale, kernel_size=1, bias=False)
            self.shuffle = nn.PixelShuffle(dim_scale)
            icnr_(self.expand_ps.weight, dim_scale)
        else:
            
            self.expand = nn.Conv2d(dim, dim // dim_scale, kernel_size=1, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)                               
        if self.mode == 'pixelshuffle':
            x = self.shuffle(self.expand_ps(x))                 
        else:
            
            
            
            input_dtype = x.dtype
            x = F.interpolate(x.float(), scale_factor=float(self.dim_scale),
                              mode='bilinear', align_corners=False).to(input_dtype)
            x = self.expand(x)                                  
        x = x.permute(0, 2, 3, 1)                               
        x = self.norm(x)                                         
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 4,
                 norm_layer=nn.LayerNorm, mode: str = 'bilinear'):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.mode = mode
        if mode == 'pixelshuffle':
            self.expand_ps = nn.Conv2d(dim, dim * dim_scale * dim_scale,
                                       kernel_size=1, bias=False)
            self.shuffle = nn.PixelShuffle(dim_scale)
            icnr_(self.expand_ps.weight, dim_scale)
            
            self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
            nn.init.zeros_(self.refine.weight)
        else:
            self.expand = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.mode == 'pixelshuffle':
            x = self.shuffle(self.expand_ps(x))
            x = x + self.refine(x)
        else:
            input_dtype = x.dtype
            x = F.interpolate(x.float(), scale_factor=float(self.dim_scale),
                              mode='bilinear', align_corners=False).to(input_dtype)
            x = self.expand(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x






class VSSLayer(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int = 16,
                 drop: float = 0., attn_drop: float = 0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint: bool = False,
                 use_fd_block: bool = False, fd_mode: str = 'add',
                 backbone_variant: str = 'transmamba',
                 strict_serial_dvss: bool = False, serial_v2: bool = False,
                 parallel_local_branch: bool = False, ffn_type: str = 'mlp',
                 kan_grid_size: int = 3, kan_spline_order: int = 3,
                 disable_global: bool = False, disable_local: bool = False,
                 disable_mha: bool = False, disable_mamba: bool = False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        block_cls = (_load_fusionmamba_cross().VSSBlock_new
                     if backbone_variant == 'fusionmamba' else SGLMBlock)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_args = dict(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=partial(norm_layer, eps=1e-6))
            if backbone_variant == 'fusionmamba':
                block_args['ssm_d_state'] = d_state
            else:
                block_args.update(attn_drop_rate=attn_drop, d_state=d_state,
                                  use_fd_block=use_fd_block, fd_mode=fd_mode,
                                  strict_serial=strict_serial_dvss, serial_v2=serial_v2,
                                  parallel_local_branch=parallel_local_branch,
                                  disable_global=disable_global,
                                  disable_local=disable_local,
                                  disable_mha=disable_mha,
                                  disable_mamba=disable_mamba,
                                  ffn_type=ffn_type,
                                  kan_grid_size=kan_grid_size,
                                  kan_spline_order=kan_spline_order)
            self.blocks.append(block_cls(**block_args))
        self.downsample = downsample(dim=dim, norm_layer=norm_layer)\
            if downsample is not None else None

        self._init_weights()

    def _init_weights(self):
        def fn(m):
            for name, p in m.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(fn)

    def forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x






class VSSLayer_up(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int = 16,
                 drop: float = 0., attn_drop: float = 0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, dim_before_up=None, use_checkpoint: bool = False,
                 use_fd_block: bool = False, fd_mode: str = 'add',
                 backbone_variant: str = 'transmamba',
                 strict_serial_dvss: bool = False, serial_v2: bool = False,
                 parallel_local_branch: bool = False, ffn_type: str = 'mlp',
                 kan_grid_size: int = 3, kan_spline_order: int = 3,
                 disable_global: bool = False, disable_local: bool = False,
                 disable_mha: bool = False, disable_mamba: bool = False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        up_dim = dim_before_up if dim_before_up is not None else dim
        self.upsample = upsample(dim=up_dim, norm_layer=norm_layer)\
            if upsample is not None else None
        block_cls = (_load_fusionmamba_cross().VSSBlock_new
                     if backbone_variant == 'fusionmamba' else SGLMBlock)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_args = dict(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=partial(norm_layer, eps=1e-6))
            if backbone_variant == 'fusionmamba':
                block_args['ssm_d_state'] = d_state
            else:
                block_args.update(attn_drop_rate=attn_drop, d_state=d_state,
                                  use_fd_block=use_fd_block, fd_mode=fd_mode,
                                  strict_serial=strict_serial_dvss, serial_v2=serial_v2,
                                  parallel_local_branch=parallel_local_branch,
                                  disable_global=disable_global,
                                  disable_local=disable_local,
                                  disable_mha=disable_mha,
                                  disable_mamba=disable_mamba,
                                  ffn_type=ffn_type,
                                  kan_grid_size=kan_grid_size,
                                  kan_spline_order=kan_spline_order)
            self.blocks.append(block_cls(**block_args))

        self._init_weights()

    def _init_weights(self):
        def fn(m):
            for name, p in m.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x




class CentralDifferenceConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = True, theta: float = 0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.theta = nn.Parameter(torch.tensor(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normal = self.conv(x)
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        diff = F.conv2d(x, kernel_diff, bias=None, stride=self.conv.stride, padding=0,
                        dilation=self.conv.dilation, groups=self.conv.groups)
        if self.conv.bias is not None:
            diff = diff + self.conv.bias.view(1, -1, 1, 1)
        return normal - torch.tanh(self.theta) * diff


class LearnableDiffConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        center = torch.zeros(kernel_size, kernel_size)
        center[kernel_size // 2, kernel_size // 2] = 1.0
        self.register_buffer('center_mask', center.view(1, 1, kernel_size, kernel_size))
        self.learnable_mask = nn.Parameter(torch.ones(out_channels, in_channels))
        self.theta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sum = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        mask = 1.0 - torch.tanh(self.theta) * self.learnable_mask[:, :, None, None] * self.center_mask * weight_sum
        return F.conv2d(x, self.conv.weight * mask, self.conv.bias,
                        stride=self.conv.stride, padding=self.conv.padding,
                        dilation=self.conv.dilation, groups=self.conv.groups)


class ShallowDetailFusion(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 1, hidden: int = 24):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)
        guide_ch = 6
        self.head = nn.Sequential(
            CentralDifferenceConv2d(guide_ch, hidden, 3, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(guide_ch, hidden // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden // 2, out_chans, 1),
            nn.Sigmoid(),
        )
        self.tail = nn.Conv2d(hidden, out_chans, 3, 1, 1)

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True) if x.shape[1] != 1 else x

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gray(x)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        x1g = self._gray(x1)
        x2g = self._gray(x2)
        base_g = self._gray(base)
        x_max = torch.max(x1g, x2g)
        x_diff = torch.abs(x1g - x2g)
        grad = torch.max(self._sobel(x1g), self._sobel(x2g)).clamp(0., 4.) / 4.0
        guide = torch.cat([x1g, x2g, x_max, x_diff, grad, base_g], dim=1)
        feat = self.head(guide)
        feat = feat + self.body(feat)
        learned_detail = torch.tanh(self.tail(feat)) * self.gate(guide)

        return learned_detail


class AdaptiveOutputRefine(nn.Module):
    def __init__(self, out_chans: int = 1, hidden: int = 16):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)

        guide_ch = 7
        self.detail_gate = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans, 1),
            nn.Sigmoid(),
        )
        self.structure = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans, 3, 1, 1),
        )

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True) if x.shape[1] != 1 else x

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gray(x)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                base: torch.Tensor, detail: torch.Tensor):
        x1g = self._gray(x1)
        x2g = self._gray(x2)
        base_g = self._gray(base)
        avg = 0.5 * (x1g + x2g)
        x_max = torch.max(x1g, x2g)
        x_diff = torch.abs(x1g - x2g)
        grad = torch.max(self._sobel(x1g), self._sobel(x2g)).clamp(0., 4.) / 4.0
        smooth_err = torch.abs(base_g - avg)
        guide = torch.cat([x1g, x2g, x_max, avg, x_diff, grad, smooth_err], dim=1)

        detail_gate = self.detail_gate(guide)
        structure_residual = torch.tanh(self.structure(guide))
        return detail_gate * detail, structure_residual


class MetricAwareBaseDetailRefine(nn.Module):
    def __init__(self, out_chans: int = 1, hidden: int = 32):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)

        guide_ch = 12
        self.shared = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            CentralDifferenceConv2d(hidden, hidden, 3, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detail_3x3 = nn.Sequential(
            LearnableDiffConv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detail_5x5 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detail_dilated = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detail_out = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans, 3, 1, 1),
        )
        self.base_expert = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans, 3, 1, 1),
        )
        self.source_gate = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans, 1),
            nn.Sigmoid(),
        )
        self.gates = nn.Sequential(
            nn.Conv2d(guide_ch, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, out_chans * 2, 1),
            nn.Sigmoid(),
        )
        self.detail_prior_coef = nn.Parameter(torch.tensor(0.0))
        self.base_target_coef = nn.Parameter(torch.tensor(0.0))

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True) if x.shape[1] != 1 else x

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        x = self._gray(x)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, base: torch.Tensor):
        x1g = self._gray(x1)
        x2g = self._gray(x2)
        base_g = self._gray(base)
        avg = 0.5 * (x1g + x2g)
        x_max = torch.max(x1g, x2g)
        x_diff = torch.abs(x1g - x2g)
        grad1 = self._sobel(x1g)
        grad2 = self._sobel(x2g)
        grad = torch.max(grad1, grad2).clamp(0., 4.) / 4.0
        grad_diff = torch.abs(grad1 - grad2).clamp(0., 4.) / 4.0

        low_x1 = F.avg_pool2d(x1g, kernel_size=9, stride=1, padding=4)
        low_x2 = F.avg_pool2d(x2g, kernel_size=9, stride=1, padding=4)
        low_base = F.avg_pool2d(base_g, kernel_size=9, stride=1, padding=4)
        lc1 = F.avg_pool2d((x1g - low_x1).abs(), kernel_size=5, stride=1, padding=2)
        lc2 = F.avg_pool2d((x2g - low_x2).abs(), kernel_size=5, stride=1, padding=2)
        w1 = lc1 / (lc1 + lc2 + 1e-6)
        low_target = w1 * low_x1 + (1.0 - w1) * low_x2
        low_err = torch.abs(low_base - low_target)
        high_x1 = x1g - low_x1
        high_x2 = x2g - low_x2

        guide = torch.cat([
            x1g, x2g, x_max, avg, x_diff, grad, grad_diff, base_g,
            low_target, low_err, high_x1.abs(), high_x2.abs()
        ], dim=1)
        feat = self.shared(guide)
        detail_gate, base_gate = self.gates(guide).chunk(2, dim=1)
        source_gate = self.source_gate(guide)
        signed_detail_prior = source_gate * high_x1 + (1.0 - source_gate) * high_x2
        envelope_detail = x_max - F.avg_pool2d(x_max, kernel_size=3, stride=1, padding=1)
        detail_prior = (0.65 * signed_detail_prior + 0.35 * envelope_detail).clamp(-0.30, 0.30)

        d3 = self.detail_3x3(feat)
        d5 = self.detail_5x5(feat)
        dd = self.detail_dilated(feat)
        learned_detail = self.detail_out(torch.cat([d3, d5, dd], dim=1))
        detail = detail_gate * torch.tanh(
            learned_detail + torch.tanh(self.detail_prior_coef) * detail_prior)
        base_residual = base_gate * torch.tanh(
            self.base_expert(feat) +
            torch.tanh(self.base_target_coef) * (low_target - low_base))
        return detail, base_residual







class VSSM_Fusion(nn.Module):

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 1,
        out_chans: int = 1,
        num_classes: int = 1000,          
        depths=(2, 2, 4, 2),              
        depths_decoder=(2, 4, 2, 2),      
        dims=(64, 128, 256, 512),         
        dims_decoder=(512, 256, 128, 64), 
        d_state: int = 16,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        
        use_edge_cross_gate: bool = False,
        use_light_version: bool = False,
        use_afe: bool = False,
        input_residual_coef: float = 0.1,
        enable_edge_hp_inject: bool = False,
        edge_hp_beta: float = 0.05,
        enable_metric_head: bool = False,
        enable_mbdg_head: bool = False,
        enable_output_refine: bool = False,
        upsample_mode: str = 'bilinear',   
        use_fusion_gate: bool = False,     
        use_cross_mamba_dffm: bool = False, 
        use_true_cross_mamba_dffm: bool = False, 
        use_deep_csif_gate: bool = False,   
        deep_csif_init: float = 0.01,       
        use_common_response: bool = False,  
        use_fd_block: bool = False,        
        fd_mode: str = 'add',              
        strict_serial_dvss: bool = False,  
        serial_v2: bool = False,           
        use_parallel_local_branch: bool = False, 
        disable_sglm_global: bool = False,       
        disable_sglm_local: bool = False,         
        disable_sglm_mha: bool = False,           
        disable_sglm_mamba: bool = False,         
        use_deep_kan: bool = False,        
        decoder_kan_stages=None,           
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        use_wavelet_hf: bool = False,      
        use_sg_ldc: bool = False,          
        true_cross_mamba_layers: str = 'all', 
        true_cross_mamba_sites=None,          
        true_cross_mamba_scale: float = 0.03,
        true_cross_mamba_trainable: bool = True,
        true_cross_state_params=('delta', 'B', 'C'),
        true_cross_static_ad: bool = True,    
        csif_bidirectional_cross_attention: bool = True,
        csif_response_aggregation: str = 'sum_product',
        csif_use_final_ssd: bool = True,
        fusion_stage_modes=None,              
        residual_csif_init: float = 0.1,      
        deep_csif_trainable: bool = True,
        backbone_variant: str = 'sglm',       
        fusion_mode: str = 'csif_figure_true_cross',
        output_activation: str = 'clamp',     
        use_source_guided_reconstruction: bool = False,
        reconstruction_hidden: int = 24,
        reconstruction_residual_scale: float = 0.25,
        **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        assert self.num_layers == len(depths_decoder) == len(dims) == len(dims_decoder),\
            "depths, depths_decoder, dims, dims_decoder 长度必须一致"

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        if isinstance(dims_decoder, int):
            dims_decoder = [int(dims_decoder * 2 ** i) for i in range(self.num_layers)]

        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.dims_decoder = dims_decoder
        self.input_residual_coef = input_residual_coef
        self.upsample_mode = upsample_mode
        self.use_fusion_gate = use_fusion_gate
        self.use_cross_mamba_dffm = use_cross_mamba_dffm
        self.use_true_cross_mamba_dffm = use_true_cross_mamba_dffm
        self.use_deep_csif_gate = bool(use_deep_csif_gate)
        self.deep_csif_init = float(deep_csif_init)
        self.true_cross_mamba_layers = true_cross_mamba_layers
        if self.num_layers == 3:
            stage_names = ('shallow', 'mid', 'deep')
        elif self.num_layers == 4:
            stage_names = ('shallow', 'mid', 'deep', 'bottleneck')
        else:
            stage_names = tuple(f'stage{i}' for i in range(self.num_layers))
        self.stage_names = stage_names
        if true_cross_mamba_sites is None:
            self.true_cross_mamba_sites = None
        else:
            if isinstance(true_cross_mamba_sites, str):
                true_cross_mamba_sites = [
                    item.strip() for item in true_cross_mamba_sites.split(',')
                    if item.strip()]
            self.true_cross_mamba_sites = frozenset(true_cross_mamba_sites)
            invalid = self.true_cross_mamba_sites.difference(stage_names)
            if invalid:
                raise ValueError(f'Unknown TrueCross sites: {sorted(invalid)}')
        self.true_cross_mamba_scale = float(true_cross_mamba_scale)
        self.true_cross_mamba_trainable = bool(true_cross_mamba_trainable)
        self.true_cross_state_params = true_cross_state_params
        self.true_cross_static_ad = bool(true_cross_static_ad)
        self.csif_bidirectional_cross_attention = bool(csif_bidirectional_cross_attention)
        self.csif_response_aggregation = str(csif_response_aggregation)
        self.csif_use_final_ssd = bool(csif_use_final_ssd)
        if not (0.0 < float(residual_csif_init) < 1.0):
            raise ValueError('residual_csif_init must lie strictly between 0 and 1')
        self.residual_csif_init = float(residual_csif_init)
        self.deep_csif_trainable = bool(deep_csif_trainable)
        self.use_common_response = use_common_response
        self.use_fd_block = use_fd_block
        self.fd_mode = fd_mode
        self.strict_serial_dvss = strict_serial_dvss
        self.serial_v2 = serial_v2
        self.use_parallel_local_branch = bool(use_parallel_local_branch)
        self.disable_sglm_global = bool(disable_sglm_global)
        self.disable_sglm_local = bool(disable_sglm_local)
        self.disable_sglm_mha = bool(disable_sglm_mha)
        self.disable_sglm_mamba = bool(disable_sglm_mamba)
        if decoder_kan_stages is None:
            parsed_kan_stages = (0,) if use_deep_kan else ()
        else:
            if isinstance(decoder_kan_stages, str):
                decoder_kan_stages = [
                    item.strip() for item in decoder_kan_stages.split(',')
                    if item.strip()]
            parsed_kan_stages = tuple(sorted({int(i) for i in decoder_kan_stages}))
        invalid_kan_stages = [
            i for i in parsed_kan_stages if i < 0 or i >= self.num_layers]
        if invalid_kan_stages:
            raise ValueError(
                f'decoder_kan_stages must be in [0,{self.num_layers - 1}], '
                f'got {invalid_kan_stages}')
        self.decoder_kan_stages = parsed_kan_stages
        self.use_deep_kan = bool(self.decoder_kan_stages)
        self.kan_grid_size = int(kan_grid_size)
        self.kan_spline_order = int(kan_spline_order)
        self.backbone_variant = backbone_variant
        self.fusion_mode = fusion_mode
        self.output_activation = output_activation
        self.use_source_guided_reconstruction = bool(use_source_guided_reconstruction)
        self.wavelet_hf = WaveletHFFusion() if use_wavelet_hf else None
        self.sg_ldc = SourceGuidedMultiScaleLDC(out_chans, hidden=16) if use_sg_ldc else None
        self.enable_metric_head = bool(enable_metric_head or enable_mbdg_head)
        self.enable_output_refine = bool(enable_output_refine or enable_mbdg_head)
        if self.use_parallel_local_branch and not (serial_v2 or strict_serial_dvss):
            raise ValueError('use_parallel_local_branch requires serial_v2 or strict_serial_dvss')
        if self.use_deep_kan and backbone_variant != 'sglm':
            raise ValueError('use_deep_kan is supported only by the sglm backbone')

        
        _patch_norm_fn = norm_layer if patch_norm else None
        self.patch_embed1 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans,
                                          embed_dim=self.embed_dim,
                                          norm_layer=_patch_norm_fn)
        self.patch_embed2 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans,
                                          embed_dim=self.embed_dim,
                                          norm_layer=_patch_norm_fn)
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_fd_block=use_fd_block,
                fd_mode=fd_mode,
                backbone_variant=backbone_variant,
                strict_serial_dvss=strict_serial_dvss,
                serial_v2=serial_v2,
                parallel_local_branch=self.use_parallel_local_branch,
                disable_global=self.disable_sglm_global,
                disable_local=self.disable_sglm_local,
                disable_mha=self.disable_sglm_mha,
                disable_mamba=self.disable_sglm_mamba,
                ffn_type='mlp',
                kan_grid_size=self.kan_grid_size,
                kan_spline_order=self.kan_spline_order,
            )
            self.layers.append(layer)

        
        def _use_true_cm(layer_idx):
            if not self.use_true_cross_mamba_dffm:
                return False
            if self.true_cross_mamba_sites is not None:
                return self.stage_names[layer_idx] in self.true_cross_mamba_sites
            if self.true_cross_mamba_layers == 'shallow':
                return layer_idx == 0
            if self.true_cross_mamba_layers == 'shallow_mid':
                return self.stage_names[layer_idx] in ('shallow', 'mid')
            return True

        self.Cross_block = nn.ModuleList()
        
        
        
        
        self.residual_csif_logits = nn.ParameterDict()
        if fusion_stage_modes is None:
            fusion_stage_modes = [fusion_mode] * self.num_layers
        elif isinstance(fusion_stage_modes, str):
            fusion_stage_modes = [
                item.strip() for item in fusion_stage_modes.split(',')]
        if len(fusion_stage_modes) != self.num_layers:
            raise ValueError('fusion_stage_modes must contain one mode per stage')
        self.fusion_stage_modes = tuple(fusion_stage_modes)
        for i in range(self.num_layers):
            stage_mode = self.fusion_stage_modes[i]
            if stage_mode == 'add':
                fusion_block = AddFusion()
            elif stage_mode == 'concat_proj':
                fusion_block = ConcatProjectionFusion(dims[i])
            elif stage_mode == 'original':
                fusion_block = _load_fusionmamba_cross().VSSBlock_Cross_new(
                    hidden_dim=dims[i], d_state=d_state,
                    norm_layer=partial(norm_layer, eps=1e-6),
                    attn_drop_rate=attn_drop_rate)
            elif stage_mode in ('csif_figure_true_cross', 'residual_csif'):
                fusion_block = FigureAlignedTrueCrossCSIF(
                    hidden_dim=dims[i], d_state=d_state,
                    norm_layer=partial(norm_layer, eps=1e-6),
                    attn_drop_rate=attn_drop_rate,
                    cross_attn_heads=max(1, dims[i] // 32),
                    true_cross_mamba_scale=self.true_cross_mamba_scale,
                    true_cross_mamba_trainable=self.true_cross_mamba_trainable,
                    true_cross_state_params=self.true_cross_state_params,
                    true_cross_static_ad=self.true_cross_static_ad,
                    use_bidirectional_cross_attention=self.csif_bidirectional_cross_attention,
                    response_aggregation=self.csif_response_aggregation,
                    use_final_ssd=self.csif_use_final_ssd)
                if stage_mode == 'residual_csif':
                    init_logit = math.log(
                        self.residual_csif_init / (1.0 - self.residual_csif_init))
                    self.residual_csif_logits[str(i)] = nn.Parameter(
                        torch.tensor(init_logit, dtype=torch.float32))
            elif stage_mode in ('proposed', 'csif', 'csif_true_cross'):
                stage_true_cross = (_use_true_cm(i) and
                                    stage_mode in ('proposed', 'csif_true_cross'))
                fusion_block = CSIF(
                    hidden_dim=dims[i], d_state=d_state,
                    norm_layer=partial(norm_layer, eps=1e-6),
                    attn_drop_rate=attn_drop_rate,
                    cross_attn_heads=max(1, dims[i] // 32),
                    use_modal_gate=self.use_fusion_gate,
                    use_cross_mamba=self.use_cross_mamba_dffm,
                    use_true_cross_mamba=stage_true_cross,
                    true_cross_mamba_scale=self.true_cross_mamba_scale,
                    true_cross_mamba_trainable=self.true_cross_mamba_trainable,
                    true_cross_state_params=self.true_cross_state_params,
                    true_cross_static_ad=self.true_cross_static_ad,
                    use_common_response=self.use_common_response)
            else:
                raise ValueError(f'Unsupported stage fusion mode: {stage_mode}')
            self.Cross_block.append(fusion_block)

        
        
        if self.use_deep_csif_gate:
            if fusion_mode not in ('proposed', 'csif_figure_true_cross'):
                raise ValueError('use_deep_csif_gate requires a CSIF fusion_mode')
            if self.fusion_stage_modes[-1] == 'add':
                raise ValueError('deep CSIF requires a non-add deep stage mode')
            if self.fusion_stage_modes[-1] == 'residual_csif':
                raise ValueError('deep CSIF gate and residual_csif cannot be enabled together')
            self.deep_csif_scale = nn.Parameter(
                torch.tensor(self.deep_csif_init, dtype=torch.float32),
                requires_grad=self.deep_csif_trainable)
        else:
            self.register_parameter('deep_csif_scale', None)

        
        
        _expand_cls = partial(PatchExpand2D, mode=self.upsample_mode)
        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            
            
            dim_before_up = dims_decoder[i - 1] if i > 0 else dims_decoder[i]
            layer = VSSLayer_up(
                dim=dims_decoder[i],
                dim_before_up=dim_before_up,
                depth=depths_decoder[i],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                norm_layer=norm_layer,
                upsample=_expand_cls if (i != 0) else None,
                use_checkpoint=use_checkpoint,
                use_fd_block=use_fd_block,
                fd_mode=fd_mode,
                backbone_variant=backbone_variant,
                strict_serial_dvss=strict_serial_dvss,
                serial_v2=serial_v2,
                parallel_local_branch=self.use_parallel_local_branch,
                disable_global=self.disable_sglm_global,
                disable_local=self.disable_sglm_local,
                disable_mha=self.disable_sglm_mha,
                disable_mamba=self.disable_sglm_mamba,
                ffn_type=('kan' if i in self.decoder_kan_stages else 'mlp'),
                kan_grid_size=self.kan_grid_size,
                kan_spline_order=self.kan_spline_order,
            )
            self.layers_up.append(layer)

        
        self.final_up = Final_PatchExpand2D(
            dim=dims_decoder[-1], dim_scale=patch_size, norm_layer=norm_layer,
            mode=self.upsample_mode)

        
        
        if self.use_fusion_gate and self.use_deep_csif_gate:
            C_deep = dims[-1]
            self.deep_gate = nn.Sequential(
                nn.Linear(2 * C_deep, C_deep), nn.GELU(),
                nn.Linear(C_deep, C_deep))
        else:
            self.deep_gate = None

        
        
        self.skip_reduce = nn.ModuleList([
            nn.Linear(dims[-(i + 1)], dims_decoder[i], bias=False)
            if dims[-(i + 1)] != dims_decoder[i] else nn.Identity()
            for i in range(self.num_layers)
        ])
        
        
        final_channels = dims_decoder[-1]
        self.output_head = nn.Conv2d(final_channels, out_chans, kernel_size=1, bias=True)
        self.source_reconstruction = SourceGuidedReconstructionHead(
            channels=out_chans, hidden=int(reconstruction_hidden),
            residual_scale=float(reconstruction_residual_scale),
        ) if self.use_source_guided_reconstruction else None
        if self.enable_output_refine:
            self.detail_fusion = ShallowDetailFusion(in_chans=in_chans, out_chans=out_chans, hidden=24)
            self.output_refine = AdaptiveOutputRefine(out_chans=out_chans, hidden=16)
            self.detail_residual_coef = nn.Parameter(torch.tensor(0.05))
            self.structure_residual_coef = nn.Parameter(torch.tensor(0.03))
        else:
            self.detail_fusion = None
            self.output_refine = None
        if self.enable_metric_head:
            self.metric_refine = MetricAwareBaseDetailRefine(out_chans=out_chans, hidden=32)
            self.metric_detail_coef = nn.Parameter(torch.tensor(0.03))
            self.metric_base_coef = nn.Parameter(torch.tensor(0.02))
        else:
            self.metric_refine = None


        
        self.diagnostics = {}
        self.enable_diagnostics = False

        self.apply(self._init_weights)
        self._zero_init_residual_heads()
        
        
        
        self._init_pixelshuffle_upsamplers()
        self._init_fusion_gates()

    def _init_pixelshuffle_upsamplers(self):
        for m in self.modules():
            if isinstance(m, (PatchExpand2D, Final_PatchExpand2D)) and\
                    getattr(m, 'mode', 'bilinear') == 'pixelshuffle':
                icnr_(m.expand_ps.weight, m.dim_scale)
                if hasattr(m, 'refine'):
                    nn.init.zeros_(m.refine.weight)

    def _init_fusion_gates(self):
        if self.deep_gate is not None:
            nn.init.zeros_(self.deep_gate[-1].weight)
            if self.deep_gate[-1].bias is not None:
                nn.init.zeros_(self.deep_gate[-1].bias)
        from models.new_cross import ModalFuseGate
        for m in self.modules():
            if isinstance(m, ModalFuseGate):
                nn.init.zeros_(m.pw_expand.weight)
                if m.pw_expand.bias is not None:
                    nn.init.zeros_(m.pw_expand.bias)
                nn.init.zeros_(m.ldc_scale)

    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'output_head') and isinstance(self.output_head, nn.Conv2d):
            nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.output_head.bias, 0.0)

    def _zero_init_residual_heads(self):
        def zero_conv(conv: nn.Conv2d):
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        if self.detail_fusion is not None:
            zero_conv(self.detail_fusion.tail)
        if self.output_refine is not None:
            zero_conv(self.output_refine.structure[-1])

        if self.metric_refine is not None:
            zero_conv(self.metric_refine.detail_out[-1])
            zero_conv(self.metric_refine.base_expert[-1])

        if self.source_reconstruction is not None:
            self.source_reconstruction.reset_safe_start()

        for module in self.modules():
            conv_detail = getattr(module, 'conv_detail', None)
            if conv_detail is not None and hasattr(conv_detail, 'pw2'):
                zero_conv(conv_detail.pw2)


    
    def forward_features_1(self, x: torch.Tensor):
        skip_list = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer.forward_blocks(x)
            skip_list.append(x)      
            if layer.downsample is not None:
                x = layer.downsample(x)
        return x, skip_list

    def forward_features_2(self, x: torch.Tensor):
        skip_list = []
        x = self.patch_embed2(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer.forward_blocks(x)
            skip_list.append(x)
            if layer.downsample is not None:
                x = layer.downsample(x)
        return x, skip_list

    
    def _residual_csif_fuse(self, stage_idx, csif, feat1, feat2):
        if self.fusion_stage_modes[stage_idx] != 'residual_csif':
            return csif
        baseline = feat1 + feat2
        alpha = torch.sigmoid(self.residual_csif_logits[str(stage_idx)])
        if self.enable_diagnostics:
            self.diagnostics[f'residual_csif_alpha_{stage_idx}'] = float(
                alpha.detach().cpu())
        return baseline + alpha * (csif - baseline)

    def Fusion_network(self, skip_list1, skip_list2):
        fused_skip_list = []
        
        
        num_skip_levels = self.num_layers - 1
        for i, (dffm, s1, s2) in enumerate(zip(
                self.Cross_block[:num_skip_levels],
                skip_list1[:num_skip_levels],
                skip_list2[:num_skip_levels])):
            fused = dffm(s1, s2)           
            fused = self._residual_csif_fuse(i, fused, s1, s2)
            fused_skip_list.append(fused)
        return fused_skip_list

    
    def forward_features_up(self, x: torch.Tensor, fused_skip_list):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                
                x = layer_up(x)
            else:
                
                skip = fused_skip_list[-(i)]        
                
                skip = self.skip_reduce[i](skip)    
                
                
                x_up = layer_up.upsample(x) if layer_up.upsample is not None else x
                
                if x_up.shape[1:3] != skip.shape[1:3]:
                    
                    skip_nchw = skip.permute(0, 3, 1, 2)
                    skip_nchw = F.interpolate(skip_nchw, size=x_up.shape[1:3],
                                              mode='bilinear', align_corners=False)
                    skip = skip_nchw.permute(0, 2, 3, 1)
                x_skip = x_up + skip               
                
                for blk in layer_up.blocks:
                    x_skip = blk(x_skip)
                x = x_skip
                continue
            
        return x

    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        
        feat1, skip_list1 = self.forward_features_1(x1)   
        feat2, skip_list2 = self.forward_features_2(x2)

        
        fused_skip_list = self.Fusion_network(skip_list1, skip_list2)

        
        deep_csif = self.Cross_block[-1](feat1, feat2)
        deep_csif = self._residual_csif_fuse(
            self.num_layers - 1, deep_csif, feat1, feat2)
        if self.deep_csif_scale is not None:
            if self.deep_gate is not None:
                g = torch.sigmoid(self.deep_gate(torch.cat([feat1, feat2], dim=-1)))
                baseline = (g * feat1 + (1.0 - g) * feat2 +
                            0.5 * (feat1 + feat2))
            else:
                baseline = feat1 + feat2
            alpha = torch.tanh(self.deep_csif_scale)
            x = baseline + alpha * (deep_csif - baseline)
            if self.enable_diagnostics:
                self.diagnostics['deep_csif_alpha'] = float(alpha.detach().cpu())
        else:
            
            x = deep_csif

        
        x = self._forward_decoder(x, fused_skip_list)

        
        x = self.final_up(x)                           

        
        x_nchw = x.permute(0, 3, 1, 2).contiguous()   
        out = self.output_head(x_nchw)                  

        
        if out.shape[-2:] != x1.shape[-2:]:
            out = F.interpolate(out, size=x1.shape[-2:],
                                mode='bilinear', align_corners=False)

        if self.source_reconstruction is not None:
            
            
            
            out = self.source_reconstruction(out, x1, x2)
        else:
            
            src = torch.max(x1, x2)
            if src.shape[1] != out.shape[1]:
                src = src.mean(dim=1, keepdim=True).expand_as(out)
            out = out + self.input_residual_coef * src

        if self.detail_fusion is not None and self.output_refine is not None:
            detail = self.detail_fusion(x1, x2, out)
            gated_detail, structure_residual = self.output_refine(x1, x2, out, detail)
            out = (out + torch.tanh(self.detail_residual_coef) * gated_detail +
                   torch.tanh(self.structure_residual_coef) * structure_residual)

        if self.metric_refine is not None:
            metric_detail, metric_base = self.metric_refine(x1, x2, out)
            out = (out + torch.tanh(self.metric_detail_coef) * metric_detail +
                   torch.tanh(self.metric_base_coef) * metric_base)

        if self.sg_ldc is not None:
            out = self.sg_ldc(out, x1, x2)

        
        if self.wavelet_hf is not None:
            out = self.wavelet_hf(out, x1, x2)

        
        
        if self.source_reconstruction is not None:
            return out.clamp(0., 1.)
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        if self.output_activation != 'clamp':
            raise ValueError(f'Unsupported output_activation: {self.output_activation}')
        return out.clamp(0., 1.)

    def _forward_decoder(self, x: torch.Tensor, fused_skip_list) -> torch.Tensor:
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                
                for blk in layer_up.blocks:
                    x = blk(x)
            else:
                
                if layer_up.upsample is not None:
                    x = layer_up.upsample(x)

                
                
                skip_idx = self.num_layers - 1 - i
                if 0 <= skip_idx < len(fused_skip_list):
                    skip = fused_skip_list[skip_idx]

                    
                    skip = self.skip_reduce[i](skip)

                    
                    if x.shape[1:3] != skip.shape[1:3]:
                        skip_nchw = skip.permute(0, 3, 1, 2)
                        skip_nchw = F.interpolate(
                            skip_nchw, size=x.shape[1:3],
                            mode='bilinear', align_corners=False)
                        skip = skip_nchw.permute(0, 2, 3, 1)

                    
                    if skip.shape[-1] != x.shape[-1]:
                        skip = F.pad(skip, (0, x.shape[-1] - skip.shape[-1]))\
                            if skip.shape[-1] < x.shape[-1] else skip[..., :x.shape[-1]]

                    x = x + skip                    

                
                for blk in layer_up.blocks:
                    x = blk(x)

        return x

















