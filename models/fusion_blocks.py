import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class EfficientKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=3,
                 spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        if self.grid_size < 1 or self.spline_order < 1:
            raise ValueError('KAN grid_size and spline_order must both be positive')

        step = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = (torch.arange(-self.spline_order,
                             self.grid_size + self.spline_order + 1)
                * step + grid_range[0])
        self.register_buffer('grid', grid.expand(self.in_features, -1).contiguous())

        n_basis = self.grid_size + self.spline_order
        self.base_weight = nn.Parameter(torch.empty(self.out_features,
                                                     self.in_features))
        self.spline_weight = nn.Parameter(torch.empty(
            self.out_features, self.in_features, n_basis))
        self.spline_scaler = nn.Parameter(torch.empty(self.out_features,
                                                       self.in_features))
        self.base_activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0,
                        std=0.02 / max(1, self.grid_size))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def _b_splines(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f'KAN expects [tokens,{self.in_features}], got {tuple(x.shape)}')
        
        
        grid = self.grid.to(dtype=x.dtype)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left_den = grid[:, order:-1] - grid[:, :-(order + 1)]
            right_den = grid[:, order + 1:] - grid[:, 1:-order]
            bases = ((x - grid[:, :-(order + 1)]) / left_den
                     * bases[:, :, :-1]
                     + (grid[:, order + 1:] - x) / right_den
                     * bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x):
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        base = F.linear(self.base_activation(x2d), self.base_weight)
        spline_basis = self._b_splines(x2d).reshape(x2d.shape[0], -1)
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(spline_basis,
                          scaled_weight.reshape(self.out_features, -1))
        return (base + spline).reshape(*original_shape[:-1], self.out_features)


class KANFFN(nn.Module):
    def __init__(self, dim, grid_size=3, spline_order=3, drop=0.):
        super().__init__()
        self.kan = EfficientKANLinear(dim, dim, grid_size=grid_size,
                                      spline_order=spline_order)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.kan(x))


class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.register_buffer('center_mask',
            torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]))
        self.learnable_mask = nn.Parameter(
            torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
            requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)

    def forward(self, x):
        
        
        mask = (1.0 - self.learnable_theta
                * self.learnable_mask[:,:,None,None]
                * self.center_mask
                * self.conv.weight.sum(2).sum(2)[:,:,None,None])
        return F.conv2d(x, self.conv.weight*mask, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        return x * self.sigmoid(y)


class TransMamba(nn.Module):
    def __init__(self, hidden_size, d_state=32, d_conv=3, expand=2, mha_heads=4):
        super().__init__()
        
        self.tf_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mha = nn.MultiheadAttention(hidden_size, mha_heads,
                                          batch_first=True, dropout=0.)

        
        
        self.mamba_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        if Mamba is not None:
            self.ssm = Mamba(d_model=hidden_size, d_state=d_state,
                             d_conv=d_conv, expand=expand)
        else:
            self.ssm = nn.Linear(hidden_size, hidden_size)

    def _mamba_block(self, x1):
        return x1 + self.ssm(self.mamba_norm(x1))

    def forward_mha(self, x):
        xn = self.tf_norm(x)
        mha_out, _ = self.mha(xn, xn, xn, need_weights=False)
        return x + mha_out

    def forward_ssm(self, x):
        return self._mamba_block(x)

    def forward_attention_mamba(self, x, legacy_post_norm=False):
        xn = self.tf_norm(x)
        mha_out, _ = self.mha(xn, xn, xn, need_weights=False)
        x1 = x + mha_out
        if legacy_post_norm:
            x1 = self.tf_norm(x1)

        
        x2 = self._mamba_block(x1)
        return x2

    def forward(self, x):
        
        return self.forward_attention_mamba(x, legacy_post_norm=False)


class ConvDetailSideBranch(nn.Module):
    def __init__(self, hidden_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.dw1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                             groups=hidden_dim, bias=False)
        self.pw1 = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2,
                             groups=hidden_dim, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        nn.init.zeros_(self.pw2.weight)

    def forward(self, x):
        
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.dw2(x)
        x = self.pw2(x)
        return x.permute(0, 2, 3, 1).contiguous()


class HighFreqBranch(nn.Module):
    def __init__(self, dim, blur_k: int = 5, split: bool = True):
        super().__init__()
        self.blur_k = blur_k
        self.split = split   
        self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        self.pw2 = nn.Conv2d(dim, dim, 1, bias=False)
        
        self.theta = nn.Parameter(torch.tensor(0.5))

    def _diff(self, x):
        w = self.dw1.weight                              
        kd = w.sum(dim=(2, 3), keepdim=True)             
        center = torch.zeros_like(w)
        center[:, :, 1, 1] = kd[:, :, 0, 0]
        w_eff = w - torch.tanh(self.theta) * center
        return F.conv2d(x, w_eff, padding=1, groups=x.shape[1])

    def forward(self, x_bhwc):
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()
        if self.split:
            low = F.avg_pool2d(x, self.blur_k, stride=1, padding=self.blur_k // 2)
            hi = x - low                                 
        else:
            hi = x                                       
        h = self.act(self.pw1(self._diff(hi)))
        h = self.pw2(self.dw2(h))
        return h.permute(0, 2, 3, 1).contiguous()


class SGLMBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0., d_state=16,
                 mlp_ratio=4.0, use_checkpoint=False, use_fd_block=False,
                 strict_serial=False, serial_v2=False,
                 parallel_local_branch=False, ffn_type='mlp',
                 kan_grid_size=3, kan_spline_order=3,
                 fd_mode='add', fd_blur_k=5,
                 disable_global=False, disable_local=False,
                 disable_mha=False, disable_mamba=False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.strict_serial = strict_serial
        self.serial_v2 = serial_v2
        self.parallel_local_branch = bool(parallel_local_branch)
        
        
        self.use_global = not bool(disable_global)
        
        
        
        self.use_mha = self.use_global and not bool(disable_mha)
        self.use_mamba = self.use_global and not bool(disable_mamba)
        self.use_local = not bool(disable_local)
        self.ffn_type = str(ffn_type)
        if self.ffn_type not in ('mlp', 'kan'):
            raise ValueError(f'Unsupported ffn_type={self.ffn_type!r}')
        
        
        
        needs_route_norm = (not serial_v2) or (
            use_fd_block and fd_mode == 'route')
        self.norm1 = norm_layer(hidden_dim) if needs_route_norm else nn.Identity()
        self.transmamba = (TransMamba(hidden_dim, d_state=min(d_state*2, 64),
                                      mha_heads=max(1, hidden_dim//32))
                           if self.use_global else None)
        if self.transmamba is not None and not self.use_mha:
            for parameter in self.transmamba.tf_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in self.transmamba.mha.parameters():
                parameter.requires_grad_(False)
        if self.transmamba is not None and not self.use_mamba:
            for parameter in self.transmamba.mamba_norm.parameters():
                parameter.requires_grad_(False)
            for parameter in self.transmamba.ssm.parameters():
                parameter.requires_grad_(False)
        self.conv_detail = None
        self.detail_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ldc = LDC(hidden_dim, hidden_dim) if self.use_local else None
        self.norm2 = norm_layer(hidden_dim) if self.use_local else nn.Identity()
        self.eca = ECA(hidden_dim) if self.use_local else None
        
        
        
        
        self.fd_mode = fd_mode
        self.fd_blur_k = fd_blur_k
        if use_fd_block:
            self.hf_branch = HighFreqBranch(hidden_dim, blur_k=fd_blur_k,
                                            split=(fd_mode != 'route'))
            self.hf_scale = nn.Parameter(torch.tensor(0.0)) if fd_mode == 'add' else None
        else:
            self.hf_branch = None
            self.hf_scale = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm3 = norm_layer(hidden_dim)
            if self.ffn_type == 'kan':
                self.mlp = KANFFN(hidden_dim, grid_size=kan_grid_size,
                                  spline_order=kan_spline_order,
                                  drop=attn_drop_rate)
            else:
                self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio),
                               drop=attn_drop_rate)

    def _forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        if self.serial_v2:
            
            
            block_input = x
            local = None
            if self.use_local and self.parallel_local_branch:
                local_nchw = self.norm2(block_input).permute(0, 3, 1, 2).contiguous()
                local = self.eca(self.ldc(local_nchw))
                local = local.permute(0, 2, 3, 1).contiguous()
            if self.use_mha:
                seq = x.view(B, H * W, C)
                seq = self.transmamba.forward_mha(seq)    
                x_after_mha = seq.view(B, H, W, C)
            else:
                x_after_mha = x
                seq = x.view(B, H * W, C)
            if self.use_mamba and self.hf_branch is not None and self.fd_mode == 'route':
                
                
                
                
                xn = self.norm1(x_after_mha)
                xn_nchw = xn.permute(0, 3, 1, 2).contiguous()
                lo = F.avg_pool2d(xn_nchw, self.fd_blur_k, stride=1,
                                  padding=self.fd_blur_k // 2)
                hi = xn_nchw - lo
                lo_seq = lo.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
                lo_out = self.transmamba.forward_ssm(lo_seq)
                ssm_delta = (lo_out - lo_seq).view(B, H, W, C)
                hf_feat = self.hf_branch(
                    hi.permute(0, 2, 3, 1).contiguous())
                x = x_after_mha + self.drop_path(ssm_delta) + self.drop_path(hf_feat)
            elif self.use_mamba:
                seq = self.transmamba.forward_ssm(seq)    
                x = seq.view(B, H, W, C)
            else:
                x = x_after_mha

            if self.use_local and self.parallel_local_branch:
                x = x + self.drop_path(local)
            elif self.use_local:
                
                x_nchw = self.norm2(x).permute(0, 3, 1, 2).contiguous()
                local = self.eca(self.ldc(x_nchw))
                x = x + self.drop_path(
                    local.permute(0, 2, 3, 1).contiguous())

            
            if self.mlp_branch:
                x = x + self.drop_path(self.mlp(self.norm3(x)))
            if self.use_mamba and self.hf_branch is not None and self.fd_mode == 'add':
                x = x + torch.tanh(self.hf_scale) * self.drop_path(
                    self.hf_branch(x))
            return x

        if self.strict_serial:
            block_input = x
            local = None
            if self.use_local and self.parallel_local_branch:
                local_nchw = self.norm2(block_input).permute(0, 3, 1, 2).contiguous()
                local = self.eca(self.ldc(local_nchw))
                local = local.permute(0, 2, 3, 1).contiguous()
            seq = x.view(B, H * W, C)
            if self.use_mha:
                seq = self.transmamba.forward_mha(seq)
            if self.use_mamba:
                seq = self.transmamba.forward_ssm(seq)
            x = seq.view(B, H, W, C) if self.use_global else x

            if self.use_local and self.parallel_local_branch:
                x = x + self.drop_path(local)
            elif self.use_local:
                x_nchw = self.norm2(x).permute(0, 3, 1, 2).contiguous()
                detail = self.eca(self.ldc(x_nchw))
                x = x + self.drop_path(
                    detail.permute(0, 2, 3, 1).contiguous())

            if self.mlp_branch:
                x = x + self.drop_path(self.mlp(self.norm3(x)))
            return x

        xn = self.norm1(x)
        if self.use_mamba and self.hf_branch is not None and self.fd_mode == 'route':
            
            
            xn_nchw = xn.permute(0, 3, 1, 2).contiguous()
            lo = F.avg_pool2d(xn_nchw, self.fd_blur_k, stride=1,
                              padding=self.fd_blur_k // 2)
            hi = xn_nchw - lo
            lo_seq = lo.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            global_feat = self.transmamba(lo_seq).view(B, H, W, C)
            hf_feat = self.hf_branch(hi.permute(0, 2, 3, 1).contiguous())
            x = shortcut + self.drop_path(global_feat) + self.drop_path(hf_feat)
        elif self.use_global:
            seq = xn.view(B, H * W, C)
            if self.use_mha:
                seq = self.transmamba.forward_mha(seq)
            if self.use_mamba:
                seq = self.transmamba.forward_ssm(seq)
            global_feat = seq.view(B, H, W, C)
            x = shortcut + self.drop_path(global_feat)
        else:
            x = shortcut

        if self.use_local:
            x_nchw = x.permute(0, 3, 1, 2).contiguous()
            x_ldc = self.ldc(x_nchw)
            x_ldc_bhwc = x_ldc.permute(0, 2, 3, 1).contiguous()
            x_ldc_norm = self.norm2(x_ldc_bhwc).permute(0, 3, 1, 2).contiguous()
            x_eca = self.eca(x_ldc_norm)
            x = x + self.drop_path(x_eca.permute(0, 2, 3, 1).contiguous())

        if self.conv_detail is not None:
            detail_feat = self.conv_detail(shortcut)
            x = x + torch.tanh(self.detail_scale) * self.drop_path(detail_feat)

        
        
        if self.use_mamba and self.hf_branch is not None and self.fd_mode == 'add':
            x = x + torch.tanh(self.hf_scale) * self.drop_path(self.hf_branch(x))

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        return self._forward(x)



DVSSBlock = SGLMBlock
VSSBlock_new = SGLMBlock


class DynamicSSD(nn.Module):
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.delta_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid())
        self.state_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(max(1, channels//16), channels), nn.SiLU())

    def forward(self, x):
        delta = self.delta_gen(x)[:,:,None,None]
        state = self.state_proj(x)
        return self.out_proj((1.-delta)*x + delta*state)



class TrueCrossMambaSelectiveScan(nn.Module):
    def __init__(self, channels, d_state=16, dt_rank=None, dropout=0.,
                 cross_params=('delta', 'B', 'C'), cross_static_ad=True):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError('mamba_ssm selective_scan_fn is required for TrueCrossMambaSelectiveScan')
        self.channels = channels
        self.d_state = d_state
        self.dt_rank = dt_rank or max(4, channels // 16)
        if isinstance(cross_params, str):
            cross_params = tuple(
                item.strip() for item in cross_params.split(',') if item.strip())
        aliases = {'dt': 'delta', 'b': 'B', 'c': 'C'}
        self.cross_params = frozenset(
            aliases.get(str(item), aliases.get(str(item).lower(), str(item)))
            for item in cross_params)
        invalid = self.cross_params.difference({'delta', 'B', 'C'})
        if invalid:
            raise ValueError(f'Unsupported cross state parameters: {sorted(invalid)}')
        self.cross_static_ad = bool(cross_static_ad)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.in_proj1 = nn.Linear(channels, channels * 2, bias=False)
        self.in_proj2 = nn.Linear(channels, channels * 2, bias=False)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, groups=channels)
        self.param_proj1 = nn.Linear(channels, self.dt_rank + 2 * d_state, bias=False)
        self.param_proj2 = nn.Linear(channels, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj1 = nn.Linear(self.dt_rank, channels, bias=True)
        self.dt_proj2 = nn.Linear(self.dt_rank, channels, bias=True)
        self.A_log1 = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(channels, 1))
        self.A_log2 = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(channels, 1))
        self.D1 = nn.Parameter(torch.ones(channels))
        self.D2 = nn.Parameter(torch.ones(channels))
        self.out_proj1 = nn.Linear(channels, channels, bias=False)
        self.out_proj2 = nn.Linear(channels, channels, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        

    def _z_and_gate(self, x, norm, in_proj, conv):
        B, H, W, C = x.shape
        L = H * W
        x = norm(x).view(B, L, C)
        z, gate = in_proj(x).chunk(2, dim=-1)
        z = z.transpose(1, 2).contiguous()
        z = F.silu(conv(z)[..., :L])
        gate = F.silu(gate).transpose(1, 2).contiguous()
        return z, gate, B, H, W, L

    def _params(self, source, norm, proj, dt_proj, A_log):
        B, H, W, C = source.shape
        L = H * W
        tokens = norm(source).view(B, L, C)
        raw = proj(tokens)
        dt, Bp, Cp = torch.split(raw, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = dt_proj(dt).transpose(1, 2).contiguous()
        Bp = Bp.transpose(1, 2).contiguous()
        Cp = Cp.transpose(1, 2).contiguous()
        A = -torch.exp(A_log.float())
        return delta, A, Bp, Cp

    def forward(self, y_m1, y_m2):
        z1, gate1, B, H, W, L = self._z_and_gate(y_m1, self.norm1, self.in_proj1, self.conv1)
        z2, gate2, _, _, _, _ = self._z_and_gate(y_m2, self.norm2, self.in_proj2, self.conv2)

        delta1, A1, B1, C1 = self._params(y_m1, self.norm1, self.param_proj1, self.dt_proj1, self.A_log1)
        delta2, A2, B2, C2 = self._params(y_m2, self.norm2, self.param_proj2, self.dt_proj2, self.A_log2)

        d_1 = delta2 if 'delta' in self.cross_params else delta1
        b_1 = B2 if 'B' in self.cross_params else B1
        c_1 = C2 if 'C' in self.cross_params else C1
        bias_1 = self.dt_proj2.bias if 'delta' in self.cross_params else self.dt_proj1.bias
        d_2 = delta1 if 'delta' in self.cross_params else delta2
        b_2 = B1 if 'B' in self.cross_params else B2
        c_2 = C1 if 'C' in self.cross_params else C2
        bias_2 = self.dt_proj1.bias if 'delta' in self.cross_params else self.dt_proj2.bias

        
        
        
        scan_A1, scan_D1 = ((A2, self.D2.float()) if self.cross_static_ad
                            else (A1, self.D1.float()))
        scan_A2, scan_D2 = ((A1, self.D1.float()) if self.cross_static_ad
                            else (A2, self.D2.float()))
        h1 = selective_scan_fn(z1, d_1, scan_A1, b_1, c_1, scan_D1, z=gate1,
                               delta_bias=bias_1.float(), delta_softplus=True)
        h2 = selective_scan_fn(z2, d_2, scan_A2, b_2, c_2, scan_D2, z=gate2,
                               delta_bias=bias_2.float(), delta_softplus=True)
        h1 = h1.transpose(1, 2).contiguous().view(B, H, W, self.channels)
        h2 = h2.transpose(1, 2).contiguous().view(B, H, W, self.channels)
        return self.drop(self.out_proj1(h1)), self.drop(self.out_proj2(h2))


class ModalFuseGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 4, 16)
        self.pw_compress = nn.Conv2d(channels * 2, mid, 1, bias=True)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.pw_expand = nn.Conv2d(mid, channels, 1, bias=True)
        self.ldc = LDC(channels, channels, kernel_size=3, padding=1)
        self.ldc_scale = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.pw_compress.weight, std=0.01)
        nn.init.zeros_(self.pw_compress.bias)
        nn.init.zeros_(self.pw_expand.weight)
        nn.init.zeros_(self.pw_expand.bias)

    def forward(self, fm1, fm2):
        m1 = fm1.permute(0, 3, 1, 2).contiguous()
        m2 = fm2.permute(0, 3, 1, 2).contiguous()
        gate = self.pw_compress(torch.cat([m1, m2], dim=1))
        gate = self.act(self.dw(gate))
        gate = torch.sigmoid(self.pw_expand(gate))

        merged = 2.0 * (gate * m1 + (1.0 - gate) * m2)
        merged = merged + torch.tanh(self.ldc_scale) * self.ldc(merged)
        return merged.permute(0, 2, 3, 1).contiguous()


class CSIF(nn.Module):
    def __init__(self, hidden_dim, d_state=16,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0., cross_attn_heads=4,
                 use_modal_gate=False, use_cross_mamba=False,
                 use_common_response=False, use_true_cross_mamba=False,
                 true_cross_mamba_scale=0.03,
                 true_cross_mamba_trainable=True,
                 true_cross_state_params=('delta', 'B', 'C'),
                 true_cross_static_ad=True, **kwargs):
        super().__init__()
        C = hidden_dim
        self.use_modal_gate = use_modal_gate
        self.common_refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(max(1, C // 16), C), nn.GELU()) if use_common_response else None
        self.common_scale = nn.Parameter(torch.zeros(1)) if use_common_response else None
        self.cross_mamba_scale = nn.Parameter(torch.zeros(1)) if use_cross_mamba else None
        self.cross_mamba = CrossMambaBlock(C, attn_drop_rate, d_state, cross_attn_heads) if use_cross_mamba else None
        self.true_cross_mamba = TrueCrossMambaSelectiveScan(
            C, d_state, dropout=attn_drop_rate,
            cross_params=true_cross_state_params,
            cross_static_ad=true_cross_static_ad) if use_true_cross_mamba else None
        self.true_cross_mamba_scale = nn.Parameter(
            torch.tensor(float(true_cross_mamba_scale)),
            requires_grad=bool(true_cross_mamba_trainable)) if use_true_cross_mamba else None

        
        self.norm_in1 = norm_layer(C)
        self.norm_in2 = norm_layer(C)
        
        self.cross_attn_12 = nn.MultiheadAttention(C, cross_attn_heads,
                                                    dropout=attn_drop_rate, batch_first=True)
        
        self.cross_attn_21 = nn.MultiheadAttention(C, cross_attn_heads,
                                                    dropout=attn_drop_rate, batch_first=True)
        self.ca_norm1 = norm_layer(C)
        self.ca_norm2 = norm_layer(C)

        
        
        self.m1_br1 = nn.Linear(C, C, bias=False)
        
        self.m1_br2_lin1 = nn.Linear(C, C, bias=False)
        self.m1_br2_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GELU())
        self.m1_br2_lin2 = nn.Linear(C, C, bias=False)

        
        self.ssd_m1 = DynamicSSD(C, d_state)

        
        self.post_norm1 = norm_layer(C)
        self.post_lin1  = nn.Linear(C, C, bias=False)
        self.post_mlp1  = Mlp(C, C * 2, C)
        self.post_conv1 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(max(1, C // 16), C), nn.GELU())

        
        self.m2_br1 = nn.Linear(C, C, bias=False)
        self.m2_br2_lin1 = nn.Linear(C, C, bias=False)
        self.m2_br2_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GELU())
        self.m2_br2_lin2 = nn.Linear(C, C, bias=False)

        
        self.ssd_m2 = DynamicSSD(C, d_state)

        
        self.post_norm2 = norm_layer(C)
        self.post_lin2  = nn.Linear(C, C, bias=False)
        self.post_mlp2  = Mlp(C, C * 2, C)
        self.post_conv2 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(max(1, C // 16), C), nn.GELU())

        
        self.final_ssd  = DynamicSSD(C, d_state)
        self.final_norm = norm_layer(C)
        
        
        self.modal_gate = ModalFuseGate(C) if use_modal_gate else None

    def forward(self, f1, f2):
        B, H, W, C = f1.shape
        t = lambda x: x.permute(0, 3, 1, 2).contiguous()  
        b = lambda x: x.permute(0, 2, 3, 1).contiguous()  

        
        f1n = self.norm_in1(f1).view(B, H * W, C)
        f2n = self.norm_in2(f2).view(B, H * W, C)
        
        y_m1, _ = self.cross_attn_12(f1n, f2n, f2n, need_weights=False)
        
        y_m2, _ = self.cross_attn_21(f2n, f1n, f1n, need_weights=False)
        Y_M1 = self.ca_norm1(f1n + y_m1).view(B, H, W, C)
        Y_M2 = self.ca_norm2(f2n + y_m2).view(B, H, W, C)

        if self.cross_mamba is not None:
            cm1, cm2 = self.cross_mamba(Y_M1, Y_M2)
            scale = torch.tanh(self.cross_mamba_scale)
            Y_M1 = Y_M1 + scale * cm1
            Y_M2 = Y_M2 + scale * cm2

        if self.true_cross_mamba is not None:
            tcm1, tcm2 = self.true_cross_mamba(Y_M1, Y_M2)
            Y_M1 = Y_M1 + self.true_cross_mamba_scale * tcm1
            Y_M2 = Y_M2 + self.true_cross_mamba_scale * tcm2

        
        br1_1 = self.m1_br1(Y_M1)                                
        br1_2 = self.m1_br2_lin2(                                 
                    b(self.m1_br2_conv(t(self.m1_br2_lin1(Y_M1)))))
        agg_m1 = br1_1 + br1_2                                    
        ssd_m1 = b(self.ssd_m1(t(agg_m1)))                       

        
        fused_m1 = ssd_m1 + Y_M1                                  
        fused_m1 = self.post_norm1(fused_m1)
        fused_m1 = fused_m1 + self.post_lin1(fused_m1)
        fused_m1 = fused_m1 + self.post_mlp1(fused_m1)
        fused_m1 = b(self.post_conv1(t(fused_m1)))

        
        br2_1 = self.m2_br1(Y_M2)
        br2_2 = self.m2_br2_lin2(
                    b(self.m2_br2_conv(t(self.m2_br2_lin1(Y_M2)))))
        agg_m2 = br2_1 + br2_2
        ssd_m2 = b(self.ssd_m2(t(agg_m2)))

        
        fused_m2 = ssd_m2 + Y_M2                                  
        fused_m2 = self.post_norm2(fused_m2)
        fused_m2 = fused_m2 + self.post_lin2(fused_m2)
        fused_m2 = fused_m2 + self.post_mlp2(fused_m2)
        fused_m2 = b(self.post_conv2(t(fused_m2)))

        if self.common_refine is not None:
            common = b(self.common_refine(t(fused_m1 * fused_m2)))
            common_scale = torch.tanh(self.common_scale)
            fused_m1 = fused_m1 + common_scale * common
            fused_m2 = fused_m2 + common_scale * common

        
        if self.modal_gate is None:
            merged = fused_m1 + fused_m2
        else:
            merged = self.modal_gate(fused_m1, fused_m2)
        out = b(self.final_ssd(t(merged)))
        return self.final_norm(out + merged)                        


def _ssd_segment_sum(a):
    length = a.size(-1)
    expanded = a.unsqueeze(-1).expand(*a.shape, length)
    strict_lower = torch.tril(
        torch.ones(length, length, device=a.device, dtype=torch.bool),
        diagonal=-1)
    expanded = expanded.masked_fill(~strict_lower, 0.)
    segment = torch.cumsum(expanded, dim=-2)
    lower = torch.tril(
        torch.ones(length, length, device=a.device, dtype=torch.bool))
    return segment.masked_fill(~lower, -torch.inf)


def _ssd_chunk_scan(x, a, b, c, block_len):
    batch, length, nheads, headdim = x.shape
    if length % block_len:
        raise ValueError(f'SSD length {length} must be divisible by block_len {block_len}')
    chunks = length // block_len

    x = x.reshape(batch, chunks, block_len, nheads, headdim)
    a = a.reshape(batch, chunks, block_len, nheads).permute(0, 3, 1, 2)
    b = b.reshape(batch, chunks, block_len, nheads, -1)
    c = c.reshape(batch, chunks, block_len, nheads, -1)
    a_cumsum = torch.cumsum(a, dim=-1)

    
    decay = torch.exp(_ssd_segment_sum(a))
    y_diag = torch.einsum(
        'bclhn,bcshn,bhcls,bcshp->bclhp', c, b, decay, x)

    
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum('bclhn,bhcl,bclhp->bchpn', b, decay_states, x)
    initial = torch.zeros_like(states[:, :1])
    states = torch.cat([initial, states], dim=1)
    chunk_decay = torch.exp(_ssd_segment_sum(F.pad(a_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum('bhzc,bchpn->bzhpn', chunk_decay, states)
    states = new_states[:, :-1]

    
    state_decay_out = torch.exp(a_cumsum)
    y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', c, states, state_decay_out)
    return (y_diag + y_off).reshape(batch, length, nheads, headdim)


class StructuredStateSpaceDuality(nn.Module):
    def __init__(self, d_model, d_state=16, headdim=64, block_len=64):
        super().__init__()
        if d_model % min(headdim, d_model) != 0:
            raise ValueError('d_model must be divisible by headdim')
        self.d_model = d_model
        self.headdim = min(headdim, d_model)
        self.nheads = d_model // self.headdim
        self.d_state = d_state
        self.block_len = block_len

        self.x_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, self.nheads, bias=True)
        self.bc_proj = nn.Linear(
            d_model, 2 * self.nheads * d_state, bias=False)
        self.A_log = nn.Parameter(
            torch.empty(self.nheads).uniform_(1., 16.).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant_(self.dt_proj.bias, -4.)

    def _block_length(self, length):
        block_len = min(self.block_len, length)
        while block_len > 1 and length % block_len:
            block_len //= 2
        return block_len

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        dtype = tokens.dtype
        x = self.x_proj(tokens).reshape(
            batch, length, self.nheads, self.headdim)
        gate = F.silu(self.gate_proj(tokens))
        dt = F.softplus(self.dt_proj(tokens).float())
        b, c = self.bc_proj(tokens).chunk(2, dim=-1)
        b = b.reshape(batch, length, self.nheads, self.d_state).float()
        c = c.reshape(batch, length, self.nheads, self.d_state).float()

        
        a = -torch.exp(self.A_log.float())
        a_discrete = dt * a.view(1, 1, self.nheads)
        x_float = x.float()
        y = _ssd_chunk_scan(
            x_float * dt.unsqueeze(-1), a_discrete, b, c,
            self._block_length(length))
        y = y + self.D.float().view(1, 1, self.nheads, 1) * x_float
        y = y.reshape(batch, length, self.d_model).to(dtype)
        return self.out_proj(y * gate)


class FigureAlignedTrueCrossCSIF(nn.Module):
    def __init__(self, hidden_dim, d_state=16,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0., cross_attn_heads=4,
                 true_cross_mamba_scale=0.03,
                 true_cross_mamba_trainable=True,
                 true_cross_state_params=('delta', 'B', 'C'),
                 true_cross_static_ad=False,
                 use_bidirectional_cross_attention=True,
                 response_aggregation='sum_product',
                 use_final_ssd=True, **kwargs):
        super().__init__()
        if Mamba is None:
            raise ImportError('mamba_ssm is required for FigureAlignedTrueCrossCSIF')

        C = hidden_dim
        if response_aggregation not in ('sum', 'sum_product'):
            raise ValueError(
                'response_aggregation must be sum or sum_product, got '
                f'{response_aggregation!r}')
        self.use_bidirectional_cross_attention = bool(use_bidirectional_cross_attention)
        self.response_aggregation = response_aggregation
        self.use_final_ssd = bool(use_final_ssd)
        self.norm_in1 = norm_layer(C)
        self.norm_in2 = norm_layer(C)
        self.cross_attn_12 = nn.MultiheadAttention(
            C, cross_attn_heads, dropout=attn_drop_rate, batch_first=True)
        self.cross_attn_21 = nn.MultiheadAttention(
            C, cross_attn_heads, dropout=attn_drop_rate, batch_first=True)
        self.ca_norm1 = norm_layer(C)
        self.ca_norm2 = norm_layer(C)

        self.cross_scan = TrueCrossMambaSelectiveScan(
            C, d_state=d_state, dropout=attn_drop_rate,
            cross_params=true_cross_state_params,
            cross_static_ad=true_cross_static_ad)
        self.cross_scale = nn.Parameter(
            torch.tensor(float(true_cross_mamba_scale)),
            requires_grad=bool(true_cross_mamba_trainable))

        self.post_norm1 = norm_layer(C)
        self.post_norm2 = norm_layer(C)
        self.post_mlp1 = Mlp(C, C * 2, C, drop=attn_drop_rate)
        self.post_mlp2 = Mlp(C, C * 2, C, drop=attn_drop_rate)
        self.post_conv1 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(max(1, C // 16), C), nn.GELU())
        self.post_conv2 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(max(1, C // 16), C), nn.GELU())

        
        
        
        self.interaction_norm1 = norm_layer(C)
        self.interaction_norm2 = norm_layer(C)
        self.pre_ssd_norm = norm_layer(C)
        self.final_ssd = (StructuredStateSpaceDuality(
            d_model=C, d_state=d_state, headdim=min(64, C), block_len=64)
            if self.use_final_ssd else None)
        self.final_norm = norm_layer(C)

    @staticmethod
    def _to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _to_bhwc(x):
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, f1, f2):
        B, H, W, C = f1.shape
        f1n = self.norm_in1(f1).reshape(B, H * W, C)
        f2n = self.norm_in2(f2).reshape(B, H * W, C)

        
        
        if self.use_bidirectional_cross_attention:
            a12, _ = self.cross_attn_12(f1n, f2n, f2n, need_weights=False)
            a21, _ = self.cross_attn_21(f2n, f1n, f1n, need_weights=False)
            y1 = self.ca_norm1(f1n + a12).reshape(B, H, W, C)
            y2 = self.ca_norm2(f2n + a21).reshape(B, H, W, C)
        else:
            y1 = self.ca_norm1(f1n).reshape(B, H, W, C)
            y2 = self.ca_norm2(f2n).reshape(B, H, W, C)

        
        c1, c2 = self.cross_scan(y1, y2)
        scale = self.cross_scale
        m1 = y1 + scale * c1
        m2 = y2 + scale * c2

        
        p1 = self.post_mlp1(self.post_norm1(m1))
        p2 = self.post_mlp2(self.post_norm2(m2))
        p1 = self._to_bhwc(self.post_conv1(self._to_nchw(p1)))
        p2 = self._to_bhwc(self.post_conv2(self._to_nchw(p2)))

        n1 = self.interaction_norm1(p1)
        n2 = self.interaction_norm2(p2)
        merged = n1 + n2
        if self.response_aggregation == 'sum_product':
            merged = merged + n1 * n2

        
        if self.use_final_ssd:
            seq = self.pre_ssd_norm(merged).reshape(B, H * W, C)
            out = self.final_ssd(seq).reshape(B, H, W, C)
            merged = merged + out
        return self.final_norm(merged)



DFFM = CSIF
VSSBlock_Cross_new = CSIF


class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0., temperature=1.0):
        super().__init__()
        self.ca_12 = nn.MultiheadAttention(hidden_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ca_21 = nn.MultiheadAttention(hidden_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, F_1, F_2):
        B,H,W,C = F_1.shape
        f1=F_1.view(B,H*W,C); f2=F_2.view(B,H*W,C)
        a12,_=self.ca_12(f1,f2,f2,need_weights=False)
        a21,_=self.ca_21(f2,f1,f1,need_weights=False)
        return self.norm1(f1+a12).view(B,H,W,C), self.norm2(f2+a21).view(B,H,W,C)


class SS2D_cross_new(nn.Module):
    def __init__(self, d_model=96, d_state=16, dropout=0., **kwargs):
        super().__init__()
        self.ssd = DynamicSSD(d_model, d_state)
        self.out_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x1, x2):
        x = x1+x2
        out = self.ssd(x.permute(0,3,1,2)).permute(0,2,3,1)
        return self.drop(self.out_norm(out+x))


class CrossMambaBlock(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0., d_state=16, cross_attn_heads=4):
        super().__init__()
        self.cross_attn = CrossAttentionModule(hidden_dim, cross_attn_heads, attn_drop)
        self.cross_ssm = SS2D_cross_new(d_model=hidden_dim, dropout=attn_drop, d_state=d_state)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim))
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(max(1,hidden_dim//16), hidden_dim), nn.ReLU(inplace=True))

    def forward(self, f1, f2):
        a1,a2 = self.cross_attn(f1,f2)
        y12 = self.cross_ssm(a1,a2)
        y21 = self.cross_ssm(a2,a1)
        y12 = y12+self.mlp(y12)
        y21 = y21+self.mlp(y21)
        y12 = self.conv(y12.permute(0,3,1,2)).permute(0,2,3,1)
        y21 = self.conv(y21.permute(0,3,1,2)).permute(0,2,3,1)
        return y12, y21
