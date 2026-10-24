from types import SimpleNamespace

import torch.nn as nn

from models.new_module import VSSM_Fusion


def build_model(modality, config):
    args = SimpleNamespace(**config)
    structure_presets = {
        'none': dict(backbone_variant='sglm', fusion_mode='proposed',
                     serial_v2=False, use_fd=False, use_true_cross=False),
        'b0': dict(backbone_variant='fusionmamba', fusion_mode='add',
                   serial_v2=False, use_fd=False, use_true_cross=False),
        'b1': dict(backbone_variant='fusionmamba', fusion_mode='original',
                   serial_v2=False, use_fd=False, use_true_cross=False),
        'a0': dict(backbone_variant='sglm', fusion_mode='add',
                   serial_v2=True, use_fd=False, use_true_cross=False),
        'a1': dict(backbone_variant='sglm', fusion_mode='add',
                   serial_v2=True, use_fd=True, use_true_cross=False),
        'a2': dict(backbone_variant='sglm', fusion_mode='proposed',
                   serial_v2=True, use_fd=False, use_true_cross=True),
        
        
        
        'full': dict(backbone_variant='sglm', fusion_mode='csif_figure_true_cross',
                     serial_v2=True, use_fd=False, use_true_cross=True),
    }
    structure_name = getattr(args, 'structure_ablation', 'none')
    structure_cfg = dict(structure_presets[structure_name])
    if structure_name == 'none':
        structure_cfg['backbone_variant'] = getattr(args, 'backbone_variant', 'sglm')
        structure_cfg['fusion_mode'] = getattr(args, 'fusion_mode', 'proposed')
    
    
    deep_csif_enabled = (
        bool(getattr(args, 'use_deep_csif_gate', False))
        if structure_name == 'none' else False
    )
    model = VSSM_Fusion(
        patch_size=4,
        in_chans=1,
        out_chans=1,
        depths=[2, 2, 4, 2],
        depths_decoder=[2, 4, 2, 2],
        dims=[64, 128, 256, 512],
        dims_decoder=[512, 256, 128, 64],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        input_residual_coef=0.0 if modality == 'CT-MRI' else (0.05 if modality == 'PET-MRI' else 0.0),
        enable_metric_head=bool(getattr(args, 'enable_metric_head', False)),
        enable_mbdg_head=bool(getattr(args, 'enable_mbdg_head', False)),
        upsample_mode=getattr(args, 'upsample_mode', 'bilinear'),
        use_fusion_gate=bool(getattr(args, 'use_fusion_gate', False)),
        use_cross_mamba_dffm=bool(getattr(args, 'use_cross_mamba_dffm', False)),
        use_true_cross_mamba_dffm=(structure_cfg['use_true_cross'] if structure_name != 'none' else bool(getattr(args, 'use_true_cross_mamba_dffm', False))),
        use_deep_csif_gate=deep_csif_enabled,
        deep_csif_init=float(getattr(args, 'deep_csif_init', 0.01)),
        true_cross_mamba_layers=getattr(args, 'true_cross_mamba_layers', 'all'),
        true_cross_mamba_sites=getattr(args, 'true_cross_mamba_sites', None),
        true_cross_mamba_scale=float(getattr(args, 'true_cross_mamba_scale', 0.03)),
        true_cross_mamba_trainable=(
            True if structure_name == 'full'
            else bool(getattr(args, 'trainable_true_cross_scale', False))),
        true_cross_state_params=getattr(args, 'true_cross_state_params', 'delta,B,C'),
        true_cross_static_ad=bool(getattr(args, 'legacy_cross_static_ad', False)),
        csif_bidirectional_cross_attention=not bool(
            getattr(args, 'csif_disable_bi_attention', False)),
        csif_response_aggregation=getattr(
            args, 'csif_response_aggregation', 'sum_product'),
        csif_use_final_ssd=not bool(getattr(args, 'csif_disable_ssd', False)),
        fusion_stage_modes=getattr(args, 'fusion_stage_modes', None),
        residual_csif_init=float(getattr(args, 'residual_csif_init', 0.1)),
        deep_csif_trainable=bool(getattr(args, 'trainable_deep_csif_scale', False)),
        use_common_response=bool(getattr(args, 'use_common_response', False)),
        use_fd_block=(structure_cfg['use_fd'] if structure_name != 'none' else bool(getattr(args, 'use_fd_block', False))),
        fd_mode=('route' if structure_name != 'none' and structure_cfg['use_fd'] else getattr(args, 'fd_mode', 'add')),
        strict_serial_dvss=bool(getattr(args, 'strict_serial_dvss', False)),
        serial_v2=(structure_cfg['serial_v2'] if structure_name != 'none' else bool(getattr(args, 'dvss_serial_v2', False))),
        use_parallel_local_branch=bool(getattr(args, 'use_parallel_local_branch', False)),
        disable_sglm_global=bool(getattr(args, 'disable_sglm_global', False)),
        disable_sglm_local=bool(getattr(args, 'disable_sglm_local', False)),
        disable_sglm_mha=bool(getattr(args, 'disable_sglm_mha', False)),
        disable_sglm_mamba=bool(getattr(args, 'disable_sglm_mamba', False)),
        use_deep_kan=bool(getattr(args, 'use_deep_kan', False)),
        decoder_kan_stages=getattr(args, 'decoder_kan_stages', None),
        kan_grid_size=int(getattr(args, 'kan_grid_size', 3)),
        kan_spline_order=int(getattr(args, 'kan_spline_order', 3)),
        use_wavelet_hf=bool(getattr(args, 'use_wavelet_hf', False)),
        use_sg_ldc=bool(getattr(args, 'use_sg_ldc', False)),
        backbone_variant=structure_cfg['backbone_variant'],
        fusion_mode=structure_cfg['fusion_mode'],
        output_activation=getattr(args, 'output_activation', 'sigmoid'),
    )
    return model
