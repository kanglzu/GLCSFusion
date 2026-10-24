


import os
import time
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from evaluator import Evaluator, VIFF
from skimage.metrics import structural_similarity

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')






PROJECT_ROOT = Path(__file__).resolve().parent
MODALITY_PATHS = {
    modality: {
        'ir_dir': str(PROJECT_ROOT / 'data' / modality / 'test' / other),
        'vis_dir': str(PROJECT_ROOT / 'data' / modality / 'test' / 'MRI'),
        'model': str(PROJECT_ROOT / 'model_last' / f'run_{modality}' / 'run00' / f'best_{modality}.pth'),
    }
    for modality, other in [('CT-MRI', 'CT'), ('PET-MRI', 'PET'), ('SPECT-MRI', 'SPECT')]
}






def preprocess(img_mri: np.ndarray,
               img_other: np.ndarray,
               modality: str):
    from TaskFusion_dataset import normalize_mri, normalize_ir

    mri = normalize_mri(img_mri, use_percentile=True)
    other = normalize_ir(img_other, modality=modality, use_percentile=True)

    def _to_tensor(arr):
        return torch.from_numpy(arr[None, None].astype(np.float32)).to(device)

    return _to_tensor(mri), _to_tensor(other)






def postprocess_visual(fusion_y8: np.ndarray,
                       ir_bgr: np.ndarray,
                       modality: str) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if modality == 'CT-MRI':
        
        y_enh = clahe.apply(fusion_y8)
        ct_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)
        y_mix = np.clip(0.7 * y_enh + 0.3 * ct_gray, 0, 255).astype(np.uint8)
        fusion_bgr = cv2.cvtColor(y_mix, cv2.COLOR_GRAY2BGR)
    else:
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        ycrcb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2YCrCb)
        y_enh = clahe.apply(fusion_y8)
        pet_y = ycrcb[..., 0].astype(np.float32)
        y_mix = np.clip(0.60 * y_enh + 0.40 * pet_y, 0, 255)
        
        y_mix = np.clip((y_mix / 255.0) ** 1.10 * 255.0 * 0.90, 0, 255).astype(np.uint8)
        
        cb = 128. + 2.0 * (ycrcb[..., 1].astype(np.float32) - 128.)
        cr = 128. + 2.0 * (ycrcb[..., 2].astype(np.float32) - 128.)
        ycrcb_out = ycrcb.copy()
        ycrcb_out[..., 0] = y_mix
        ycrcb_out[..., 1] = np.clip(cb, 0, 255).astype(np.uint8)
        ycrcb_out[..., 2] = np.clip(cr, 0, 255).astype(np.uint8)
        fusion_bgr = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)

    return fusion_bgr


def make_pseudo_color(gray8: np.ndarray, modality: str) -> np.ndarray:
    norm = cv2.normalize(gray8, None, 0, 255, cv2.NORM_MINMAX)
    cmap = cv2.COLORMAP_TURBO if modality in ('PET-MRI', 'SPECT-MRI') else cv2.COLORMAP_BONE
    return cv2.applyColorMap(norm, cmap)






def tta_forward(model, t_mri: torch.Tensor,
                t_other: torch.Tensor) -> torch.Tensor:
    def _fwd(x1, x2):
        with torch.no_grad():
            return model(x1, x2).clamp(0., 1.)

    preds = []
    for k in range(4):
        x1 = torch.rot90(t_mri,   k, dims=[-2, -1])
        x2 = torch.rot90(t_other, k, dims=[-2, -1])
        y  = _fwd(x1, x2)
        preds.append(torch.rot90(y, (4 - k) % 4, dims=[-2, -1]))
        
        x1f = torch.flip(x1, dims=[-1])
        x2f = torch.flip(x2, dims=[-1])
        yf  = _fwd(x1f, x2f)
        preds.append(torch.flip(torch.rot90(yf, (4 - k) % 4, dims=[-2, -1]),
                                dims=[-1]))

    return torch.stack(preds, dim=0).mean(dim=0)






def compute_metrics(fusion_f32: np.ndarray,
                    mri_f32: np.ndarray,
                    other_f32: np.ndarray) -> dict:
    f8   = (fusion_f32  * 255).clip(0, 255).astype(np.uint8)
    m8   = (mri_f32     * 255).clip(0, 255).astype(np.uint8)
    o8   = (other_f32   * 255).clip(0, 255).astype(np.uint8)

    vif   = VIFF(fusion_f32, mri_f32, other_f32)
    scd   = Evaluator.SCD(fusion_f32, mri_f32, other_f32)
    qabf  = Evaluator.Qabf(fusion_f32, mri_f32, other_f32)
    from skimage.metrics import structural_similarity as _ssim
    msssim = (_ssim(fusion_f32, mri_f32, data_range=1.0)
            + _ssim(fusion_f32, other_f32, data_range=1.0))
    fmi   = Evaluator.MI(f8, m8, o8)

    return dict(VIF=vif, SCD=scd, QABF=qabf, MSSSIM=msssim, FMI=fmi)






def fusion_inference(ir_dir: str,
                     vis_dir: str,
                     out_dir: str,
                     modality: str = 'CT-MRI',
                     model_path: str = None,
                     config_path: str = None,
                     use_tta: bool = False):
    
    result_dir  = os.path.join(out_dir, f'results_{modality}')
    fusion_dir  = os.path.join(result_dir, 'fusion_img')
    compare_dir = os.path.join(result_dir, 'compare_img')
    os.makedirs(fusion_dir,  exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    
    from model_factory import build_model

    if model_path is None:
        model_path = MODALITY_PATHS[modality]['model']
    config_file = Path(config_path) if config_path else Path(model_path).with_name('experiment_config.json')
    if not config_file.is_file():
        raise FileNotFoundError(f'Missing model configuration: {config_file}. Pass --config_path.')
    with config_file.open(encoding='utf-8') as f:
        model_config = json.load(f)
    model = build_model(modality, model_config).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f'[{modality}] 模型加载自：{model_path}')

    
    valid_ext = ('.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png')
    ir_files  = sorted([f for f in os.listdir(ir_dir)
                        if f.lower().endswith(valid_ext)])
    vis_files = sorted([f for f in os.listdir(vis_dir)
                        if f.lower().endswith(valid_ext)])
    assert len(ir_files) == len(vis_files),\
        f'图像数量不匹配：ir={len(ir_files)}, vis={len(vis_files)}'
    print(f'共 {len(ir_files)} 对图像')

    all_metrics = []
    t0 = time.time()

    for ir_name, vis_name in zip(ir_files, vis_files):
        
        ir_path  = os.path.join(ir_dir,  ir_name)
        vis_path = os.path.join(vis_dir, vis_name)

        img_mri   = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)   
        img_other = cv2.imread(ir_path,  cv2.IMREAD_GRAYSCALE)   

        if img_mri is None or img_other is None:
            print(f'[跳过] 无法读取：{ir_name} 或 {vis_name}')
            continue

        
        H, W = img_mri.shape[:2]
        if img_other.shape[:2] != (H, W):
            img_other = cv2.resize(img_other, (W, H), interpolation=cv2.INTER_CUBIC)

        
        t_mri, t_other = preprocess(img_mri, img_other, modality)

        
        with torch.no_grad():
            if use_tta:
                out = tta_forward(model, t_mri, t_other)
            else:
                out = model(t_mri, t_other).clamp(0., 1.)

        fusion_f32 = out.squeeze().cpu().numpy()               
        
        f_min, f_max = fusion_f32.min(), fusion_f32.max()
        if f_max - f_min > 1e-6:
            fusion_f32 = (fusion_f32 - f_min) / (f_max - f_min)
        fusion_y8  = (fusion_f32 * 255.).clip(0, 255).astype(np.uint8)

        
        
        ir_bgr = cv2.imread(ir_path, cv2.IMREAD_COLOR)
        if ir_bgr is None:
            ir_bgr = cv2.cvtColor(img_other, cv2.COLOR_GRAY2BGR)
        else:
            
            b, g, r = cv2.split(ir_bgr)
            if np.allclose(b, g) and np.allclose(g, r):
                ir_bgr = make_pseudo_color(img_other, modality)

        fusion_bgr = postprocess_visual(fusion_y8, ir_bgr, modality)

        
        stem = os.path.splitext(ir_name)[0]
        cv2.imwrite(os.path.join(fusion_dir, f'{stem}_fusion.png'), fusion_bgr)

        
        mri_bgr = cv2.cvtColor(img_mri, cv2.COLOR_GRAY2BGR)
        compare = np.concatenate([mri_bgr, ir_bgr, fusion_bgr], axis=1)
        cv2.imwrite(os.path.join(compare_dir, f'{stem}_compare.png'), compare)

        
        mri_f32   = t_mri.squeeze().cpu().numpy()
        other_f32 = t_other.squeeze().cpu().numpy()
        m = compute_metrics(fusion_f32, mri_f32, other_f32)
        m['image'] = ir_name
        all_metrics.append(m)

        print(f'{ir_name}: '
              f'VIF={m["VIF"]:.4f}  SCD={m["SCD"]:.4f}  '
              f'QABF={m["QABF"]:.4f}  MSSSIM={m["MSSSIM"]:.4f}  '
              f'FMI={m["FMI"]:.4f}')

    
    if not all_metrics:
        print('[错误] 未计算任何指标，请检查输入路径和图像格式。')
        return

    keys = ['VIF', 'SCD', 'QABF', 'MSSSIM', 'FMI']

    def _arr(k):
        return np.array([m[k] for m in all_metrics], dtype=np.float64)

    mean_m = {k: float(_arr(k).mean()) for k in keys}
    std_m  = {k: float(_arr(k).std())  for k in keys}

    sep = '=' * 65
    def _row(label, d):
        return (f'{label:<6} | VIF={d["VIF"]:.4f} | SCD={d["SCD"]:.4f} | '
                f'QABF={d["QABF"]:.4f} | MSSSIM={d["MSSSIM"]:.4f} | '
                f'FMI={d["FMI"]:.4f}')

    print(f'\n{sep}')
    print(f'  融合指标汇总 [{modality}]  共 {len(all_metrics)} 张')
    print(sep)
    print(_row('MEAN', mean_m))
    print(_row('STD',  std_m))
    print(f'{sep}\n')

    
    csv_path = os.path.join(result_dir, f'metrics_{modality}.csv')
    with open(csv_path, 'w') as f:
        f.write('image,VIF,SCD,QABF,MSSSIM,FMI\n')
        for m in all_metrics:
            f.write(f'{m["image"]},{m["VIF"]:.6f},{m["SCD"]:.6f},'
                    f'{m["QABF"]:.6f},{m["MSSSIM"]:.6f},{m["FMI"]:.6f}\n')
        f.write(f'MEAN,{mean_m["VIF"]:.6f},{mean_m["SCD"]:.6f},'
                f'{mean_m["QABF"]:.6f},{mean_m["MSSSIM"]:.6f},{mean_m["FMI"]:.6f}\n')
        f.write(f'STD,{std_m["VIF"]:.6f},{std_m["SCD"]:.6f},'
                f'{std_m["QABF"]:.6f},{std_m["MSSSIM"]:.6f},{std_m["FMI"]:.6f}\n')

    txt_path = os.path.join(result_dir, f'metrics_summary_{modality}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Fusion Metrics Summary [{modality}]\n{sep}\n')
        f.write(_row('MEAN', mean_m) + '\n')
        f.write(_row('STD',  std_m)  + '\n')
        f.write(f'{sep}\n')

    print(f'结果已保存至：{result_dir}')
    print(f'总耗时：{time.time() - t0:.1f}s')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGLM+CSIF 图像融合推理脚本')
    parser.add_argument('--modality', '-m', type=str, default='CT-MRI',
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'])
    parser.add_argument('--all_modalities', action='store_true',
                        help='依次推理全部三种模态')
    parser.add_argument('--ir_dir',  type=str, default=None,
                        help='CT/PET/SPECT 图像文件夹（默认使用内置路径）')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='MRI 图像文件夹（默认使用内置路径）')
    parser.add_argument('--out_dir', type=str, default='./outputs',
                        help='输出根目录')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径（默认使用内置路径）')
    parser.add_argument('--use_tta', action='store_true',
                        help='启用测试时增强（8 次翻转/旋转均值）')
    args = parser.parse_args()

    modalities = (['CT-MRI', 'PET-MRI', 'SPECT-MRI']
                  if args.all_modalities else [args.modality])

    for mod in modalities:
        print(f'\n{"="*65}')
        print(f'  推理模态：{mod}')
        print(f'{"="*65}')
        ir_dir  = args.ir_dir  or MODALITY_PATHS[mod]['ir_dir']
        vis_dir = args.vis_dir or MODALITY_PATHS[mod]['vis_dir']
        mpath   = args.model_path or MODALITY_PATHS[mod]['model']
        fusion_inference(
            ir_dir=ir_dir,
            vis_dir=vis_dir,
            out_dir=args.out_dir,
            modality=mod,
            model_path=mpath,
            config_path=args.config_path,
            use_tta=args.use_tta,
        )