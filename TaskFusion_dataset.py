


import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from numpy import asarray
from pathlib import Path






def imresize(arr: np.ndarray, size, interp: str = 'bilinear') -> np.ndarray:
    im = Image.fromarray(asarray(arr))
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        size = tuple((np.array(im.size) * (size / 100.0)).astype(int))
    elif np.issubdtype(ts, np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])   
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    return np.array(im.resize(size, resample=func[interp]))


def prepare_data_path(dataset_path: str):
    exts = ('*.bmp', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dataset_path, ext)))
    paths.sort()
    filenames = [os.path.basename(p) for p in paths]
    return paths, filenames






def normalize_mri(arr: np.ndarray, use_percentile: bool = True) -> np.ndarray:
    arr = arr.astype(np.float32)
    if use_percentile:
        nz = arr[arr > 0]
        p1, p99 = (np.percentile(nz, [1, 99]) if nz.size > 100
                   else (0.0, 255.0))
        denom = max(p99 - p1, 1e-8)
        arr = np.clip((arr - p1) / denom, 0., 1.) if p99 > p1 else arr / 255.
    else:
        arr = arr / 255.
    return np.clip(arr, 0., 1.)


def normalize_ir(arr: np.ndarray, modality: str,
                 use_percentile: bool = True) -> np.ndarray:
    arr = arr.astype(np.float32)

    if modality == 'PET-MRI':
        arr = np.log1p(arr)
        nz = arr[arr > 0]
        p1, p99 = (np.percentile(nz, [1, 99]) if nz.size > 100
                   else (0.0, 5.420535))
        denom = max(p99 - p1, 1e-8)
        arr = ((arr - p1) / denom if p99 > p1
               else (arr - 0.) / 5.420535)

    elif modality == 'SPECT-MRI':
        if use_percentile:
            nz = arr[arr > 0]
            p1, p99 = (np.percentile(nz, [1, 99]) if nz.size > 100
                       else (0.0, 255.0))
            denom = max(p99 - p1, 1e-8)
            arr = (arr - p1) / denom if p99 > p1 else arr / 255.
        else:
            arr = arr / 255.

    else:  
        if use_percentile:
            nz = arr[arr > 0]
            p1, p99 = (np.percentile(nz, [1, 99]) if nz.size > 100
                       else (0.0, 255.0))
            denom = max(p99 - p1, 1e-8)
            arr = (arr - p1) / denom if p99 > p1 else arr / 255.
        else:
            arr = arr / 255.

    return np.clip(arr, 0., 1.)







_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_TRAIN_DIRS = {
    modality: {'vis': str(_PROJECT_ROOT / 'data' / modality / 'train' / 'MRI'),
               'ir': str(_PROJECT_ROOT / 'data' / modality / 'train' / other)}
    for modality, other in [('CT-MRI', 'CT'), ('PET-MRI', 'PET'), ('SPECT-MRI', 'SPECT')]
}


class Fusion_dataset(Dataset):

    def __init__(self,
                 split: str,
                 ir_path: str = None,
                 vi_path: str = None,
                 length: int = 0,
                 modality: str = 'PET-MRI',
                 use_percentile_norm: bool = True,
                 resize_to=(256, 256),
                 use_augmentation: bool = True,
                 aug_level: str = 'basic'):

        super().__init__()
        assert split in ('train', 'test'), 'split must be "train" or "test"'
        assert aug_level in ('basic', 'strong'), "aug_level must be 'basic' or 'strong'"

        self.split              = split
        self.modality           = modality
        self.use_percentile_norm = use_percentile_norm
        self.resize_to          = resize_to
        self.use_augmentation   = use_augmentation and (split == 'train')
        
        self.aug_level          = aug_level

        
        if split == 'train':
            defaults = _DEFAULT_TRAIN_DIRS[modality]
            data_dir_vis = vi_path or defaults['vis']
            data_dir_ir  = ir_path or defaults['ir']
        else:
            assert ir_path and vi_path,\
                'test split 必须显式提供 ir_path 和 vi_path'
            data_dir_vis = vi_path
            data_dir_ir  = ir_path

        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir,  self.filenames_ir  = prepare_data_path(data_dir_ir)

        assert len(self.filepath_vis) == len(self.filepath_ir), (
            f'[{modality}] 图像数量不一致：'
            f'MRI={len(self.filepath_vis)}, IR={len(self.filepath_ir)}'
        )

        
        self.length = len(self.filepath_vis)

    

    def __getitem__(self, index: int):
        vis_path = self.filepath_vis[index]
        ir_path  = self.filepath_ir[index]

        
        img_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        img_ir  = cv2.imread(ir_path,  cv2.IMREAD_GRAYSCALE)

        if img_vis is None:
            raise ValueError(f'无法读取 MRI 图像：{vis_path}')
        if img_ir is None:
            raise ValueError(f'无法读取 IR 图像：{ir_path}')

        
        H, W = img_vis.shape[:2]
        if img_ir.shape[:2] != (H, W):
            img_ir = cv2.resize(img_ir, (W, H), interpolation=cv2.INTER_CUBIC)

        
        if self.resize_to is not None:
            img_ir,  img_vis = (
                imresize(img_ir,  list(self.resize_to), interp='bicubic'),
                imresize(img_vis, list(self.resize_to), interp='bicubic'),
            )

        
        if self.use_augmentation:
            img_ir, img_vis = self._augment(img_ir, img_vis)

        
        vis_norm = normalize_mri(img_vis, self.use_percentile_norm)
        ir_norm  = normalize_ir(img_ir,   self.modality, self.use_percentile_norm)

        
        t_vis = torch.from_numpy(vis_norm[None].astype(np.float32))
        t_ir  = torch.from_numpy(ir_norm [None].astype(np.float32))

        return t_vis, t_ir

    def __len__(self) -> int:
        return self.length

    

    def split_train_val(self, train_ratio: float = 0.8, seed: int = 42):
        assert self.split == 'train', '只有 train split 可以划分验证集'
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.length).tolist()
        cut = int(self.length * train_ratio)
        return indices[:cut], indices[cut:]

    def _augment(self, img_ir: np.ndarray,
                 img_vis: np.ndarray):
        
        if random.random() < 0.5:                       
            img_ir  = cv2.flip(img_ir,  1)
            img_vis = cv2.flip(img_vis, 1)

        if random.random() < 0.3:                       
            img_ir  = cv2.flip(img_ir,  0)
            img_vis = cv2.flip(img_vis, 0)

        if random.random() < 0.3:                       
            angle = random.uniform(-15, 15)
            h, w  = img_ir.shape[:2]
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_ir  = cv2.warpAffine(img_ir,  M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
            img_vis = cv2.warpAffine(img_vis, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

        
        
        
        if self.aug_level == 'strong':
            if random.random() < 0.5:                   
                img_ir, img_vis = self._affine_pair(img_ir, img_vis)
            if random.random() < 0.5:                   
                img_ir, img_vis = self._elastic_pair(img_ir, img_vis)
            if random.random() < 0.4:                   
                img_ir, img_vis = self._scale_crop_pair(img_ir, img_vis)

        
        if self.modality == 'CT-MRI':
            
            '''
            if random.random() < 0.2:
                alpha = random.uniform(0.97, 1.03)
                img_ir  = np.clip(img_ir  * alpha, 0, 255).astype(np.uint8)
                img_vis = np.clip(img_vis * alpha, 0, 255).astype(np.uint8)
            '''
            pass
        elif self.modality == 'PET-MRI':
            
            if random.random() < 0.2:
                alpha = random.uniform(0.97, 1.03)
                img_ir  = np.clip(img_ir  * alpha, 0, 255).astype(np.uint8)
                img_vis = np.clip(img_vis * alpha, 0, 255).astype(np.uint8)

        elif self.modality == 'SPECT-MRI':
            
            if random.random() < 0.3:
                alpha = random.uniform(0.92, 1.08)
                img_ir  = np.clip(img_ir  * alpha, 0, 255).astype(np.uint8)
                img_vis = np.clip(img_vis * alpha, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                beta = random.uniform(-8, 8)
                img_ir  = np.clip(img_ir  + beta, 0, 255).astype(np.uint8)
                img_vis = np.clip(img_vis + beta, 0, 255).astype(np.uint8)
        return img_ir, img_vis

    
    def _affine_pair(self, img_ir, img_vis):
        h, w = img_ir.shape[:2]
        tx = random.uniform(-0.06, 0.06) * w            
        ty = random.uniform(-0.06, 0.06) * h
        scale = random.uniform(0.9, 1.1)                
        shear = random.uniform(-0.05, 0.05)             
        M = np.array([[scale, shear, tx],
                      [shear * 0.5, scale, ty]], dtype=np.float32)
        kw = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return (cv2.warpAffine(img_ir,  M, (w, h), **kw),
                cv2.warpAffine(img_vis, M, (w, h), **kw))

    def _elastic_pair(self, img_ir, img_vis, alpha=None, sigma=None):
        from scipy.ndimage import gaussian_filter
        h, w = img_ir.shape[:2]
        alpha = alpha if alpha is not None else random.uniform(20.0, 40.0)
        sigma = sigma if sigma is not None else random.uniform(5.0, 8.0)
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)
        kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return (cv2.remap(img_ir,  map_x, map_y, **kw),
                cv2.remap(img_vis, map_x, map_y, **kw))

    def _scale_crop_pair(self, img_ir, img_vis):
        h, w = img_ir.shape[:2]
        s = random.uniform(0.80, 0.98)                  
        ch, cw = int(h * s), int(w * s)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        crop_ir  = img_ir [y0:y0 + ch, x0:x0 + cw]
        crop_vis = img_vis[y0:y0 + ch, x0:x0 + cw]
        return (cv2.resize(crop_ir,  (w, h), interpolation=cv2.INTER_CUBIC),
                cv2.resize(crop_vis, (w, h), interpolation=cv2.INTER_CUBIC))