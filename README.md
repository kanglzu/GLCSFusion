# GLCSFusion

GLCSFusion is a medical image fusion model for paired CT–MRI, PET–MRI, and SPECT–MRI images. It extracts features from each input, fuses them at multiple scales, and uses a decoder with skip connections to reconstruct the fused image.

During development, the model was called VSSMFusion and was later renamed GLCSFusion. The former name still appears in code identifiers such as `VSSM_Fusion`.

## Contents

- `train.py`: training and validation for the three modality pairs.
- `test.py`: inference, visualization, and fusion metrics.
- `model_factory.py`: shared model construction for training and inference.
- `TaskFusion_dataset.py`: paired image loading, normalization, and augmentation.
- `models/`: network.
- `loss.py`, `evaluator.py`: training losses and evaluation metrics.

## Environment

The original experiments used Python 3.8.10, PyTorch 2.0.0 with CUDA 11.8, `mamba-ssm` 1.0.1, and `causal-conv1d` 1.0.0. Install a PyTorch build appropriate for your CUDA environment, then install `requirements.txt`. The Mamba packages require a compatible Linux/CUDA build.

## Data layout

The medical fusion experiments use paired images from the public [Harvard Whole Brain Atlas](http://www.med.harvard.edu/AANLIB/home.html) ([repository record](https://www.re3data.org/repository/r3d100010274)). Arrange the downloaded CT–MRI, PET–MRI, and SPECT–MRI images with matching sorted filenames under:

```text
data/
  CT-MRI/{train,test}/{MRI,CT}/
  PET-MRI/{train,test}/{MRI,PET}/
  SPECT-MRI/{train,test}/{MRI,SPECT}/
```

Training also accepts `--vis_dir` and `--ir_dir` to use another location. Inference accepts the same two options. Both scripts should be run from the repository root.

## Additional experiments

We also evaluated the model's applicability beyond the medical fusion setting on the [MSRS infrared–visible image fusion dataset](https://github.com/Linfeng-Tang/MSRS) and through medical image segmentation experiments.

## Train

```bash
python train.py --modality CT-MRI
python train.py --modality PET-MRI
python train.py --modality SPECT-MRI
```

Training writes checkpoints and `experiment_config.json` under `model_last/`, split indices under `outputs/splits/`, and TensorBoard files under `runs/`. Set a seed with `--seed` and record any additional model flags used for a reported experiment. The pretrained weights woule be supplied in BaiduYun.

## Inference

```bash
python test.py --modality CT-MRI \
  --vis_dir data/CT-MRI/test/MRI \
  --ir_dir data/CT-MRI/test/CT \
  --model_path model_last/run_CT-MRI/run00/best_CT-MRI.pth
```

`test.py` reads `experiment_config.json` from the checkpoint directory. Pass `--config_path` if the configuration is stored elsewhere. The inference model is built from that configuration, and checkpoint loading is strict so an architecture mismatch fails clearly.

