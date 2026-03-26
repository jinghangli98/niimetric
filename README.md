# NiiMetric

A Python CLI tool for evaluating image quality metrics between NIfTI (.nii/.nii.gz) medical images.

[![PyPI version](https://badge.fury.io/py/niimetric.svg)](https://badge.fury.io/py/niimetric)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index (slice-based with configurable dimension)
- **MAE** - Mean Absolute Error
- **LPIPS** - Learned Perceptual Image Patch Similarity (slice-based)
- **Image Augmentations** - Apply realistic MRI noise (coil-based), standard Rician noise, blur, bias field, motion, ghosting, and GRAPPA artifacts.
- **Auto-cropping** - Automatically crops to brain region based on reference image (30% mean threshold)
- **Foreground masking** - Evaluates only on non-air regions (excludes background)
- **Configurable slice dimension** - Evaluate on sagittal, coronal, or axial slices
- **CSV output** - Save results to CSV file
- **Mask export** - Optionally save the foreground mask as NIfTI

## Installation

```bash
pip install niimetric
```

## Usage

### Basic Usage

```bash
# Single metric
niimetric -a reference.nii.gz -b image.nii.gz --ssim -o output.csv
niimetric -a reference.nii.gz -b image.nii.gz --psnr -o output.csv
niimetric -a reference.nii.gz -b image.nii.gz --mae -o output.csv
niimetric -a reference.nii.gz -b image.nii.gz --lpips -o output.csv

# All metrics at once
niimetric -a reference.nii.gz -b image.nii.gz --all -o output.csv
```

### Slice Dimension Selection

```bash
# Evaluate on axial slices (default)
niimetric -a reference.nii.gz -b image.nii.gz --all --dim 2 -o output.csv

# Evaluate on sagittal slices
niimetric -a reference.nii.gz -b image.nii.gz --all --dim 0 -o output.csv

# Evaluate on coronal slices
niimetric -a reference.nii.gz -b image.nii.gz --all --dim 1 -o output.csv
```

### Save Foreground Mask

```bash
niimetric -a reference.nii.gz -b image.nii.gz --all -o output.csv --save-mask mask.nii.gz
```

### Image Augmentations (Artifact Generation)

You can generate degraded NIfTI images using included augmentation scripts powered by TorchIO. All augmentations accept an input `-i`, output `-o`, and severity `-s` (1 to 5).

```bash
# Add random noise (mostly on brain)
noise -i input.nii.gz -o noise_1.nii.gz -s 1

# Add realistic coil-based noise (supports --coils and --dir)
noise_robust -i input.nii.gz -o robust.nii.gz -s 3

# Add GRAPPA reconstruction artifacts
grappa -i input.nii.gz -o grappy.nii.gz -s 3

# Add blur artifact
blur -i input.nii.gz -o blur_3.nii.gz -s 3

# Add bias field (inhomogeneity) artifact
bias -i input.nii.gz -o bias_5.nii.gz -s 5

# Add motion artifact
motion -i input.nii.gz -o motion_2.nii.gz -s 2

# Add ghosting artifact
ghost -i input.nii.gz -o ghost_4.nii.gz -s 4
```

## Arguments

| Argument | Description |
|----------|-------------|
| `-a, --reference` | Path to reference NIfTI image (used for cropping boundaries) |
| `-b, --image` | Path to comparison NIfTI image |
| `-o, --output` | Path to output CSV file |
| `--ssim` | Calculate Structural Similarity Index |
| `--psnr` | Calculate Peak Signal-to-Noise Ratio |
| `--mae` | Calculate Mean Absolute Error |
| `--lpips` | Calculate Learned Perceptual Image Patch Similarity |
| `--all` | Calculate all metrics |
| `--dim` | Dimension for slice-based evaluation: 0=sagittal, 1=coronal, 2=axial (default: 2) |
| `--save-mask` | Path to save the foreground mask as NIfTI file |

## Output Format

Results are saved in CSV format:

```csv
reference,image,metric,value
reference.nii.gz,image.nii.gz,PSNR,25.432100
reference.nii.gz,image.nii.gz,SSIM,0.921500
reference.nii.gz,image.nii.gz,MAE,0.045200
reference.nii.gz,image.nii.gz,LPIPS,0.032100
```

## How It Works

1. **Load** both NIfTI images
2. **Auto-crop** based on reference image (30% mean threshold per slice)
3. **Normalize** both images to 0-1 range
4. **Create foreground mask** from reference (10% intensity threshold)
5. **Compute metrics** only on foreground regions
6. **Save results** to CSV

## Dependencies

- `nibabel` - NIfTI file I/O
- `numpy` - Array operations
- `scikit-image` - SSIM, PSNR calculations
- `torch` - LPIPS backend
- `lpips` - Perceptual similarity metric

## Python API

You can also use niimetric as a Python library:

```python
from niimetric import load_nifti, compute_ssim, compute_psnr, compute_mae, compute_lpips
from niimetric import auto_crop_volumes
from niimetric.metrics import create_foreground_mask
from niimetric.utils import normalize_to_range

# Load images
ref = load_nifti("reference.nii.gz")
img = load_nifti("image.nii.gz")

# Auto-crop
ref_cropped, img_cropped, bbox = auto_crop_volumes(ref, img)

# Normalize
ref_norm = normalize_to_range(ref_cropped, 0, 1)
img_norm = normalize_to_range(img_cropped, 0, 1)

# Create mask
mask = create_foreground_mask(ref_norm)

# Compute metrics
psnr = compute_psnr(ref_norm, img_norm, mask=mask)
ssim = compute_ssim(ref_norm, img_norm, mask=mask, dim=2)
mae = compute_mae(ref_norm, img_norm, mask=mask)
lpips = compute_lpips(ref_norm, img_norm, mask=mask, dim=2)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
