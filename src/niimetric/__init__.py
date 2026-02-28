"""NiiMetric - NIfTI Image Quality Metrics Package."""

__version__ = "0.1.2"

from .metrics import (
    compute_ssim,
    compute_psnr,
    compute_mae,
    compute_lpips,
    compute_tsnr,
    compute_flickering_index,
)
from .cropping import auto_crop_volumes
from .utils import load_nifti

__all__ = [
    "compute_ssim",
    "compute_psnr",
    "compute_mae",
    "compute_lpips",
    "compute_tsnr",
    "compute_flickering_index",
    "auto_crop_volumes",
    "load_nifti",
]
