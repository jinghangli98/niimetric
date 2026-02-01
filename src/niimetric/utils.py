"""Utility functions for NIfTI file handling."""

import nibabel as nib
import numpy as np
from pathlib import Path


def load_nifti(filepath: str) -> np.ndarray:
    """
    Load a NIfTI file and return the image data as a numpy array.
    
    Args:
        filepath: Path to the .nii or .nii.gz file
        
    Returns:
        numpy array of image data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid NIfTI file
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")
    
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        return data.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file: {filepath}. Error: {e}")


def validate_shapes(ref_data: np.ndarray, img_data: np.ndarray) -> None:
    """
    Validate that two images have the same shape.
    
    Args:
        ref_data: Reference image data
        img_data: Comparison image data
        
    Raises:
        ValueError: If shapes don't match
    """
    if ref_data.shape != img_data.shape:
        raise ValueError(
            f"Image shapes do not match. "
            f"Reference: {ref_data.shape}, Image: {img_data.shape}"
        )


def normalize_to_range(data: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
    """
    Normalize array to specified range.
    
    Args:
        data: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range
        
    Returns:
        Normalized array
    """
    data_min = data.min()
    data_max = data.max()
    
    if data_max - data_min == 0:
        return np.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val
