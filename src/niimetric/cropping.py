"""Auto-cropping functionality for 3D NIfTI volumes."""

import numpy as np
from typing import Tuple


def find_bounding_box(data: np.ndarray, mean_threshold_ratio: float = 0.3) -> Tuple[slice, slice, slice]:
    """
    Find the bounding box of meaningful slices in a 3D volume.
    
    Uses a mean-based threshold: slices are included if their mean intensity
    is at least `mean_threshold_ratio` times the overall volume mean.
    
    Args:
        data: 3D numpy array (X, Y, Z)
        mean_threshold_ratio: Ratio of volume mean to use as threshold (default 0.3 = 30%)
        
    Returns:
        Tuple of slices for each axis (x_slice, y_slice, z_slice)
    """
    # Compute overall volume mean
    volume_mean = np.mean(np.abs(data))
    threshold = volume_mean * mean_threshold_ratio
    
    # Find bounding box along each axis using slice means
    # Axis 0 (X - Sagittal slices)
    x_means = np.array([np.mean(np.abs(data[i, :, :])) for i in range(data.shape[0])])
    x_valid = x_means >= threshold
    x_indices = np.where(x_valid)[0]
    if len(x_indices) == 0:
        x_slice = slice(0, data.shape[0])
    else:
        x_slice = slice(x_indices[0], x_indices[-1] + 1)
    
    # Axis 1 (Y - Coronal slices)
    y_means = np.array([np.mean(np.abs(data[:, j, :])) for j in range(data.shape[1])])
    y_valid = y_means >= threshold
    y_indices = np.where(y_valid)[0]
    if len(y_indices) == 0:
        y_slice = slice(0, data.shape[1])
    else:
        y_slice = slice(y_indices[0], y_indices[-1] + 1)
    
    # Axis 2 (Z - Axial slices)
    z_means = np.array([np.mean(np.abs(data[:, :, k])) for k in range(data.shape[2])])
    z_valid = z_means >= threshold
    z_indices = np.where(z_valid)[0]
    if len(z_indices) == 0:
        z_slice = slice(0, data.shape[2])
    else:
        z_slice = slice(z_indices[0], z_indices[-1] + 1)
    
    return (x_slice, y_slice, z_slice)


def auto_crop_volumes(
    ref_data: np.ndarray, 
    img_data: np.ndarray,
    mean_threshold_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice, slice]]:
    """
    Auto-crop both volumes based on reference image bounding box.
    
    The bounding box is computed from the reference image using mean-based
    thresholding, and the same crop is applied to both images.
    
    Args:
        ref_data: Reference 3D volume
        img_data: Comparison 3D volume  
        mean_threshold_ratio: Ratio of volume mean for slice threshold (default 0.3 = 30%)
        
    Returns:
        Tuple of (cropped_ref, cropped_img, bounding_box_slices)
    """
    # Find bounding box from reference
    bbox = find_bounding_box(ref_data, mean_threshold_ratio=mean_threshold_ratio)
    
    # Apply same crop to both volumes
    cropped_ref = ref_data[bbox[0], bbox[1], bbox[2]]
    cropped_img = img_data[bbox[0], bbox[1], bbox[2]]
    
    return cropped_ref, cropped_img, bbox


def get_crop_info(bbox: Tuple[slice, slice, slice], original_shape: Tuple[int, ...]) -> dict:
    """
    Get human-readable crop information.
    
    Args:
        bbox: Bounding box slices
        original_shape: Original volume shape
        
    Returns:
        Dictionary with crop information
    """
    return {
        "original_shape": original_shape,
        "cropped_shape": (
            bbox[0].stop - bbox[0].start,
            bbox[1].stop - bbox[1].start,
            bbox[2].stop - bbox[2].start,
        ),
        "x_range": (bbox[0].start, bbox[0].stop),
        "y_range": (bbox[1].start, bbox[1].stop),
        "z_range": (bbox[2].start, bbox[2].stop),
    }
