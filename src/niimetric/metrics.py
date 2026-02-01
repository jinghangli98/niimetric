"""Image quality metrics for NIfTI volumes."""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Optional


def create_foreground_mask(ref: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Create a binary mask of foreground (non-air) regions.
    
    Args:
        ref: Reference image
        threshold_ratio: Ratio of max intensity to use as threshold
        
    Returns:
        Binary mask where True = foreground
    """
    threshold = ref.max() * threshold_ratio
    return ref > threshold


def compute_psnr(ref: np.ndarray, img: np.ndarray, mask: Optional[np.ndarray] = None, 
                 data_range: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        data_range: The data range of the images. If None, computed from reference.
        
    Returns:
        PSNR value in dB
    """
    if mask is not None:
        ref = ref[mask]
        img = img[mask]
    
    if data_range is None:
        data_range = ref.max() - ref.min()
    
    # Manual PSNR calculation for masked arrays
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(ref: np.ndarray, img: np.ndarray, mask: Optional[np.ndarray] = None,
                 data_range: Optional[float] = None, dim: int = 2) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    For masked evaluation, computes SSIM per slice along the specified dimension
    and averages only over foreground pixels.
    
    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        data_range: The data range of the images. If None, computed from reference.
        dim: Dimension for slice-based evaluation (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        SSIM value between -1 and 1 (higher is better)
    """
    if data_range is None:
        data_range = ref.max() - ref.min()
    
    if mask is None:
        # Full 3D SSIM
        min_dim = min(ref.shape)
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        
        return float(structural_similarity(
            ref, img, 
            data_range=data_range,
            win_size=win_size,
            channel_axis=None
        ))
    else:
        # Masked SSIM: compute per-slice and weight by foreground pixels
        ssim_values = []
        weights = []
        
        for i in range(ref.shape[dim]):
            # Get slices along specified dimension
            if dim == 0:
                ref_slice = ref[i, :, :]
                img_slice = img[i, :, :]
                mask_slice = mask[i, :, :]
            elif dim == 1:
                ref_slice = ref[:, i, :]
                img_slice = img[:, i, :]
                mask_slice = mask[:, i, :]
            else:  # dim == 2
                ref_slice = ref[:, :, i]
                img_slice = img[:, :, i]
                mask_slice = mask[:, :, i]
            
            fg_count = np.sum(mask_slice)
            if fg_count < 49:  # Need at least 7x7 pixels for SSIM
                continue
            
            # Compute full slice SSIM and get SSIM map
            _, ssim_map = structural_similarity(
                ref_slice, img_slice,
                data_range=data_range,
                win_size=7,
                full=True
            )
            
            # Average SSIM only over foreground regions
            masked_ssim = ssim_map[mask_slice].mean()
            ssim_values.append(masked_ssim)
            weights.append(fg_count)
        
        if not ssim_values:
            return 0.0
        
        # Weighted average by foreground pixel count
        return float(np.average(ssim_values, weights=weights))


def compute_mae(ref: np.ndarray, img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        
    Returns:
        MAE value (lower is better)
    """
    if mask is not None:
        ref = ref[mask]
        img = img[mask]
    
    return float(np.mean(np.abs(ref - img)))


def compute_lpips(ref: np.ndarray, img: np.ndarray, mask: Optional[np.ndarray] = None, dim: int = 2) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    LPIPS is designed for 2D images, so for 3D volumes we compute
    the average LPIPS across all slices along the specified dimension,
    weighted by foreground content.
    
    Args:
        ref: Reference 3D volume
        img: Comparison 3D volume
        mask: Optional binary mask for foreground regions
        dim: Dimension for slice-based evaluation (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        Average LPIPS value (lower is better)
    """
    import torch
    import lpips
    
    # Initialize LPIPS model (using AlexNet by default)
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    
    # Normalize to [-1, 1] range as expected by LPIPS
    ref_min, ref_max = ref.min(), ref.max()
    if ref_max - ref_min > 0:
        ref_norm = 2 * (ref - ref_min) / (ref_max - ref_min) - 1
        img_norm = 2 * (img - ref_min) / (ref_max - ref_min) - 1
    else:
        ref_norm = np.zeros_like(ref)
        img_norm = np.zeros_like(img)
    
    # Compute LPIPS slice by slice along specified dimension
    lpips_values = []
    weights = []
    
    with torch.no_grad():
        for i in range(ref.shape[dim]):
            # Get slices along specified dimension
            if dim == 0:
                ref_slice = ref_norm[i, :, :]
                img_slice = img_norm[i, :, :]
                mask_slice = mask[i, :, :] if mask is not None else None
            elif dim == 1:
                ref_slice = ref_norm[:, i, :]
                img_slice = img_norm[:, i, :]
                mask_slice = mask[:, i, :] if mask is not None else None
            else:  # dim == 2
                ref_slice = ref_norm[:, :, i]
                img_slice = img_norm[:, :, i]
                mask_slice = mask[:, :, i] if mask is not None else None
            
            # Check if slice has enough foreground
            if mask_slice is not None:
                fg_count = np.sum(mask_slice)
                if fg_count < 100:  # Skip slices with very little foreground
                    continue
                weights.append(fg_count)
            else:
                weights.append(1)
            
            # Convert to tensor: (1, 3, H, W) - replicate grayscale to 3 channels
            ref_tensor = torch.from_numpy(ref_slice.copy()).float().unsqueeze(0).unsqueeze(0)
            ref_tensor = ref_tensor.repeat(1, 3, 1, 1).to(device)
            
            img_tensor = torch.from_numpy(img_slice.copy()).float().unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.repeat(1, 3, 1, 1).to(device)
            
            # Compute LPIPS
            lpips_val = loss_fn(ref_tensor, img_tensor)
            lpips_values.append(lpips_val.item())
    
    if not lpips_values:
        return 0.0
    
    # Weighted average by foreground pixel count
    return float(np.average(lpips_values, weights=weights))
