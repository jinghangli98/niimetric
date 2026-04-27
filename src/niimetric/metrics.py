"""Image quality metrics for NIfTI volumes."""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage import gaussian_filter as _gaussian_filter, convolve as _convolve
from typing import Optional


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ssim_components_2d(img1: np.ndarray, img2: np.ndarray,
                         data_range: float, sigma: float = 1.5):
    """Return (ssim_map, cs_map) for a 2D image pair."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = _gaussian_filter(img1, sigma=sigma, mode='reflect')
    mu2 = _gaussian_filter(img2, sigma=sigma, mode='reflect')
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sig1_sq = np.maximum(_gaussian_filter(img1 ** 2, sigma=sigma, mode='reflect') - mu1_sq, 0)
    sig2_sq = np.maximum(_gaussian_filter(img2 ** 2, sigma=sigma, mode='reflect') - mu2_sq, 0)
    sig12 = _gaussian_filter(img1 * img2, sigma=sigma, mode='reflect') - mu12

    cs_map = (2 * sig12 + C2) / (sig1_sq + sig2_sq + C2)
    ssim_map = ((2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    return ssim_map, cs_map


def _gradient_magnitude_2d(img: np.ndarray) -> np.ndarray:
    """Prewitt gradient magnitude for a 2D image."""
    hx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
    gx = _convolve(img.astype(np.float64), hx, mode='reflect')
    gy = _convolve(img.astype(np.float64), hx.T, mode='reflect')
    return np.sqrt(gx ** 2 + gy ** 2)


def _phase_congruency_2d(img: np.ndarray, n_scales: int = 4,
                          n_orientations: int = 4, min_wavelength: int = 6,
                          mult: float = 2.0, sigma_on_f: float = 0.65) -> np.ndarray:
    """
    Phase congruency via log-Gabor filters (Kovesi 1999).
    Returns a feature-strength map in [0, 1].
    """
    rows, cols = img.shape
    img = img.astype(np.float64)

    ux = np.fft.fftfreq(cols)
    uy = np.fft.fftfreq(rows)
    u, v = np.meshgrid(ux, uy)

    radius = np.sqrt(u ** 2 + v ** 2)
    radius[0, 0] = 1.0  # avoid log(0) at DC
    theta_grid = np.arctan2(v, u)

    log_sigma_sq = np.log(sigma_on_f) ** 2
    angle_sigma = np.pi / (n_orientations * 1.5)

    img_fft = np.fft.fft2(img)
    PC = np.zeros((rows, cols))

    for o in range(n_orientations):
        theta_o = o * np.pi / n_orientations
        d_theta = np.abs(np.mod(theta_grid - theta_o + np.pi / 2, np.pi) - np.pi / 2)
        angular = np.exp(-d_theta ** 2 / (2 * angle_sigma ** 2))

        E = np.zeros((rows, cols))
        O = np.zeros((rows, cols))
        A = np.zeros((rows, cols))

        for s in range(n_scales):
            f0 = 1.0 / (min_wavelength * mult ** s)
            log_gabor = np.exp(-(np.log(radius / f0)) ** 2 / (2 * log_sigma_sq))
            log_gabor[0, 0] = 0.0

            response = np.fft.ifft2(img_fft * (log_gabor * angular))
            e, o_resp = np.real(response), np.imag(response)
            E += e
            O += o_resp
            A += np.sqrt(e ** 2 + o_resp ** 2)

        PC = np.maximum(PC, np.sqrt(E ** 2 + O ** 2) / (A + 1e-5))

    return PC


def _vif_2d(ref: np.ndarray, img: np.ndarray,
            sigma_nsq: float = 2.0 / 255.0 ** 2) -> float:
    """
    Scalar VIF for a 2D image pair (Sheikh & Bovik 2006).
    Both images must be in [0, 1].
    """
    EPS = 1e-10
    ref = ref.astype(np.float64)
    img = img.astype(np.float64)

    num_total = 0.0
    den_total = 0.0
    ref_cur, img_cur = ref.copy(), img.copy()

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1   # 17, 9, 5, 3
        win_sigma = N / 5.0              # 3.4, 1.8, 1.0, 0.6

        if scale > 1:
            ref_cur = _gaussian_filter(ref_cur, sigma=win_sigma, mode='reflect')[::2, ::2]
            img_cur = _gaussian_filter(img_cur, sigma=win_sigma, mode='reflect')[::2, ::2]

        if min(ref_cur.shape) < 3:
            break

        mu1 = _gaussian_filter(ref_cur, sigma=0.6, mode='reflect')
        mu2 = _gaussian_filter(img_cur, sigma=0.6, mode='reflect')
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sig1_sq = np.maximum(
            _gaussian_filter(ref_cur ** 2, sigma=0.6, mode='reflect') - mu1_sq, 0)
        sig2_sq = np.maximum(
            _gaussian_filter(img_cur ** 2, sigma=0.6, mode='reflect') - mu2_sq, 0)
        sig12 = _gaussian_filter(ref_cur * img_cur, sigma=0.6, mode='reflect') - mu12

        g = sig12 / (sig1_sq + EPS)
        sv_sq = sig2_sq - g * sig12
        sv_sq = np.where(g < 0, sig2_sq, sv_sq)
        g = np.maximum(g, 0)
        sv_sq = np.maximum(sv_sq, EPS)

        num_total += np.sum(np.log10(1.0 + g ** 2 * sig1_sq / (sv_sq + sigma_nsq)))
        den_total += np.sum(np.log10(1.0 + sig1_sq / sigma_nsq))

    if den_total < EPS:
        return 1.0
    return float(num_total / den_total)


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


def compute_tsnr(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dim: int = 2,
) -> float:
    """
    Compute Temporal Signal-to-Noise Ratio (tSNR).

    Treats the 3D volume as a pseudo-temporal sequence of 2D slices stacked
    along ``dim``.  For every spatial position across the slice stack the
    local "temporal" mean and standard deviation are computed; tSNR is then
    the voxel-wise mean / std ratio, averaged over all (foreground) voxels.

    A high tSNR indicates low slice-to-slice noise relative to the signal
    level.  This is a no-reference metric.

    Args:
        img:  3D image volume (H × W × D or similar ordering).
        mask: Optional binary mask selecting foreground voxels.
              Applied to the 2-D spatial dimensions that are *not* ``dim``.
        dim:  Axis treated as the "temporal" (slice) direction.
              0 = sagittal, 1 = coronal, 2 = axial (default).

    Returns:
        Mean tSNR value (higher is better).
    """
    # Move the "temporal" axis to position 0 so shape is (T, *spatial)
    vol = np.moveaxis(img, dim, 0).astype(np.float64)  # (T, A, B)

    temporal_mean = vol.mean(axis=0)   # (A, B)
    temporal_std  = vol.std(axis=0)    # (A, B)

    # Avoid division by zero: only compute where std > 0
    valid = temporal_std > 0
    if mask is not None:
        # Project mask onto the two remaining spatial axes
        spatial_mask = np.moveaxis(mask, dim, 0)
        # Collapse across the temporal axis to get a 2-D spatial mask
        spatial_mask_2d = spatial_mask.any(axis=0) if spatial_mask.ndim == 3 else spatial_mask
        valid = valid & spatial_mask_2d

    if not valid.any():
        return 0.0

    tsnr_map = temporal_mean[valid] / temporal_std[valid]
    return float(tsnr_map.mean())


def compute_flickering_index(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dim: int = 2,
) -> float:
    """
    Compute Temporal Flickering Index (TFI).

    Measures slice-to-slice intensity variability as the mean absolute
    difference between consecutive slices, normalised by the overall mean
    intensity.  Lower values indicate smoother, more consistent slice
    transitions; higher values indicate "flickering" artefacts.

    This is a no-reference metric.

    Args:
        img:  3D image volume.
        mask: Optional binary mask.  Per-slice foreground pixels only are
              used when computing the mean absolute difference.
        dim:  Axis along which consecutive slices are compared.
              0 = sagittal, 1 = coronal, 2 = axial (default).

    Returns:
        Flickering index (lower is better; 0 = perfectly uniform).
    """
    vol = np.moveaxis(img, dim, 0).astype(np.float64)  # (T, A, B)
    n_slices = vol.shape[0]

    if n_slices < 2:
        return 0.0

    # Build a spatial mask projected to (A, B)
    if mask is not None:
        spatial_mask = np.moveaxis(mask, dim, 0)
        spatial_mask_2d = spatial_mask.any(axis=0) if spatial_mask.ndim == 3 else spatial_mask
    else:
        spatial_mask_2d = np.ones(vol.shape[1:], dtype=bool)

    abs_diffs = []
    for i in range(n_slices - 1):
        s1 = vol[i][spatial_mask_2d]
        s2 = vol[i + 1][spatial_mask_2d]
        if s1.size == 0:
            continue
        abs_diffs.append(np.mean(np.abs(s2 - s1)))

    if not abs_diffs:
        return 0.0

    mean_abs_diff = np.mean(abs_diffs)

    # Normalise by the global foreground mean to make the metric scale-invariant
    global_mean = vol[:, spatial_mask_2d].mean()
    if global_mean == 0:
        return 0.0

    return float(mean_abs_diff / global_mean)


def compute_ms_ssim(ref: np.ndarray, img: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    data_range: Optional[float] = None,
                    dim: int = 2) -> float:
    """
    Compute Multi-Scale Structural Similarity (MS-SSIM).

    Evaluates SSIM across 5 resolution levels for better alignment with
    human visual perception (Wang et al. 2003).

    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        data_range: Data range of the images. If None, computed from reference.
        dim: Dimension for slice-based evaluation (0=sagittal, 1=coronal, 2=axial)

    Returns:
        MS-SSIM value between 0 and 1 (higher is better)
    """
    if data_range is None:
        data_range = ref.max() - ref.min()

    _WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    def _ms_ssim_2d(r, i, m):
        min_side = min(r.shape)
        n = min(len(_WEIGHTS), max(1, int(np.log2(min_side / 11))))
        w = _WEIGHTS[:n] / _WEIGHTS[:n].sum()

        r_cur = r.astype(np.float64)
        i_cur = i.astype(np.float64)
        m_cur = m

        cs_vals = []
        for _ in range(n - 1):
            _, cs_map = _ssim_components_2d(r_cur, i_cur, data_range)
            region = cs_map[m_cur] if (m_cur is not None and m_cur.any()) else cs_map.ravel()
            cs_vals.append(region.mean())
            r_cur = _gaussian_filter(r_cur, sigma=0.5, mode='reflect')[::2, ::2]
            i_cur = _gaussian_filter(i_cur, sigma=0.5, mode='reflect')[::2, ::2]
            if m_cur is not None:
                m_cur = m_cur[::2, ::2]

        ssim_map, _ = _ssim_components_2d(r_cur, i_cur, data_range)
        region = ssim_map[m_cur] if (m_cur is not None and m_cur.any()) else ssim_map.ravel()
        cs_vals.append(region.mean())

        result = 1.0
        for j, val in enumerate(cs_vals):
            result *= np.abs(val) ** w[j]
        return float(result)

    ms_ssim_values, weights = [], []
    for i in range(ref.shape[dim]):
        if dim == 0:
            ref_s, img_s = ref[i, :, :], img[i, :, :]
            mask_s = mask[i, :, :] if mask is not None else None
        elif dim == 1:
            ref_s, img_s = ref[:, i, :], img[:, i, :]
            mask_s = mask[:, i, :] if mask is not None else None
        else:
            ref_s, img_s = ref[:, :, i], img[:, :, i]
            mask_s = mask[:, :, i] if mask is not None else None

        fg_count = int(np.sum(mask_s)) if mask_s is not None else ref_s.size
        if fg_count < 49:
            continue
        ms_ssim_values.append(_ms_ssim_2d(ref_s, img_s, mask_s))
        weights.append(fg_count)

    if not ms_ssim_values:
        return 0.0
    return float(np.average(ms_ssim_values, weights=weights))


def compute_gmsd(ref: np.ndarray, img: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 dim: int = 2) -> float:
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD).

    Uses the standard deviation of the per-pixel GMS map as a pooling
    strategy (Xue et al. 2014). Lower values indicate higher quality.

    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        dim: Dimension for slice-based evaluation

    Returns:
        GMSD value (lower is better)
    """
    c = 0.0026  # stability constant calibrated for [0, 1] images

    gmsd_values, weights = [], []
    for i in range(ref.shape[dim]):
        if dim == 0:
            ref_s, img_s = ref[i, :, :], img[i, :, :]
            mask_s = mask[i, :, :] if mask is not None else None
        elif dim == 1:
            ref_s, img_s = ref[:, i, :], img[:, i, :]
            mask_s = mask[:, i, :] if mask is not None else None
        else:
            ref_s, img_s = ref[:, :, i], img[:, :, i]
            mask_s = mask[:, :, i] if mask is not None else None

        fg_count = int(np.sum(mask_s)) if mask_s is not None else ref_s.size
        if fg_count < 49:
            continue

        gm_ref = _gradient_magnitude_2d(ref_s)
        gm_img = _gradient_magnitude_2d(img_s)
        gms = (2 * gm_ref * gm_img + c) / (gm_ref ** 2 + gm_img ** 2 + c)

        region = gms[mask_s] if mask_s is not None else gms.ravel()
        gmsd_values.append(float(np.std(region)))
        weights.append(fg_count)

    if not gmsd_values:
        return 0.0
    return float(np.average(gmsd_values, weights=weights))


def compute_vif(ref: np.ndarray, img: np.ndarray,
                mask: Optional[np.ndarray] = None,
                dim: int = 2) -> float:
    """
    Compute Visual Information Fidelity (VIF).

    Measures how much visual information from the reference is preserved
    in the distorted image using natural scene statistics (Sheikh & Bovik 2006).
    Values near 1.0 indicate no information loss; >1 possible for enhanced images.

    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        dim: Dimension for slice-based evaluation

    Returns:
        VIF value (higher is better; 1.0 = perfect fidelity)
    """
    vif_values, weights = [], []
    for i in range(ref.shape[dim]):
        if dim == 0:
            ref_s, img_s = ref[i, :, :], img[i, :, :]
            mask_s = mask[i, :, :] if mask is not None else None
        elif dim == 1:
            ref_s, img_s = ref[:, i, :], img[:, i, :]
            mask_s = mask[:, i, :] if mask is not None else None
        else:
            ref_s, img_s = ref[:, :, i], img[:, :, i]
            mask_s = mask[:, :, i] if mask is not None else None

        fg_count = int(np.sum(mask_s)) if mask_s is not None else ref_s.size
        if fg_count < 49:
            continue
        vif_values.append(_vif_2d(ref_s, img_s))
        weights.append(fg_count)

    if not vif_values:
        return 0.0
    return float(np.average(vif_values, weights=weights))


def compute_fsim(ref: np.ndarray, img: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 dim: int = 2) -> float:
    """
    Compute Feature Similarity Index (FSIM).

    Combines phase congruency and gradient magnitude similarity, giving
    strong emphasis on edges and structural details (Zhang et al. 2011).

    Args:
        ref: Reference image
        img: Comparison image
        mask: Optional binary mask for foreground regions
        dim: Dimension for slice-based evaluation

    Returns:
        FSIM value between 0 and 1 (higher is better)
    """
    T1 = 0.85                         # phase congruency stability constant
    T2 = 160.0 / (255.0 ** 2)        # GM stability constant scaled for [0, 1] input

    def _fsim_2d(ref_s, img_s, mask_s):
        pc1 = _phase_congruency_2d(ref_s)
        pc2 = _phase_congruency_2d(img_s)
        gm1 = _gradient_magnitude_2d(ref_s)
        gm2 = _gradient_magnitude_2d(img_s)

        S_PC = (2 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
        S_GM = (2 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)
        S_F = S_PC * S_GM
        PC_m = np.maximum(pc1, pc2)

        if mask_s is not None:
            num = np.sum(PC_m[mask_s] * S_F[mask_s])
            den = np.sum(PC_m[mask_s])
        else:
            num = np.sum(PC_m * S_F)
            den = np.sum(PC_m)

        return float(num / den) if den > 1e-10 else 1.0

    fsim_values, weights = [], []
    for i in range(ref.shape[dim]):
        if dim == 0:
            ref_s, img_s = ref[i, :, :], img[i, :, :]
            mask_s = mask[i, :, :] if mask is not None else None
        elif dim == 1:
            ref_s, img_s = ref[:, i, :], img[:, i, :]
            mask_s = mask[:, i, :] if mask is not None else None
        else:
            ref_s, img_s = ref[:, :, i], img[:, :, i]
            mask_s = mask[:, :, i] if mask is not None else None

        fg_count = int(np.sum(mask_s)) if mask_s is not None else ref_s.size
        if fg_count < 49:
            continue
        fsim_values.append(_fsim_2d(ref_s, img_s, mask_s))
        weights.append(fg_count)

    if not fsim_values:
        return 0.0
    return float(np.average(fsim_values, weights=weights))
