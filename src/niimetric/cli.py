"""Command-line interface for niimetric."""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import nibabel as nib
import numpy as np

from .utils import load_nifti, validate_shapes, normalize_to_range
from .cropping import auto_crop_volumes, get_crop_info
from .metrics import (
    compute_ssim,
    compute_psnr,
    compute_mae,
    compute_lpips,
    compute_tsnr,
    compute_flickering_index,
    create_foreground_mask,
)


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="niimetric",
        description="Evaluate image quality metrics for NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  niimetric -a reference.nii.gz -b image1.nii.gz --ssim -o output.csv
  niimetric -a reference.nii.gz -b image1.nii.gz --all -o results.csv
  niimetric -a reference.nii.gz -b image1.nii.gz --all --dim 0 -o results.csv  # sagittal slices
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-a", "--reference",
        required=False,
        default=None,
        help="Path to reference NIfTI image (used for cropping boundaries). "
             "Not needed for --tsnr / --flickering."
    )
    parser.add_argument(
        "-b", "--image",
        required=True,
        help="Path to comparison NIfTI image"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output CSV file"
    )
    
    # Metric selection
    metric_group = parser.add_argument_group("Metrics")
    metric_group.add_argument(
        "--ssim",
        action="store_true",
        help="Calculate Structural Similarity Index (SSIM)"
    )
    metric_group.add_argument(
        "--psnr",
        action="store_true",
        help="Calculate Peak Signal-to-Noise Ratio (PSNR)"
    )
    metric_group.add_argument(
        "--mae",
        action="store_true",
        help="Calculate Mean Absolute Error (MAE)"
    )
    metric_group.add_argument(
        "--lpips",
        action="store_true",
        help="Calculate Learned Perceptual Image Patch Similarity (LPIPS)"
    )
    metric_group.add_argument(
        "--tsnr",
        action="store_true",
        help="Calculate Temporal SNR (no reference required)"
    )
    metric_group.add_argument(
        "--flickering",
        action="store_true",
        help="Calculate Temporal Flickering Index (no reference required)"
    )
    metric_group.add_argument(
        "--all",
        action="store_true",
        help="Calculate all metrics"
    )
    
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Evaluate metrics only on foreground regions (auto-cropped and masked)"
    )
    
    # Optional outputs
    parser.add_argument(
        "--save-mask",
        type=str,
        default=None,
        help="Path to save the foreground mask as NIfTI file (only used with --foreground)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Dimension for slice-based evaluation: 0=sagittal, 1=coronal, 2=axial (default: 2)"
    )
    
    return parser.parse_args(args)


def compute_metrics(
    img_cropped: np.ndarray,
    ref_cropped: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    dim: int = 2,
    ssim: bool = False,
    psnr: bool = False,
    mae: bool = False,
    lpips: bool = False,
    tsnr: bool = False,
    flickering: bool = False,
    all_metrics: bool = False,
) -> List[Tuple[str, float]]:
    """
    Compute requested metrics on cropped volumes.

    Reference-based metrics (PSNR, SSIM, MAE, LPIPS) require ``ref_cropped``.
    No-reference metrics (tSNR, flickering) operate on ``img_cropped`` alone.

    Args:
        img_cropped:  Comparison / generated volume.
        ref_cropped:  Reference volume (required for PSNR/SSIM/MAE/LPIPS).
        mask:         Optional binary mask for foreground-only evaluation.
        dim:          Dimension for slice-based evaluation.

    Returns:
        List of (metric_name, value) tuples.
    """
    dim_names = {0: "sagittal", 1: "coronal", 2: "axial"}
    results = []

    # --- Reference-required metrics ---
    if (psnr or all_metrics) and ref_cropped is not None:
        print("  Computing PSNR...", flush=True)
        value = compute_psnr(ref_cropped, img_cropped, mask=mask)
        results.append(("PSNR", value))
        print(f"    PSNR: {value:.4f} dB")

    if (ssim or all_metrics) and ref_cropped is not None:
        print(f"  Computing SSIM (on {dim_names[dim]} slices)...", flush=True)
        value = compute_ssim(ref_cropped, img_cropped, mask=mask, dim=dim)
        results.append(("SSIM", value))
        print(f"    SSIM: {value:.4f}")

    if (mae or all_metrics) and ref_cropped is not None:
        print("  Computing MAE...", flush=True)
        value = compute_mae(ref_cropped, img_cropped, mask=mask)
        results.append(("MAE", value))
        print(f"    MAE: {value:.4f}")

    if (lpips or all_metrics) and ref_cropped is not None:
        print(f"  Computing LPIPS (on {dim_names[dim]} slices, this may take a while)...", flush=True)
        value = compute_lpips(ref_cropped, img_cropped, mask=mask, dim=dim)
        results.append(("LPIPS", value))
        print(f"    LPIPS: {value:.4f}")

    # --- No-reference temporal metrics ---
    if tsnr or all_metrics:
        print(f"  Computing tSNR (across {dim_names[dim]} slices)...", flush=True)
        value = compute_tsnr(img_cropped, mask=mask, dim=dim)
        results.append(("tSNR", value))
        print(f"    tSNR: {value:.4f}")

    if flickering or all_metrics:
        print(f"  Computing Flickering Index (across {dim_names[dim]} slices)...", flush=True)
        value = compute_flickering_index(img_cropped, mask=mask, dim=dim)
        results.append(("FlickeringIndex", value))
        print(f"    FlickeringIndex: {value:.4f}")

    return results


def write_csv(
    output_path: str,
    reference: str,
    image: str,
    results: List[Tuple[str, float]]
) -> None:
    """Write results to CSV file."""
    path = Path(output_path)
    file_exists = path.exists()
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(["reference", "image", "metric", "value"])
        
        # Write results
        ref_name = Path(reference).name
        img_name = Path(image).name
        for metric_name, value in results:
            writer.writerow([ref_name, img_name, metric_name, f"{value:.6f}"])


def main(args: List[str] = None) -> int:
    """Main entry point for the CLI."""
    parsed = parse_args(args)

    # Identify which metrics were requested
    ref_metrics_requested = any([parsed.ssim, parsed.psnr, parsed.mae, parsed.lpips])
    noref_metrics_requested = any([parsed.tsnr, parsed.flickering])

    # Check that at least one metric is selected
    if not any([parsed.ssim, parsed.psnr, parsed.mae, parsed.lpips,
                parsed.tsnr, parsed.flickering, parsed.all]):
        print(
            "Error: Please specify at least one metric "
            "(--ssim, --psnr, --mae, --lpips, --tsnr, --flickering, or --all)"
        )
        return 1

    # Reference is required for reference-based metrics
    if (ref_metrics_requested or parsed.all) and parsed.reference is None:
        print("Error: -a/--reference is required for SSIM, PSNR, MAE, and LPIPS metrics.")
        return 1
    
    try:
        # Load image under evaluation
        print(f"Loading image: {parsed.image}")
        img_data = load_nifti(parsed.image)
        print(f"  Shape: {img_data.shape}")

        # Load reference only when needed
        ref_data = None
        if parsed.reference is not None:
            print(f"Loading reference: {parsed.reference}")
            ref_data = load_nifti(parsed.reference)
            print(f"  Shape: {ref_data.shape}")
            validate_shapes(ref_data, img_data)

        ref_final = ref_data
        img_final = img_data
        mask = None
        
        if parsed.foreground:
            if ref_data is None:
                print("Warning: --foreground requires a reference image (-a). Skipping cropping.")
            else:
                # Auto-crop based on reference
                print("Auto-cropping volumes based on reference...")
                ref_final, img_final, bbox = auto_crop_volumes(ref_data, img_data)
                crop_info = get_crop_info(bbox, ref_data.shape)
                print(f"  Original shape: {crop_info['original_shape']}")
                print(f"  Cropped shape: {crop_info['cropped_shape']}")
                print(f"  X range: {crop_info['x_range']}")
                print(f"  Y range: {crop_info['y_range']}")
                print(f"  Z range: {crop_info['z_range']}")
        else:
            print("Using full volumes (no cropping)...")
        
        # Normalize both images to 0-1 range
        print("Normalizing image to 0-1 range...")
        img_normalized = normalize_to_range(img_final, 0, 1)
        print(f"  Image range: [{img_final.min():.2f}, {img_final.max():.2f}] -> [0, 1]")
        ref_normalized = None
        if ref_final is not None:
            ref_normalized = normalize_to_range(ref_final, 0, 1)
            print(f"  Reference range: [{ref_final.min():.2f}, {ref_final.max():.2f}] -> [0, 1]")
        
        if parsed.foreground and ref_normalized is not None:
            # Create foreground mask (non-air regions based on reference)
            print("Creating foreground mask (excluding air regions)...")
            mask = create_foreground_mask(ref_normalized, threshold_ratio=0.1)
            fg_percentage = 100 * mask.sum() / mask.size
            print(f"  Foreground pixels: {mask.sum():,} / {mask.size:,} ({fg_percentage:.1f}%)")

            # Save mask if requested
            if parsed.save_mask:
                print(f"Saving mask to: {parsed.save_mask}")
                ref_nii = nib.load(parsed.reference)
                mask_nii = nib.Nifti1Image(mask.astype(np.uint8), ref_nii.affine, ref_nii.header)
                nib.save(mask_nii, parsed.save_mask)
                print(f"  Mask saved successfully")
        
        # Compute metrics
        dim_names = {0: "sagittal", 1: "coronal", 2: "axial"}
        if parsed.foreground:
            print(f"Computing metrics on foreground regions (slicing on {dim_names[parsed.dim]} dimension)...")
        else:
            print(f"Computing metrics on full volume (slicing on {dim_names[parsed.dim]} dimension)...")
            
        results = compute_metrics(
            img_normalized,
            ref_cropped=ref_normalized,
            mask=mask,
            dim=parsed.dim,
            ssim=parsed.ssim,
            psnr=parsed.psnr,
            mae=parsed.mae,
            lpips=parsed.lpips,
            tsnr=parsed.tsnr,
            flickering=parsed.flickering,
            all_metrics=parsed.all,
        )

        # Write to CSV
        ref_label = parsed.reference if parsed.reference else "(none)"
        write_csv(parsed.output, ref_label, parsed.image, results)
        print(f"\nResults written to: {parsed.output}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
