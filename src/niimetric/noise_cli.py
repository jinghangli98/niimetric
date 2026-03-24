import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .utils import load_nifti, normalize_to_range
from .metrics import create_foreground_mask

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Add noise to a NIfTI image, mostly on the brain region.")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI image")
    parser.add_argument("-o", "--output", required=True, help="Output NIfTI image")
    parser.add_argument("--std", type=float, help="Standard deviation of the Rician noise. Defaults to 0.05 if neither --std nor -s is provided.")
    parser.add_argument("-s", "--severity", type=int, choices=[1, 2, 3, 4, 5], help="Severity of noise (1-5). Overrides --std. 1: 0.01, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.09")
    parser.add_argument("--uniform", action="store_true", help="Apply uniform noise across the entire image (equivalent to --bg-ratio 1.0).")
    parser.add_argument("--bg-ratio", type=float, default=0.4, help="Ratio of background noise compared to the brain region. 0.0 = no noise outside, 1.0 = uniform noise. Default is 0.1.")
    return parser.parse_args(args)

def main(args=None):
    parsed = parse_args(args)
    
    if parsed.severity is not None:
        severity_map = {1: 0.01, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.09}
        noise_std = severity_map[parsed.severity]
    else:
        noise_std = parsed.std if parsed.std is not None else 0.05
    
    print(f"Loading image: {parsed.input}")
    try:
        img = nib.load(parsed.input)
        img_data = img.get_fdata().astype(np.float32)
    except Exception as e:
        print(f"Error loading {parsed.input}: {e}")
        return 1
        
    print("Normalizing image intensity between 0 and 1...")
    img_norm = normalize_to_range(img_data, 0, 1)
    
    bg_ratio = 1.0 if parsed.uniform else parsed.bg_ratio

    if bg_ratio >= 1.0:
        print(f"Adding uniform Rician noise (std={noise_std}) to the entire image...")
        noise_scale = 1.0
    else:
        print("Identifying brain region for targeted noise addition...")
        mask = create_foreground_mask(img_norm, threshold_ratio=0.1)
        print(f"Adding Rician noise (std={noise_std}) with {bg_ratio*100:.0f}% strength outside the brain...")
        # Scale noise: 1.0 on brain, bg_ratio on background
        noise_scale = np.where(mask, 1.0, bg_ratio)
        
    scaled_std = noise_std * noise_scale
    
    n1 = np.random.normal(loc=0.0, scale=scaled_std, size=img_norm.shape)
    n2 = np.random.normal(loc=0.0, scale=scaled_std, size=img_norm.shape)
    
    img_noisy = np.sqrt((img_norm + n1)**2 + n2**2)
    
    print("Rescaling image intensity to 0-255...")
    img_rescaled = np.clip(img_noisy * 255.0, 0, 255).astype(np.int64)
    
    print(f"Saving output to {parsed.output}...")
    # Preserve header and affine from original image
    out_img = nib.Nifti1Image(img_rescaled, img.affine, img.header)
    
    # We might need to update the data type in the header if we want it to be considered floats
    out_img.header.set_data_dtype(np.float32)
    
    nib.save(out_img, parsed.output)
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
