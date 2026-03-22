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
    parser.add_argument("--std", type=float, help="Standard deviation of the Gaussian noise. Defaults to 0.05 if neither --std nor -s is provided.")
    parser.add_argument("-s", "--severity", type=int, choices=[1, 2, 3, 4, 5], help="Severity of noise (1-5). Overrides --std. 1: 0.01, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.09")
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
    
    print("Identifying brain region for targeted noise addition...")
    mask = create_foreground_mask(img_norm, threshold_ratio=0.1)
    
    # Optional: we can apply a slight smoothing to the mask so noise doesn't abruptly stop at the edge
    # but for simplicity, we just use the mask directly or add noise everywhere but scale it based on mask
    # To add noise mostly on the brain, we can have a base noise and a brain noise.
    # The prompt says "mostly on the brain", so we can put 100% noise on brain and 10% on background.
    print(f"Adding Gaussian noise (std={noise_std}) mostly on the brain...")
    noise = np.random.normal(loc=0.0, scale=noise_std, size=img_norm.shape)
    
    # Scale noise: 1.0 on brain, 0.1 on background
    noise_scale = np.where(mask, 1.0, 0.1)
    noise = noise * noise_scale
    
    img_noisy = img_norm + noise
    
    print("Rescaling image intensity to 0-255...")
    img_rescaled = np.clip(img_noisy * 255.0, 0, 255).astype(np.float32)
    
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
