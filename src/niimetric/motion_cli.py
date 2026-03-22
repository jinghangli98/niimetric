import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio

from .utils import normalize_to_range

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Add random motion artifacts to a NIfTI image using TorchIO.")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI image")
    parser.add_argument("-o", "--output", required=True, help="Output NIfTI image")
    parser.add_argument("-s", "--severity", type=int, choices=[1, 2, 3, 4, 5], default=3, help="Severity of motion (1-5). Default is 3.")
    return parser.parse_args(args)

def main(args=None):
    parsed = parse_args(args)
    
    severity_map = {
        1: {'degrees': 1, 'translation': 1},
        2: {'degrees': 2, 'translation': 2},
        3: {'degrees': 3, 'translation': 3},
        4: {'degrees': 4, 'translation': 4},
        5: {'degrees': 5, 'translation': 5}
    }
    params = severity_map[parsed.severity]
    
    print(f"Loading image: {parsed.input}")
    try:
        img = nib.load(parsed.input)
        img_data = img.get_fdata().astype(np.float32)
    except Exception as e:
        print(f"Error loading {parsed.input}: {e}")
        return 1
        
    print("Normalizing image intensity between 0 and 1...")
    img_norm = normalize_to_range(img_data, 0, 1)
    
    print(f"Applying TorchIO RandomMotion (severity={parsed.severity}, params={params})...")
    # TorchIO requires (C, W, H, D)
    tensor = img_norm[np.newaxis, ...]
    subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor, affine=img.affine))
    
    transform = tio.RandomMotion(**params)
    transformed_subject = transform(subject)
    
    img_augmented = transformed_subject['t1'].data.squeeze(0).numpy()
    
    print("Rescaling image intensity to 0-255...")
    img_rescaled = np.clip(img_augmented * 255.0, 0, 255).astype(np.float32)
    
    print(f"Saving output to {parsed.output}...")
    out_img = nib.Nifti1Image(img_rescaled, img.affine, img.header)
    out_img.header.set_data_dtype(np.float32)
    
    nib.save(out_img, parsed.output)
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
