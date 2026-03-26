import argparse
import sys
import os
import nibabel as nib
import numpy as np
from .utils import normalize_to_range
from .mri_sim import customgauss, sensitivity_map, simulate_grappa

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Simulate GRAPPA reconstruction artifacts on a NIfTI image.")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI image")
    parser.add_argument("-o", "--output", required=True, help="Output NIfTI image")
    parser.add_argument("-s", "--severity", type=int, choices=[1, 2, 3, 4, 5], default=3, help="Severity of noise (1-5). Default is 3.")
    parser.add_argument("--noise", type=float, help="Manual noise level. Overrides --severity.")
    parser.add_argument("--coils", type=int, default=32, help="Number of coils. Default is 32.")
    parser.add_argument("--accel", type=int, default=2, help="Acceleration factor. Default is 2.")
    parser.add_argument("--acs", type=int, default=32, help="Number of ACS lines. Default is 32.")
    parser.add_argument("--dir", choices=["RL", "AP"], default="RL", help="Phase encoding direction. Default is RL (1st axis).")
    return parser.parse_args(args)

def main(args=None):
    parsed = parse_args(args)
    
    if not os.path.exists(parsed.input):
        print(f"Error: Input file {parsed.input} not found.")
        return 1

    print(f"Loading {parsed.input}...")
    nii = nib.load(parsed.input)
    data = nii.get_fdata()
    header = nii.header
    affine = nii.affine
    
    # Normalization
    max_grayscale = 255.0
    d_min, d_max = np.min(data), np.max(data)
    data_norm = (data - d_min) / (d_max - d_min) * max_grayscale
    
    X, Y, Z = data_norm.shape[:3]
    
    # Noise attenuation map
    g_size = [X, Z]
    sig = X // 2
    sm1_z_2d = customgauss(g_size, sig, sig, 0.0, 0.5, 0.5, [1, 1])
    sigma_attenuation = sm1_z_2d[X // 2, :]
    
    # Determine noise level based on severity (1-5)
    if parsed.noise is not None:
        noise_level = parsed.noise
    else:
        # Range-based random noise: 0 to (severity/5)*0.15
        max_for_severity = (parsed.severity / 5.0) * 0.15
        noise_level = np.random.uniform(0, max_for_severity)
            
    print(f"Simulating GRAPPA (severity={parsed.severity}, target noise={noise_level*100:.2f}%, accel={parsed.accel}, acs={parsed.acs})...")
    
    sigma_base = noise_level * max_grayscale
    rho = 0.05
    
    # We will generate sensitivity maps INSIDE the loop for Z-variation
    
    processed_data = np.zeros_like(data_norm)
    
    # Pre-calculated g-factor for noise adjustment
    print("Processing slices with G-factor adaptive noise...")
    # Optimization: Calculate g-avg once using a middle slice
    print("Estimating G-factor for adaptive noise adjustment...")
    z_mid = Z // 2
    map_w_mid = sensitivity_map((Y, X) if parsed.dir == "AP" else (X, Y), parsed.coils, z=z_mid, nz=Z)
    slice_mid = data_norm[:, :, z_mid]
    if parsed.dir == "AP": slice_mid = slice_mid.T
    
    _, g_map_mid = simulate_grappa(slice_mid, parsed.coils, sigma_base, rho, parsed.accel, parsed.acs, map_w_mid)
    g_avg_volume = np.mean(g_map_mid)
    R = parsed.accel
    
    print(f"Volume G-factor estimate: {g_avg_volume:.2f}. Processing slices...")
    
    for z in range(Z):
        sigma_z_target = sigma_base * sigma_attenuation[z]
        sigma_z_adj = sigma_z_target / (g_avg_volume * np.sqrt(R))
        
        # Generate sensitivity map with Z-variation
        if parsed.dir == "AP":
            map_w = sensitivity_map((Y, X), parsed.coils, z=z, nz=Z)
            slice_2d = data_norm[:, :, z].T
            final_recon, _ = simulate_grappa(slice_2d, parsed.coils, sigma_z_adj, rho, parsed.accel, parsed.acs, map_w)
            processed_data[:, :, z] = final_recon.T
        else:
            map_w = sensitivity_map((X, Y), parsed.coils, z=z, nz=Z)
            slice_2d = data_norm[:, :, z]
            final_recon, _ = simulate_grappa(slice_2d, parsed.coils, sigma_z_adj, rho, parsed.accel, parsed.acs, map_w)
            processed_data[:, :, z] = final_recon
            
        if (z + 1) % 50 == 0:
            print(f"  Processed {z+1}/{Z} slices...")

    # Save reconstructed image
    out_img = nib.Nifti1Image(processed_data.astype(np.int16), affine, header)
    out_img.header.set_data_dtype(np.int16)
    nib.save(out_img, parsed.output)
    
    print(f"Saved reconstructed image to {parsed.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
