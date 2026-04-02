"""Utility functions for NIfTI file handling."""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional


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


# ---------------------------------------------------------------------------
# SynthStrip skull-stripping (model bundled with the package)
# ---------------------------------------------------------------------------

def _build_synthstrip_model():
    """Build the SynthStrip UNet model architecture."""
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
            super().__init__()
            Conv = getattr(nn, 'Conv%dd' % ndims)
            self.conv = Conv(in_channels, out_channels, 3, stride, 1)
            if activation == 'leaky':
                self.activation = nn.LeakyReLU(0.2)
            elif activation is None:
                self.activation = None
            else:
                raise ValueError(f'Unknown activation: {activation}')

        def forward(self, x):
            out = self.conv(x)
            if self.activation is not None:
                out = self.activation(out)
            return out

    class StripModel(nn.Module):
        def __init__(self,
                     nb_features=16,
                     nb_levels=7,
                     feat_mult=2,
                     max_features=64,
                     nb_conv_per_level=2,
                     max_pool=2,
                     return_mask=False):
            super().__init__()
            ndims = 3
            if isinstance(nb_features, int):
                feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
                feats = np.clip(feats, 1, max_features)
                nb_features = [
                    np.repeat(feats[:-1], nb_conv_per_level),
                    np.repeat(np.flip(feats), nb_conv_per_level),
                ]
            enc_nf, dec_nf = nb_features
            nb_dec_convs = len(enc_nf)
            final_convs = dec_nf[nb_dec_convs:]
            dec_nf = dec_nf[:nb_dec_convs]
            self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

            if isinstance(max_pool, int):
                max_pool = [max_pool] * self.nb_levels

            MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
            self.pooling = [MaxPooling(s) for s in max_pool]
            self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

            prev_nf = 1
            encoder_nfs = [prev_nf]
            self.encoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = enc_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.encoder.append(convs)
                encoder_nfs.append(prev_nf)

            encoder_nfs = np.flip(encoder_nfs)
            self.decoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = dec_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.decoder.append(convs)
                if level < (self.nb_levels - 1):
                    prev_nf += encoder_nfs[level]

            self.remaining = nn.ModuleList()
            for num, nf in enumerate(final_convs):
                self.remaining.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            if return_mask:
                self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
                self.remaining.append(nn.Softmax(dim=1))
            else:
                self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

        def forward(self, x):
            x_history = [x]
            for level, convs in enumerate(self.encoder):
                for conv in convs:
                    x = conv(x)
                x_history.append(x)
                x = self.pooling[level](x)
            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if level < (self.nb_levels - 1):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history.pop()], dim=1)
            for conv in self.remaining:
                x = conv(x)
            return x

    return StripModel


def skull_strip(
    data: np.ndarray,
    affine: np.ndarray,
    gpu: bool = False,
    border: float = 1.0,
) -> np.ndarray:
    """
    Apply SynthStrip skull stripping to a volumetric image in memory.

    Uses the with-CSF model bundled with the package (``models/synthstrip.1.pt``).
    FreeSurfer does **not** need to be installed.

    Args:
        data:   3-D (or 4-D with frames as last axis) float32 numpy array.
        affine: 4x4 affine matrix from the NIfTI file (used to build a surfa
                Volume for conformation).
        gpu:    If True, run inference on CUDA GPU.
        border: Mask border threshold in mm (passed to SynthStrip, default 1).

    Returns:
        Skull-stripped numpy array with the same shape as *data*.
        Background (non-brain) voxels are set to 0.
    """
    try:
        import surfa as sf
        import torch
    except ImportError as e:
        raise ImportError(
            "skull_strip requires 'surfa' and 'torch'. "
            "Install with: pip install surfa torch"
        ) from e

    # ---- device ----
    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # ---- locate bundled model ----
    model_path = Path(__file__).parent / 'models' / 'synthstrip.1.pt'
    if not model_path.exists():
        raise FileNotFoundError(
            f"Bundled SynthStrip model not found at {model_path}. "
            "Please re-install the package."
        )

    # ---- build and load model ----
    StripModel = _build_synthstrip_model()
    with torch.no_grad():
        model = StripModel()
        model.to(device)
        model.eval()
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # ---- handle 3-D vs 4-D input ----
    squeeze = False
    if data.ndim == 3:
        data = data[..., np.newaxis]
        squeeze = True
    nframes = data.shape[-1]

    # ---- wrap in a surfa Volume so we can use .conform() / .resample_like() ----
    # surfa expects (X, Y, Z) data and a 4x4 affine
    sf_vol = sf.Volume(data[..., 0], geometry=sf.ImageGeometry(data.shape[:3], vox2world=affine))

    # ---- run per-frame ----
    stripped = np.zeros_like(data)
    for f in range(nframes):
        frame = sf_vol.new(data[..., f].astype(np.float32))

        # conform to isotropic 1mm, fit to multiples of 64
        conformed = frame.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA')
        conformed = conformed.crop_to_bbox()
        target_shape = np.clip(
            np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320
        )
        conformed = conformed.reshape(target_shape)

        # normalize
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)
        inp = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)

        # predict signed distance transform
        with torch.no_grad():
            sdt = model(inp).squeeze().cpu()

        # extend SDT if needed and resample back to original space
        sdt_vol = conformed.new(sdt.numpy())
        if border >= int(sdt_vol.max()):
            # recompute outer EDT for large borders
            mask_inner = sdt_vol < 1
            keep = np.nonzero(mask_inner)
            if keep[0].size > 0:
                low = np.min(keep, axis=-1)
                upp = np.max(keep, axis=-1)
                gap = int(border + 0.5)
                low = tuple(max(i - gap, 0) for i in low)
                upp = tuple(min(i + gap, d - 1) for i, d in zip(upp, mask_inner.shape))
                ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
                out = np.full_like(sdt_vol.data, fill_value=100)
                out[ind] = sf.Volume(mask_inner.data[ind]).distance()
                out[np.nonzero(mask_inner)] = sdt_vol.data[np.nonzero(mask_inner)]
                sdt_vol = sdt_vol.new(out)

        sdt_resampled = sdt_vol.resample_like(frame, fill=100)
        brain_mask = (sdt_resampled < border).connected_component_mask(k=1, fill=True)

        frame_data = data[..., f].copy()
        frame_data[np.array(brain_mask.data) == 0] = 0
        stripped[..., f] = frame_data

    if squeeze:
        stripped = stripped[..., 0]

    return stripped
