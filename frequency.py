"""Low-level frequency feature extraction for the hybrid detector.

The hybrid model's low-level branch has two sub-streams: a *spatial* stream
(high-pass residual, computed inside the model) and a *frequency* stream fed by
the per-patch FFT spectrogram produced here.

Why per-patch instead of a single whole-image FFT:
  * Generator artifacts (upsampling grids, periodic checkerboard patterns,
    denoiser residuals) are *local* and roughly stationary within a small
    window. A per-patch transform keeps that locality -- a 224x224 FFT would
    blur every patch's signature into one global spectrum.
  * Laying the patch spectra back out on a grid gives a single-channel
    "spectrogram image" that an ordinary small CNN can consume, and lets that
    CNN learn how the spectral signature varies across the frame.

The transform is deliberately cheap (grid*grid small FFTs, all NumPy) so it can
run in DataLoader worker processes without becoming the training bottleneck, and
so we never depend on FFT support on the training accelerator (historically
flaky on Apple MPS).
"""

import numpy as np
import torch


def compute_freq_spectrogram(rgb01, grid=7):
    """Per-patch FFT log-magnitude spectrogram of an image.

    Args:
        rgb01: float tensor/array shaped (3, H, W) with values in [0, 1]
               (an un-normalised image, e.g. straight out of ToTensor()).
        grid:  the image is split into a grid x grid array of equal patches;
               each patch gets its own 2D FFT. H and W should be divisible by
               `grid` (224 / 7 = 32); any remainder is left as a zero border.

    Returns:
        torch.FloatTensor shaped (1, H, W): each patch cell holds
        log(1 + |fftshift(FFT2(patch - mean))|). No per-patch normalisation is
        applied -- absolute spectral energy differs between real and generated
        images and is a useful signal; the frequency CNN's input BatchNorm
        handles global scaling.
    """
    if torch.is_tensor(rgb01):
        arr = rgb01.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb01, dtype=np.float32)

    # Work on luminance: spectral generator artifacts are largely achromatic,
    # and one channel keeps the FFT cost (and the CNN) small.
    lum = arr.mean(axis=0).astype(np.float32)
    h, w = lum.shape
    ph, pw = h // grid, w // grid

    spec = np.zeros((h, w), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            y0, x0 = i * ph, j * pw
            patch = lum[y0:y0 + ph, x0:x0 + pw]
            # Remove the DC term so the (0,0) bin doesn't dominate the log-mag.
            patch = patch - float(patch.mean())
            mag = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
            spec[y0:y0 + ph, x0:x0 + pw] = np.log1p(mag).astype(np.float32)

    return torch.from_numpy(spec).unsqueeze(0)


def stack_image_and_spectrogram(rgb01, grid=7):
    """Concatenate the [0,1] RGB image with its spectrogram along channel 0.

    Produces the (4, H, W) tensor the hybrid dataset hands to the DataLoader:
    channels 0-2 are the un-normalised RGB image (the model applies CLIP's own
    normalisation and the high-pass residual), channel 3 is the FFT
    spectrogram. Packing it this way keeps every `for x, y in loader` /
    `model(x)` call site in the codebase unchanged.
    """
    if not torch.is_tensor(rgb01):
        rgb01 = torch.as_tensor(np.asarray(rgb01, dtype=np.float32))
    spec = compute_freq_spectrogram(rgb01, grid=grid)
    return torch.cat([rgb01, spec], dim=0)
