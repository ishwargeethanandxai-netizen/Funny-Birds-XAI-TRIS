#!/usr/bin/env python3
"""
suppressors.py - background/noise families that create suppressor variables, and the alpha blending.

Idea (XAI-TRIS): a background pixel is a suppressor if it carries no class information itself but
is statistically correlated with pixels of the object, so a model can use it to cancel noise on the
object. Spatially correlated noise creates exactly that. Two independent "knobs" per dataset:
    alpha       signal vs. noise strength   (alpha = 1 -> clean image, small alpha -> noise dominates)
    correlation how far the noise is correlated (sigma_frac, in fractions of the image size)

Additive families (x = alpha * H(signal) + (1 - alpha) * noise_image):
    white       uncorrelated Gaussian noise                          -> control, no suppressors
    corr        Gaussian-smoothed noise (sigma_frac)                 -> the XAI-TRIS CORR background
    oriented    different smoothing along x and y (sigma_x_frac, sigma_y_frac) -> stripe-like noise
    color_corr  corr noise whose RGB channels are correlated (rho)
    pink        1/f^beta noise (beta), natural-image-like spectrum
    natural     real photographs from a folder (dir), zero-centred
Multiplicative family:
    illum       smooth grey illumination field L: x = signal * (1 + (1 - alpha) * gain * L).
                With the 'default' signal the constant background reveals L, so background pixels
                are perfect suppressors of the lighting on the bird.

Extra options for additive families:
    gain          scales the noise amplitude (default 1.0)
    exclude_bird  true -> noise is removed inside the bird mask (ablation: no suppressor relation)

Noise is generated from a per-sample seed and scaled to unit standard deviation using a fixed
calibration, so alpha means the same thing for every image (dataset-level normalisation, like the
Frobenius normalisation of the paper). The noise image is  center + gain * std(signal) * noise, where center is
the signal mean, moved away from 0/1 if needed so that the noise is not clipped.
"""
import json
import os
import threading

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


# --------------------------------------------------------------------------- noise families
def _white(shape, rng, p):
    return rng.standard_normal(shape).astype(np.float32)


def _smooth_channels(eta, sigma):
    if np.all(np.asarray(sigma) <= 0):
        return eta
    out = np.empty_like(eta)
    for c in range(eta.shape[-1]):
        out[..., c] = ndi.gaussian_filter(eta[..., c], sigma=sigma, mode='reflect')
    return out


def _corr(shape, rng, p):
    return _smooth_channels(_white(shape, rng, p), float(p.get('sigma_frac', 0.04)) * shape[0])


def _oriented(shape, rng, p):
    sy = float(p.get('sigma_y_frac', 0.01)) * shape[0]
    sx = float(p.get('sigma_x_frac', 0.08)) * shape[1]
    return _smooth_channels(_white(shape, rng, p), (sy, sx))


def _color_corr(shape, rng, p):
    rho = float(p.get('rho', 0.8))
    shared = rng.standard_normal(shape[:2]).astype(np.float32)[..., None]
    indep = rng.standard_normal(shape).astype(np.float32)
    eta = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * indep
    return _smooth_channels(eta, float(p.get('sigma_frac', 0.04)) * shape[0])


def _pink(shape, rng, p):
    beta = float(p.get('beta', 1.0))
    h, w = shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = np.inf                       # remove the DC component
    amp = f ** (-beta / 2.0)
    out = np.empty(shape, dtype=np.float32)
    for c in range(shape[-1]):
        spec = np.fft.rfft2(rng.standard_normal((h, w))) * amp
        out[..., c] = np.fft.irfft2(spec, s=(h, w))
    return out


_NAT_CACHE = {}


def _natural(shape, rng, p):
    d = p['dir']
    if d not in _NAT_CACHE:
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(exts))
        if not files:
            raise FileNotFoundError(f'no background images found in {d}')
        _NAT_CACHE[d] = files
    files = _NAT_CACHE[d]
    img = Image.open(files[int(rng.integers(len(files)))]).convert('L' if p.get('gray', True) else 'RGB')
    side = min(img.size)
    left, top = (img.size[0] - side) // 2, (img.size[1] - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((shape[1], shape[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], shape[-1], axis=-1)
    return (arr - arr.mean()).astype(np.float32)


def _illum(shape, rng, p):
    field = ndi.gaussian_filter(rng.standard_normal(shape[:2]).astype(np.float32),
                                sigma=float(p.get('sigma_frac', 0.15)) * shape[0], mode='reflect')
    return np.repeat(field[..., None], shape[-1], axis=-1)


FAMILIES = {
    'white': _white, 'corr': _corr, 'oriented': _oriented, 'color_corr': _color_corr,
    'pink': _pink, 'natural': _natural, 'illum': _illum,
}
MULTIPLICATIVE = {'illum'}


def default_name(spec):
    t = spec['type']
    parts = [t]
    if 'sigma_frac' in spec and t in ('corr', 'color_corr', 'illum'):
        parts.append(f"s{round(float(spec['sigma_frac']) * 100, 3):g}")
    if t == 'oriented':
        parts.append(f"x{round(float(spec.get('sigma_x_frac', 0.08)) * 100, 3):g}"
                     f"y{round(float(spec.get('sigma_y_frac', 0.01)) * 100, 3):g}")
    if t == 'pink':
        parts.append(f"b{float(spec.get('beta', 1.0)):g}")
    if t == 'color_corr':
        parts.append(f"rho{float(spec.get('rho', 0.8)):g}")
    if spec.get('exclude_bird'):
        parts.append('nobird')
    return '_'.join(parts)


# --------------------------------------------------------------------------- noise generation
_SCALE_CACHE = {}
_LOCK = threading.Lock()


def _scale_key(spec, shape):
    keep = {k: v for k, v in spec.items() if k not in ('alphas', 'name', 'gain', 'exclude_bird')}
    return json.dumps(keep, sort_keys=True, default=str) + str(tuple(shape))


def prepare(spec, shape):
    """Calibrate the unit-std scale of a family once (fixed seed, so it is reproducible)."""
    key = _scale_key(spec, shape)
    with _LOCK:
        if key not in _SCALE_CACHE:
            rng = np.random.default_rng(12345)
            draws = [FAMILIES[spec['type']](shape, rng, spec) for _ in range(6)]
            rms = float(np.sqrt(np.mean(np.concatenate([d.ravel() for d in draws]) ** 2)))
            _SCALE_CACHE[key] = 1.0 / max(rms, 1e-8)
    return _SCALE_CACHE[key]


def make_noise(spec, shape, seed):
    """Unit-std noise (H, W, 3) for one sample; identical for every alpha of the same sample."""
    scale = prepare(spec, shape)
    rng = np.random.default_rng(int(seed))
    return (FAMILIES[spec['type']](shape, rng, spec) * scale).astype(np.float32)


# --------------------------------------------------------------------------- signal and blending
def smooth_image(img_u8, sigma_px, threshold):
    """XAI-TRIS filter H: Gaussian smoothing per channel, values below `threshold` x channel max set to 0."""
    sig = img_u8.astype(np.float32) / 255.0
    if sigma_px <= 0:
        return sig
    out = np.empty_like(sig)
    for c in range(sig.shape[-1]):
        sm = ndi.gaussian_filter(sig[..., c], sigma=float(sigma_px))
        mx = sm.max()
        if mx > 1e-6:
            sm[sm < float(threshold) * mx] = 0.0
        out[..., c] = sm
    return out


def prepare_signal(clean_u8, sig_cfg):
    """Stored clean image (uint8) -> float signal in [0,1] ready for blending.

    source 'default' / 'foreground': H is applied here to the whole image.
    source 'layered': H was already applied to the bird alone in stage 1 (generate.py), and the
                      distractors were pasted afterwards, so nothing is smoothed here.
    """
    if sig_cfg.get('source') == 'layered':
        return clean_u8.astype(np.float32) / 255.0
    return smooth_image(clean_u8, float(sig_cfg['sigma_smooth_frac']) * clean_u8.shape[0],
                        sig_cfg['threshold_support'])


def signal_stats(signals):
    """Per-channel mean and pooled std of smoothed signals (dataset-level normalisation constants)."""
    n = 0
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    for sig in signals:
        flat = sig.reshape(-1, 3).astype(np.float64)
        n += flat.shape[0]
        s1 += flat.sum(0)
        s2 += (flat ** 2).sum(0)
    mean = s1 / n
    var = s2 / n - mean ** 2
    return {'mean': mean.tolist(), 'std': float(np.sqrt(var.mean())), 'n_pixels': int(n)}


def compose(signal, spec, alpha, noise, stats, bird_mask=None):
    """Blend signal and noise for one alpha. Returns an RGB uint8 image."""
    alpha = float(alpha)
    gain = float(spec.get('gain', 0.5 if spec['type'] in MULTIPLICATIVE else 1.0))
    if spec['type'] in MULTIPLICATIVE:
        light = np.clip(1.0 + (1.0 - alpha) * gain * noise, 0.05, None)
        x = signal * light
    else:
        n = noise
        if spec.get('exclude_bird') and bird_mask is not None:
            n = n * (~bird_mask.astype(bool))[..., None]
        amp = gain * float(stats['std'])
        # centre the noise at the signal mean, but keep 3 amplitudes away from 0 and 1 so that a dark
        # signal (foreground render on black) does not get its noise cut off at 0
        center = np.clip(np.asarray(stats['mean'], dtype=np.float32), min(3 * amp, 0.5), max(1 - 3 * amp, 0.5))
        x = alpha * signal + (1.0 - alpha) * (center + amp * n)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def nominal_snr_db(spec, alpha):
    """Closed-form SNR of the additive blend (signal std == noise std at alpha = 0.5 when gain = 1)."""
    if spec['type'] in MULTIPLICATIVE:
        return None
    alpha = float(alpha)
    if alpha >= 1.0:
        return float('inf')
    gain = float(spec.get('gain', 1.0))
    return float(20.0 * np.log10(alpha / ((1.0 - alpha) * gain)))
