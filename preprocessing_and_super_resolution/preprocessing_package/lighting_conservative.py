"""Bounded illumination correction for OCR; no denoise, stretch or CLAHE.

The same gain multiplies all three colour channels. A smooth field removes
some brightness variation without turning board texture into hard edges.
This is a candidate operator, not evidence that every dim photo benefits.
"""
from __future__ import annotations
import cv2
import numpy as np
from . import fix_lighting as legacy


def fix(image, strength='light', **kwargs):
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError('Expected a nonempty uint8 image')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Expected grayscale or three-channel BGR')
    small = legacy._downscale(image)
    gray = legacy._luma(small)
    field = legacy._background_field(gray, 'dark_ink')
    target = float(np.clip(np.median(field), 170., 235.))
    full = cv2.resize(field, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_LINEAR)
    gain = np.clip(1. + .5*(target/np.maximum(full,1.)-1.), .85, 1.35)
    # Do not clip a colour channel when lifting brightness; preserve its ratio
    # to the others. Already bright/clipped pixels are not reconstructed.
    peak = image.max(axis=2) if image.ndim == 3 else image
    headroom = np.maximum(1.,250./np.maximum(peak.astype(np.float32),1.))
    gain = np.minimum(gain,headroom)
    result = np.clip(np.rint(image.astype(np.float32) *
                            (gain[:,:,None] if image.ndim == 3 else gain)),0,255).astype(np.uint8)
    return result, dict(method='bounded_shared_channel_flat_field',target=target,
                        min_gain=float(gain.min()),max_gain=float(gain.max()),
                        actions=['partial illumination field only'])
