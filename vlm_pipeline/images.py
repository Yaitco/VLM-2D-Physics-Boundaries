from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def apply_mask_to_image(image: Image.Image, mask: Image.Image, bg_mode: str = "black") -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    out = rgb.copy()
    out[~m] = 255 if bg_mode == "white" else 0
    return Image.fromarray(out, mode="RGB")


def apply_mask_overlay_to_image(
    image: Image.Image,
    mask: Image.Image,
    tint_rgb: Tuple[int, int, int] = (255, 96, 96),
    tint_alpha: float = 0.22,
    boundary_rgb: Tuple[int, int, int] = (255, 32, 32),
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    out = rgb.astype(np.float32)
    if m.any():
        tint = np.asarray(tint_rgb, dtype=np.float32)
        out[m] = (1.0 - tint_alpha) * out[m] + tint_alpha * tint
        up = np.zeros_like(m)
        down = np.zeros_like(m)
        left = np.zeros_like(m)
        right = np.zeros_like(m)
        up[1:] = m[:-1]
        down[:-1] = m[1:]
        left[:, 1:] = m[:, :-1]
        right[:, :-1] = m[:, 1:]
        interior = m & up & down & left & right
        boundary = m & (~interior)
        out[boundary] = np.asarray(boundary_rgb, dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def load_variant_image(sample_meta: Dict[str, Any], variant: str, mask_background_mode: str) -> Image.Image:
    image = Image.open(Path(sample_meta["path"])).convert("RGB")
    if variant == "raw":
        return image
    mask_path = sample_meta.get("mask_path")
    if not mask_path:
        raise FileNotFoundError("mask_path is missing in sample metadata")
    mask = Image.open(Path(mask_path)).convert("L")
    if variant == "masked":
        return apply_mask_to_image(image, mask, bg_mode=mask_background_mode)
    if variant == "mask_overlay":
        return apply_mask_overlay_to_image(image, mask)
    raise ValueError(f"Unknown variant: {variant}")


def load_variant_image_cached(
    sample_meta: Dict[str, Any],
    variant: str,
    mask_background_mode: str,
    image_cache: Optional[Dict[Tuple[str, str, str, str], Image.Image]] = None,
) -> Image.Image:
    if image_cache is None:
        return load_variant_image(sample_meta, variant, mask_background_mode)
    key = (
        str(sample_meta.get("path") or ""),
        str(sample_meta.get("mask_path") or ""),
        variant,
        mask_background_mode,
    )
    cached = image_cache.get(key)
    if cached is not None:
        return cached
    image = load_variant_image(sample_meta, variant, mask_background_mode)
    image_cache[key] = image
    return image
