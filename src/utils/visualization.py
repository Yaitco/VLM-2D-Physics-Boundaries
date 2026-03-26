from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(slots=True)
class CollagePanel:
    title: str
    image: Image.Image
    lines: list[str] = field(default_factory=list)


def render_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.35,
    tint_rgb: tuple[int, int, int] = (255, 88, 88),
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    binary = mask.astype(bool)
    if binary.any():
        tint = np.asarray(tint_rgb, dtype=np.float32)
        rgb[binary] = ((1.0 - alpha) * rgb[binary]) + (alpha * tint)
        boundary = _extract_boundary(binary)
        rgb[boundary] = np.asarray((255, 255, 255), dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def render_binary_mask(mask: np.ndarray) -> Image.Image:
    array = mask.astype(np.uint8) * 255
    stacked = np.stack([array, array, array], axis=-1)
    return Image.fromarray(stacked, mode="RGB")


def save_collage(
    panels: list[CollagePanel],
    output_path: Path,
    panel_width: int = 420,
    panel_height: int = 320,
    columns: int = 3,
) -> None:
    rows = int(math.ceil(len(panels) / max(1, columns)))
    canvas = Image.new("RGB", (panel_width * columns, panel_height * rows), color=(245, 245, 245))

    for index, panel in enumerate(panels):
        rendered = _render_panel(panel, panel_width, panel_height)
        row = index // columns
        col = index % columns
        canvas.paste(rendered, (col * panel_width, row * panel_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.expand(canvas, border=8, fill=(220, 220, 220)).save(output_path)


def _render_panel(panel: CollagePanel, width: int, height: int) -> Image.Image:
    margin = 12
    text_height = 88
    image_height = max(60, height - text_height - (2 * margin))
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    fitted = ImageOps.contain(panel.image.convert("RGB"), (width - (2 * margin), image_height))
    canvas.paste(fitted, ((width - fitted.width) // 2, margin))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, height - text_height, width, height), fill=(248, 248, 248), outline=(220, 220, 220))
    draw.text((margin, height - text_height + 8), panel.title, fill=(20, 20, 20), font=font)
    if panel.lines:
        draw.multiline_text(
            (margin, height - text_height + 28),
            "\n".join(panel.lines[:4]),
            fill=(60, 60, 60),
            font=font,
            spacing=2,
        )
    return canvas


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:] = mask[:-1]
    down[:-1] = mask[1:]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    interior = mask & up & down & left & right
    return mask & (~interior)
