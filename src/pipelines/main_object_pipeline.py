from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from src.models.sam_wrapper import SamWrapper


@dataclass(slots=True)
class MainObjectPrediction:
    mask: np.ndarray
    sam_score: float | None
    inference_time: float
    candidate_count: int
    heuristic_score: float | None


class MainObjectPipeline:
    """Select the most plausible primary object from SAM automatic masks."""

    def __init__(
        self,
        sam_wrapper: SamWrapper,
        area_weight: float = 0.60,
        center_weight: float = 0.30,
        border_weight: float = 0.20,
    ) -> None:
        self.sam_wrapper = sam_wrapper
        self.area_weight = area_weight
        self.center_weight = center_weight
        self.border_weight = border_weight

    def run(self, image: Image.Image) -> MainObjectPrediction:
        start_time = time.perf_counter()
        annotations = self.sam_wrapper.generate_masks(image)
        elapsed = time.perf_counter() - start_time
        if not annotations:
            height, width = image.size[1], image.size[0]
            return MainObjectPrediction(
                mask=np.zeros((height, width), dtype=bool),
                sam_score=None,
                inference_time=elapsed,
                candidate_count=0,
                heuristic_score=None,
            )

        height, width = image.size[1], image.size[0]
        best_mask = np.zeros((height, width), dtype=bool)
        best_score = -float("inf")
        best_sam_score: float | None = None

        for annotation in annotations:
            mask = np.asarray(annotation.get("segmentation"), dtype=bool)
            score = self._score_mask(mask)
            predicted_iou = annotation.get("predicted_iou")
            if predicted_iou is not None:
                score += 0.05 * float(predicted_iou)
            if score > best_score:
                best_score = score
                best_mask = mask
                best_sam_score = float(predicted_iou) if predicted_iou is not None else None

        return MainObjectPrediction(
            mask=best_mask,
            sam_score=best_sam_score,
            inference_time=elapsed,
            candidate_count=len(annotations),
            heuristic_score=best_score,
        )

    def _score_mask(self, mask: np.ndarray) -> float:
        if mask.size == 0 or not mask.any():
            return -1e9

        height, width = mask.shape
        area_ratio = float(mask.mean())
        cy, cx = self._mask_centroid(mask)
        image_center_y = (height - 1) / 2.0
        image_center_x = (width - 1) / 2.0
        distance = math.sqrt(((cy - image_center_y) ** 2) + ((cx - image_center_x) ** 2))
        max_distance = math.sqrt((image_center_y ** 2) + (image_center_x ** 2)) or 1.0
        center_score = 1.0 - (distance / max_distance)
        border_penalty = self._border_touch_ratio(mask)
        return (
            (self.area_weight * area_ratio)
            + (self.center_weight * center_score)
            - (self.border_weight * border_penalty)
        )

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            height, width = mask.shape
            return (height / 2.0, width / 2.0)
        return float(np.mean(ys)), float(np.mean(xs))

    @staticmethod
    def _border_touch_ratio(mask: np.ndarray, border_width: int = 4) -> float:
        if mask.size == 0:
            return 0.0
        border = np.zeros_like(mask, dtype=bool)
        border[:border_width, :] = True
        border[-border_width:, :] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True
        border_pixels = int(np.logical_and(mask, border).sum())
        mask_pixels = int(mask.sum())
        if mask_pixels == 0:
            return 0.0
        return float(border_pixels / mask_pixels)
