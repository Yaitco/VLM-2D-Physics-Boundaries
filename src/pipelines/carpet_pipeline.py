from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.models.grounding_dino import GroundingDinoAdapter
from src.models.sam_wrapper import SamWrapper


@dataclass(slots=True)
class CarpetPrediction:
    prompt: str
    method_name: str
    mask: np.ndarray
    dino_score: float | None
    sam_score: float | None
    inference_time: float
    box_xyxy: np.ndarray | None


class CarpetPipeline:
    """Grounding DINO text-to-box followed by SAM box-to-mask."""

    def __init__(self, grounding_dino: GroundingDinoAdapter, sam_wrapper: SamWrapper) -> None:
        self.grounding_dino = grounding_dino
        self.sam_wrapper = sam_wrapper

    def run_prompt(self, image: Image.Image, prompt: str) -> CarpetPrediction:
        start_time = time.perf_counter()
        detections = self.grounding_dino.detect(image, prompt)
        if not detections:
            elapsed = time.perf_counter() - start_time
            empty_mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
            return CarpetPrediction(
                prompt=prompt,
                method_name=_slug(prompt),
                mask=empty_mask,
                dino_score=None,
                sam_score=None,
                inference_time=elapsed,
                box_xyxy=None,
            )

        best_detection = detections[0]
        sam_result = self.sam_wrapper.predict_from_box(image, best_detection.box_xyxy)
        elapsed = time.perf_counter() - start_time
        return CarpetPrediction(
            prompt=prompt,
            method_name=_slug(prompt),
            mask=sam_result.mask,
            dino_score=best_detection.score,
            sam_score=sam_result.score,
            inference_time=elapsed,
            box_xyxy=best_detection.box_xyxy.copy(),
        )


def _slug(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "_")
