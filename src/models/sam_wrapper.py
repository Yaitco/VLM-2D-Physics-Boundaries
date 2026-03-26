from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class SamBoxResult:
    mask: np.ndarray
    score: float | None
    mask_index: int | None
    all_scores: list[float]


class SamWrapper:
    """Adapter around the official Segment Anything package."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str,
        device: str,
        automatic_mask_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.device = device
        self.automatic_mask_kwargs = automatic_mask_kwargs or {}
        self.model: Any | None = None
        self.predictor: Any | None = None
        self.automatic_mask_generator: Any | None = None
        self._current_image_key: tuple[int, int, int] | None = None

    def load(self) -> None:
        if self.model is not None and self.predictor is not None and self.automatic_mask_generator is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint was not found: {self.checkpoint_path}. "
                "Download a SAM checkpoint and pass --sam_checkpoint or place it in checkpoints/."
            )

        self._ensure_local_package_paths()
        try:
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError(
                "segment-anything is not installed. Install it from the official repository or clone it locally."
            ) from exc

        model_constructor = sam_model_registry.get(self.model_type)
        if model_constructor is None:
            available = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(f"Unsupported SAM model type '{self.model_type}'. Available: {available}")

        self.model = model_constructor(checkpoint=str(self.checkpoint_path))
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        self.automatic_mask_generator = SamAutomaticMaskGenerator(model=self.model, **self.automatic_mask_kwargs)

    def set_image(self, image: Image.Image | np.ndarray) -> None:
        self.load()
        assert self.predictor is not None
        image_array = _to_rgb_array(image)
        image_key = (id(image), image_array.shape[0], image_array.shape[1])
        if self._current_image_key == image_key:
            return
        self.predictor.set_image(image_array)
        self._current_image_key = image_key

    def predict_from_box(self, image: Image.Image | np.ndarray, box_xyxy: np.ndarray) -> SamBoxResult:
        self.load()
        assert self.predictor is not None
        self.set_image(image)

        masks, scores, _ = self.predictor.predict(
            box=np.asarray(box_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        if len(masks) == 0:
            height, width = _to_rgb_array(image).shape[:2]
            return SamBoxResult(
                mask=np.zeros((height, width), dtype=bool),
                score=None,
                mask_index=None,
                all_scores=[],
            )

        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        best_index = int(np.argmax(scores_array))
        return SamBoxResult(
            mask=np.asarray(masks[best_index], dtype=bool),
            score=float(scores_array[best_index]),
            mask_index=best_index,
            all_scores=[float(value) for value in scores_array.tolist()],
        )

    def generate_masks(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        self.load()
        assert self.automatic_mask_generator is not None
        image_array = _to_rgb_array(image)
        annotations = self.automatic_mask_generator.generate(image_array)
        return list(annotations)

    @staticmethod
    def _ensure_local_package_paths() -> None:
        candidates = [
            Path.cwd() / "segment-anything",
            Path.cwd() / "segment_anything",
        ]
        for candidate in candidates:
            if candidate.exists():
                path_value = str(candidate)
                if path_value not in sys.path:
                    sys.path.insert(0, path_value)


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape {array.shape}")
    return array.astype(np.uint8)
