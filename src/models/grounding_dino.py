from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class Detection:
    box_xyxy: np.ndarray
    score: float
    label: str


class GroundingDinoAdapter:
    """Thin adapter around the Hugging Face Grounding DINO implementation."""

    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO dependencies are missing. Install transformers and huggingface_hub."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_id_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image, prompt: str) -> list[Detection]:
        self.load()
        assert self.processor is not None
        assert self.model is not None

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required to run Grounding DINO.") from exc

        normalized_prompt = _normalize_prompt(prompt)
        text_payload = [[normalized_prompt]]
        try:
            inputs = self.processor(images=image, text=text_payload, return_tensors="pt")
        except Exception:
            inputs = self.processor(images=image, text=normalized_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        post_process = getattr(self.processor, "post_process_grounded_object_detection", None)
        if post_process is None:
            raise RuntimeError("Grounding DINO processor does not expose post_process_grounded_object_detection.")

        kwargs = {
            "outputs": outputs,
            "input_ids": inputs.input_ids,
            "target_sizes": [image.size[::-1]],
            "text_threshold": self.text_threshold,
        }
        parameters = inspect.signature(post_process).parameters
        if "threshold" in parameters:
            kwargs["threshold"] = self.box_threshold
        elif "box_threshold" in parameters:
            kwargs["box_threshold"] = self.box_threshold
        else:
            raise RuntimeError("Unsupported Grounding DINO post-processing signature.")

        results = post_process(**kwargs)
        if not results:
            return []

        result = results[0]
        boxes = _to_numpy(result.get("boxes", np.zeros((0, 4), dtype=np.float32)))
        scores = _to_numpy(result.get("scores", np.zeros((0,), dtype=np.float32))).reshape(-1)

        labels_raw = result.get("text_labels")
        if labels_raw is None:
            labels_raw = result.get("labels")
        labels = _normalize_labels(labels_raw)

        detections: list[Detection] = []
        for index in range(len(scores)):
            label = prompt if index >= len(labels) else str(labels[index])
            detections.append(
                Detection(
                    box_xyxy=np.asarray(boxes[index], dtype=np.float32),
                    score=float(scores[index]),
                    label=label,
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections


def ensure_local_hf_cache_dir(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def _normalize_prompt(prompt: str) -> str:
    value = prompt.strip().lower()
    if not value.endswith("."):
        value = f"{value}."
    return value


def _normalize_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    array = _to_numpy(value).reshape(-1)
    return [str(item) for item in array.tolist()]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
