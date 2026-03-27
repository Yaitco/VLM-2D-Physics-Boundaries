from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils.metrics import area_ratio
from src.utils.visualization import render_mask_overlay

CLASS_METHODS = ("carpet", "rug", "area_rug")
UPDATE_METHODS = ("main_object", "carpet", "rug", "area_rug", "best_class", "best_available", "hint")


class DatasetUpdater:
    """Write selected masks back into a dataset and update meta.json safely."""

    def __init__(
        self,
        dataset_dir: Path,
        images_dir: Path,
        model_name: str,
        update_method: str = "main_object",
        meta_path: Path | None = None,
        masks_dir_name: str = "masks",
        preview_dir_name: str = "masks_preview",
        write_previews: bool = False,
        save_every: int = 10,
        field_suffix: str = "",
        summary_name: str | None = None,
    ) -> None:
        if update_method not in UPDATE_METHODS:
            raise ValueError(f"Unsupported dataset update method: {update_method}")

        self.dataset_dir = dataset_dir.resolve()
        self.images_dir = images_dir.resolve()
        self.model_name = model_name
        self.update_method = update_method
        self.meta_path = (meta_path or (self.dataset_dir / "meta.json")).resolve()
        self.masks_dir_name = masks_dir_name
        self.preview_dir_name = preview_dir_name
        self.write_previews = write_previews
        self.save_every = max(1, int(save_every))
        self.dataset_name = self.dataset_dir.name
        self.field_suffix = field_suffix or ""

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Dataset meta.json was not found: {self.meta_path}")

        self.meta: list[dict[str, Any]] = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.index: dict[str, dict[str, Any]] = {}
        for item in self.meta:
            self._normalize_path_fields_inplace(item)
            path_value = str(item.get("path") or "").strip()
            if path_value:
                self.index[self._normalize_key(path_value)] = item

        suffix_tag = self.field_suffix if self.field_suffix else ""
        self.summary_path = self.dataset_dir / (summary_name or f"grounded_sam{suffix_tag}_summary.json")
        self.backup_path = self.dataset_dir / "meta.before_grounded_sam.json"
        if not self.backup_path.exists():
            self.backup_path.write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self.counters = {
            "num_items": len(self.meta),
            "updated": 0,
            "skipped_missing_meta": 0,
            "skipped_selection_failed": 0,
            "skipped_empty_mask": 0,
            "image_failures": 0,
        }
        self.selected_counts: dict[str, int] = {}
        self.written_area_ratios: list[float] = []
        self._num_updates_since_flush = 0

    def update_image(
        self,
        image_path: Path,
        image: Image.Image,
        method_results: dict[str, dict[str, Any]],
    ) -> None:
        item = self._find_meta_item(image_path)
        if item is None:
            self.counters["skipped_missing_meta"] += 1
            return

        selection = self._select_result(method_results)
        if selection is None:
            self.counters["skipped_selection_failed"] += 1
            item[self._field("seg_status")] = "selection_failed"
            item[self._field("seg_error")] = f"No valid result for dataset update method '{self.update_method}'"
            self._flush_if_needed(force=False)
            return

        selected_method, result = selection
        mask = np.asarray(result.get("mask"), dtype=bool)
        if mask.size == 0 or not mask.any():
            self.counters["skipped_empty_mask"] += 1
            item[self._field("seg_status")] = "empty_prediction"
            item[self._field("seg_error")] = f"Selected method '{selected_method}' returned an empty mask"
            item[self._field("seg_prompt_mode")] = selected_method
            item[self._field("seg_query_text")] = str(result.get("query_text") or "")
            self._flush_if_needed(force=False)
            return

        relative = image_path.resolve().relative_to(self.images_dir)
        mask_abs = self.dataset_dir / self.masks_dir_name / relative.with_suffix(".png")
        mask_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_abs)

        mask_rel = self._dataset_rel_path(self.masks_dir_name, relative.with_suffix(".png"))
        preview_rel = None
        if self.write_previews:
            preview_abs = self.dataset_dir / self.preview_dir_name / relative.with_suffix(".jpg")
            preview_abs.parent.mkdir(parents=True, exist_ok=True)
            render_mask_overlay(image, mask, alpha=0.35).save(preview_abs)
            preview_rel = self._dataset_rel_path(self.preview_dir_name, relative.with_suffix(".jpg"))

        previous_mask_path = item.get("mask_path")
        previous_mask_source = item.get("mask_source")
        item[self._field("seg_seed_mask_path")] = self._normalize_optional_path(previous_mask_path)
        item[self._field("seg_seed_mask_source")] = previous_mask_source
        item[self._field("mask_path")] = mask_rel
        item[self._field("mask_source")] = self._build_mask_source(selected_method)
        item[self._field("seg_prompt_mode")] = selected_method
        item[self._field("seg_selected_method")] = selected_method
        item[self._field("seg_query_text")] = str(result.get("query_text") or "")
        item[self._field("seg_box_xyxy")] = self._serialize_box(result.get("box_xyxy"))
        item[self._field("seg_model_name")] = self.model_name
        item[self._field("seg_status")] = "written"
        item[self._field("seg_mask_area_ratio")] = round(float(area_ratio(mask)), 6)
        item[self._field("seg_dino_score")] = self._round_or_none(result.get("dino_score"))
        item[self._field("seg_sam_score")] = self._round_or_none(result.get("sam_score"))
        item[self._field("seg_inference_time")] = self._round_or_none(result.get("inference_time"))
        if preview_rel is not None:
            item[self._field("seg_preview_path")] = preview_rel
        error_field = self._field("seg_error")
        if error_field in item:
            item.pop(error_field, None)

        self.counters["updated"] += 1
        self.selected_counts[selected_method] = self.selected_counts.get(selected_method, 0) + 1
        self.written_area_ratios.append(float(area_ratio(mask)))
        self._num_updates_since_flush += 1
        self._flush_if_needed(force=False)

    def mark_image_failure(self, image_path: Path, reason: str) -> None:
        item = self._find_meta_item(image_path)
        if item is None:
            self.counters["skipped_missing_meta"] += 1
            return
        self.counters["image_failures"] += 1
        item[self._field("seg_status")] = reason
        item[self._field("seg_error")] = reason
        self._flush_if_needed(force=False)

    def finalize(self) -> None:
        self._flush_if_needed(force=True)

    def _find_meta_item(self, image_path: Path) -> dict[str, Any] | None:
        relative = image_path.resolve().relative_to(self.images_dir)
        key = self._normalize_key(self._dataset_rel_path("images", relative))
        return self.index.get(key)

    def _select_result(self, method_results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
        if self.update_method in method_results:
            result = method_results[self.update_method]
            return (self.update_method, result)
        if self.update_method == "best_class":
            return self._select_best_class(method_results)
        if self.update_method == "best_available":
            class_selection = self._select_best_class(method_results)
            if class_selection is not None:
                return class_selection
            fallback = method_results.get("main_object")
            if fallback is None:
                return None
            return ("main_object", fallback)
        return None

    def _select_best_class(self, method_results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for method in CLASS_METHODS:
            result = method_results.get(method)
            if result is None:
                continue
            mask = np.asarray(result.get("mask"), dtype=bool)
            if mask.size == 0 or not mask.any():
                continue
            candidates.append((method, result))
        if not candidates:
            return None
        candidates.sort(key=self._class_result_sort_key, reverse=True)
        return candidates[0]

    @staticmethod
    def _class_result_sort_key(candidate: tuple[str, dict[str, Any]]) -> tuple[float, float, float]:
        _, result = candidate
        dino_score = float(result.get("dino_score") or -1.0)
        sam_score = float(result.get("sam_score") or -1.0)
        mask = np.asarray(result.get("mask"), dtype=bool)
        mask_ratio = float(mask.mean()) if mask.size else -1.0
        return (dino_score, sam_score, mask_ratio)

    def _flush_if_needed(self, force: bool) -> None:
        if not force and self._num_updates_since_flush < self.save_every:
            return
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary_path.write_text(
            json.dumps(self._build_summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._num_updates_since_flush = 0

    def _build_summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": self._path_to_posix(self.dataset_dir),
            "meta_path": self._path_to_posix(self.meta_path),
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "update_method": self.update_method,
            "field_suffix": self.field_suffix,
            "masks_dir_name": self.masks_dir_name,
            "preview_dir_name": self.preview_dir_name,
            "write_previews": self.write_previews,
            "counters": self.counters,
            "selected_counts": self.selected_counts,
            "mean_written_mask_area_ratio": self._round_or_none(
                float(np.mean(self.written_area_ratios)) if self.written_area_ratios else None
            ),
        }

    def _build_mask_source(self, selected_method: str) -> str:
        return f"grounded_sam:{selected_method}:{self.model_name}"

    def _dataset_rel_path(self, root_name: str, relative: Path) -> str:
        return f"{self.dataset_name}/{root_name}/{relative.as_posix()}"

    def _field(self, base_name: str) -> str:
        return f"{base_name}{self.field_suffix}" if self.field_suffix else base_name

    @staticmethod
    def _serialize_box(box_xyxy: Any) -> list[int] | None:
        if box_xyxy is None:
            return None
        array = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if array.size != 4:
            return None
        return [int(round(float(value))) for value in array.tolist()]

    @staticmethod
    def _round_or_none(value: Any) -> float | None:
        if value is None:
            return None
        return round(float(value), 6)

    @staticmethod
    def _normalize_optional_path(value: Any) -> Any:
        if value is None:
            return None
        return str(value).replace("\\", "/")

    @classmethod
    def _normalize_path_fields_inplace(cls, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, str) and (key == "path" or key.endswith("_path")):
                    payload[key] = cls._normalize_optional_path(value)
                else:
                    cls._normalize_path_fields_inplace(value)
            return
        if isinstance(payload, list):
            for item in payload:
                cls._normalize_path_fields_inplace(item)

    @staticmethod
    def _path_to_posix(path: Path) -> str:
        return path.as_posix()

    @staticmethod
    def _normalize_key(value: str) -> str:
        return value.replace("\\", "/").strip().lower()


