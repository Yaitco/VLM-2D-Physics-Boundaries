from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.models.grounding_dino import GroundingDinoAdapter
from src.models.sam_wrapper import SamWrapper
from src.pipelines.carpet_pipeline import CarpetPipeline
from src.pipelines.main_object_pipeline import MainObjectPipeline
from src.utils.dataset_update import DatasetUpdater
from src.utils.metrics import area_ratio, compute_metrics, empty_metrics, mask_area
from src.utils.postprocess import postprocess_mask
from src.utils.visualization import CollagePanel, render_binary_mask, render_mask_overlay, save_collage

pd: Any = None

CLASS_PROMPTS = ["carpet", "rug", "area rug"]
CLASS_METHODS = ["carpet", "rug", "area_rug"]
BASE_METHODS = ["main_object", *CLASS_METHODS]
DATASET_UPDATE_METHODS = ["main_object", "carpet", "rug", "area_rug", "best_class", "best_available", "hint"]
HINT_FIELDS = ["primary_object", "product_type", "title"]
HINT_METHOD = "hint"
HINT_FIELD_SUFFIX = "_hint"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
RESULT_COLUMNS = [
    "image_name",
    "method",
    "iou",
    "dice",
    "precision",
    "recall",
    "mask_area",
    "area_ratio",
    "dino_score",
    "sam_score",
    "inference_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO + SAM carpet segmentation experiment.")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with input images.")
    parser.add_argument("--masks_dir", type=Path, default=None, help="Optional directory with GT masks.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--grounding_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=Path,
        default=Path("checkpoints/sam_vit_b_01ec64.pth"),
        help="Path to the SAM checkpoint.",
    )
    parser.add_argument(
        "--sam_model_type",
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_b",
        help="SAM backbone type.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.25, help="Grounding DINO box threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Grounding DINO text threshold.")
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=500,
        help="Remove connected components smaller than this area.",
    )
    parser.add_argument("--closing_kernel_size", type=int, default=5, help="Morphological closing kernel size.")
    parser.add_argument("--closing_iterations", type=int, default=1, help="Morphological closing iterations.")
    parser.add_argument("--overlay_alpha", type=float, default=0.35, help="Overlay alpha for saved visualizations.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--amg_points_per_side", type=int, default=32)
    parser.add_argument("--amg_pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--amg_stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--amg_crop_n_layers", type=int, default=1)
    parser.add_argument(
        "--enable_hint_method",
        action="store_true",
        help="Run an extra Grounding DINO -> SAM pass using object-name hints from dataset meta.json.",
    )
    parser.add_argument(
        "--hint_field",
        choices=HINT_FIELDS,
        default="primary_object",
        help="Metadata field to use for the hint prompt. Falls back to the other supported field if missing.",
    )
    parser.add_argument(
        "--update_dataset",
        action="store_true",
        help="Write selected masks back into dataset masks/meta.json.",
    )
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Dataset root containing meta.json, images/, masks/.")
    parser.add_argument("--dataset_meta_path", type=Path, default=None, help="Optional explicit meta.json path for dataset updates.")
    parser.add_argument(
        "--dataset_update_method",
        choices=DATASET_UPDATE_METHODS,
        default="main_object",
        help="Which prediction to write back into dataset metadata.",
    )
    parser.add_argument("--dataset_masks_dir_name", type=str, default="masks", help="Directory name inside dataset for written masks.")
    parser.add_argument(
        "--dataset_preview_dir_name",
        type=str,
        default="masks_preview",
        help="Directory name inside dataset for preview overlays.",
    )
    parser.add_argument(
        "--dataset_field_suffix",
        type=str,
        default="",
        help="Optional suffix for dataset fields and default output dirs, for example _hint_title.",
    )
    parser.add_argument("--dataset_save_every", type=int, default=10, help="Flush updated meta.json every N written items.")
    parser.add_argument("--write_dataset_previews", action="store_true", help="Write overlay previews into the dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    global pd
    if pd is None:
        try:
            import pandas as _pd
        except Exception as exc:
            raise RuntimeError("pandas is required to run the experiment. Install dependencies from requirements.txt.") from exc
        pd = _pd

    output_dir = args.output_dir
    ensure_dir(output_dir)
    logger = setup_logging(output_dir / "logs" / "experiment.log", args.log_level)
    device = resolve_device(args.device, logger)
    logger.info("Using device: %s", device)

    image_paths = collect_image_paths(args.images_dir)
    logger.info("Found %d images in %s", len(image_paths), args.images_dir)

    hint_enabled = bool(args.enable_hint_method or args.dataset_update_method == HINT_METHOD)
    active_methods = list(BASE_METHODS)
    if hint_enabled:
        active_methods.append(HINT_METHOD)

    hint_meta_index: dict[str, dict[str, Any]] = {}
    hint_dataset_name: str | None = None
    if hint_enabled:
        hint_meta_index, hint_dataset_name = load_hint_meta_index(
            images_dir=args.images_dir,
            dataset_dir=args.dataset_dir,
            dataset_meta_path=args.dataset_meta_path,
        )
        logger.info(
            "Hint mode enabled: field=%s, indexed_items=%d",
            args.hint_field,
            len({id(item): item for item in hint_meta_index.values()}),
        )

    dino = GroundingDinoAdapter(
        model_id_or_path=args.grounding_model_id,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    sam = SamWrapper(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=device,
        automatic_mask_kwargs={
            "points_per_side": args.amg_points_per_side,
            "pred_iou_thresh": args.amg_pred_iou_thresh,
            "stability_score_thresh": args.amg_stability_score_thresh,
            "crop_n_layers": args.amg_crop_n_layers,
            "min_mask_region_area": 0,
        },
    )
    main_pipeline = MainObjectPipeline(sam)
    carpet_pipeline = CarpetPipeline(dino, sam)

    dataset_updater: DatasetUpdater | None = None
    if args.update_dataset:
        dataset_dir = args.dataset_dir.resolve() if args.dataset_dir is not None else infer_dataset_dir(args.images_dir)
        dataset_masks_dir_name = args.dataset_masks_dir_name
        dataset_preview_dir_name = args.dataset_preview_dir_name
        dataset_field_suffix = normalize_field_suffix(args.dataset_field_suffix)
        if args.dataset_update_method == HINT_METHOD:
            if not dataset_field_suffix:
                dataset_field_suffix = default_hint_field_suffix(args.hint_field)
            if dataset_masks_dir_name == "masks":
                dataset_masks_dir_name = build_dataset_dir_name("masks", dataset_field_suffix)
            if dataset_preview_dir_name == "masks_preview":
                dataset_preview_dir_name = build_dataset_dir_name("masks_preview", dataset_field_suffix)

        dataset_updater = DatasetUpdater(
            dataset_dir=dataset_dir,
            images_dir=args.images_dir,
            meta_path=args.dataset_meta_path,
            model_name=build_model_name(args.grounding_model_id, args.sam_model_type),
            update_method=args.dataset_update_method,
            masks_dir_name=dataset_masks_dir_name,
            preview_dir_name=dataset_preview_dir_name,
            write_previews=args.write_dataset_previews,
            save_every=args.dataset_save_every,
            field_suffix=dataset_field_suffix,
        )
        logger.info(
            "Dataset update mode enabled: dataset=%s, method=%s, suffix=%s, masks_dir=%s",
            dataset_dir,
            args.dataset_update_method,
            dataset_field_suffix or "<none>",
            dataset_masks_dir_name,
        )
        if args.masks_dir is not None and args.masks_dir.resolve() == (dataset_dir / dataset_masks_dir_name).resolve():
            logger.info(
                "masks_dir points to the dataset masks directory, so reported metrics compare new masks against the previous dataset masks."
            )

    try:
        rows: list[dict[str, Any]] = []
        for image_path in image_paths:
            image_key = image_path.relative_to(args.images_dir).as_posix()
            logger.info("Processing %s", image_key)

            try:
                image = load_rgb_image(image_path)
            except Exception:
                logger.exception("Failed to load image %s", image_path)
                rows.extend(build_failure_rows(image_key, active_methods))
                if dataset_updater is not None:
                    dataset_updater.mark_image_failure(image_path, "image_load_failed")
                continue

            gt_mask = load_gt_mask_if_available(image_path, args.images_dir, args.masks_dir, image.size, logger)
            method_to_overlay: dict[str, Image.Image] = {}
            method_to_metrics: dict[str, dict[str, float]] = {}
            method_results: dict[str, dict[str, Any]] = {}

            try:
                main_prediction = main_pipeline.run(image)
                main_mask = postprocess_mask(
                    main_prediction.mask,
                    min_component_area=args.min_component_area,
                    closing_kernel_size=args.closing_kernel_size,
                    closing_iterations=args.closing_iterations,
                )
                save_prediction_outputs(output_dir, image_path, args.images_dir, "main_object", image, main_mask, args.overlay_alpha)
                main_metrics = build_metric_dict(main_mask, gt_mask)
                method_to_metrics["main_object"] = main_metrics
                method_to_overlay["main_object"] = render_mask_overlay(image, main_mask, alpha=args.overlay_alpha)
                rows.append(
                    build_result_row(
                        image_name=image_key,
                        method="main_object",
                        mask=main_mask,
                        metrics=main_metrics,
                        dino_score=None,
                        sam_score=main_prediction.sam_score,
                        inference_time=main_prediction.inference_time,
                    )
                )
                method_results["main_object"] = build_method_result(
                    mask=main_mask,
                    query_text=None,
                    dino_score=None,
                    sam_score=main_prediction.sam_score,
                    inference_time=main_prediction.inference_time,
                    box_xyxy=None,
                )
            except Exception:
                logger.exception("Main object pipeline failed for %s", image_path)
                empty_mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
                save_prediction_outputs(output_dir, image_path, args.images_dir, "main_object", image, empty_mask, args.overlay_alpha)
                method_to_metrics["main_object"] = build_metric_dict(empty_mask, gt_mask)
                method_to_overlay["main_object"] = render_mask_overlay(image, empty_mask, alpha=args.overlay_alpha)
                rows.append(
                    build_result_row(
                        image_name=image_key,
                        method="main_object",
                        mask=empty_mask,
                        metrics=method_to_metrics["main_object"],
                        dino_score=None,
                        sam_score=None,
                        inference_time=float("nan"),
                    )
                )
                method_results["main_object"] = build_method_result(
                    mask=empty_mask,
                    query_text=None,
                    dino_score=None,
                    sam_score=None,
                    inference_time=float("nan"),
                    box_xyxy=None,
                )

            if hint_enabled:
                hint_prompt = resolve_hint_prompt(
                    image_path=image_path,
                    images_dir=args.images_dir,
                    dataset_name=hint_dataset_name,
                    hint_meta_index=hint_meta_index,
                    hint_field=args.hint_field,
                )
                if hint_prompt is None:
                    logger.warning("No hint prompt found for %s", image_key)
                    hint_mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
                    hint_dino_score = None
                    hint_sam_score = None
                    hint_inference_time = float("nan")
                    hint_box = None
                else:
                    try:
                        hint_prediction = carpet_pipeline.run_prompt(image, hint_prompt)
                        hint_mask = postprocess_mask(
                            hint_prediction.mask,
                            min_component_area=args.min_component_area,
                            closing_kernel_size=args.closing_kernel_size,
                            closing_iterations=args.closing_iterations,
                        )
                        hint_dino_score = hint_prediction.dino_score
                        hint_sam_score = hint_prediction.sam_score
                        hint_inference_time = hint_prediction.inference_time
                        hint_box = hint_prediction.box_xyxy
                    except Exception:
                        logger.exception("Hint pipeline failed for %s with prompt '%s'", image_path, hint_prompt)
                        hint_mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
                        hint_dino_score = None
                        hint_sam_score = None
                        hint_inference_time = float("nan")
                        hint_box = None

                save_prediction_outputs(output_dir, image_path, args.images_dir, HINT_METHOD, image, hint_mask, args.overlay_alpha)
                hint_metrics = build_metric_dict(hint_mask, gt_mask)
                method_to_metrics[HINT_METHOD] = hint_metrics
                method_to_overlay[HINT_METHOD] = render_mask_overlay(image, hint_mask, alpha=args.overlay_alpha)
                rows.append(
                    build_result_row(
                        image_name=image_key,
                        method=HINT_METHOD,
                        mask=hint_mask,
                        metrics=hint_metrics,
                        dino_score=hint_dino_score,
                        sam_score=hint_sam_score,
                        inference_time=hint_inference_time,
                    )
                )
                method_results[HINT_METHOD] = build_method_result(
                    mask=hint_mask,
                    query_text=hint_prompt,
                    dino_score=hint_dino_score,
                    sam_score=hint_sam_score,
                    inference_time=hint_inference_time,
                    box_xyxy=hint_box,
                )

            for prompt in CLASS_PROMPTS:
                method_name = slugify(prompt)
                try:
                    prediction = carpet_pipeline.run_prompt(image, prompt)
                    final_mask = postprocess_mask(
                        prediction.mask,
                        min_component_area=args.min_component_area,
                        closing_kernel_size=args.closing_kernel_size,
                        closing_iterations=args.closing_iterations,
                    )
                    save_prediction_outputs(output_dir, image_path, args.images_dir, method_name, image, final_mask, args.overlay_alpha)
                    metrics = build_metric_dict(final_mask, gt_mask)
                    method_to_metrics[method_name] = metrics
                    method_to_overlay[method_name] = render_mask_overlay(image, final_mask, alpha=args.overlay_alpha)
                    rows.append(
                        build_result_row(
                            image_name=image_key,
                            method=method_name,
                            mask=final_mask,
                            metrics=metrics,
                            dino_score=prediction.dino_score,
                            sam_score=prediction.sam_score,
                            inference_time=prediction.inference_time,
                        )
                    )
                    method_results[method_name] = build_method_result(
                        mask=final_mask,
                        query_text=prompt,
                        dino_score=prediction.dino_score,
                        sam_score=prediction.sam_score,
                        inference_time=prediction.inference_time,
                        box_xyxy=prediction.box_xyxy,
                    )
                except Exception:
                    logger.exception("Class-specific pipeline failed for %s with prompt '%s'", image_path, prompt)
                    empty_mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
                    save_prediction_outputs(output_dir, image_path, args.images_dir, method_name, image, empty_mask, args.overlay_alpha)
                    metrics = build_metric_dict(empty_mask, gt_mask)
                    method_to_metrics[method_name] = metrics
                    method_to_overlay[method_name] = render_mask_overlay(image, empty_mask, alpha=args.overlay_alpha)
                    rows.append(
                        build_result_row(
                            image_name=image_key,
                            method=method_name,
                            mask=empty_mask,
                            metrics=metrics,
                            dino_score=None,
                            sam_score=None,
                            inference_time=float("nan"),
                        )
                    )
                    method_results[method_name] = build_method_result(
                        mask=empty_mask,
                        query_text=prompt,
                        dino_score=None,
                        sam_score=None,
                        inference_time=float("nan"),
                        box_xyxy=None,
                    )

            save_image_collage(
                output_dir,
                image_path,
                args.images_dir,
                image,
                gt_mask,
                method_to_overlay,
                method_to_metrics,
                active_methods,
            )
            if dataset_updater is not None:
                dataset_updater.update_image(image_path, image, method_results)

        results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        tables_dir = ensure_dir(output_dir / "tables")
        results_df.to_csv(tables_dir / "results.csv", index=False)
        build_summary_table(results_df).to_csv(tables_dir / "summary_by_method.csv", index=False)

        prompt_winners, prompt_counts = build_prompt_winner_tables(results_df)
        prompt_winners.to_csv(tables_dir / "best_prompt_per_image.csv", index=False)
        prompt_counts.to_csv(tables_dir / "prompt_win_counts.csv", index=False)

        comparison_df, comparison_summary = build_main_vs_best_class_tables(results_df)
        comparison_df.to_csv(tables_dir / "main_object_vs_best_class.csv", index=False)
        comparison_summary.to_csv(tables_dir / "main_object_vs_best_class_summary.csv", index=False)

        logger.info("Saved experiment outputs to %s", tables_dir)
        return 0
    finally:
        if dataset_updater is not None:
            dataset_updater.finalize()


def build_model_name(grounding_model_id: str, sam_model_type: str) -> str:
    grounding_name = grounding_model_id.split("/")[-1].replace(" ", "_")
    return f"{grounding_name}+sam_{sam_model_type}"


def infer_dataset_dir(images_dir: Path) -> Path:
    resolved = images_dir.resolve()
    if resolved.name.lower() != "images":
        raise ValueError("When dataset metadata is needed without --dataset_dir, images_dir must point to <dataset>/images")
    return resolved.parent


def normalize_field_suffix(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    if not text.startswith("_"):
        text = f"_{text}"
    return text


def default_hint_field_suffix(hint_field: str) -> str:
    if hint_field == "primary_object":
        return HINT_FIELD_SUFFIX
    return f"{HINT_FIELD_SUFFIX}_{hint_field}"


def build_dataset_dir_name(base_name: str, field_suffix: str) -> str:
    normalized_suffix = normalize_field_suffix(field_suffix)
    if not normalized_suffix:
        return base_name
    return f"{base_name}{normalized_suffix}"


def load_hint_meta_index(
    images_dir: Path,
    dataset_dir: Path | None,
    dataset_meta_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], str]:
    resolved_dataset_dir = dataset_dir.resolve() if dataset_dir is not None else infer_dataset_dir(images_dir)
    meta_path = dataset_meta_path.resolve() if dataset_meta_path is not None else (resolved_dataset_dir / "meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Hint mode requires a dataset meta.json file: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_name = resolved_dataset_dir.name
    prefix = f"{dataset_name}/images/"
    index: dict[str, dict[str, Any]] = {}
    for item in meta:
        path_value = str(item.get("path") or "").strip()
        if not path_value:
            continue
        normalized = normalize_meta_key(path_value)
        index[normalized] = item
        if normalized.startswith(prefix):
            index[normalized[len(prefix) :]] = item
    if not index:
        raise ValueError(f"Hint mode could not build an index from {meta_path}")
    return index, dataset_name


def resolve_hint_prompt(
    image_path: Path,
    images_dir: Path,
    dataset_name: str | None,
    hint_meta_index: dict[str, dict[str, Any]],
    hint_field: str,
) -> str | None:
    relative = image_path.resolve().relative_to(images_dir.resolve()).as_posix()
    candidates = [normalize_meta_key(relative)]
    if dataset_name:
        candidates.append(normalize_meta_key(f"{dataset_name}/images/{relative}"))

    for key in candidates:
        item = hint_meta_index.get(key)
        if item is None:
            continue
        prompt = build_hint_prompt(item, hint_field)
        if prompt is not None:
            return prompt
    return None


def build_hint_prompt(meta_item: dict[str, Any], hint_field: str) -> str | None:
    if hint_field not in HINT_FIELDS:
        raise ValueError(f"Unsupported hint field: {hint_field}")

    abo_meta = meta_item.get("abo_meta") or {}
    if hint_field == "primary_object":
        raw_candidates = [
            meta_item.get("primary_object"),
            abo_meta.get("product_type"),
            abo_meta.get("title"),
        ]
    elif hint_field == "product_type":
        raw_candidates = [
            abo_meta.get("product_type"),
            meta_item.get("primary_object"),
            abo_meta.get("title"),
        ]
    else:
        raw_candidates = [
            abo_meta.get("title"),
            meta_item.get("primary_object"),
            abo_meta.get("product_type"),
        ]

    seen: set[str] = set()
    for raw_value in raw_candidates:
        prompt = normalize_hint_prompt(raw_value)
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        return prompt
    return None


def normalize_hint_prompt(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text or None


def normalize_meta_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def build_method_result(
    *,
    mask: np.ndarray,
    query_text: str | None,
    dino_score: float | None,
    sam_score: float | None,
    inference_time: float,
    box_xyxy: Any,
) -> dict[str, Any]:
    return {
        "mask": np.asarray(mask, dtype=bool),
        "query_text": query_text,
        "dino_score": dino_score,
        "sam_score": sam_score,
        "inference_time": inference_time,
        "box_xyxy": box_xyxy,
    }


def setup_logging(log_path: Path, level: str) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("grounded_sam_experiment")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_device(requested_device: str, logger: logging.Logger) -> str:
    try:
        import torch
    except Exception:
        if requested_device == "cuda":
            logger.warning("PyTorch is unavailable, falling back to CPU.")
        return "cpu"

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory was not found: {images_dir}")
    paths = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda item: item.relative_to(images_dir).as_posix())


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_binary_mask(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if target_size is not None and mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        array = np.asarray(mask, dtype=np.uint8)
    return array > 127


def load_gt_mask_if_available(
    image_path: Path,
    images_dir: Path,
    masks_dir: Path | None,
    image_size: tuple[int, int],
    logger: logging.Logger,
) -> np.ndarray | None:
    if masks_dir is None or not masks_dir.exists():
        return None

    relative = image_path.relative_to(images_dir)
    candidates = [(masks_dir / relative).with_suffix(ext) for ext in MASK_EXTENSIONS]
    candidates.extend((masks_dir / image_path.stem).with_suffix(ext) for ext in MASK_EXTENSIONS)

    mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if mask_path is None:
        return None

    try:
        return load_binary_mask(mask_path, target_size=image_size)
    except Exception:
        logger.exception("Failed to load GT mask %s", mask_path)
        return None


def save_prediction_outputs(
    output_dir: Path,
    image_path: Path,
    images_dir: Path,
    method: str,
    image: Image.Image,
    mask: np.ndarray,
    overlay_alpha: float,
) -> None:
    relative = image_path.relative_to(images_dir)
    mask_path = (output_dir / "masks" / method / relative).with_suffix(".png")
    overlay_path = (output_dir / "overlays" / method / relative).with_suffix(".png")
    ensure_dir(mask_path.parent)
    ensure_dir(overlay_path.parent)

    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    render_mask_overlay(image, mask, alpha=overlay_alpha).save(overlay_path)


def build_metric_dict(mask: np.ndarray, gt_mask: np.ndarray | None) -> dict[str, float]:
    if gt_mask is None:
        return empty_metrics()
    return compute_metrics(mask, gt_mask).as_dict()


def build_result_row(
    image_name: str,
    method: str,
    mask: np.ndarray,
    metrics: dict[str, float],
    dino_score: float | None,
    sam_score: float | None,
    inference_time: float,
) -> dict[str, Any]:
    return {
        "image_name": image_name,
        "method": method,
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mask_area": mask_area(mask),
        "area_ratio": area_ratio(mask),
        "dino_score": _safe_float(dino_score),
        "sam_score": _safe_float(sam_score),
        "inference_time": _safe_float(inference_time),
    }


def build_failure_rows(image_name: str, active_methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    empty = empty_metrics()
    for method in active_methods:
        rows.append(
            {
                "image_name": image_name,
                "method": method,
                "iou": empty["iou"],
                "dice": empty["dice"],
                "precision": empty["precision"],
                "recall": empty["recall"],
                "mask_area": 0,
                "area_ratio": 0.0,
                "dino_score": float("nan"),
                "sam_score": float("nan"),
                "inference_time": float("nan"),
            }
        )
    return rows


def save_image_collage(
    output_dir: Path,
    image_path: Path,
    images_dir: Path,
    image: Image.Image,
    gt_mask: np.ndarray | None,
    method_to_overlay: dict[str, Image.Image],
    method_to_metrics: dict[str, dict[str, float]],
    active_methods: list[str],
) -> None:
    relative = image_path.relative_to(images_dir)
    collage_path = (output_dir / "collages" / relative).with_suffix(".jpg")
    panels = [
        CollagePanel(
            title="Original",
            image=image,
            lines=[relative.as_posix(), f"size: {image.size[0]}x{image.size[1]}"],
        )
    ]
    if gt_mask is None:
        panels.append(CollagePanel(title="GT", image=image.copy(), lines=["mask not provided"]))
    else:
        panels.append(
            CollagePanel(
                title="GT",
                image=render_binary_mask(gt_mask),
                lines=[f"area: {int(gt_mask.sum())} px"],
            )
        )

    for method in active_methods:
        overlay = method_to_overlay.get(method, image.copy())
        metrics = method_to_metrics.get(method, empty_metrics())
        panels.append(
            CollagePanel(
                title=method,
                image=overlay,
                lines=build_collage_lines(metrics),
            )
        )

    save_collage(panels, collage_path)


def build_collage_lines(metrics: dict[str, float]) -> list[str]:
    if math.isnan(float(metrics["iou"])):
        return ["GT unavailable"]
    return [
        f"IoU: {metrics['iou']:.4f}",
        f"Dice: {metrics['dice']:.4f}",
        f"Prec: {metrics['precision']:.4f}",
        f"Rec: {metrics['recall']:.4f}",
    ]


def build_summary_table(df: Any) -> Any:
    if df.empty:
        return pd.DataFrame(columns=["method"])
    numeric_columns = [
        "iou",
        "dice",
        "precision",
        "recall",
        "mask_area",
        "area_ratio",
        "dino_score",
        "sam_score",
        "inference_time",
    ]
    grouped = df.groupby("method", sort=False)
    base = grouped.agg(num_rows=("image_name", "size"), num_images=("image_name", "nunique"))
    frames = [base]
    for suffix, agg_name in (("mean", "mean"), ("std", "std"), ("median", "median")):
        frames.append(grouped[numeric_columns].agg(agg_name).add_suffix(f"_{suffix}"))
    return pd.concat(frames, axis=1).reset_index()


def build_prompt_winner_tables(df: Any) -> tuple[Any, Any]:
    scored = df[df["method"].isin(CLASS_METHODS) & df["iou"].notna()].copy()
    if scored.empty:
        winners = pd.DataFrame(columns=["image_name", "method", "iou", "dice", "precision", "recall"])
        counts = pd.DataFrame(columns=["method", "win_count", "win_ratio"])
        return winners, counts

    scored["sam_score"] = scored["sam_score"].fillna(-1.0)
    scored["dino_score"] = scored["dino_score"].fillna(-1.0)
    winners = (
        scored.sort_values(
            by=["image_name", "iou", "dice", "sam_score", "dino_score", "method"],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("image_name", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    counts = winners["method"].value_counts().rename_axis("method").reset_index(name="win_count")
    counts["win_ratio"] = counts["win_count"] / max(1, len(winners))
    return winners, counts


def build_main_vs_best_class_tables(df: Any) -> tuple[Any, Any]:
    main_df = df[(df["method"] == "main_object") & df["iou"].notna()].copy()
    best_class_df, _ = build_prompt_winner_tables(df)
    if main_df.empty or best_class_df.empty:
        comparison = pd.DataFrame(
            columns=["image_name", "main_object_iou", "best_class_method", "best_class_iou", "winner"]
        )
        summary = pd.DataFrame(
            columns=[
                "main_object_win_count",
                "best_class_win_count",
                "tie_count",
                "main_object_mean_iou",
                "best_class_mean_iou",
                "main_object_mean_dice",
                "best_class_mean_dice",
            ]
        )
        return comparison, summary

    merged = main_df.merge(
        best_class_df[["image_name", "method", "iou", "dice"]],
        on="image_name",
        how="inner",
        suffixes=("_main_object", "_best_class"),
    )
    merged = merged.rename(
        columns={
            "method_best_class": "best_class_method",
            "iou_main_object": "main_object_iou",
            "dice_main_object": "main_object_dice",
            "iou_best_class": "best_class_iou",
            "dice_best_class": "best_class_dice",
        }
    )
    merged["winner"] = merged.apply(determine_winner, axis=1)

    summary = pd.DataFrame(
        [
            {
                "main_object_win_count": int((merged["winner"] == "main_object").sum()),
                "best_class_win_count": int((merged["winner"] == "best_class").sum()),
                "tie_count": int((merged["winner"] == "tie").sum()),
                "main_object_mean_iou": float(merged["main_object_iou"].mean()),
                "best_class_mean_iou": float(merged["best_class_iou"].mean()),
                "main_object_mean_dice": float(merged["main_object_dice"].mean()),
                "best_class_mean_dice": float(merged["best_class_dice"].mean()),
            }
        ]
    )
    comparison = merged[["image_name", "main_object_iou", "best_class_method", "best_class_iou", "winner"]]
    return comparison, summary


def determine_winner(row: Any) -> str:
    main_iou = float(row["main_object_iou"])
    class_iou = float(row["best_class_iou"])
    if main_iou > class_iou:
        return "main_object"
    if class_iou > main_iou:
        return "best_class"
    return "tie"


def slugify(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "_")


def _safe_float(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())




