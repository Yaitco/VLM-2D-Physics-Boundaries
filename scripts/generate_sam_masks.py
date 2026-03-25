from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class SamRuntime:
    model: Any
    processor: Any
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SAM masks for an existing ABO subset.")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/abo_physics_natural_bg_v2"),
        help="Path to dataset directory containing meta.json and images/.",
    )
    p.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional explicit meta.json path. Defaults to <dataset-dir>/meta.json",
    )
    p.add_argument(
        "--sam-model-id",
        type=str,
        default="facebook/sam-vit-base",
        help="Hugging Face SAM model id.",
    )
    p.add_argument(
        "--input-mask-field",
        type=str,
        default="mask_path",
        help="Metadata field used as a weak seed mask prompt for SAM.",
    )
    p.add_argument(
        "--input-source-field",
        type=str,
        default="mask_source",
        help="Metadata field describing the current seed-mask source.",
    )
    p.add_argument(
        "--output-mask-field",
        type=str,
        default="mask_path",
        help="Metadata field where SAM mask path will be stored.",
    )
    p.add_argument(
        "--output-source-field",
        type=str,
        default="mask_source",
        help="Metadata field where SAM source string will be stored.",
    )
    p.add_argument(
        "--output-dir-name",
        type=str,
        default="masks",
        help="Directory name inside dataset dir where SAM masks will be saved.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Write updated metadata and summary every N processed items.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    p.add_argument(
        "--bbox-margin-ratio",
        type=float,
        default=0.08,
        help="Relative padding added around the seed-mask bounding box.",
    )
    p.add_argument(
        "--min-seed-area-ratio",
        type=float,
        default=0.003,
        help="Skip samples whose seed mask is smaller than this image-area ratio.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute SAM masks even if output-mask-field already exists.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device. 'auto' prefers CUDA when available.",
    )
    p.add_argument(
        "--write-previews",
        action="store_true",
        help="Also save simple overlay previews next to masks for QA.",
    )
    return p.parse_args()


def load_runtime(model_id: str, device_pref: str) -> SamRuntime:
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except Exception as exc:
        raise RuntimeError(
            "SAM dependencies are missing. Install 'transformers' and 'torch' in the target environment."
        ) from exc

    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return SamRuntime(model=model, processor=processor, device=device)


def load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_dataset_path(dataset_dir: Path, stored_path: str) -> Path:
    p = Path(stored_path)
    if p.is_absolute():
        return p
    return dataset_dir.parent / p


def load_binary_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127


def mask_to_box(mask: np.ndarray, margin_ratio: float = 0.08) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(2, int(round(bw * margin_ratio)))
    pad_y = max(2, int(round(bh * margin_ratio)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _squeeze_mask_candidates(mask_batch: Any) -> np.ndarray:
    arr = np.asarray(mask_batch)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SAM mask shape: {arr.shape}")
    return arr


def choose_best_candidate(
    candidate_masks: np.ndarray,
    iou_scores: Iterable[float],
    seed_mask: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    best_idx = 0
    best_value = -1e9
    best_overlap = 0.0
    best_model_iou = 0.0
    seed_area = float(seed_mask.mean())
    scores = list(float(x) for x in iou_scores)
    for idx in range(candidate_masks.shape[0]):
        cand = candidate_masks[idx] > 0
        overlap = binary_iou(cand, seed_mask)
        area_delta = abs(float(cand.mean()) - seed_area)
        model_iou = scores[idx] if idx < len(scores) else 0.0
        blended = 0.65 * overlap + 0.30 * model_iou - 0.05 * area_delta
        if blended > best_value:
            best_value = blended
            best_idx = idx
            best_overlap = overlap
            best_model_iou = model_iou
    return candidate_masks[best_idx] > 0, best_model_iou, best_overlap


def save_mask(mask: np.ndarray, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)
    return float(mask.mean())


def save_preview(image: Image.Image, mask: np.ndarray, out_path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).astype(np.float32)
    m = mask.astype(bool)
    if m.any():
        tint = np.asarray([255, 96, 96], dtype=np.float32)
        rgb[m] = 0.72 * rgb[m] + 0.28 * tint
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def run_sam_on_image(
    runtime: SamRuntime,
    image: Image.Image,
    seed_mask: np.ndarray,
    box_xyxy: List[int],
) -> Tuple[np.ndarray, float, float]:
    import torch

    inputs = runtime.processor(
        images=image,
        input_boxes=[[[box_xyxy]]],
        return_tensors="pt",
    )
    inputs = {k: (v.to(runtime.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = runtime.model(**inputs)

    processed_masks = runtime.processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    candidate_masks = _squeeze_mask_candidates(processed_masks[0])
    iou_scores = np.asarray(outputs.iou_scores.detach().cpu()).reshape(-1)
    chosen_mask, predicted_iou, overlap_iou = choose_best_candidate(candidate_masks, iou_scores, seed_mask)
    return chosen_mask, predicted_iou, overlap_iou


def build_summary(
    *,
    dataset_dir: Path,
    meta_path: Path,
    runtime_device: str,
    args: argparse.Namespace,
    counters: Dict[str, int],
    area_ratios: List[float],
    seed_area_ratios: List[float],
    overlaps: List[float],
    predicted_ious: List[float],
) -> Dict[str, Any]:
    def mean_or_none(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return round(float(np.mean(values)), 6)

    return {
        "dataset_dir": str(dataset_dir),
        "meta_path": str(meta_path),
        "sam_model_id": args.sam_model_id,
        "device": runtime_device,
        "input_mask_field": args.input_mask_field,
        "input_source_field": args.input_source_field,
        "output_mask_field": args.output_mask_field,
        "output_source_field": args.output_source_field,
        "output_dir_name": args.output_dir_name,
        "bbox_margin_ratio": args.bbox_margin_ratio,
        "min_seed_area_ratio": args.min_seed_area_ratio,
        "max_samples": args.max_samples,
        "counters": counters,
        "mean_seed_area_ratio": mean_or_none(seed_area_ratios),
        "mean_sam_area_ratio": mean_or_none(area_ratios),
        "mean_seed_overlap_iou": mean_or_none(overlaps),
        "mean_predicted_iou": mean_or_none(predicted_ious),
    }


def checkpoint(
    *,
    meta: List[Dict[str, Any]],
    meta_path: Path,
    summary_path: Path,
    summary_payload: Dict[str, Any],
) -> None:
    write_json(meta_path, meta)
    write_json(summary_path, summary_payload)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    meta_path = args.meta_path.resolve() if args.meta_path is not None else dataset_dir / "meta.json"
    summary_path = dataset_dir / "sam_summary.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    all_meta = load_meta(meta_path)
    work_items = all_meta[: args.max_samples] if args.max_samples is not None else all_meta

    runtime = load_runtime(args.sam_model_id, args.device)
    dataset_name = dataset_dir.name
    backup_meta_path = dataset_dir / "meta.before_sam.json"
    if not backup_meta_path.exists():
        write_json(backup_meta_path, all_meta)

    counters = {
        "num_items": len(work_items),
        "written": 0,
        "skipped_existing": 0,
        "skipped_missing_seed_mask": 0,
        "skipped_missing_image": 0,
        "skipped_small_seed_mask": 0,
        "failures": 0,
    }
    area_ratios: List[float] = []
    seed_area_ratios: List[float] = []
    overlaps: List[float] = []
    predicted_ious: List[float] = []

    progress = tqdm(work_items, desc=f"Generating SAM masks [{dataset_name}]")
    for idx, item in enumerate(progress, start=1):
        existing = item.get(args.output_mask_field)
        existing_source = str(item.get(args.output_source_field) or "")
        if args.output_mask_field == args.input_mask_field:
            already_sam = existing and existing_source.startswith("sam:")
        else:
            already_sam = bool(existing)
        if already_sam and not args.overwrite:
            counters["skipped_existing"] += 1
            continue

        image_rel = item.get("path")
        seed_mask_rel = item.get(args.input_mask_field)
        seed_mask_source = item.get(args.input_source_field)
        if not image_rel:
            counters["skipped_missing_image"] += 1
            item["sam_status"] = "missing_image_path"
            continue
        if not seed_mask_rel:
            counters["skipped_missing_seed_mask"] += 1
            item["sam_status"] = f"missing_{args.input_mask_field}"
            continue

        image_path = resolve_dataset_path(dataset_dir, image_rel)
        seed_mask_path = resolve_dataset_path(dataset_dir, seed_mask_rel)
        if not image_path.exists():
            counters["skipped_missing_image"] += 1
            item["sam_status"] = "image_not_found"
            continue
        if not seed_mask_path.exists():
            counters["skipped_missing_seed_mask"] += 1
            item["sam_status"] = "seed_mask_not_found"
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            seed_mask = load_binary_mask(seed_mask_path)
            seed_area_ratio = float(seed_mask.mean())
            seed_area_ratios.append(seed_area_ratio)
            if seed_area_ratio < args.min_seed_area_ratio:
                counters["skipped_small_seed_mask"] += 1
                item["sam_status"] = "seed_mask_too_small"
                continue

            box_xyxy = mask_to_box(seed_mask, margin_ratio=args.bbox_margin_ratio)
            if box_xyxy is None:
                counters["failures"] += 1
                item["sam_status"] = "seed_box_failed"
                continue

            sam_mask, predicted_iou, overlap_iou = run_sam_on_image(runtime, image, seed_mask, box_xyxy)
            if image_rel.startswith(f"{dataset_name}/images/"):
                image_subpath = Path(image_rel).relative_to(Path(dataset_name) / "images")
            elif image_rel.startswith(f"{dataset_name}/"):
                image_subpath = Path(image_rel).relative_to(dataset_name)
            else:
                image_subpath = Path(Path(image_rel).name)
            out_rel = Path(dataset_name) / args.output_dir_name / image_subpath.with_suffix(".png")
            out_abs = dataset_dir.parent / out_rel
            sam_area_ratio = save_mask(sam_mask, out_abs)

            item["sam_seed_mask_path"] = seed_mask_rel
            item["sam_seed_mask_source"] = seed_mask_source
            item[args.output_mask_field] = str(out_rel)
            item[args.output_source_field] = f"sam:{args.sam_model_id}"
            item["sam_prompt_mask_field"] = args.input_mask_field
            item["sam_prompt_source_field"] = args.input_source_field
            item["sam_prompt_box_xyxy"] = [int(v) for v in box_xyxy]
            item["sam_predicted_iou"] = round(float(predicted_iou), 6)
            item["sam_overlap_with_seed_iou"] = round(float(overlap_iou), 6)
            item["sam_mask_area_ratio"] = round(float(sam_area_ratio), 6)
            item["sam_status"] = "written"

            counters["written"] += 1
            area_ratios.append(sam_area_ratio)
            overlaps.append(float(overlap_iou))
            predicted_ious.append(float(predicted_iou))

            if args.write_previews:
                preview_rel = Path(dataset_name) / f"{args.output_dir_name}_preview" / image_subpath.with_suffix(".jpg")
                preview_abs = dataset_dir.parent / preview_rel
                save_preview(image, sam_mask, preview_abs)
                item["sam_preview_path"] = str(preview_rel)
        except Exception as exc:
            counters["failures"] += 1
            item["sam_status"] = f"failed:{type(exc).__name__}"
            item["sam_error"] = str(exc)[:500]

        if idx % max(1, args.save_every) == 0:
            summary_payload = build_summary(
                dataset_dir=dataset_dir,
                meta_path=meta_path,
                runtime_device=runtime.device,
                args=args,
                counters=counters,
                area_ratios=area_ratios,
                seed_area_ratios=seed_area_ratios,
                overlaps=overlaps,
                predicted_ious=predicted_ious,
            )
            checkpoint(meta=all_meta, meta_path=meta_path, summary_path=summary_path, summary_payload=summary_payload)
            progress.set_postfix(
                written=counters["written"],
                skipped=counters["skipped_existing"],
                failed=counters["failures"],
            )

    summary_payload = build_summary(
        dataset_dir=dataset_dir,
        meta_path=meta_path,
        runtime_device=runtime.device,
        args=args,
        counters=counters,
        area_ratios=area_ratios,
        seed_area_ratios=seed_area_ratios,
        overlaps=overlaps,
        predicted_ious=predicted_ious,
    )
    checkpoint(meta=all_meta, meta_path=meta_path, summary_path=summary_path, summary_payload=summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
