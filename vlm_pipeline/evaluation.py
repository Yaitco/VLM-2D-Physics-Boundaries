from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .images import load_variant_image_cached
from .parsing import normalize_pred, parse_model_output
from .reporting import save_report
from .runtime import (
    VLMRuntime,
    infer_runtime_batch,
    infer_runtime_messages_batch,
    load_runtime,
    unload_runtime,
)
from .specs import PropertySpec, is_known_value, serialize_value, values_equal
from .utils import chunked, column_prefix


def select_property_keys_for_sample(
    gt_properties: Dict[str, Any],
    property_specs: Dict[str, PropertySpec],
    include_only_gt_known: bool,
) -> List[str]:
    keys = list(property_specs.keys())
    if include_only_gt_known:
        selected = [key for key in keys if is_known_value(property_specs[key], gt_properties.get(key))]
    else:
        selected = keys
    return selected or keys[: min(12, len(keys))]


def build_property_prompt(image_id: str, property_key: str, spec: PropertySpec) -> str:
    if spec.value_type == "categorical":
        value_hint = "string"
        rule_line = f'- "{property_key}" must be one of [{"|".join(spec.allowed_values)}]'
    elif spec.value_type == "boolean":
        value_hint = "string"
        rule_line = f'- "{property_key}" must be one of [true|false|unknown]'
    else:
        value_hint = "array"
        rule_line = f'- "{property_key}" must be an array with values from [{"|".join(spec.allowed_values)}], or []'

    desc_line = f"Property description: {spec.description}" if spec.description else ""

    return f"""
You are given an image of one product.
The image_id is "{image_id}". The output JSON must contain exactly this image_id.

Predict ONLY this property: "{property_key}".

Return ONLY JSON with this structure:
{{
  "image_id": "{image_id}",
  "primary_object": "<short noun phrase>",
  "properties": {{
    "{property_key}": <{value_hint}>
  }},
  "notes": "<short visual evidence>"
}}

Rules:
{rule_line}
- Do not add any other property keys.
- For unknown categorical/boolean value use "unknown".
- For unknown list property use [].
- No markdown, no code fences, no extra text.
{desc_line}
""".strip()


def build_demo_response_payload(
    image_id: str,
    property_key: str,
    gt_properties: Dict[str, Any],
    property_specs: Dict[str, PropertySpec],
) -> Dict[str, Any]:
    spec = property_specs[property_key]
    default = [] if spec.value_type == "multi_categorical" else "unknown"
    return {
        "image_id": image_id,
        "primary_object": "object",
        "properties": {property_key: gt_properties.get(property_key, default)},
        "notes": "reference example",
    }


def select_few_shot_examples(
    current_image_id: str,
    samples: Sequence[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    few_shot_k: int,
    property_key: str,
    selection_mode: str = "fixed",
    fixed_candidates: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if few_shot_k <= 0:
        return []

    if selection_mode == "fixed":
        base_candidates = fixed_candidates if fixed_candidates is not None else samples
        selected: List[Dict[str, Any]] = []
        for sample in base_candidates:
            if str(sample.get("image_id")) == current_image_id:
                continue
            selected.append(sample)
            if len(selected) >= few_shot_k:
                break
        return selected

    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for sample in samples:
        image_id = str(sample.get("image_id"))
        if image_id == current_image_id:
            continue
        score = int(
            is_known_value(property_specs[property_key], sample.get("gt_properties", {}).get(property_key))
        )
        if score > 0:
            scored.append((score, image_id, sample))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sample for _, _, sample in scored[:few_shot_k]]


def build_fixed_few_shot_candidates(
    samples: Sequence[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for sample in samples:
        image_id = str(sample.get("image_id"))
        gt_props = sample.get("gt_properties", {})
        score = sum(1 for key, spec in property_specs.items() if is_known_value(spec, gt_props.get(key)))
        if score > 0:
            scored.append((score, image_id, sample))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sample for _, _, sample in scored]


def _build_per_property_few_shot_messages(
    image_id: str,
    image: Image.Image,
    property_key: str,
    spec: PropertySpec,
    demos: Sequence[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    variant: str,
    mask_background_mode: str,
    image_cache: Optional[Dict[Tuple[str, str, str, str], Image.Image]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    messages: List[Dict[str, Any]] = []
    images: List[Image.Image] = []

    for demo in demos:
        demo_prompt = build_property_prompt(str(demo["image_id"]), property_key, spec)
        demo_image = load_variant_image_cached(demo, variant, mask_background_mode, image_cache)
        demo_payload = build_demo_response_payload(
            str(demo["image_id"]),
            property_key,
            demo.get("gt_properties", {}),
            property_specs,
        )
        messages.append(
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": demo_prompt}]}
        )
        messages.append({"role": "assistant", "content": json.dumps(demo_payload, ensure_ascii=False)})
        images.append(demo_image)

    query_prompt = build_property_prompt(image_id, property_key, spec)
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query_prompt}]})
    images.append(image)
    return messages, images


def evaluate_one(
    runtime: VLMRuntime,
    sample_meta: Dict[str, Any],
    few_shot_pool: Sequence[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    variant: str,
    property_batch_size: int,
    include_only_gt_known: bool,
    few_shot_k: int,
    few_shot_selection_mode: str,
    fixed_few_shot_candidates: Optional[Sequence[Dict[str, Any]]],
    mask_background_mode: str,
    image_cache: Optional[Dict[Tuple[str, str, str, str], Image.Image]],
    save_raw_output: bool,
    json_success_threshold: float,
    image_id_success_threshold: Optional[float],
) -> Dict[str, Any]:
    image_id = str(sample_meta["image_id"])
    image = load_variant_image_cached(sample_meta, variant, mask_background_mode, image_cache)
    gt_props = sample_meta["gt_properties"]
    selected_keys = select_property_keys_for_sample(gt_props, property_specs, include_only_gt_known)

    pred_props = {
        key: ([] if spec.value_type == "multi_categorical" else "unknown")
        for key, spec in property_specs.items()
    }
    raw_map: Dict[str, str] = {}
    parse_errors: List[str] = []
    valid_json_count = 0
    image_id_match_count = 0
    primary_object_pred = None
    notes_pred = None
    few_shot_demo_ids: List[str] = []

    shared_demos: Optional[List[Dict[str, Any]]] = None
    if few_shot_k > 0 and few_shot_selection_mode == "fixed":
        shared_demos = select_few_shot_examples(
            current_image_id=image_id,
            samples=few_shot_pool,
            property_specs=property_specs,
            few_shot_k=few_shot_k,
            property_key=selected_keys[0],
            selection_mode=few_shot_selection_mode,
            fixed_candidates=fixed_few_shot_candidates,
        )
        few_shot_demo_ids = [str(x["image_id"]) for x in shared_demos]

    effective_batch_size = int(property_batch_size)
    if runtime.prompt_batch_cap is not None:
        effective_batch_size = min(effective_batch_size, int(runtime.prompt_batch_cap))
    effective_batch_size = max(1, effective_batch_size)

    for key_chunk in chunked(selected_keys, effective_batch_size):
        if few_shot_k > 0:
            messages_batch: List[List[Dict[str, Any]]] = []
            images_batch: List[List[Image.Image]] = []
            for key in key_chunk:
                demos = shared_demos
                if demos is None:
                    demos = select_few_shot_examples(
                        current_image_id=image_id,
                        samples=few_shot_pool,
                        property_specs=property_specs,
                        few_shot_k=few_shot_k,
                        property_key=key,
                        selection_mode=few_shot_selection_mode,
                        fixed_candidates=fixed_few_shot_candidates,
                    )
                    if not few_shot_demo_ids:
                        few_shot_demo_ids = [str(x["image_id"]) for x in demos]
                messages, images = _build_per_property_few_shot_messages(
                    image_id=image_id,
                    image=image,
                    property_key=key,
                    spec=property_specs[key],
                    demos=demos,
                    property_specs=property_specs,
                    variant=variant,
                    mask_background_mode=mask_background_mode,
                    image_cache=image_cache,
                )
                messages_batch.append(messages)
                images_batch.append(images)
            raw_chunk = infer_runtime_messages_batch(runtime, messages_batch, images_batch)
        else:
            prompts = [build_property_prompt(image_id, key, property_specs[key]) for key in key_chunk]
            raw_chunk = infer_runtime_batch(runtime, image, prompts)

        for key, raw in zip(key_chunk, raw_chunk):
            raw_map[key] = raw
            parsed_json, parse_error = parse_model_output(raw)
            if parse_error:
                parse_errors.append(f"{key}: {parse_error}")
            else:
                valid_json_count += 1

            pred = normalize_pred(parsed_json, {key: property_specs[key]})
            pred_props[key] = pred["properties"][key]
            if pred.get("image_id") == image_id:
                image_id_match_count += 1
            if primary_object_pred is None and pred.get("primary_object") is not None:
                primary_object_pred = pred.get("primary_object")
            if notes_pred is None and pred.get("notes") is not None:
                notes_pred = pred.get("notes")

    requested_count = len(selected_keys)
    expected_response_count = requested_count
    valid_json_ratio = float(valid_json_count) / float(expected_response_count) if expected_response_count > 0 else 0.0
    image_id_match_ratio = (
        float(image_id_match_count) / float(expected_response_count) if expected_response_count > 0 else 0.0
    )
    has_valid_json_all = valid_json_count == expected_response_count if expected_response_count else False
    image_id_matched_all = image_id_match_count == expected_response_count if expected_response_count else False
    image_id_threshold = json_success_threshold if image_id_success_threshold is None else image_id_success_threshold

    row = {
        "variant": variant,
        "image_id": image_id,
        "path": sample_meta.get("path"),
        "mask_path": sample_meta.get("mask_path"),
        "has_valid_json": valid_json_ratio >= json_success_threshold,
        "has_valid_json_all": has_valid_json_all,
        "parse_error": None if not parse_errors else " | ".join(parse_errors[:30]),
        "image_id_matched": image_id_match_ratio >= image_id_threshold,
        "image_id_matched_all": image_id_matched_all,
        "valid_json_count": valid_json_count,
        "image_id_match_count": image_id_match_count,
        "valid_json_ratio": round(valid_json_ratio, 6),
        "image_id_match_ratio": round(image_id_match_ratio, 6),
        "json_success_threshold": json_success_threshold,
        "image_id_success_threshold": image_id_threshold,
        "primary_object_pred": primary_object_pred,
        "notes_pred": notes_pred,
        "requested_property_count": requested_count,
        "expected_response_count": expected_response_count,
        "requested_property_keys": "|".join(selected_keys),
        "property_batch_size": int(property_batch_size),
        "effective_property_batch_size": int(effective_batch_size),
        "few_shot_k": int(few_shot_k),
        "few_shot_selection_mode": few_shot_selection_mode,
        "few_shot_demo_ids": "|".join(few_shot_demo_ids),
    }

    for key, spec in property_specs.items():
        col = column_prefix(key)
        gt_val = gt_props[key]
        pred_val = pred_props[key]
        gt_known = is_known_value(spec, gt_val)
        pred_known = is_known_value(spec, pred_val)
        row[f"{col}_gt"] = serialize_value(gt_val)
        row[f"{col}_pred"] = serialize_value(pred_val)
        row[f"{col}_gt_known"] = gt_known
        row[f"{col}_pred_known"] = pred_known
        row[f"{col}_missed_when_gt_known"] = gt_known and (not pred_known)
        row[f"{col}_exact_match"] = gt_known and values_equal(spec, pred_val, gt_val)

    if save_raw_output:
        row["raw_output"] = json.dumps(raw_map, ensure_ascii=False)
    return row


def run_validation(
    model_key: str,
    samples: List[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    model_registry: Dict[str, Dict[str, Any]],
    variant: str = "raw",
    property_batch_size: int = 8,
    include_only_gt_known: bool = True,
    few_shot_k: int = 0,
    few_shot_selection_mode: str = "fixed",
    mask_background_mode: str = "black",
    save_raw_output: bool = True,
    json_success_threshold: float = 1.0,
    image_id_success_threshold: Optional[float] = None,
) -> pd.DataFrame:
    runtime = None
    rows: List[Dict[str, Any]] = []
    image_cache: Dict[Tuple[str, str, str, str], Image.Image] = {}
    fixed_few_shot_candidates = (
        build_fixed_few_shot_candidates(samples, property_specs)
        if few_shot_k > 0 and few_shot_selection_mode == "fixed"
        else None
    )

    try:
        runtime = load_runtime(model_key, model_registry)
        for sample_meta in tqdm(samples, desc=f"Evaluating {model_key} [{variant}]"):
            try:
                row = evaluate_one(
                    runtime=runtime,
                    sample_meta=sample_meta,
                    few_shot_pool=samples,
                    property_specs=property_specs,
                    variant=variant,
                    property_batch_size=property_batch_size,
                    include_only_gt_known=include_only_gt_known,
                    few_shot_k=few_shot_k,
                    few_shot_selection_mode=few_shot_selection_mode,
                    fixed_few_shot_candidates=fixed_few_shot_candidates,
                    mask_background_mode=mask_background_mode,
                    image_cache=image_cache,
                    save_raw_output=save_raw_output,
                    json_success_threshold=json_success_threshold,
                    image_id_success_threshold=image_id_success_threshold,
                )
            except Exception as exc:
                image_id = str(sample_meta.get("image_id", "unknown"))
                row = {
                    "variant": variant,
                    "image_id": image_id,
                    "path": sample_meta.get("path"),
                    "mask_path": sample_meta.get("mask_path"),
                    "has_valid_json": False,
                    "has_valid_json_all": False,
                    "parse_error": f"runtime_error: {exc}",
                    "image_id_matched": False,
                    "image_id_matched_all": False,
                    "valid_json_count": 0,
                    "image_id_match_count": 0,
                    "valid_json_ratio": 0.0,
                    "image_id_match_ratio": 0.0,
                    "json_success_threshold": json_success_threshold,
                    "image_id_success_threshold": (
                        image_id_success_threshold
                        if image_id_success_threshold is not None
                        else json_success_threshold
                    ),
                    "primary_object_pred": None,
                    "notes_pred": None,
                    "requested_property_count": 0,
                    "expected_response_count": 0,
                    "requested_property_keys": "",
                    "property_batch_size": int(property_batch_size),
                    "effective_property_batch_size": int(
                        min(property_batch_size, runtime.prompt_batch_cap)
                        if runtime and runtime.prompt_batch_cap is not None
                        else property_batch_size
                    ),
                    "few_shot_k": int(few_shot_k),
                    "few_shot_selection_mode": few_shot_selection_mode,
                    "few_shot_demo_ids": "",
                }
                gt_props = sample_meta.get("gt_properties", {})
                for key, spec in property_specs.items():
                    col = column_prefix(key)
                    gt_val = gt_props.get(key, [] if spec.value_type == "multi_categorical" else "unknown")
                    gt_known = is_known_value(spec, gt_val)
                    row[f"{col}_gt"] = serialize_value(gt_val)
                    row[f"{col}_pred"] = "" if spec.value_type == "multi_categorical" else "unknown"
                    row[f"{col}_gt_known"] = gt_known
                    row[f"{col}_pred_known"] = False
                    row[f"{col}_missed_when_gt_known"] = gt_known
                    row[f"{col}_exact_match"] = False
                if save_raw_output:
                    row["raw_output"] = None
            rows.append(row)
        if runtime is not None:
            print(
                f"Runtime stats for {model_key} [{variant}]: "
                f"{json.dumps(runtime.stats, ensure_ascii=False, sort_keys=True)}"
            )
    finally:
        unload_runtime(runtime)
    df = pd.DataFrame(rows)
    if runtime is not None:
        df.attrs["runtime_stats"] = dict(runtime.stats)
    return df


def get_available_variants(requested_variants: Sequence[str], samples: List[Dict[str, Any]]) -> List[str]:
    variants: List[str] = []
    has_masks = any("mask_path" in sample for sample in samples)
    if "raw" in requested_variants:
        variants.append("raw")
    if "mask_overlay" in requested_variants and has_masks:
        variants.append("mask_overlay")
    if "masked" in requested_variants and has_masks:
        variants.append("masked")
    return variants


def run_many_models(
    model_keys: List[str],
    samples: List[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    model_registry: Dict[str, Dict[str, Any]],
    reports_dir: Path,
    variants: Optional[List[str]] = None,
    property_batch_size: int = 8,
    include_only_gt_known: bool = True,
    few_shot_k: int = 0,
    few_shot_selection_mode: str = "fixed",
    mask_background_mode: str = "black",
    save_raw_output: bool = True,
    json_success_threshold: float = 1.0,
    image_id_success_threshold: Optional[float] = None,
    comet_experiment=None,
) -> pd.DataFrame:
    if variants is None:
        variants = get_available_variants(["raw", "mask_overlay", "masked"], samples)

    rows: List[Dict[str, Any]] = []
    for model_key in model_keys:
        for variant in variants:
            print(f"\n######## Running {model_key} [{variant}] ########")
            df = run_validation(
                model_key=model_key,
                samples=samples,
                property_specs=property_specs,
                model_registry=model_registry,
                variant=variant,
                property_batch_size=property_batch_size,
                include_only_gt_known=include_only_gt_known,
                few_shot_k=few_shot_k,
                few_shot_selection_mode=few_shot_selection_mode,
                mask_background_mode=mask_background_mode,
                save_raw_output=save_raw_output,
                json_success_threshold=json_success_threshold,
                image_id_success_threshold=image_id_success_threshold,
            )
            pm, summary = save_report(
                df=df,
                model_key=model_key,
                variant=variant,
                model_registry=model_registry,
                property_specs=property_specs,
                reports_dir=reports_dir,
                comet_experiment=comet_experiment,
            )
            for _, row in pm.iterrows():
                rows.append(
                    {
                        "model_key": model_key,
                        "variant": variant,
                        "property_batch_size": property_batch_size,
                        "few_shot_k": few_shot_k,
                        "few_shot_selection_mode": few_shot_selection_mode,
                        "property": row["property"],
                        "group": row["group"],
                        "value_type": row["value_type"],
                        "coverage_pct": row["coverage_pct"],
                        "accuracy_pct": row["accuracy_pct"],
                        "macro_f1_pct": row["macro_f1_pct"],
                        "selective_accuracy_pct": row["selective_accuracy_pct"],
                        "coverage_on_gt_known_pct": row["coverage_on_gt_known_pct"],
                        "exact_match_on_gt_known_pct": row["exact_match_on_gt_known_pct"],
                        "num_samples": summary["num_samples"],
                    }
                )
    return pd.DataFrame(rows)
