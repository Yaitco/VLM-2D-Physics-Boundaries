"""Validation pipeline for dataset/abo_150_expanded in Colab."""

from __future__ import annotations

import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
except Exception:  # pragma: no cover - optional during non-inference tooling
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


MODEL_REGISTRY = {
    "qwen3_vl_8b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 1024,
    },
    "qwen2_5_vl_7b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 1024,
    },
    "llava_onevision_1_5_8b": {
        "backend": "hf_chat",
        "model_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 1024,
    },
}


@dataclass
class PropertySpec:
    key: str
    group: str
    name: str
    value_type: str  # categorical | boolean | multi_categorical
    allowed_values: List[str]
    description: str


@dataclass
class VLMRuntime:
    name: str
    backend: str
    model_id: str
    processor: Any
    model: Any
    gen_kwargs: Dict[str, Any]


def get_secret_value(secret_name: str) -> Optional[str]:
    try:
        from google.colab import userdata

        value = userdata.get(secret_name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(secret_name.upper()) or os.getenv(secret_name)


def init_comet_experiment(
    run_tag: str,
    run_params: Dict[str, Any],
    enabled: bool = True,
    default_project: str = "vlm-physics-validation",
):
    if not enabled:
        print("Comet: disabled.")
        return None

    api_key = get_secret_value("comet_api_key")
    workspace = get_secret_value("comet_workspace")
    project_name = get_secret_value("comet_project_name") or default_project

    if not api_key:
        print(
            "Comet: API key is missing. "
            "Add 'comet_api_key' in Colab userdata (Secrets) or set COMET_API_KEY."
        )
        return None

    try:
        from comet_ml import Experiment
    except Exception as exc:
        print(f"Comet: comet_ml import failed: {exc}")
        return None

    kwargs = {
        "api_key": api_key,
        "project_name": project_name,
        "auto_output_logging": "simple",
    }
    if workspace:
        kwargs["workspace"] = workspace

    try:
        exp = Experiment(**kwargs)
    except Exception as exc:
        print(f"Comet: failed to initialize Experiment: {exc}")
        return None

    exp.set_name(run_tag)
    exp.log_parameters(run_params)
    print(f"Comet: logging to project='{project_name}' workspace='{workspace}'")
    return exp


def _normalize_token(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower()).replace("-", "_")


def _load_enums(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    enums = schema.get("enums", {})
    out: Dict[str, List[str]] = {}
    if not isinstance(enums, dict):
        return out

    for enum_name, values in enums.items():
        if isinstance(values, list):
            out[str(enum_name)] = [_normalize_token(str(v)) for v in values]
    return out


def load_property_specs(
    schema_path: Path,
    include_groups: Optional[Sequence[str]] = None,
) -> Dict[str, PropertySpec]:
    """Load evaluable properties from ontology (categorical, boolean, list[categorical])."""
    with schema_path.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    groups = schema.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"Invalid schema: groups are missing in {schema_path}")

    enums_map = _load_enums(schema)
    specs: Dict[str, PropertySpec] = {}

    for group_name, group_cfg in groups.items():
        if include_groups and group_name not in include_groups:
            continue

        properties = {}
        if isinstance(group_cfg, dict):
            properties = group_cfg.get("properties", {})
        if not isinstance(properties, dict):
            continue

        for prop_name, prop_cfg in properties.items():
            if not isinstance(prop_cfg, dict):
                continue

            prop_type = prop_cfg.get("type")
            key = f"{group_name}.{prop_name}"
            desc = str(prop_cfg.get("description") or "").strip()

            if prop_type == "categorical":
                enum_ref = prop_cfg.get("enum_ref")
                allowed = list(enums_map.get(str(enum_ref), []))
                if "unknown" not in allowed:
                    allowed.append("unknown")
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type="categorical",
                    allowed_values=allowed,
                    description=desc,
                )
                continue

            if prop_type == "boolean":
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type="boolean",
                    allowed_values=["true", "false", "unknown"],
                    description=desc,
                )
                continue

            if prop_type == "list":
                items = prop_cfg.get("items", {})
                if not isinstance(items, dict):
                    continue
                if items.get("type") != "categorical":
                    continue
                enum_ref = items.get("enum_ref")
                allowed = list(enums_map.get(str(enum_ref), []))
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type="multi_categorical",
                    allowed_values=allowed,
                    description=desc,
                )

    if not specs:
        raise ValueError("No evaluable properties found in physics schema.")
    return specs


def _normalize_categorical(value: Any, allowed: Sequence[str]) -> str:
    allowed_set = set(allowed)
    if value is None:
        return "unknown"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)

    if not isinstance(value, str):
        return "unknown"

    token = _normalize_token(value)
    if token in {"", "none", "null", "nan"}:
        return "unknown"
    if token in allowed_set:
        return token

    token_alt = token.replace("/", "_")
    return token_alt if token_alt in allowed_set else "unknown"


def _normalize_boolean(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if value is None:
        return "unknown"

    token = _normalize_token(str(value))
    if token in {"true", "1", "yes", "y"}:
        return "true"
    if token in {"false", "0", "no", "n"}:
        return "false"
    return "unknown"


def _normalize_multi_categorical(value: Any, allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)

    if value is None:
        return []

    raw_values: Iterable[Any]
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        if value.strip() in {"", "none", "null", "unknown"}:
            return []
        raw_values = [v.strip() for v in value.split(",")]
    else:
        raw_values = [value]

    out: List[str] = []
    for item in raw_values:
        token = _normalize_token(str(item))
        if token in {"", "none", "null", "unknown"}:
            continue
        if token in allowed_set and token not in out:
            out.append(token)
    return sorted(out)


def normalize_value(spec: PropertySpec, value: Any) -> Any:
    if spec.value_type == "categorical":
        return _normalize_categorical(value, spec.allowed_values)
    if spec.value_type == "boolean":
        return _normalize_boolean(value)
    if spec.value_type == "multi_categorical":
        return _normalize_multi_categorical(value, spec.allowed_values)
    raise ValueError(f"Unsupported spec type: {spec.value_type}")


def is_known_value(spec: PropertySpec, value: Any) -> bool:
    if spec.value_type == "multi_categorical":
        return isinstance(value, list) and len(value) > 0
    return value in {"true", "false"} if spec.value_type == "boolean" else value != "unknown"


def values_equal(spec: PropertySpec, pred_value: Any, gt_value: Any) -> bool:
    if spec.value_type == "multi_categorical":
        pred_set = set(pred_value or [])
        gt_set = set(gt_value or [])
        return pred_set == gt_set
    return pred_value == gt_value


def serialize_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(value)
    if value is None:
        return ""
    return str(value)


def _resolve_panel_path(
    panel_path_raw: str,
    dataset_dir: Path,
    panels_dir: Path,
    obj_id: str,
) -> Path:
    candidates: List[Path] = []

    if panel_path_raw:
        p = Path(panel_path_raw)
        if p.is_absolute():
            candidates.append(p)
            candidates.append(panels_dir / p.name)
        else:
            candidates.append(dataset_dir / p)
            candidates.append(panels_dir / p.name)

    candidates.append(panels_dir / f"{obj_id}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Panel image not found for obj_id={obj_id}. "
        f"Tried: {[str(x) for x in candidates]}"
    )


def _extract_gt_properties(
    record: Dict[str, Any],
    property_specs: Dict[str, PropertySpec],
) -> Dict[str, Any]:
    groups = record.get("groups", {})
    if not isinstance(groups, dict):
        groups = {}

    gt: Dict[str, Any] = {}
    for key, spec in property_specs.items():
        group_payload = groups.get(spec.group, {})
        raw_value = None
        if isinstance(group_payload, dict):
            raw_value = group_payload.get(spec.name)
        gt[key] = normalize_value(spec, raw_value)
    return gt


def load_abo150_samples(
    annotations_path: Path,
    dataset_dir: Path,
    property_specs: Dict[str, PropertySpec],
    max_samples: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    panels_dir = dataset_dir / "selected_150_photos" / "panels"
    rows: List[Dict[str, Any]] = []

    with annotations_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            obj_id = str(record.get("obj_id") or "")
            if not obj_id:
                continue

            panels = record.get("panels", {})
            panel_path_raw = ""
            if isinstance(panels, dict):
                panel_path_raw = str(panels.get("panel_path") or "")

            panel_path = _resolve_panel_path(
                panel_path_raw=panel_path_raw,
                dataset_dir=dataset_dir,
                panels_dir=panels_dir,
                obj_id=obj_id,
            )

            sample = {
                "image_id": obj_id,
                "path": str(panel_path),
                "split": record.get("split"),
                "caption": record.get("caption"),
                "gt_properties": _extract_gt_properties(record, property_specs),
            }

            mask_path = None
            if isinstance(panels, dict):
                if isinstance(panels.get("mask_path"), str):
                    mask_path = panels["mask_path"]
            if isinstance(mask_path, str) and mask_path.strip():
                mp = Path(mask_path)
                if not mp.is_absolute():
                    mp = dataset_dir / mp
                if mp.exists():
                    sample["mask_path"] = str(mp.resolve())

            rows.append(sample)

    if max_samples is not None and len(rows) > max_samples:
        rng = random.Random(random_seed)
        rows = rng.sample(rows, k=max_samples)

    if not rows:
        raise RuntimeError("Loaded 0 samples from annotations.")
    return rows


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def find_first_json_object(text: str) -> Optional[str]:
    s = strip_code_fences(text)
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_model_output(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    blob = find_first_json_object(raw_text)
    if blob is None:
        return None, "json_not_found"
    try:
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            return None, "json_not_object"
        return parsed, None
    except Exception as exc:
        return None, f"json_parse_error: {exc}"


def _fetch_pred_raw_value(parsed_json: Dict[str, Any], key: str) -> Any:
    group, name = key.split(".", 1)

    props = parsed_json.get("properties", {})
    if isinstance(props, dict):
        if key in props:
            return props[key]
        g_payload = props.get(group)
        if isinstance(g_payload, dict) and name in g_payload:
            return g_payload[name]

    groups = parsed_json.get("groups", {})
    if isinstance(groups, dict):
        g_payload = groups.get(group)
        if isinstance(g_payload, dict) and name in g_payload:
            return g_payload[name]

    return None


def normalize_pred(
    parsed_json: Optional[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
) -> Dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return {
            "image_id": None,
            "primary_object": None,
            "notes": None,
            "properties": {
                key: [] if spec.value_type == "multi_categorical" else "unknown"
                for key, spec in property_specs.items()
            },
        }

    pred_props: Dict[str, Any] = {}
    for key, spec in property_specs.items():
        raw_value = _fetch_pred_raw_value(parsed_json, key)
        pred_props[key] = normalize_value(spec, raw_value)

    return {
        "image_id": str(parsed_json.get("image_id")).strip() if parsed_json.get("image_id") is not None else None,
        "primary_object": parsed_json.get("primary_object"),
        "notes": parsed_json.get("notes"),
        "properties": pred_props,
    }


def build_prompt_for_sample(
    image_id: str,
    gt_properties: Dict[str, Any],
    property_specs: Dict[str, PropertySpec],
    include_only_gt_known: bool = True,
    max_properties_per_sample: Optional[int] = None,
) -> Tuple[str, List[str]]:
    property_keys = list(property_specs.keys())

    if include_only_gt_known:
        selected = [
            key
            for key in property_keys
            if is_known_value(property_specs[key], gt_properties.get(key))
        ]
    else:
        selected = property_keys

    if not selected:
        selected = property_keys[: min(12, len(property_keys))]

    if max_properties_per_sample is not None and len(selected) > max_properties_per_sample:
        selected = selected[:max_properties_per_sample]

    lines = []
    for key in selected:
        spec = property_specs[key]
        if spec.value_type == "categorical":
            allowed = "|".join(spec.allowed_values)
            lines.append(f"- {key}: one of [{allowed}]")
        elif spec.value_type == "boolean":
            lines.append(f"- {key}: one of [true|false|unknown]")
        else:
            allowed = "|".join(spec.allowed_values)
            lines.append(f"- {key}: JSON array of values from [{allowed}], or []")
        if spec.description:
            lines.append(f"  desc: {spec.description}")

    prop_block = "\n".join(lines)
    key_block = ", ".join(selected)

    prompt = f"""
You are given an image of one product.
The image_id is "{image_id}". The output JSON must contain exactly this image_id.

Return ONLY JSON with this structure:
{{
  "image_id": "{image_id}",
  "primary_object": "<short noun phrase>",
  "properties": {{
    "<property_key>": "<value or array>"
  }},
  "notes": "<short visual evidence>"
}}

Rules:
- Fill ONLY these property keys: {key_block}
- Do not add any extra property keys.
- For unknown categorical/boolean value use "unknown".
- For unknown list property use [].
- Use only values allowed per property definition below.
- No markdown, no code fences, no extra text.

Property definitions:
{prop_block}
""".strip()

    return prompt, selected


def make_bnb_config(use_4bit: bool):
    if BitsAndBytesConfig is None:
        return None
    if not use_4bit:
        return None
    if not torch.cuda.is_available():
        print("CUDA is not available. 4-bit quantization disabled.")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_hf_chat_runtime(name: str, cfg: Dict[str, Any]) -> VLMRuntime:
    if AutoProcessor is None or AutoModelForImageTextToText is None or AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is not installed. Install notebook dependencies first."
        )
    model_id = cfg["model_id"]
    use_4bit = bool(cfg.get("use_4bit", True))
    gen_kwargs = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "do_sample": bool(cfg.get("do_sample", False)),
    }

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    bnb = make_bnb_config(use_4bit)
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        print(f"AutoModelForImageTextToText failed: {exc}")
        print("Falling back to AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    return VLMRuntime(
        name=name,
        backend="hf_chat",
        model_id=model_id,
        processor=processor,
        model=model,
        gen_kwargs=gen_kwargs,
    )


def infer_hf_chat(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if hasattr(runtime.processor, "apply_chat_template"):
        try:
            text = runtime.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            text = runtime.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
    else:
        text = prompt

    inputs = runtime.processor(
        text=text,
        images=image,
        return_tensors="pt",
        truncation=False,
    )

    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {
            key: value.to(runtime.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    with torch.no_grad():
        out = runtime.model.generate(**inputs, **runtime.gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = out[:, input_len:]
    return runtime.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]


BACKEND_LOADERS = {"hf_chat": load_hf_chat_runtime}
BACKEND_INFER = {"hf_chat": infer_hf_chat}


def load_runtime(model_key: str, model_registry: Dict[str, Dict[str, Any]]) -> VLMRuntime:
    cfg = model_registry[model_key]
    loader = BACKEND_LOADERS[cfg["backend"]]
    runtime = loader(model_key, cfg)
    print(f"Loaded model: {runtime.name} -> {runtime.model_id}")
    return runtime


def infer_runtime(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    return BACKEND_INFER[runtime.backend](runtime, image, prompt)


def unload_runtime(runtime: Optional[VLMRuntime]) -> None:
    if runtime is None:
        return
    try:
        del runtime.model
    except Exception:
        pass
    try:
        del runtime.processor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_mask_to_image(image: Image.Image, mask: Image.Image, bg_mode: str = "black") -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    out = rgb.copy()
    if bg_mode == "white":
        out[~m] = 255
    else:
        out[~m] = 0
    return Image.fromarray(out, mode="RGB")


def load_variant_image(sample_meta: Dict[str, Any], variant: str, mask_background_mode: str) -> Image.Image:
    image_path = Path(sample_meta["path"])
    image = Image.open(image_path).convert("RGB")

    if variant == "raw":
        return image
    if variant == "masked":
        mask_path = sample_meta.get("mask_path")
        if not mask_path:
            raise FileNotFoundError("mask_path is missing in sample metadata")
        mask = Image.open(Path(mask_path)).convert("L")
        return apply_mask_to_image(image, mask, bg_mode=mask_background_mode)
    raise ValueError(f"Unknown variant: {variant}")


def _column_prefix(property_key: str) -> str:
    return property_key.replace(".", "__")


def evaluate_one(
    runtime: VLMRuntime,
    sample_meta: Dict[str, Any],
    property_specs: Dict[str, PropertySpec],
    variant: str,
    include_only_gt_known: bool,
    max_properties_per_sample: Optional[int],
    mask_background_mode: str,
    save_raw_output: bool,
) -> Dict[str, Any]:
    image_id = str(sample_meta["image_id"])
    image = load_variant_image(
        sample_meta=sample_meta,
        variant=variant,
        mask_background_mode=mask_background_mode,
    )

    gt_props = sample_meta["gt_properties"]
    prompt, selected_keys = build_prompt_for_sample(
        image_id=image_id,
        gt_properties=gt_props,
        property_specs=property_specs,
        include_only_gt_known=include_only_gt_known,
        max_properties_per_sample=max_properties_per_sample,
    )

    raw = infer_runtime(runtime, image, prompt)
    parsed_json, parse_error = parse_model_output(raw)
    pred = normalize_pred(parsed_json, property_specs)

    row = {
        "variant": variant,
        "image_id": image_id,
        "path": sample_meta.get("path"),
        "mask_path": sample_meta.get("mask_path"),
        "has_valid_json": parse_error is None,
        "parse_error": parse_error,
        "image_id_matched": pred.get("image_id") == image_id,
        "primary_object_pred": pred.get("primary_object"),
        "notes_pred": pred.get("notes"),
        "requested_property_count": len(selected_keys),
        "requested_property_keys": "|".join(selected_keys),
    }

    pred_props = pred["properties"]
    for key, spec in property_specs.items():
        col = _column_prefix(key)
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
        row["raw_output"] = raw

    return row


def run_validation(
    model_key: str,
    samples: List[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    model_registry: Dict[str, Dict[str, Any]],
    variant: str = "raw",
    include_only_gt_known: bool = True,
    max_properties_per_sample: Optional[int] = None,
    mask_background_mode: str = "black",
    save_raw_output: bool = True,
) -> pd.DataFrame:
    runtime = None
    rows = []

    try:
        runtime = load_runtime(model_key, model_registry)
        for sample_meta in tqdm(samples, desc=f"Evaluating {model_key} [{variant}]"):
            try:
                row = evaluate_one(
                    runtime=runtime,
                    sample_meta=sample_meta,
                    property_specs=property_specs,
                    variant=variant,
                    include_only_gt_known=include_only_gt_known,
                    max_properties_per_sample=max_properties_per_sample,
                    mask_background_mode=mask_background_mode,
                    save_raw_output=save_raw_output,
                )
            except Exception as exc:
                image_id = str(sample_meta.get("image_id", "unknown"))
                row = {
                    "variant": variant,
                    "image_id": image_id,
                    "path": sample_meta.get("path"),
                    "mask_path": sample_meta.get("mask_path"),
                    "has_valid_json": False,
                    "parse_error": f"runtime_error: {exc}",
                    "image_id_matched": False,
                    "primary_object_pred": None,
                    "notes_pred": None,
                    "requested_property_count": 0,
                    "requested_property_keys": "",
                }

                gt_props = sample_meta.get("gt_properties", {})
                for key, spec in property_specs.items():
                    col = _column_prefix(key)
                    gt_val = gt_props.get(
                        key, [] if spec.value_type == "multi_categorical" else "unknown"
                    )
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
    finally:
        unload_runtime(runtime)

    return pd.DataFrame(rows)


def build_property_metrics(df: pd.DataFrame, property_specs: Dict[str, PropertySpec]) -> pd.DataFrame:
    total = len(df)
    metrics = []

    for key, spec in property_specs.items():
        col = _column_prefix(key)
        pred_yes = int(df[f"{col}_pred_known"].sum())
        pred_no = int(total - pred_yes)

        gt_known = int(df[f"{col}_gt_known"].sum())
        missed = int(df[f"{col}_missed_when_gt_known"].sum())
        extracted_when_gt_known = int(gt_known - missed)
        exact = int(df[f"{col}_exact_match"].sum())

        metrics.append(
            {
                "property": key,
                "group": spec.group,
                "value_type": spec.value_type,
                "pred_yes_count": pred_yes,
                "pred_no_count": pred_no,
                "pred_yes_pct": round(100.0 * pred_yes / total, 2) if total else 0.0,
                "gt_known_count": gt_known,
                "extracted_when_gt_known": extracted_when_gt_known,
                "missed_when_gt_known": missed,
                "coverage_on_gt_known_pct": round(100.0 * extracted_when_gt_known / gt_known, 2)
                if gt_known
                else None,
                "exact_match_on_gt_known_pct": round(100.0 * exact / gt_known, 2)
                if gt_known
                else None,
            }
        )

    return pd.DataFrame(metrics)


def _safe_metric_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


def log_report_to_comet(
    comet_experiment,
    model_key: str,
    variant: str,
    property_metrics: pd.DataFrame,
    summary: Dict[str, Any],
    per_sample_path: Path,
    property_path: Path,
    summary_path: Path,
):
    if comet_experiment is None:
        return

    summary_metrics = {
        f"{model_key}/{variant}/num_samples": summary["num_samples"],
        f"{model_key}/{variant}/valid_json_pct": summary["valid_json_pct"],
        f"{model_key}/{variant}/image_id_match_pct": summary["image_id_match_pct"],
    }
    comet_experiment.log_metrics(summary_metrics)

    for _, row in property_metrics.iterrows():
        prop = _safe_metric_key(str(row["property"]))
        for col in [
            "pred_yes_pct",
            "coverage_on_gt_known_pct",
            "exact_match_on_gt_known_pct",
            "pred_yes_count",
            "pred_no_count",
            "missed_when_gt_known",
        ]:
            val = row[col]
            if pd.isna(val):
                continue
            comet_experiment.log_metric(f"{model_key}/{variant}/{prop}/{col}", float(val))

    comet_experiment.log_asset(
        str(per_sample_path),
        file_name=f"{model_key}_{variant}_per_sample_predictions.csv",
    )
    comet_experiment.log_asset(
        str(property_path),
        file_name=f"{model_key}_{variant}_property_metrics.csv",
    )
    comet_experiment.log_asset(
        str(summary_path),
        file_name=f"{model_key}_{variant}_summary.json",
    )


def save_report(
    df: pd.DataFrame,
    model_key: str,
    variant: str,
    model_registry: Dict[str, Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    reports_dir: Path,
    comet_experiment=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_dir = reports_dir / model_key / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    property_metrics = build_property_metrics(df, property_specs)

    per_sample_path = run_dir / "per_sample_predictions.csv"
    property_path = run_dir / "property_metrics.csv"
    summary_path = run_dir / "summary.json"

    df.to_csv(per_sample_path, index=False)
    property_metrics.to_csv(property_path, index=False)

    summary = {
        "model_key": model_key,
        "model_id": model_registry[model_key]["model_id"],
        "variant": variant,
        "num_samples": int(len(df)),
        "valid_json_count": int(df["has_valid_json"].sum()),
        "valid_json_pct": round(100.0 * float(df["has_valid_json"].mean()), 2) if len(df) else 0.0,
        "image_id_match_count": int(df["image_id_matched"].sum()),
        "image_id_match_pct": round(100.0 * float(df["image_id_matched"].mean()), 2) if len(df) else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Property coverage ({variant}) ===")
    try:
        from IPython.display import display

        display(property_metrics)
    except Exception:
        print(property_metrics.head(30).to_string(index=False))

    print("\n=== Run summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"- {per_sample_path}")
    print(f"- {property_path}")
    print(f"- {summary_path}")

    log_report_to_comet(
        comet_experiment=comet_experiment,
        model_key=model_key,
        variant=variant,
        property_metrics=property_metrics,
        summary=summary,
        per_sample_path=per_sample_path,
        property_path=property_path,
        summary_path=summary_path,
    )

    return property_metrics, summary


def get_available_variants(
    requested_variants: Sequence[str],
    samples: List[Dict[str, Any]],
) -> List[str]:
    variants: List[str] = []
    if "raw" in requested_variants:
        variants.append("raw")
    if "masked" in requested_variants and any("mask_path" in x for x in samples):
        variants.append("masked")
    return variants


def run_many_models(
    model_keys: List[str],
    samples: List[Dict[str, Any]],
    property_specs: Dict[str, PropertySpec],
    model_registry: Dict[str, Dict[str, Any]],
    reports_dir: Path,
    variants: Optional[List[str]] = None,
    include_only_gt_known: bool = True,
    max_properties_per_sample: Optional[int] = None,
    mask_background_mode: str = "black",
    save_raw_output: bool = True,
    comet_experiment=None,
) -> pd.DataFrame:
    if variants is None:
        variants = get_available_variants(["raw", "masked"], samples)

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
                include_only_gt_known=include_only_gt_known,
                max_properties_per_sample=max_properties_per_sample,
                mask_background_mode=mask_background_mode,
                save_raw_output=save_raw_output,
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
            for _, r in pm.iterrows():
                rows.append(
                    {
                        "model_key": model_key,
                        "variant": variant,
                        "property": r["property"],
                        "group": r["group"],
                        "value_type": r["value_type"],
                        "pred_yes_pct": r["pred_yes_pct"],
                        "coverage_on_gt_known_pct": r["coverage_on_gt_known_pct"],
                        "exact_match_on_gt_known_pct": r["exact_match_on_gt_known_pct"],
                        "num_samples": summary["num_samples"],
                    }
                )

    return pd.DataFrame(rows)
