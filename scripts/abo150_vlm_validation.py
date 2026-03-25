"""Structured per-property validation pipeline for ABO150 in Colab."""

from __future__ import annotations

import ast
import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


MODEL_REGISTRY = {
    "qwen3_vl_8b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "qwen2_5_vl_7b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "llava_onevision_1_5_8b": {
        "backend": "hf_chat",
        "model_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
}

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_COMPACT_SCHEMA_PATH = ROOT_DIR / "configs" / "pdf_protocol_properties.yaml"
NARROW_CORE_KEYS_PATH = ROOT_DIR / "configs" / "narrow_core_property_keys.yaml"


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


def chunked(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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
            "Comet: API key is missing. Add 'comet_api_key' in Colab userdata or set COMET_API_KEY."
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


def _load_allowed_values(
    cfg: Dict[str, Any],
    enums_map: Dict[str, List[str]],
    enum_ref_key: str = "enum_ref",
    inline_enum_key: str = "enum",
) -> List[str]:
    enum_ref = cfg.get(enum_ref_key)
    if enum_ref is not None:
        values = list(enums_map.get(str(enum_ref), []))
        if values:
            return values
    inline_values = cfg.get(inline_enum_key)
    if isinstance(inline_values, list):
        return [_normalize_token(str(v)) for v in inline_values]
    return []


def load_property_specs(
    schema_path: Path,
    include_groups: Optional[Sequence[str]] = None,
) -> Dict[str, PropertySpec]:
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
        properties = group_cfg.get("properties", {}) if isinstance(group_cfg, dict) else {}
        if not isinstance(properties, dict):
            continue

        for prop_name, prop_cfg in properties.items():
            if not isinstance(prop_cfg, dict):
                continue

            key = f"{group_name}.{prop_name}"
            desc = str(prop_cfg.get("description") or "").strip()
            prop_type = prop_cfg.get("type")

            if prop_type == "categorical":
                allowed = _load_allowed_values(prop_cfg, enums_map)
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
                if not isinstance(items, dict) or items.get("type") != "categorical":
                    continue
                allowed = [value for value in _load_allowed_values(items, enums_map) if value != "unknown"]
                specs[key] = PropertySpec(
                    key=key,
                    group=group_name,
                    name=prop_name,
                    value_type="multi_categorical",
                    allowed_values=allowed,
                    description=desc,
                )

    if not specs:
        raise ValueError("No evaluable properties found in schema.")
    return specs


def load_property_subset_specs(schema_path: Path, subset_keys_path: Path) -> Dict[str, PropertySpec]:
    base_specs = load_property_specs(schema_path=schema_path)
    with subset_keys_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    include_keys = cfg.get("include_keys", [])
    if not isinstance(include_keys, list) or not include_keys:
        raise ValueError(f"Invalid subset config: include_keys are missing in {subset_keys_path}")

    subset: Dict[str, PropertySpec] = {}
    missing: List[str] = []
    for key in include_keys:
        key = str(key)
        spec = base_specs.get(key)
        if spec is None:
            missing.append(key)
            continue
        subset[key] = spec

    if missing:
        raise ValueError(f"Unknown property keys in subset config {subset_keys_path}: {missing}")
    return subset


def load_compact_property_specs(schema_path: Path) -> Dict[str, PropertySpec]:
    with schema_path.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"Invalid compact schema: properties are missing in {schema_path}")

    specs: Dict[str, PropertySpec] = {}
    for key, cfg in properties.items():
        if not isinstance(cfg, dict) or str(cfg.get("type")) != "categorical":
            continue
        allowed = _load_allowed_values(cfg, enums_map={})
        if "unknown" not in allowed:
            allowed.append("unknown")
        specs[str(key)] = PropertySpec(
            key=str(key),
            group="pdf_compact",
            name=str(key),
            value_type="categorical",
            allowed_values=allowed,
            description=str(cfg.get("description") or "").strip(),
        )

    if not specs:
        raise ValueError(f"No properties found in compact schema: {schema_path}")
    return specs


def load_protocol_property_specs(
    protocol_name: str,
    schema_path: Optional[Path] = None,
    include_groups: Optional[Sequence[str]] = None,
) -> Dict[str, PropertySpec]:
    if protocol_name in {"expanded_ontology", "full_expanded"}:
        if schema_path is None:
            raise ValueError("schema_path is required for expanded ontology protocol")
        return load_property_specs(schema_path=schema_path, include_groups=include_groups)

    if protocol_name == "narrow_core":
        if schema_path is None:
            raise ValueError("schema_path is required for narrow_core protocol")
        return load_property_subset_specs(schema_path=schema_path, subset_keys_path=NARROW_CORE_KEYS_PATH)

    if protocol_name == "pdf_compact":
        return load_compact_property_specs(PDF_COMPACT_SCHEMA_PATH)

    raise ValueError(f"Unknown protocol_name: {protocol_name}")


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
    if isinstance(value, list):
        raw_values: Iterable[Any] = value
    elif isinstance(value, str):
        if value.strip() in {"", "none", "null", "unknown"}:
            return []
        raw_values = [v.strip() for v in value.split(",")]
    else:
        return []

    out: List[str] = []
    for raw in raw_values:
        token = _normalize_token(str(raw))
        if token in allowed_set and token not in out:
            out.append(token)
    return out


def normalize_value(spec: PropertySpec, value: Any) -> Any:
    if spec.value_type == "categorical":
        return _normalize_categorical(value, spec.allowed_values)
    if spec.value_type == "boolean":
        return _normalize_boolean(value)
    if spec.value_type == "multi_categorical":
        return _normalize_multi_categorical(value, spec.allowed_values)
    raise ValueError(f"Unsupported value_type: {spec.value_type}")


def is_known_value(spec: PropertySpec, value: Any) -> bool:
    if spec.value_type == "multi_categorical":
        return isinstance(value, list) and len(value) > 0
    return value not in {None, "unknown"}


def values_equal(spec: PropertySpec, pred: Any, gt: Any) -> bool:
    if not is_known_value(spec, gt):
        return False
    if spec.value_type == "multi_categorical":
        return sorted(pred or []) == sorted(gt or [])
    return pred == gt


def serialize_value(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _compute_accuracy_pct(gt_values: Sequence[str], pred_values: Sequence[str]) -> Optional[float]:
    if not gt_values:
        return None
    correct = sum(1 for gt, pred in zip(gt_values, pred_values) if gt == pred)
    return round(100.0 * correct / len(gt_values), 2)


def _compute_macro_f1_pct(
    gt_values: Sequence[str],
    pred_values: Sequence[str],
    labels: Sequence[str],
) -> Optional[float]:
    if not gt_values or not labels:
        return None

    f1_values: List[float] = []
    for label in labels:
        tp = sum(1 for gt, pred in zip(gt_values, pred_values) if gt == label and pred == label)
        fp = sum(1 for gt, pred in zip(gt_values, pred_values) if gt != label and pred == label)
        fn = sum(1 for gt, pred in zip(gt_values, pred_values) if gt == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
        f1_values.append(f1)

    return round(100.0 * float(sum(f1_values) / len(f1_values)), 2)


def _compute_selective_accuracy_pct(
    gt_values: Sequence[str],
    pred_values: Sequence[str],
    unknown_label: str = "unknown",
) -> Optional[float]:
    covered = [(gt, pred) for gt, pred in zip(gt_values, pred_values) if pred != unknown_label]
    if not covered:
        return None
    correct = sum(1 for gt, pred in covered if gt == pred)
    return round(100.0 * correct / len(covered), 2)


def _resolve_panel_path(panel_path_raw: str, dataset_dir: Path, panels_dir: Path, obj_id: str) -> Path:
    candidates: List[Path] = []
    if panel_path_raw:
        p = Path(panel_path_raw)
        if p.is_absolute():
            candidates.extend([p, panels_dir / p.name])
        else:
            candidates.extend([dataset_dir / p, panels_dir / p.name])
    candidates.append(panels_dir / f"{obj_id}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Panel image not found for obj_id={obj_id}. Tried: {[str(x) for x in candidates]}")


def _extract_gt_properties(record: Dict[str, Any], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
    groups = record.get("groups", {})
    groups = groups if isinstance(groups, dict) else {}

    gt: Dict[str, Any] = {}
    for key, spec in property_specs.items():
        group_payload = groups.get(spec.group, {}) if "." in key else record
        raw_value = group_payload.get(spec.name) if isinstance(group_payload, dict) else None
        gt[key] = normalize_value(spec, raw_value)
    return gt


def _map_pdf_material(raw_value: Any) -> str:
    mapping = {
        "wood": "wood",
        "metal": "metal",
        "glass": "glass",
        "plastic": "plastic",
        "fabric": "fabric",
        "paper": "paper/cardboard",
        "cardboard": "paper/cardboard",
        "ceramic": "ceramic/stone",
        "stone": "ceramic/stone",
        "rubber": "rubber",
        "leather": "leather",
        "unknown": "unknown",
    }
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    return mapping.get(token, "other")


def _map_pdf_reflectance(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token == "matte":
        return "matte"
    if token in {"glossy", "semi_glossy", "very_glossy", "mirror_like"}:
        return "glossy"
    return "unknown"


def _map_pdf_surface_roughness(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token in {"smooth", "very_smooth"}:
        return "smooth"
    if token in {"rough", "very_rough"}:
        return "rough"
    return "unknown"


def _map_pdf_rigidity(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token == "rigid":
        return "rigid"
    if token in {"flexible", "floppy"}:
        return "deformable"
    return "unknown"


def _map_pdf_fragility(intrinsic: Dict[str, Any], affordance: Dict[str, Any]) -> str:
    breakable = affordance.get("breakable")
    if isinstance(breakable, bool):
        return "fragile" if breakable else "not_fragile"
    token = _normalize_token(str(intrinsic.get("brittleness_class"))) if intrinsic.get("brittleness_class") is not None else "unknown"
    if token in {"brittle", "very_brittle", "slightly_brittle"}:
        return "fragile"
    if token == "non_brittle":
        return "not_fragile"
    return "unknown"


def _map_pdf_state(state_payload: Dict[str, Any]) -> str:
    is_dirty = state_payload.get("is_dirty")
    if isinstance(is_dirty, bool):
        return "dirty" if is_dirty else "clean"
    return "unknown"


def _map_pdf_weight_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token in {"very_light", "light"}:
        return "light"
    if token in {"heavy", "very_heavy"}:
        return "heavy"
    return "unknown"


def _map_pdf_temperature_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token in {"room", "room_temp"}:
        return "room"
    if token in {"warm", "hot"}:
        return "hot"
    if token in {"cold", "chilled", "frozen"}:
        return "cold"
    return "unknown"


def _map_pdf_phase(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    return token if token in {"solid", "liquid", "gas"} else "unknown"


def _map_pdf_filled_state(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token == "empty":
        return "empty"
    if token in {"almost_empty", "half_full", "almost_full"}:
        return "partially_filled"
    if token == "full":
        return "filled"
    return "unknown"


def _map_pdf_slipperiness_hint(raw_value: Any) -> str:
    token = _normalize_token(str(raw_value)) if raw_value is not None else "unknown"
    if token == "slippery":
        return "slippery"
    if token in {"high", "normal"}:
        return "not_slippery"
    return "unknown"


def extract_pdf_protocol_properties(record: Dict[str, Any], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
    groups = record.get("groups", {}) if isinstance(record.get("groups"), dict) else {}
    intrinsic = groups.get("intrinsic", {}) if isinstance(groups.get("intrinsic"), dict) else {}
    state = groups.get("state", {}) if isinstance(groups.get("state"), dict) else {}
    affordance = groups.get("affordance", {}) if isinstance(groups.get("affordance"), dict) else {}

    compact_values = {
        "material": _map_pdf_material(intrinsic.get("main_material")),
        "transparency": _normalize_token(str(intrinsic.get("transparency_class"))) if intrinsic.get("transparency_class") is not None else "unknown",
        "reflectance": _map_pdf_reflectance(intrinsic.get("glossiness_class")),
        "surface_roughness": _map_pdf_surface_roughness(intrinsic.get("surface_roughness_class")),
        "rigidity": _map_pdf_rigidity(intrinsic.get("rigidity_class")),
        "fragility": _map_pdf_fragility(intrinsic, affordance),
        "wetness": "unknown",
        "state": _map_pdf_state(state),
        "weight_hint": _map_pdf_weight_hint(intrinsic.get("mass_class")),
        "temperature_hint": _map_pdf_temperature_hint(state.get("object_temperature_class")),
        "phase": _map_pdf_phase(intrinsic.get("state_of_matter")),
        "filled_state": _map_pdf_filled_state(state.get("fill_state_class")),
        "slipperiness_hint": _map_pdf_slipperiness_hint(intrinsic.get("friction_class")),
    }
    return {key: normalize_value(spec, compact_values.get(key)) for key, spec in property_specs.items()}


def load_abo150_samples(
    annotations_path: Path,
    dataset_dir: Path,
    property_specs: Dict[str, PropertySpec],
    protocol_name: str = "expanded_ontology",
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

            panels = record.get("panels", {}) if isinstance(record.get("panels"), dict) else {}
            panel_path = _resolve_panel_path(
                panel_path_raw=str(panels.get("panel_path") or ""),
                dataset_dir=dataset_dir,
                panels_dir=panels_dir,
                obj_id=obj_id,
            )

            if protocol_name == "pdf_compact":
                gt_properties = extract_pdf_protocol_properties(record, property_specs)
            else:
                gt_properties = _extract_gt_properties(record, property_specs)

            sample = {
                "image_id": obj_id,
                "path": str(panel_path),
                "split": record.get("split"),
                "caption": record.get("caption"),
                "gt_properties": gt_properties,
            }

            mask_path = panels.get("mask_path") if isinstance(panels.get("mask_path"), str) else None
            if mask_path and mask_path.strip():
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


def parse_model_output(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None, "empty_output"

    blob = find_first_json_object(raw_output)
    if blob is None:
        return None, "json_not_found"

    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_not_object"
    except Exception as exc_json:
        try:
            parsed = yaml.safe_load(blob)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(blob)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass
        return None, f"json_parse_error: {exc_json}"


def _fetch_pred_raw_value(parsed_json: Dict[str, Any], key: str) -> Any:
    if "." in key:
        group, name = key.split(".", 1)
    else:
        group, name = None, key

    if key in parsed_json:
        return parsed_json[key]
    if name in parsed_json:
        return parsed_json[name]

    props = parsed_json.get("properties", {})
    if isinstance(props, dict):
        if key in props:
            return props[key]
        group_payload = props.get(group) if group is not None else None
        if isinstance(group_payload, dict) and name in group_payload:
            return group_payload[name]

    groups = parsed_json.get("groups", {})
    if isinstance(groups, dict):
        group_payload = groups.get(group) if group is not None else None
        if isinstance(group_payload, dict) and name in group_payload:
            return group_payload[name]
    return None


def normalize_pred(parsed_json: Optional[Dict[str, Any]], property_specs: Dict[str, PropertySpec]) -> Dict[str, Any]:
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
        pred_props[key] = normalize_value(spec, _fetch_pred_raw_value(parsed_json, key))

    return {
        "image_id": str(parsed_json.get("image_id")).strip() if parsed_json.get("image_id") is not None else None,
        "primary_object": parsed_json.get("primary_object"),
        "notes": parsed_json.get("notes"),
        "properties": pred_props,
    }


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
        score = int(is_known_value(property_specs[property_key], sample.get("gt_properties", {}).get(property_key)))
        if score > 0:
            scored.append((score, image_id, sample))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sample for _, _, sample in scored[:few_shot_k]]


def build_fixed_few_shot_candidates(samples: Sequence[Dict[str, Any]], property_specs: Dict[str, PropertySpec]) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for sample in samples:
        image_id = str(sample.get("image_id"))
        gt_props = sample.get("gt_properties", {})
        score = sum(1 for key, spec in property_specs.items() if is_known_value(spec, gt_props.get(key)))
        if score > 0:
            scored.append((score, image_id, sample))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sample for _, _, sample in scored]


def make_bnb_config(use_4bit: bool):
    if BitsAndBytesConfig is None or not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_hf_chat_runtime(name: str, cfg: Dict[str, Any]) -> VLMRuntime:
    if AutoProcessor is None or AutoModelForImageTextToText is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is not installed. Install notebook dependencies first.")

    model_id = cfg["model_id"]
    use_4bit = bool(cfg.get("use_4bit", True))
    gen_kwargs = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "do_sample": bool(cfg.get("do_sample", False)),
    }

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
        try:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
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

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        try:
            generation_config.do_sample = gen_kwargs["do_sample"]
        except Exception:
            pass
        if not gen_kwargs["do_sample"]:
            for attr in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                if hasattr(generation_config, attr):
                    try:
                        setattr(generation_config, attr, None)
                    except Exception:
                        pass

    model.eval()
    return VLMRuntime(name=name, backend="hf_chat", model_id=model_id, processor=processor, model=model, gen_kwargs=gen_kwargs)


def _simple_user_message(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _messages_to_fallback_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = message.get("content")
        parts: List[str] = []
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text = str(chunk.get("text") or "").strip()
                    if text:
                        parts.append(text)
        if parts:
            lines.append(f"{role}: " + "\n".join(parts))
    return "\n\n".join(lines).strip()


def _apply_chat_template_text(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]]) -> str:
    if hasattr(runtime.processor, "apply_chat_template"):
        try:
            text = runtime.processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            if isinstance(text, str) and text.strip():
                return text
        except TypeError:
            text = runtime.processor.apply_chat_template(list(messages), add_generation_prompt=True)
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
    return _messages_to_fallback_text(messages)


def infer_hf_chat_messages(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]], images: Sequence[Image.Image]) -> str:
    if not images:
        raise ValueError("images must not be empty")

    image_input: Any = images[0] if len(images) == 1 else list(images)
    text = _apply_chat_template_text(runtime, messages)

    inputs = runtime.processor(text=text, images=image_input, return_tensors="pt", truncation=False)
    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = runtime.model.generate(**inputs, **runtime.gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    if out.shape[1] > input_len:
        gen_tokens = out[:, input_len:]
    else:
        gen_tokens = out
    return runtime.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]


def infer_hf_chat(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    return infer_hf_chat_messages(runtime, _simple_user_message(prompt), [image])


def infer_hf_chat_messages_batch(
    runtime: VLMRuntime,
    messages_batch: Sequence[Sequence[Dict[str, Any]]],
    images_batch: Sequence[Sequence[Image.Image]],
) -> List[str]:
    if not messages_batch:
        return []
    if len(messages_batch) == 1:
        return [infer_hf_chat_messages(runtime, messages_batch[0], images_batch[0])]
    if "llava" in runtime.model_id.lower():
        return [infer_hf_chat_messages(runtime, messages, images) for messages, images in zip(messages_batch, images_batch)]

    texts: List[str] = []
    image_inputs: List[Any] = []
    for messages, images in zip(messages_batch, images_batch):
        text = _apply_chat_template_text(runtime, messages)
        if not text.strip():
            return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]
        texts.append(text)
        image_inputs.append(images[0] if len(images) == 1 else list(images))

    try:
        inputs = runtime.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=False)
    except Exception:
        return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]

    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = runtime.model.generate(**inputs, **runtime.gen_kwargs)

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lens = [inputs["input_ids"].shape[1]] * len(texts)

    generated = []
    for i, input_len in enumerate(input_lens):
        if out.shape[1] > int(input_len):
            generated.append(out[i, int(input_len):].detach().cpu())
        else:
            generated.append(out[i].detach().cpu())
    decoded = runtime.processor.batch_decode(generated, skip_special_tokens=True)
    if any((not isinstance(item, str)) or (not item.strip()) for item in decoded):
        return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]
    return decoded


def infer_hf_chat_batch(runtime: VLMRuntime, image: Image.Image, prompts: Sequence[str]) -> List[str]:
    return infer_hf_chat_messages_batch(
        runtime,
        [_simple_user_message(prompt) for prompt in prompts],
        [[image] for _ in prompts],
    )


BACKEND_LOADERS = {"hf_chat": load_hf_chat_runtime}
BACKEND_INFER = {"hf_chat": infer_hf_chat}
BACKEND_INFER_BATCH = {"hf_chat": infer_hf_chat_batch}
BACKEND_INFER_MESSAGES = {"hf_chat": infer_hf_chat_messages}
BACKEND_INFER_MESSAGES_BATCH = {"hf_chat": infer_hf_chat_messages_batch}


def load_runtime(model_key: str, model_registry: Dict[str, Dict[str, Any]]) -> VLMRuntime:
    cfg = model_registry[model_key]
    runtime = BACKEND_LOADERS[cfg["backend"]](model_key, cfg)
    print(f"Loaded model: {runtime.name} -> {runtime.model_id}")
    return runtime


def infer_runtime(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    return BACKEND_INFER[runtime.backend](runtime, image, prompt)


def infer_runtime_batch(runtime: VLMRuntime, image: Image.Image, prompts: Sequence[str]) -> List[str]:
    fn = BACKEND_INFER_BATCH.get(runtime.backend)
    if fn is None:
        return [infer_runtime(runtime, image, prompt) for prompt in prompts]
    return fn(runtime, image, prompts)


def infer_runtime_messages(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]], images: Sequence[Image.Image]) -> str:
    fn = BACKEND_INFER_MESSAGES.get(runtime.backend)
    if fn is not None:
        return fn(runtime, messages, images)
    if not images:
        raise ValueError("images must not be empty")
    last_prompt = ""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        for chunk in message.get("content", []):
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                last_prompt = str(chunk.get("text") or "")
                break
        if last_prompt:
            break
    return infer_runtime(runtime, images[-1], last_prompt)


def infer_runtime_messages_batch(
    runtime: VLMRuntime,
    messages_batch: Sequence[Sequence[Dict[str, Any]]],
    images_batch: Sequence[Sequence[Image.Image]],
) -> List[str]:
    fn = BACKEND_INFER_MESSAGES_BATCH.get(runtime.backend)
    if fn is not None:
        return fn(runtime, messages_batch, images_batch)
    return [infer_runtime_messages(runtime, messages, images) for messages, images in zip(messages_batch, images_batch)]


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
    out[~m] = 255 if bg_mode == "white" else 0
    return Image.fromarray(out, mode="RGB")


def apply_mask_overlay_to_image(
    image: Image.Image,
    mask: Image.Image,
    tint_rgb: Tuple[int, int, int] = (255, 96, 96),
    tint_alpha: float = 0.22,
    boundary_rgb: Tuple[int, int, int] = (255, 32, 32),
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    out = rgb.astype(np.float32)
    if m.any():
        tint = np.asarray(tint_rgb, dtype=np.float32)
        out[m] = (1.0 - tint_alpha) * out[m] + tint_alpha * tint
        up = np.zeros_like(m)
        down = np.zeros_like(m)
        left = np.zeros_like(m)
        right = np.zeros_like(m)
        up[1:] = m[:-1]
        down[:-1] = m[1:]
        left[:, 1:] = m[:, :-1]
        right[:, :-1] = m[:, 1:]
        interior = m & up & down & left & right
        boundary = m & (~interior)
        out[boundary] = np.asarray(boundary_rgb, dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def load_variant_image(sample_meta: Dict[str, Any], variant: str, mask_background_mode: str) -> Image.Image:
    image = Image.open(Path(sample_meta["path"])).convert("RGB")
    if variant == "raw":
        return image
    mask_path = sample_meta.get("mask_path")
    if not mask_path:
        raise FileNotFoundError("mask_path is missing in sample metadata")
    mask = Image.open(Path(mask_path)).convert("L")
    if variant == "masked":
        return apply_mask_to_image(image, mask, bg_mode=mask_background_mode)
    if variant == "mask_overlay":
        return apply_mask_overlay_to_image(image, mask)
    raise ValueError(f"Unknown variant: {variant}")


def load_variant_image_cached(
    sample_meta: Dict[str, Any],
    variant: str,
    mask_background_mode: str,
    image_cache: Optional[Dict[Tuple[str, str, str, str], Image.Image]] = None,
) -> Image.Image:
    if image_cache is None:
        return load_variant_image(sample_meta, variant, mask_background_mode)
    key = (
        str(sample_meta.get("path") or ""),
        str(sample_meta.get("mask_path") or ""),
        variant,
        mask_background_mode,
    )
    cached = image_cache.get(key)
    if cached is not None:
        return cached
    image = load_variant_image(sample_meta, variant, mask_background_mode)
    image_cache[key] = image
    return image


def _column_prefix(property_key: str) -> str:
    return property_key.replace(".", "__")


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
        demo_payload = build_demo_response_payload(str(demo["image_id"]), property_key, demo.get("gt_properties", {}), property_specs)
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": demo_prompt}]})
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

    pred_props = {key: ([] if spec.value_type == "multi_categorical" else "unknown") for key, spec in property_specs.items()}
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

    for key_chunk in chunked(selected_keys, property_batch_size):
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
    image_id_match_ratio = float(image_id_match_count) / float(expected_response_count) if expected_response_count > 0 else 0.0
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
        "few_shot_k": int(few_shot_k),
        "few_shot_selection_mode": few_shot_selection_mode,
        "few_shot_demo_ids": "|".join(few_shot_demo_ids),
    }

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
    fixed_few_shot_candidates = build_fixed_few_shot_candidates(samples, property_specs) if few_shot_k > 0 and few_shot_selection_mode == "fixed" else None

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
                    "image_id_success_threshold": image_id_success_threshold if image_id_success_threshold is not None else json_success_threshold,
                    "primary_object_pred": None,
                    "notes_pred": None,
                    "requested_property_count": 0,
                    "expected_response_count": 0,
                    "requested_property_keys": "",
                    "property_batch_size": int(property_batch_size),
                    "few_shot_k": int(few_shot_k),
                    "few_shot_selection_mode": few_shot_selection_mode,
                    "few_shot_demo_ids": "",
                }
                gt_props = sample_meta.get("gt_properties", {})
                for key, spec in property_specs.items():
                    col = _column_prefix(key)
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
    finally:
        unload_runtime(runtime)
    return pd.DataFrame(rows)


def build_property_metrics(df: pd.DataFrame, property_specs: Dict[str, PropertySpec]) -> pd.DataFrame:
    total = len(df)
    metrics = []
    for key, spec in property_specs.items():
        col = _column_prefix(key)
        pred_yes = int(df[f"{col}_pred_known"].sum())
        gt_known = int(df[f"{col}_gt_known"].sum())
        missed = int(df[f"{col}_missed_when_gt_known"].sum())
        extracted_when_gt_known = int(gt_known - missed)
        exact = int(df[f"{col}_exact_match"].sum())

        gt_known_mask = df[f"{col}_gt_known"].fillna(False).astype(bool)
        gt_values = df.loc[gt_known_mask, f"{col}_gt"].astype(str).tolist()
        pred_values = df.loc[gt_known_mask, f"{col}_pred"].astype(str).tolist()
        accuracy_pct = _compute_accuracy_pct(gt_values, pred_values)
        selective_accuracy_pct = _compute_selective_accuracy_pct(gt_values, pred_values)
        macro_f1_pct = None
        if spec.value_type in {"categorical", "boolean"}:
            labels = [x for x in spec.allowed_values if x != "unknown"]
            macro_f1_pct = _compute_macro_f1_pct(gt_values, pred_values, labels)

        metrics.append(
            {
                "property": key,
                "group": spec.group,
                "value_type": spec.value_type,
                "pred_yes_count": pred_yes,
                "pred_no_count": int(total - pred_yes),
                "pred_yes_pct": round(100.0 * pred_yes / total, 2) if total else 0.0,
                "coverage_pct": round(100.0 * pred_yes / total, 2) if total else 0.0,
                "gt_known_count": gt_known,
                "extracted_when_gt_known": extracted_when_gt_known,
                "missed_when_gt_known": missed,
                "coverage_on_gt_known_pct": round(100.0 * extracted_when_gt_known / gt_known, 2) if gt_known else None,
                "accuracy_pct": accuracy_pct,
                "macro_f1_pct": macro_f1_pct,
                "selective_accuracy_pct": selective_accuracy_pct,
                "exact_match_on_gt_known_pct": round(100.0 * exact / gt_known, 2) if gt_known else None,
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
    for key in [
        "mean_coverage_pct",
        "mean_accuracy_pct",
        "mean_macro_f1_pct",
        "mean_selective_accuracy_pct",
    ]:
        if summary.get(key) is not None:
            summary_metrics[f"{model_key}/{variant}/{key}"] = summary[key]
    if summary.get("valid_json_all_pct") is not None:
        summary_metrics[f"{model_key}/{variant}/valid_json_all_pct"] = summary["valid_json_all_pct"]
    if summary.get("image_id_match_all_pct") is not None:
        summary_metrics[f"{model_key}/{variant}/image_id_match_all_pct"] = summary["image_id_match_all_pct"]
    if summary.get("valid_json_per_response_pct") is not None:
        summary_metrics[f"{model_key}/{variant}/valid_json_per_response_pct"] = summary["valid_json_per_response_pct"]
    comet_experiment.log_metrics(summary_metrics)

    for _, row in property_metrics.iterrows():
        prop = _safe_metric_key(str(row["property"]))
        for col in [
            "pred_yes_pct",
            "coverage_pct",
            "coverage_on_gt_known_pct",
            "accuracy_pct",
            "macro_f1_pct",
            "selective_accuracy_pct",
            "exact_match_on_gt_known_pct",
            "pred_yes_count",
            "pred_no_count",
            "missed_when_gt_known",
        ]:
            val = row[col]
            if pd.isna(val):
                continue
            comet_experiment.log_metric(f"{model_key}/{variant}/{prop}/{col}", float(val))

    comet_experiment.log_asset(str(per_sample_path), file_name=f"{model_key}_{variant}_per_sample_predictions.csv")
    comet_experiment.log_asset(str(property_path), file_name=f"{model_key}_{variant}_property_metrics.csv")
    comet_experiment.log_asset(str(summary_path), file_name=f"{model_key}_{variant}_summary.json")


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
        "property_batch_size": int(df["property_batch_size"].iloc[0]) if "property_batch_size" in df.columns and len(df) else None,
        "few_shot_k": int(df["few_shot_k"].iloc[0]) if "few_shot_k" in df.columns and len(df) else 0,
        "few_shot_selection_mode": str(df["few_shot_selection_mode"].iloc[0]) if "few_shot_selection_mode" in df.columns and len(df) else "fixed",
        "num_samples": int(len(df)),
        "valid_json_count": int(df["has_valid_json"].sum()),
        "valid_json_pct": round(100.0 * float(df["has_valid_json"].mean()), 2) if len(df) else 0.0,
        "valid_json_all_count": int(df["has_valid_json_all"].sum()) if "has_valid_json_all" in df.columns else None,
        "valid_json_all_pct": round(100.0 * float(df["has_valid_json_all"].mean()), 2) if "has_valid_json_all" in df.columns and len(df) else None,
        "image_id_match_count": int(df["image_id_matched"].sum()),
        "image_id_match_pct": round(100.0 * float(df["image_id_matched"].mean()), 2) if len(df) else 0.0,
        "image_id_match_all_count": int(df["image_id_matched_all"].sum()) if "image_id_matched_all" in df.columns else None,
        "image_id_match_all_pct": round(100.0 * float(df["image_id_matched_all"].mean()), 2) if "image_id_matched_all" in df.columns and len(df) else None,
        "mean_coverage_pct": round(float(property_metrics["coverage_pct"].dropna().mean()), 2) if len(property_metrics) else None,
        "mean_accuracy_pct": round(float(property_metrics["accuracy_pct"].dropna().mean()), 2) if property_metrics["accuracy_pct"].notna().any() else None,
        "mean_macro_f1_pct": round(float(property_metrics["macro_f1_pct"].dropna().mean()), 2) if property_metrics["macro_f1_pct"].notna().any() else None,
        "mean_selective_accuracy_pct": round(float(property_metrics["selective_accuracy_pct"].dropna().mean()), 2) if property_metrics["selective_accuracy_pct"].notna().any() else None,
    }
    if {"valid_json_count", "expected_response_count"}.issubset(df.columns) and len(df):
        total_expected = float(df["expected_response_count"].sum())
        total_valid = float(df["valid_json_count"].sum())
        summary["valid_json_per_response_pct"] = round(100.0 * total_valid / total_expected, 2) if total_expected > 0 else None

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

    log_report_to_comet(comet_experiment, model_key, variant, property_metrics, summary, per_sample_path, property_path, summary_path)
    return property_metrics, summary


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
