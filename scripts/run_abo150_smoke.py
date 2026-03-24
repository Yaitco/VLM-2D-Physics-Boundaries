#!/usr/bin/env python3
"""Day-1 smoke run for the ABO150 validation pipeline without downloading a real VLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import abo150_vlm_validation as val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end smoke test on ABO150 pipeline.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/abo_150_expanded"),
        help="Path to ABO150 dataset root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Number of samples for smoke run.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports_smoke_abo150"),
        help="Directory where smoke-run reports will be written.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="per_property",
        choices=["joint", "per_property"],
        help="Prompt mode to exercise during the smoke test.",
    )
    parser.add_argument(
        "--protocol-name",
        type=str,
        default="expanded_ontology",
        choices=["expanded_ontology", "full_expanded", "narrow_core", "pdf_compact"],
        help="Evaluation protocol to exercise during the smoke test.",
    )
    parser.add_argument(
        "--include-only-gt-known",
        action="store_true",
        help="If passed, request only GT-known properties.",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=0,
        help="Number of demo examples to include in few-shot mode.",
    )
    parser.add_argument(
        "--few-shot-selection-mode",
        type=str,
        default="fixed",
        choices=["fixed", "dynamic"],
        help="How demo examples are selected for few-shot runs.",
    )
    return parser.parse_args()


def _extract_image_id(prompt: str) -> str:
    match = re.search(r'The image_id is "([^"]+)"', prompt)
    return match.group(1) if match else "unknown_image"


def _extract_joint_keys(prompt: str) -> List[str]:
    match = re.search(r"Fill ONLY these property keys:\s*(.+)", prompt)
    if not match:
        return []
    keys = [part.strip() for part in match.group(1).split(",")]
    return [key for key in keys if key]


def _extract_single_key(prompt: str) -> str:
    match = re.search(r'Predict ONLY this property:\s*"([^"]+)"', prompt)
    return match.group(1) if match else ""


def _mock_value_for_spec(spec: val.PropertySpec):
    if spec.value_type == "boolean":
        return "unknown"
    if spec.value_type == "multi_categorical":
        return []
    return "unknown"


def load_mock_runtime(name: str, cfg: Dict[str, object]) -> val.VLMRuntime:
    return val.VLMRuntime(
        name=name,
        backend="mock_json",
        model_id=str(cfg["model_id"]),
        processor=None,
        model=None,
        gen_kwargs={},
    )


def infer_mock(runtime: val.VLMRuntime, image, prompt: str) -> str:
    image_id = _extract_image_id(prompt)
    keys = _extract_joint_keys(prompt)
    properties = {}
    for key in keys:
        properties[key] = "unknown"
    payload = {
        "image_id": image_id,
        "primary_object": "object",
        "properties": properties,
        "notes": "smoke run",
    }
    return json.dumps(payload, ensure_ascii=False)


def infer_mock_batch(runtime: val.VLMRuntime, image, prompts: List[str]) -> List[str]:
    outputs = []
    for prompt in prompts:
        image_id = _extract_image_id(prompt)
        key = _extract_single_key(prompt)
        payload = {
            "image_id": image_id,
            "primary_object": "object",
            "properties": {key: "unknown"},
            "notes": "smoke run",
        }
        outputs.append(json.dumps(payload, ensure_ascii=False))
    return outputs


def infer_mock_messages(runtime: val.VLMRuntime, messages, images) -> str:
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

    property_key = _extract_single_key(last_prompt)
    if property_key:
        payload = {
            "image_id": _extract_image_id(last_prompt),
            "primary_object": "object",
            "properties": {property_key: "unknown"},
            "notes": "smoke run",
        }
        return json.dumps(payload, ensure_ascii=False)

    fallback_image = images[-1] if images else None
    return infer_mock(runtime, fallback_image, last_prompt)


def main() -> None:
    args = parse_args()

    schema_path = args.dataset_dir / "physics_properties.yaml"
    annotations_path = args.dataset_dir / "selected_150_annotations.jsonl"

    property_specs = val.load_protocol_property_specs(
        protocol_name=args.protocol_name,
        schema_path=schema_path if args.protocol_name in {"expanded_ontology", "full_expanded", "narrow_core"} else None,
    )
    samples = val.load_abo150_samples(
        annotations_path=annotations_path,
        dataset_dir=args.dataset_dir,
        property_specs=property_specs,
        protocol_name=args.protocol_name,
        max_samples=args.max_samples,
        random_seed=42,
    )

    val.BACKEND_LOADERS["mock_json"] = load_mock_runtime
    val.BACKEND_INFER["mock_json"] = infer_mock
    val.BACKEND_INFER_BATCH["mock_json"] = infer_mock_batch
    val.BACKEND_INFER_MESSAGES["mock_json"] = infer_mock_messages

    model_registry = {
        "mock_json": {
            "backend": "mock_json",
            "model_id": "mock://day1_smoke",
        }
    }

    df = val.run_validation(
        model_key="mock_json",
        samples=samples,
        property_specs=property_specs,
        model_registry=model_registry,
        variant="raw",
        prompt_mode=args.prompt_mode,
        property_batch_size=4,
        include_only_gt_known=args.include_only_gt_known,
        few_shot_k=args.few_shot_k,
        few_shot_selection_mode=args.few_shot_selection_mode,
        max_properties_per_sample=12 if args.prompt_mode == "joint" else None,
        mask_background_mode="black",
        save_raw_output=True,
        json_success_threshold=1.0,
        image_id_success_threshold=1.0,
    )
    _, summary = val.save_report(
        df=df,
        model_key="mock_json",
        variant="raw",
        model_registry=model_registry,
        property_specs=property_specs,
        reports_dir=args.reports_dir,
        comet_experiment=None,
    )

    print("\nSmoke run completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
