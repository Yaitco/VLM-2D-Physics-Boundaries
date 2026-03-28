from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pipeline import (
    build_demo_response_payload,
    build_property_prompt,
    filter_property_specs,
    get_dataset_context,
    is_known_value,
    load_protocol_property_specs,
    load_samples_for_dataset,
    resolve_protocol_schema_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-property multimodal SFT dataset from dataset/abo_150_expanded "
            "for QLoRA / LoRA fine-tuning."
        )
    )
    parser.add_argument("--dataset-name", type=str, default="abo_150_expanded", choices=["abo_150_expanded"])
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--protocol-name", type=str, default="full_expanded")
    parser.add_argument(
        "--property-keys",
        type=str,
        default="",
        help="Comma-separated property keys. Empty means all properties from the chosen protocol.",
    )
    parser.add_argument(
        "--train-ids-path",
        type=Path,
        required=True,
        help="TXT/JSON file with image_ids for the training split.",
    )
    parser.add_argument(
        "--val-ids-path",
        type=Path,
        default=None,
        help="Optional TXT/JSON file with image_ids for validation split export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include samples where GT is unknown. By default only known GT values are exported.",
    )
    return parser.parse_args()


def _parse_property_keys(raw: str) -> List[str]:
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    return keys


def _build_examples(
    samples: List[Dict[str, Any]],
    property_specs: Dict[str, Any],
    include_unknown: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        image_id = str(sample["image_id"])
        image_path = str(sample["path"])
        gt_properties = sample["gt_properties"]
        for key, spec in property_specs.items():
            gt_value = gt_properties.get(key)
            if not include_unknown and not is_known_value(spec, gt_value):
                continue
            prompt = build_property_prompt(image_id=image_id, property_key=key, spec=spec)
            response_payload = build_demo_response_payload(
                image_id=image_id,
                property_key=key,
                gt_properties=gt_properties,
                property_specs=property_specs,
            )
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "property_key": key,
                    "prompt": prompt,
                    "response": json.dumps(response_payload, ensure_ascii=False),
                    "gt_value": response_payload["properties"][key],
                }
            )
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset = get_dataset_context(args.dataset_name, dataset_root=args.dataset_root)
    schema_path = resolve_protocol_schema_path(dataset, args.protocol_name)
    property_specs = load_protocol_property_specs(protocol_name=args.protocol_name, schema_path=schema_path)
    selected_keys = _parse_property_keys(args.property_keys)
    property_specs = filter_property_specs(property_specs, selected_keys or None)

    train_samples = load_samples_for_dataset(
        dataset=dataset,
        property_specs=property_specs,
        protocol_name=args.protocol_name,
        sample_ids_path=args.train_ids_path,
    )
    train_rows = _build_examples(train_samples, property_specs, include_unknown=args.include_unknown)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train_rows)

    manifest = {
        "dataset_name": args.dataset_name,
        "protocol_name": args.protocol_name,
        "property_keys": list(property_specs.keys()),
        "include_unknown": bool(args.include_unknown),
        "train_ids_path": str(args.train_ids_path.resolve()),
        "train_num_images": len(train_samples),
        "train_num_examples": len(train_rows),
    }

    if args.val_ids_path is not None:
        val_samples = load_samples_for_dataset(
            dataset=dataset,
            property_specs=property_specs,
            protocol_name=args.protocol_name,
            sample_ids_path=args.val_ids_path,
        )
        val_rows = _build_examples(val_samples, property_specs, include_unknown=args.include_unknown)
        _write_jsonl(output_dir / "val.jsonl", val_rows)
        manifest["val_ids_path"] = str(args.val_ids_path.resolve())
        manifest["val_num_images"] = len(val_samples)
        manifest["val_num_examples"] = len(val_rows)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
