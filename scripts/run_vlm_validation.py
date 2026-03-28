from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pipeline import (
    MODEL_REGISTRY,
    filter_property_specs,
    get_available_variants,
    get_dataset_context,
    init_comet_experiment,
    load_protocol_property_specs,
    load_samples_for_dataset,
    resolve_protocol_schema_path,
    run_validation,
    save_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the unified VLM validation pipeline on a supported dataset.")
    p.add_argument("--custom-model-key", type=str, default=None)
    p.add_argument("--model-config-path", type=Path, default=None)
    p.add_argument(
        "--dataset-name",
        type=str,
        default="abo_physics_natural_bg_v2",
        choices=["abo_150_expanded", "abo_physics_natural_bg_v2"],
    )
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--protocol-name", type=str, default="natural_bg_v2")
    p.add_argument("--model-key", type=str, default="qwen2_5_vl_7b", choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--variants", type=str, default="raw", help="Comma-separated list: raw,mask_overlay,masked")
    p.add_argument("--reports-dir", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--sample-ids-path",
        type=Path,
        default=None,
        help="Optional TXT/JSON file with explicit image_ids to evaluate.",
    )
    p.add_argument(
        "--meta-override-path",
        type=Path,
        default=None,
        help="Optional alternate meta.json path, useful for running only manually approved samples.",
    )
    p.add_argument(
        "--mask-field",
        type=str,
        default="mask_path",
        help=(
            "Mask field or alias to use from dataset metadata. "
            "Examples: mask_path, masks_hint, mask_path_hint, masks_hint_title."
        ),
    )
    p.add_argument(
        "--mask-preview-field",
        type=str,
        default=None,
        help=(
            "Optional preview field or alias from dataset metadata. "
            "Examples: seg_preview_path, masks_preview_hint, seg_preview_path_hint_title."
        ),
    )
    p.add_argument("--property-batch-size", type=int, default=8)
    p.add_argument(
        "--property-keys",
        type=str,
        default="",
        help="Optional comma-separated property keys to evaluate. Empty means all properties from the protocol.",
    )
    p.add_argument(
        "--property-keys-manifest-path",
        type=Path,
        default=None,
        help="Optional manifest.json path with a 'property_keys' list, e.g. from build_abo150_qlora_dataset.py.",
    )
    p.add_argument("--include-only-gt-known", action="store_true")
    p.add_argument("--few-shot-k", type=int, default=0)
    p.add_argument("--few-shot-selection-mode", type=str, default="fixed", choices=["fixed", "dynamic"])
    p.add_argument("--mask-background-mode", type=str, default="black", choices=["black", "white"])
    p.add_argument("--save-raw-output", action="store_true")
    p.add_argument("--json-success-threshold", type=float, default=1.0)
    p.add_argument("--image-id-success-threshold", type=float, default=None)
    p.add_argument("--disable-comet", action="store_true")
    p.add_argument("--comet-project-name", type=str, default="vlm-physics-validation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_registry = dict(MODEL_REGISTRY)
    selected_model_key = args.model_key
    if args.model_config_path is not None:
        selected_model_key = args.custom_model_key or f"{args.model_key}_custom"
        cfg = json.loads(args.model_config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Expected a JSON object in {args.model_config_path}")
        model_registry[selected_model_key] = cfg
    dataset = get_dataset_context(args.dataset_name, dataset_root=args.dataset_root)
    schema_path = resolve_protocol_schema_path(dataset, args.protocol_name)
    property_specs = load_protocol_property_specs(protocol_name=args.protocol_name, schema_path=schema_path)
    manifest_property_keys = None
    if args.property_keys_manifest_path is not None:
        manifest_cfg = json.loads(args.property_keys_manifest_path.read_text(encoding="utf-8"))
        manifest_property_keys = manifest_cfg.get("property_keys")
        if not isinstance(manifest_property_keys, list) or not manifest_property_keys:
            raise ValueError(
                f"Expected a non-empty 'property_keys' list in {args.property_keys_manifest_path}"
            )
    explicit_property_keys = [key.strip() for key in args.property_keys.split(",") if key.strip()]
    selected_property_keys = explicit_property_keys or manifest_property_keys
    property_specs = filter_property_specs(property_specs, selected_property_keys)
    samples = load_samples_for_dataset(
        dataset=dataset,
        property_specs=property_specs,
        protocol_name=args.protocol_name,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        mask_field=args.mask_field,
        mask_preview_field=args.mask_preview_field,
        meta_override_path=args.meta_override_path,
        sample_ids_path=args.sample_ids_path,
    )

    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    available_variants = get_available_variants(requested_variants, samples)
    reports_dir = args.reports_dir or dataset.reports_dir or Path("reports")
    run_params = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset.dataset_dir),
        "protocol_name": args.protocol_name,
        "selected_model_key": selected_model_key,
        "selected_model_id": model_registry[selected_model_key]["model_id"],
        "eval_variants_requested": ",".join(requested_variants),
        "eval_variants_actual": ",".join(available_variants),
        "sample_ids_path": str(args.sample_ids_path) if args.sample_ids_path is not None else None,
        "meta_override_path": str(args.meta_override_path) if args.meta_override_path is not None else None,
        "property_keys_manifest_path": str(args.property_keys_manifest_path)
        if args.property_keys_manifest_path is not None
        else None,
        "selected_property_keys": "|".join(property_specs.keys()),
        "mask_field": args.mask_field,
        "mask_preview_field": args.mask_preview_field,
        "mask_background_mode": args.mask_background_mode,
        "max_samples": args.max_samples,
        "random_seed": args.random_seed,
        "include_only_gt_known": args.include_only_gt_known,
        "property_batch_size": args.property_batch_size,
        "few_shot_k": args.few_shot_k,
        "few_shot_selection_mode": args.few_shot_selection_mode,
        "json_success_threshold": args.json_success_threshold,
        "image_id_success_threshold": args.image_id_success_threshold,
        "num_properties": len(property_specs),
    }
    comet_experiment = init_comet_experiment(
        run_tag=f"{selected_model_key}_{args.dataset_name}_{args.protocol_name}",
        run_params=run_params,
        enabled=not bool(args.disable_comet),
        default_project=args.comet_project_name,
    )

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Protocol: {args.protocol_name}")
    print(f"Loaded samples: {len(samples)}")
    print(f"Variants: {available_variants}")
    print(f"Model: {selected_model_key} -> {model_registry[selected_model_key]['model_id']}")
    print(f"Meta override path: {args.meta_override_path}")
    print(f"Sample ids path: {args.sample_ids_path}")
    print(f"Property keys manifest path: {args.property_keys_manifest_path}")
    print(f"Selected property keys: {list(property_specs.keys())}")
    print(f"Mask field: {args.mask_field}")
    print(f"Mask preview field: {args.mask_preview_field}")
    try:
        for variant in available_variants:
            df = run_validation(
                model_key=selected_model_key,
                samples=samples,
                property_specs=property_specs,
                model_registry=model_registry,
                variant=variant,
                property_batch_size=args.property_batch_size,
                include_only_gt_known=args.include_only_gt_known,
                few_shot_k=args.few_shot_k,
                few_shot_selection_mode=args.few_shot_selection_mode,
                mask_background_mode=args.mask_background_mode,
                save_raw_output=args.save_raw_output,
                json_success_threshold=args.json_success_threshold,
                image_id_success_threshold=args.image_id_success_threshold,
            )
            save_report(
                df=df,
                model_key=selected_model_key,
                variant=variant,
                model_registry=model_registry,
                property_specs=property_specs,
                reports_dir=reports_dir / args.protocol_name,
                comet_experiment=comet_experiment,
            )
    finally:
        if comet_experiment is not None:
            try:
                comet_experiment.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
