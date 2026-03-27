from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pipeline import (
    MODEL_REGISTRY,
    get_available_variants,
    get_dataset_context,
    load_protocol_property_specs,
    load_samples_for_dataset,
    resolve_protocol_schema_path,
    run_validation,
    save_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the unified VLM validation pipeline on a supported dataset.")
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
    p.add_argument("--include-only-gt-known", action="store_true")
    p.add_argument("--few-shot-k", type=int, default=0)
    p.add_argument("--few-shot-selection-mode", type=str, default="fixed", choices=["fixed", "dynamic"])
    p.add_argument("--mask-background-mode", type=str, default="black", choices=["black", "white"])
    p.add_argument("--save-raw-output", action="store_true")
    p.add_argument("--json-success-threshold", type=float, default=1.0)
    p.add_argument("--image-id-success-threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset_context(args.dataset_name, dataset_root=args.dataset_root)
    schema_path = resolve_protocol_schema_path(dataset, args.protocol_name)
    property_specs = load_protocol_property_specs(protocol_name=args.protocol_name, schema_path=schema_path)
    samples = load_samples_for_dataset(
        dataset=dataset,
        property_specs=property_specs,
        protocol_name=args.protocol_name,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        mask_field=args.mask_field,
        mask_preview_field=args.mask_preview_field,
        meta_override_path=args.meta_override_path,
    )

    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    available_variants = get_available_variants(requested_variants, samples)
    reports_dir = args.reports_dir or dataset.reports_dir or Path("reports")

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Protocol: {args.protocol_name}")
    print(f"Loaded samples: {len(samples)}")
    print(f"Variants: {available_variants}")
    print(f"Model: {args.model_key} -> {MODEL_REGISTRY[args.model_key]['model_id']}")
    print(f"Meta override path: {args.meta_override_path}")
    print(f"Mask field: {args.mask_field}")
    print(f"Mask preview field: {args.mask_preview_field}")

    for variant in available_variants:
        df = run_validation(
            model_key=args.model_key,
            samples=samples,
            property_specs=property_specs,
            model_registry=MODEL_REGISTRY,
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
            model_key=args.model_key,
            variant=variant,
            model_registry=MODEL_REGISTRY,
            property_specs=property_specs,
            reports_dir=reports_dir / args.protocol_name,
            comet_experiment=None,
        )


if __name__ == "__main__":
    main()
