from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from .specs import PropertySpec
from .utils import column_prefix


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

    f1_values = []
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


def build_property_metrics(df: pd.DataFrame, property_specs: Dict[str, PropertySpec]) -> pd.DataFrame:
    total = len(df)
    metrics = []
    for key, spec in property_specs.items():
        col = column_prefix(key)
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
                "coverage_on_gt_known_pct": round(100.0 * extracted_when_gt_known / gt_known, 2)
                if gt_known
                else None,
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
) -> None:
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
        "property_batch_size": int(df["property_batch_size"].iloc[0])
        if "property_batch_size" in df.columns and len(df)
        else None,
        "effective_property_batch_size": int(df["effective_property_batch_size"].iloc[0])
        if "effective_property_batch_size" in df.columns and len(df)
        else None,
        "few_shot_k": int(df["few_shot_k"].iloc[0]) if "few_shot_k" in df.columns and len(df) else 0,
        "few_shot_selection_mode": str(df["few_shot_selection_mode"].iloc[0])
        if "few_shot_selection_mode" in df.columns and len(df)
        else "fixed",
        "num_samples": int(len(df)),
        "valid_json_count": int(df["has_valid_json"].sum()),
        "valid_json_pct": round(100.0 * float(df["has_valid_json"].mean()), 2) if len(df) else 0.0,
        "valid_json_all_count": int(df["has_valid_json_all"].sum()) if "has_valid_json_all" in df.columns else None,
        "valid_json_all_pct": round(100.0 * float(df["has_valid_json_all"].mean()), 2)
        if "has_valid_json_all" in df.columns and len(df)
        else None,
        "image_id_match_count": int(df["image_id_matched"].sum()),
        "image_id_match_pct": round(100.0 * float(df["image_id_matched"].mean()), 2) if len(df) else 0.0,
        "image_id_match_all_count": int(df["image_id_matched_all"].sum())
        if "image_id_matched_all" in df.columns
        else None,
        "image_id_match_all_pct": round(100.0 * float(df["image_id_matched_all"].mean()), 2)
        if "image_id_matched_all" in df.columns and len(df)
        else None,
        "mean_coverage_pct": round(float(property_metrics["coverage_pct"].dropna().mean()), 2)
        if len(property_metrics)
        else None,
        "mean_accuracy_pct": round(float(property_metrics["accuracy_pct"].dropna().mean()), 2)
        if property_metrics["accuracy_pct"].notna().any()
        else None,
        "mean_macro_f1_pct": round(float(property_metrics["macro_f1_pct"].dropna().mean()), 2)
        if property_metrics["macro_f1_pct"].notna().any()
        else None,
        "mean_selective_accuracy_pct": round(float(property_metrics["selective_accuracy_pct"].dropna().mean()), 2)
        if property_metrics["selective_accuracy_pct"].notna().any()
        else None,
    }
    runtime_stats = df.attrs.get("runtime_stats")
    if runtime_stats:
        summary["runtime_stats"] = runtime_stats
    if {"valid_json_count", "expected_response_count"}.issubset(df.columns) and len(df):
        total_expected = float(df["expected_response_count"].sum())
        total_valid = float(df["valid_json_count"].sum())
        summary["valid_json_per_response_pct"] = (
            round(100.0 * total_valid / total_expected, 2) if total_expected > 0 else None
        )

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
        comet_experiment,
        model_key,
        variant,
        property_metrics,
        summary,
        per_sample_path,
        property_path,
        summary_path,
    )
    return property_metrics, summary
