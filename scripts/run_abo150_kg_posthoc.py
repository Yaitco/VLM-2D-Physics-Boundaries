# scripts/run_abo150_kg_posthoc.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from vlm_pipeline.kg import make_kg_reranker


REQUIRED_COLUMNS = {
    "image_id",
    "model",
    "property_key",
    "gt_value",
    "pred_value",
}


def _norm(v: Any) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if s in {"", "none", "null", "nan"}:
        return "unknown"
    return s


def load_predictions_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list):
                rows = obj["rows"]
            else:
                raise ValueError("Unsupported JSON format. Expected list[...] or {'rows': [...]}")

        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Predictions file must contain columns {sorted(REQUIRED_COLUMNS)}. Missing: {sorted(missing)}"
        )

    for col in ["image_id", "model", "property_key", "gt_value", "pred_value"]:
        df[col] = df[col].map(_norm)

    if "variant" not in df.columns:
        df["variant"] = "default"

    return df


def compute_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (model, variant, property_key), part in df.groupby(["model", "variant", "property_key"], dropna=False):
        gt = part["gt_value"].tolist()
        pred = part[pred_col].tolist()

        acc = sum(int(g == p) for g, p in zip(gt, pred)) / len(gt) if gt else None

        covered = [(g, p) for g, p in zip(gt, pred) if p != "unknown"]
        sel_acc = (sum(int(g == p) for g, p in covered) / len(covered)) if covered else None

        labels = sorted(set(gt) | set(pred) - {"unknown"})
        f1s = []
        for lab in labels:
            tp = sum(1 for g, p in zip(gt, pred) if g == lab and p == lab)
            fp = sum(1 for g, p in zip(gt, pred) if g != lab and p == lab)
            fn = sum(1 for g, p in zip(gt, pred) if g == lab and p != lab)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        macro_f1 = (sum(f1s) / len(f1s)) if f1s else None

        rows.append(
            {
                "model": model,
                "variant": variant,
                "property_key": property_key,
                "n": len(part),
                "accuracy": None if acc is None else round(100 * acc, 2),
                "macro_f1": None if macro_f1 is None else round(100 * macro_f1, 2),
                "selective_accuracy": None if sel_acc is None else round(100 * sel_acc, 2),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "variant", "property_key"]).reset_index(drop=True)


def bundle_rows_to_dict(part: pd.DataFrame, value_col: str) -> Dict[str, str]:
    return {row["property_key"]: row[value_col] for _, row in part.iterrows()}


def apply_kg_posthoc(df: pd.DataFrame, kg_config: str | Path) -> pd.DataFrame:
    reranker = make_kg_reranker(kg_config)

    corrected_rows: List[Dict[str, Any]] = []
    group_cols = ["model", "variant", "image_id"]

    for group_key, part in df.groupby(group_cols, dropna=False):
        pred_bundle = bundle_rows_to_dict(part, "pred_value")
        corrected_bundle = reranker.rerank_bundle(pred_bundle)

        for _, row in part.iterrows():
            out = dict(row)
            out["pred_value_kg"] = corrected_bundle.get(row["property_key"], row["pred_value"])
            corrected_rows.append(out)

    return pd.DataFrame(corrected_rows)


def build_delta_table(raw_metrics: pd.DataFrame, kg_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = raw_metrics.merge(
        kg_metrics,
        on=["model", "variant", "property_key", "n"],
        suffixes=("_raw", "_kg"),
        how="inner",
    )

    for metric in ["accuracy", "macro_f1", "selective_accuracy"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_kg"] - merged[f"{metric}_raw"]

    return merged.sort_values(["model", "variant", "property_key"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="CSV/JSON/JSONL with per-property predictions")
    parser.add_argument("--kg_config", type=str, default="configs/kg_physical_priors.yaml")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions_table(args.predictions)
    corrected = apply_kg_posthoc(df, args.kg_config)

    raw_metrics = compute_metrics(corrected, pred_col="pred_value")
    kg_metrics = compute_metrics(corrected, pred_col="pred_value_kg")
    delta = build_delta_table(raw_metrics, kg_metrics)

    corrected.to_csv(out_dir / "predictions_with_kg.csv", index=False)
    raw_metrics.to_csv(out_dir / "metrics_raw.csv", index=False)
    kg_metrics.to_csv(out_dir / "metrics_kg.csv", index=False)
    delta.to_csv(out_dir / "metrics_delta.csv", index=False)

    print(f"Saved:\n- {out_dir / 'predictions_with_kg.csv'}")
    print(f"- {out_dir / 'metrics_raw.csv'}")
    print(f"- {out_dir / 'metrics_kg.csv'}")
    print(f"- {out_dir / 'metrics_delta.csv'}")


if __name__ == "__main__":
    main()
