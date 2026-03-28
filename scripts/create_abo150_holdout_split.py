from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic 50/100 holdout split for dataset/abo_150_expanded. "
            "The default val split matches notebook-style random sampling with seed 42."
        )
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=Path("dataset/abo_150_expanded/selected_150_annotations.jsonl"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/abo_150_expanded/splits/seed42_val50_train100"),
    )
    return parser.parse_args()


def load_image_ids(annotations_path: Path) -> List[str]:
    ids: List[str] = []
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = str(record.get("obj_id") or "").strip()
            if image_id:
                ids.append(image_id)
    if not ids:
        raise RuntimeError(f"No obj_id entries found in {annotations_path}")
    return ids


def write_lines(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_ids = load_image_ids(args.annotations_path)
    if args.val_size <= 0 or args.val_size >= len(image_ids):
        raise ValueError(f"val_size must be in [1, {len(image_ids) - 1}]")

    rng = random.Random(args.seed)
    val_ids = rng.sample(image_ids, k=args.val_size)
    val_set = set(val_ids)
    train_ids = [image_id for image_id in image_ids if image_id not in val_set]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_lines(output_dir / "val_ids.txt", val_ids)
    write_lines(output_dir / "train_ids.txt", train_ids)
    (output_dir / "val_ids.json").write_text(json.dumps(val_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "train_ids.json").write_text(
        json.dumps(train_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "annotations_path": str(args.annotations_path.resolve()),
        "seed": int(args.seed),
        "total_size": len(image_ids),
        "val_size": len(val_ids),
        "train_size": len(train_ids),
        "val_ids_path": str((output_dir / "val_ids.txt").resolve()),
        "train_ids_path": str((output_dir / "train_ids.txt").resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
