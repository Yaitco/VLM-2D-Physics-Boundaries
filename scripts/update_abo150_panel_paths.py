#!/usr/bin/env python3
"""Rewrite panel_path entries in ABO-150 annotations to local paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update panels.panel_path in selected_150_annotations.jsonl."
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=Path("dataset/abo_150_expanded/selected_150_annotations.jsonl"),
        help="Path to selected_150_annotations.jsonl",
    )
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=Path("dataset/abo_150_expanded/selected_150_photos/panels"),
        help="Directory with local panel images.",
    )
    parser.add_argument(
        "--relative-prefix",
        type=Path,
        default=Path("selected_150_photos/panels"),
        help="Prefix to use when --path-mode=relative.",
    )
    parser.add_argument(
        "--path-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="How to store panel_path in JSONL.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak file before rewriting annotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output file.",
    )
    return parser.parse_args()


def choose_filename(record: Dict[str, object]) -> str:
    panels = record.get("panels", {}) if isinstance(record.get("panels"), dict) else {}
    old_path = panels.get("panel_path")
    if isinstance(old_path, str) and old_path.strip():
        return Path(old_path).name

    obj_id = record.get("obj_id")
    if isinstance(obj_id, str) and obj_id.strip():
        return f"{obj_id}.png"

    raise ValueError("Cannot derive filename: both panels.panel_path and obj_id are missing.")


def build_new_path(filename: str, panels_dir: Path, relative_prefix: Path, mode: str) -> str:
    if mode == "relative":
        return (relative_prefix / filename).as_posix()
    return str((panels_dir / filename).resolve())


def update_records(
    annotations_path: Path,
    panels_dir: Path,
    relative_prefix: Path,
    path_mode: str,
) -> Tuple[List[Dict[str, object]], int, int]:
    updated: List[Dict[str, object]] = []
    changed = 0
    missing_files = 0

    with annotations_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = choose_filename(record)
            panel_file = panels_dir / filename
            if not panel_file.exists():
                missing_files += 1

            new_path = build_new_path(
                filename=filename,
                panels_dir=panels_dir,
                relative_prefix=relative_prefix,
                mode=path_mode,
            )

            panels = record.get("panels")
            if not isinstance(panels, dict):
                panels = {}
                record["panels"] = panels

            old_path = panels.get("panel_path")
            if old_path != new_path:
                changed += 1

            panels["panel_path"] = new_path
            updated.append(record)

    return updated, changed, missing_files


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    annotations_path = args.annotations_path
    panels_dir = args.panels_dir
    relative_prefix = args.relative_prefix

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file does not exist: {annotations_path}")
    if not panels_dir.exists():
        raise FileNotFoundError(f"Panels dir does not exist: {panels_dir}")

    rows, changed, missing_files = update_records(
        annotations_path=annotations_path,
        panels_dir=panels_dir,
        relative_prefix=relative_prefix,
        path_mode=args.path_mode,
    )

    print(f"Rows processed: {len(rows)}")
    print(f"Rows changed: {changed}")
    print(f"Missing panel files: {missing_files}")
    print(f"Path mode: {args.path_mode}")

    if args.dry_run:
        print("Dry-run mode: file was not modified.")
        return

    if args.backup:
        backup_path = annotations_path.with_suffix(annotations_path.suffix + ".bak")
        backup_path.write_bytes(annotations_path.read_bytes())
        print(f"Backup written: {backup_path}")

    write_jsonl(annotations_path, rows)
    print(f"Annotations updated in-place: {annotations_path}")


if __name__ == "__main__":
    main()
