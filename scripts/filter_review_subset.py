from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pipeline.images import apply_mask_overlay_to_image, apply_mask_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive keep/drop filter for an already reviewed segmentation subset. "
            "Reads approved_meta.json and exports a final kept subset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/abo_physics_natural_bg_v2"),
        help="Dataset directory containing image and mask files.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("dataset/abo_physics_natural_bg_v2/review_outputs/segmentation_review_approved_meta.json"),
        help="Path to the already approved meta JSON that should be filtered further.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/abo_physics_natural_bg_v2/review_outputs"),
        help="Directory for filter outputs.",
    )
    parser.add_argument(
        "--review-name",
        type=str,
        default="segmentation_review_final",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=460,
        help="Maximum width per preview panel in pixels.",
    )
    parser.add_argument(
        "--image-max-height",
        type=int,
        default=340,
        help="Maximum height per preview panel in pixels.",
    )
    parser.add_argument(
        "--mask-background-mode",
        type=str,
        default="black",
        choices=["black", "white"],
        help="Background color for generated masked previews.",
    )
    parser.add_argument(
        "--start-image-id",
        type=str,
        default=None,
        help="Optional image_id to jump to at startup.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_dir: Path, value: Any) -> Optional[Path]:
    if not isinstance(value, str):
        return None
    path_str = value.strip().replace("\\", "/")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (dataset_dir.parent / path).resolve()
    return path


def load_meta_rows(meta_path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {meta_path}")
    clean_rows = [row for row in rows if isinstance(row, dict) and row.get("image_id") and row.get("path")]
    if not clean_rows:
        raise RuntimeError(f"No valid rows found in {meta_path}")
    return clean_rows


def load_decisions(decisions_path: Path) -> Dict[str, Dict[str, Any]]:
    if not decisions_path.exists():
        return {}
    payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {decisions_path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_width, max_height), (245, 245, 245))
    x = (max_width - fitted.width) // 2
    y = (max_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


@dataclass
class ReviewView:
    raw_image: Image.Image
    preview_image: Image.Image
    masked_image: Image.Image
    image_path: Path
    mask_path: Optional[Path]
    preview_path: Optional[Path]


def build_review_view(
    row: Dict[str, Any],
    dataset_dir: Path,
    mask_background_mode: str,
) -> ReviewView:
    image_path = resolve_dataset_path(dataset_dir, row.get("path"))
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Image file not found for image_id={row.get('image_id')}: {row.get('path')}")

    raw_image = Image.open(image_path).convert("RGB")
    mask_path = resolve_dataset_path(dataset_dir, row.get("mask_path"))
    preview_path = resolve_dataset_path(dataset_dir, row.get("seg_preview_path"))

    if mask_path is not None and mask_path.exists():
        mask_image = Image.open(mask_path).convert("L")
        overlay_image = apply_mask_overlay_to_image(raw_image, mask_image)
        masked_image = apply_mask_to_image(raw_image, mask_image, bg_mode=mask_background_mode)
    else:
        overlay_image = raw_image.copy()
        masked_image = raw_image.copy()

    if preview_path is not None and preview_path.exists():
        preview_image = Image.open(preview_path).convert("RGB")
    else:
        preview_image = overlay_image

    return ReviewView(
        raw_image=raw_image,
        preview_image=preview_image,
        masked_image=masked_image,
        image_path=image_path,
        mask_path=mask_path if mask_path is not None and mask_path.exists() else None,
        preview_path=preview_path if preview_path is not None and preview_path.exists() else None,
    )


def build_kept_meta(rows: List[Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(row.get("image_id")): row for row in rows}
    kept_rows: List[Dict[str, Any]] = []
    for image_id, decision in decisions.items():
        if decision.get("decision") != "keep":
            continue
        row = by_id.get(image_id)
        if row is None:
            continue
        row_copy = copy.deepcopy(row)
        row_copy["final_review_decision"] = "keep"
        row_copy["final_review_timestamp"] = decision.get("reviewed_at")
        kept_rows.append(row_copy)
    return kept_rows


class ReviewFilterApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dataset_dir = args.dataset_dir.resolve()
        self.meta_path = args.meta_path.resolve()
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_name = args.review_name
        self.image_max_width = int(args.image_max_width)
        self.image_max_height = int(args.image_max_height)
        self.mask_background_mode = args.mask_background_mode

        self.decisions_path = self.output_dir / f"{self.review_name}_decisions.json"
        self.kept_meta_path = self.output_dir / f"{self.review_name}_kept_meta.json"
        self.kept_ids_path = self.output_dir / f"{self.review_name}_kept_ids.txt"

        self.rows = load_meta_rows(self.meta_path)
        self.decisions = load_decisions(self.decisions_path)
        self.index = self._initial_index(args.start_image_id)

        self.root = tk.Tk()
        self.root.title("Segmentation Review Filter")
        self.root.geometry("1540x980")
        self.root.configure(bg="#f5f5f5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.header_label = ttk.Label(self.root, text="", font=("Arial", 15, "bold"), anchor="w", justify="left")
        self.header_label.pack(fill="x", padx=16, pady=(12, 4))

        self.meta_label = ttk.Label(self.root, text="", anchor="w", justify="left")
        self.meta_label.pack(fill="x", padx=16, pady=(0, 8))

        self.shortcuts_label = ttk.Label(
            self.root,
            text="Хоткеи: K оставить, D удалить, S пропустить, ←/→ назад/вперёд",
            anchor="w",
            justify="left",
        )
        self.shortcuts_label.pack(fill="x", padx=16, pady=(0, 12))

        self.panel_frame = ttk.Frame(self.root)
        self.panel_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=16, pady=(0, 16))
        ttk.Button(controls, text="← Назад", command=self.go_prev).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Пропустить", command=self.skip_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Удалить", command=self.drop_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Оставить", command=self.keep_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="→ Вперёд", command=self.go_next).pack(side="left", padx=(0, 8))

        self.root.bind("<Key>", self.on_key)
        self.render_current()

    def _initial_index(self, start_image_id: Optional[str]) -> int:
        if start_image_id:
            for idx, row in enumerate(self.rows):
                if str(row.get("image_id")) == start_image_id:
                    return idx
        for idx, row in enumerate(self.rows):
            if str(row.get("image_id")) not in self.decisions:
                return idx
        return 0

    def current_row(self) -> Dict[str, Any]:
        return self.rows[self.index]

    def current_decision(self) -> Optional[Dict[str, Any]]:
        return self.decisions.get(str(self.current_row().get("image_id")))

    def make_photo(self, image: Image.Image) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(fit_image(image, self.image_max_width, self.image_max_height))

    def on_key(self, event: tk.Event) -> None:
        key = (event.keysym or "").lower()
        if key == "k":
            self.keep_current()
        elif key == "d":
            self.drop_current()
        elif key == "s":
            self.skip_current()
        elif key == "left":
            self.go_prev()
        elif key == "right":
            self.go_next()

    def render_current(self) -> None:
        row = self.current_row()
        decision = self.current_decision()
        status = decision.get("decision") if decision else "unreviewed"
        kept = sum(1 for item in self.decisions.values() if item.get("decision") == "keep")
        dropped = sum(1 for item in self.decisions.values() if item.get("decision") == "drop")
        skipped = sum(1 for item in self.decisions.values() if item.get("decision") == "skipped")

        for widget in self.panel_frame.winfo_children():
            widget.destroy()

        title = (row.get("abo_meta") or {}).get("title") or row.get("primary_object") or "Untitled"
        product_type = (row.get("abo_meta") or {}).get("product_type") or "unknown"
        self.header_label.configure(
            text=(
                f"[{self.index + 1}/{len(self.rows)}] image_id={row.get('image_id')} | "
                f"status={status} | kept={kept}, dropped={dropped}, skipped={skipped}"
            )
        )
        self.meta_label.configure(
            text=(
                f"primary_object: {row.get('primary_object') or 'unknown'}\n"
                f"product_type: {product_type}\n"
                f"title: {title}\n"
                f"mask_source: {row.get('mask_source') or row.get('review_selected_mask_source') or 'unknown'}\n"
                f"selected_mask_field: {row.get('review_selected_mask_field') or 'mask_path'}"
            )
        )

        view = build_review_view(
            row=row,
            dataset_dir=self.dataset_dir,
            mask_background_mode=self.mask_background_mode,
        )

        panels = [
            ("Оригинал", view.raw_image),
            ("Preview / Overlay", view.preview_image),
            ("Masked", view.masked_image),
        ]
        for col, (label, image) in enumerate(panels):
            panel = ttk.LabelFrame(self.panel_frame, text=label)
            panel.grid(row=0, column=col, padx=8, pady=8, sticky="n")
            photo = self.make_photo(image)
            image_label = ttk.Label(panel, image=photo)
            image_label.image = photo
            image_label.pack(padx=8, pady=8)

    def save_outputs(self) -> None:
        write_json(self.decisions_path, self.decisions)
        kept_meta = build_kept_meta(self.rows, self.decisions)
        write_json(self.kept_meta_path, kept_meta)
        kept_ids = [str(row.get("image_id")) for row in kept_meta]
        self.kept_ids_path.write_text("\n".join(kept_ids) + ("\n" if kept_ids else ""), encoding="utf-8")

    def record_decision(self, decision: str) -> None:
        row = self.current_row()
        image_id = str(row.get("image_id"))
        self.decisions[image_id] = {
            "image_id": image_id,
            "decision": decision,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_outputs()

    def go_prev(self) -> None:
        if self.index > 0:
            self.index -= 1
        self.render_current()

    def go_next(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
        self.render_current()

    def keep_current(self) -> None:
        self.record_decision("keep")
        self.go_next()

    def drop_current(self) -> None:
        self.record_decision("drop")
        self.go_next()

    def skip_current(self) -> None:
        self.record_decision("skipped")
        self.go_next()

    def on_close(self) -> None:
        self.save_outputs()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    app = ReviewFilterApp(args)
    app.run()


if __name__ == "__main__":
    main()
