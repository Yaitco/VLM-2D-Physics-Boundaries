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


DEFAULT_MASK_VARIANTS = [
    {
        "label": "base",
        "mask_field": "mask_path",
        "preview_field": "seg_preview_path",
        "fallback_preview_field": "sam_preview_path",
        "source_field": "mask_source",
    },
    {
        "label": "hint",
        "mask_field": "mask_path_hint",
        "preview_field": "seg_preview_path_hint",
        "fallback_preview_field": None,
        "source_field": "mask_source_hint",
    },
    {
        "label": "hint_title",
        "mask_field": "mask_path_hint_title",
        "preview_field": "seg_preview_path_hint_title",
        "fallback_preview_field": None,
        "source_field": "mask_source_hint_title",
    },
]


@dataclass
class VariantView:
    label: str
    mask_field: str
    preview_field: Optional[str]
    source_field: Optional[str]
    mask_path: Optional[Path]
    preview_path: Optional[Path]
    source_value: Optional[str]
    masked_image: Image.Image
    preview_image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive manual review tool for segmentation masks. "
            "Shows the raw image and multiple mask variants, then exports approved metadata."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/abo_physics_natural_bg_v2"),
        help="Dataset directory containing meta.json and image/mask files.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional explicit path to meta.json. Defaults to <dataset-dir>/meta.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for review outputs. Defaults to <dataset-dir>/review_outputs.",
    )
    parser.add_argument(
        "--review-name",
        type=str,
        default="segmentation_review",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=440,
        help="Maximum width per preview panel in pixels.",
    )
    parser.add_argument(
        "--image-max-height",
        type=int,
        default=300,
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


def build_variant_views(
    row: Dict[str, Any],
    dataset_dir: Path,
    max_width: int,
    max_height: int,
    mask_background_mode: str,
) -> List[VariantView]:
    image_path = resolve_dataset_path(dataset_dir, row.get("path"))
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Image file not found for image_id={row.get('image_id')}: {row.get('path')}")

    raw_image = Image.open(image_path).convert("RGB")
    views: List[VariantView] = []
    for config in DEFAULT_MASK_VARIANTS:
        mask_path = resolve_dataset_path(dataset_dir, row.get(config["mask_field"]))
        if mask_path is None or not mask_path.exists():
            continue

        preview_path = resolve_dataset_path(dataset_dir, row.get(config["preview_field"]))
        if (preview_path is None or not preview_path.exists()) and config.get("fallback_preview_field"):
            preview_path = resolve_dataset_path(dataset_dir, row.get(config["fallback_preview_field"]))

        mask_image = Image.open(mask_path).convert("L")
        overlay_image = apply_mask_overlay_to_image(raw_image, mask_image)
        masked_image = apply_mask_to_image(raw_image, mask_image, bg_mode=mask_background_mode)
        preview_image = (
            Image.open(preview_path).convert("RGB")
            if preview_path is not None and preview_path.exists()
            else overlay_image
        )

        views.append(
            VariantView(
                label=config["label"],
                mask_field=config["mask_field"],
                preview_field=config["preview_field"],
                source_field=config["source_field"],
                mask_path=mask_path,
                preview_path=preview_path,
                source_value=row.get(config["source_field"]),
                masked_image=masked_image.copy(),
                preview_image=preview_image.copy(),
            )
        )

    if not views:
        raise RuntimeError(f"No mask variants available for image_id={row.get('image_id')}")
    return views


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_width, max_height), (245, 245, 245))
    x = (max_width - fitted.width) // 2
    y = (max_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def build_approved_meta(
    rows: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    approved_rows: List[Dict[str, Any]] = []
    by_id = {str(row.get("image_id")): row for row in rows}
    for image_id, decision in decisions.items():
        if decision.get("decision") != "approved":
            continue
        row = by_id.get(image_id)
        if row is None:
            continue

        selected_mask_field = decision.get("selected_mask_field")
        selected_preview_field = decision.get("selected_preview_field")
        selected_source_field = decision.get("selected_source_field")

        row_copy = copy.deepcopy(row)
        if selected_mask_field and row_copy.get(selected_mask_field):
            row_copy["mask_path"] = row_copy[selected_mask_field]
        if selected_source_field and row_copy.get(selected_source_field):
            row_copy["mask_source"] = row_copy[selected_source_field]
        if selected_preview_field and row_copy.get(selected_preview_field):
            row_copy["seg_preview_path"] = row_copy[selected_preview_field]

        row_copy["review_decision"] = "approved"
        row_copy["review_selected_mask_field"] = selected_mask_field
        row_copy["review_selected_mask_source_field"] = selected_source_field
        row_copy["review_selected_preview_field"] = selected_preview_field
        row_copy["review_selected_mask_source"] = decision.get("selected_mask_source")
        row_copy["review_selected_mask_path"] = decision.get("selected_mask_path")
        row_copy["review_selected_preview_path"] = decision.get("selected_preview_path")
        row_copy["review_timestamp"] = decision.get("reviewed_at")
        approved_rows.append(row_copy)
    return approved_rows


class SegmentationReviewApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dataset_dir = args.dataset_dir.resolve()
        self.meta_path = (args.meta_path or (self.dataset_dir / "meta.json")).resolve()
        self.output_dir = (args.output_dir or (self.dataset_dir / "review_outputs")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_name = args.review_name
        self.image_max_width = int(args.image_max_width)
        self.image_max_height = int(args.image_max_height)
        self.mask_background_mode = args.mask_background_mode

        self.decisions_path = self.output_dir / f"{self.review_name}_decisions.json"
        self.approved_meta_path = self.output_dir / f"{self.review_name}_approved_meta.json"
        self.approved_ids_path = self.output_dir / f"{self.review_name}_approved_ids.txt"

        self.rows = load_meta_rows(self.meta_path)
        self.decisions = load_decisions(self.decisions_path)
        self.index = self._initial_index(args.start_image_id)

        self.root = tk.Tk()
        self.root.title("Segmentation Review")
        self.root.geometry("1720x980")
        self.root.configure(bg="#f5f5f5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected_variant = tk.StringVar(value="")

        self.header_label = ttk.Label(self.root, text="", font=("Arial", 15, "bold"), anchor="w", justify="left")
        self.header_label.pack(fill="x", padx=16, pady=(12, 4))

        self.meta_label = ttk.Label(self.root, text="", anchor="w", justify="left")
        self.meta_label.pack(fill="x", padx=16, pady=(0, 8))

        self.shortcuts_label = ttk.Label(
            self.root,
            text="Хоткеи: 1/2/3 выбрать вариант, A одобрить, R отклонить, S пропустить, ←/→ назад/вперёд",
            anchor="w",
            justify="left",
        )
        self.shortcuts_label.pack(fill="x", padx=16, pady=(0, 12))

        self.raw_frame = ttk.LabelFrame(self.root, text="Оригинальное изображение")
        self.raw_frame.pack(fill="x", padx=16, pady=(0, 12))
        self.raw_image_label = ttk.Label(self.raw_frame)
        self.raw_image_label.pack(padx=8, pady=8)

        self.variants_frame = ttk.Frame(self.root)
        self.variants_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=16, pady=(0, 16))
        ttk.Button(controls, text="← Назад", command=self.go_prev).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Пропустить", command=self.skip_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Отклонить", command=self.reject_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Одобрить выбранный вариант", command=self.approve_current).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="→ Вперёд", command=self.go_next).pack(side="left", padx=(0, 8))

        self.root.bind("<Key>", self.on_key)
        self.variant_widgets: List[Dict[str, Any]] = []
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

    def on_key(self, event: tk.Event) -> None:
        key = (event.keysym or "").lower()
        if key in {"1", "2", "3"}:
            pos = int(key) - 1
            if pos < len(self.variant_widgets):
                self.selected_variant.set(self.variant_widgets[pos]["mask_field"])
        elif key == "a":
            self.approve_current()
        elif key == "r":
            self.reject_current()
        elif key == "s":
            self.skip_current()
        elif key == "left":
            self.go_prev()
        elif key == "right":
            self.go_next()

    def make_photo(self, image: Image.Image) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(fit_image(image, self.image_max_width, self.image_max_height))

    def render_current(self) -> None:
        row = self.current_row()
        decision = self.current_decision()
        raw_path = resolve_dataset_path(self.dataset_dir, row.get("path"))
        if raw_path is None or not raw_path.exists():
            raise FileNotFoundError(f"Image file not found: {row.get('path')}")
        raw_image = Image.open(raw_path).convert("RGB")
        raw_photo = self.make_photo(raw_image)
        self.raw_image_label.configure(image=raw_photo)
        self.raw_image_label.image = raw_photo

        for widget in self.variants_frame.winfo_children():
            widget.destroy()
        self.variant_widgets = []

        views = build_variant_views(
            row=row,
            dataset_dir=self.dataset_dir,
            max_width=self.image_max_width,
            max_height=self.image_max_height,
            mask_background_mode=self.mask_background_mode,
        )

        default_variant = views[0].mask_field
        if decision and decision.get("selected_mask_field"):
            default_variant = decision["selected_mask_field"]
        self.selected_variant.set(default_variant)

        approved = sum(1 for item in self.decisions.values() if item.get("decision") == "approved")
        rejected = sum(1 for item in self.decisions.values() if item.get("decision") == "rejected")
        skipped = sum(1 for item in self.decisions.values() if item.get("decision") == "skipped")
        status = decision.get("decision") if decision else "unreviewed"
        title = (row.get("abo_meta") or {}).get("title") or row.get("primary_object") or "Untitled"
        product_type = (row.get("abo_meta") or {}).get("product_type") or "unknown"

        self.header_label.configure(
            text=(
                f"[{self.index + 1}/{len(self.rows)}] image_id={row.get('image_id')} | "
                f"status={status} | approved={approved}, rejected={rejected}, skipped={skipped}"
            )
        )
        self.meta_label.configure(
            text=(
                f"primary_object: {row.get('primary_object') or 'unknown'}\n"
                f"product_type: {product_type}\n"
                f"title: {title}"
            )
        )

        for col, view in enumerate(views):
            panel = ttk.LabelFrame(
                self.variants_frame,
                text=f"{col + 1}. {view.label} | {view.mask_field}",
            )
            panel.grid(row=0, column=col, padx=8, pady=8, sticky="n")
            ttk.Radiobutton(
                panel,
                text="Выбрать этот вариант",
                variable=self.selected_variant,
                value=view.mask_field,
            ).pack(anchor="w", padx=8, pady=(8, 4))

            ttk.Label(
                panel,
                text=f"source: {view.source_value or 'unknown'}",
                justify="left",
                wraplength=self.image_max_width,
            ).pack(anchor="w", padx=8, pady=(0, 8))

            overlay_photo = self.make_photo(view.preview_image)
            overlay_label = ttk.Label(panel, image=overlay_photo)
            overlay_label.image = overlay_photo
            overlay_label.pack(padx=8, pady=(0, 6))
            ttk.Label(panel, text="preview / overlay").pack(anchor="center", pady=(0, 8))

            masked_photo = self.make_photo(view.masked_image)
            masked_label = ttk.Label(panel, image=masked_photo)
            masked_label.image = masked_photo
            masked_label.pack(padx=8, pady=(0, 6))
            ttk.Label(panel, text="masked").pack(anchor="center", pady=(0, 8))

            self.variant_widgets.append(
                {
                    "mask_field": view.mask_field,
                    "preview_field": view.preview_field,
                    "source_field": view.source_field,
                    "mask_path": str(view.mask_path) if view.mask_path else None,
                    "preview_path": str(view.preview_path) if view.preview_path else None,
                    "source_value": view.source_value,
                }
            )

    def selected_variant_payload(self) -> Dict[str, Any]:
        selected = self.selected_variant.get()
        for item in self.variant_widgets:
            if item["mask_field"] == selected:
                return item
        raise RuntimeError("No variant selected")

    def save_outputs(self) -> None:
        write_json(self.decisions_path, self.decisions)
        approved_meta = build_approved_meta(self.rows, self.decisions)
        write_json(self.approved_meta_path, approved_meta)
        approved_ids = [str(row.get("image_id")) for row in approved_meta]
        self.approved_ids_path.write_text("\n".join(approved_ids) + ("\n" if approved_ids else ""), encoding="utf-8")

    def record_decision(self, decision: str) -> None:
        row = self.current_row()
        image_id = str(row.get("image_id"))
        selected = self.selected_variant_payload()
        self.decisions[image_id] = {
            "image_id": image_id,
            "decision": decision,
            "selected_mask_field": selected.get("mask_field"),
            "selected_preview_field": selected.get("preview_field"),
            "selected_source_field": selected.get("source_field"),
            "selected_mask_path": selected.get("mask_path"),
            "selected_preview_path": selected.get("preview_path"),
            "selected_mask_source": selected.get("source_value"),
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

    def approve_current(self) -> None:
        self.record_decision("approved")
        self.go_next()

    def reject_current(self) -> None:
        self.record_decision("rejected")
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
    app = SegmentationReviewApp(args)
    app.run()


if __name__ == "__main__":
    main()
