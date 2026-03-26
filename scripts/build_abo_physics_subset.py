#!/usr/bin/env python3
"""Build a physics-oriented ABO validation subset.

Output format is compatible with notebooks/CourseWork.ipynb:
- dataset/abo_physics_val/meta.json
- dataset/abo_physics_val/images/<xx>/<file>.jpg
- dataset/abo_physics_val/masks/<xx>/<file>.png (optional)
- dataset/abo_physics_val/summary.json

The script supports two modes:
1) local-only mode: reuse already downloaded images from `--source-images-dir`
2) download mode: if `--download-missing` is set, missing images are fetched from
   public ABO S3 bucket.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover - boto3 can be absent in local mode
    boto3 = None
    UNSIGNED = None
    Config = None
    ClientError = None


CANONICAL_MATERIALS = {
    "glass",
    "metal",
    "wood",
    "plastic",
    "rubber",
    "fabric",
    "paper",
    "ceramic",
    "stone",
    "mixed",
    "unknown",
}

PROPERTY_ORDER = ["material", "rigidity", "transparency", "surface", "fragility"]


MATERIAL_SYNONYMS = {
    "glass": [
        "glass",
        "crystal",
        "стек",
        "glas",
        "verre",
        "vidrio",
    ],
    "metal": [
        "metal",
        "steel",
        "iron",
        "aluminium",
        "aluminum",
        "alloy",
        "brass",
        "bronze",
        "chrome",
        "stainless",
        "металл",
    ],
    "wood": [
        "wood",
        "wooden",
        "bamboo",
        "plywood",
        "timber",
        "madera",
        "holz",
        "legno",
        "木",
    ],
    "plastic": [
        "plastic",
        "pvc",
        "acrylic",
        "polycarbonate",
        "polypropylene",
        "polyethylene",
        "polyurethane",
        "pu ",
        "abs",
        "пластик",
        "plast",
    ],
    "rubber": [
        "rubber",
        "latex",
        "silicone",
        "silicon",
        "tpu",
        "silikon",
        "goma",
        "кауч",
    ],
    "fabric": [
        "fabric",
        "textile",
        "cloth",
        "cotton",
        "polyester",
        "linen",
        "wool",
        "canvas",
        "nylon",
        "denim",
        "suede",
        "velvet",
        "mesh",
        "leather",
        "faux leather",
        "tela",
        "stoff",
        "algod",
        "baumwoll",
        "皮革",
        "涤纶",
    ],
    "paper": [
        "paper",
        "cardboard",
        "karton",
        "papier",
        "papel",
        "бумаг",
    ],
    "ceramic": [
        "ceramic",
        "porcelain",
        "earthenware",
        "clay",
        "keramik",
    ],
    "stone": [
        "stone",
        "marble",
        "granite",
        "slate",
        "concrete",
        "cement",
        "камень",
        "piedra",
    ],
}

SOFT_HINTS = {
    "soft",
    "flex",
    "flexible",
    "elastic",
    "gel",
    "stretch",
}
HARD_HINTS = {"hard", "rigid", "solid", "stiff"}

TRANSPARENT_HINTS = {
    "transparent",
    "clear",
    "see-through",
    "see through",
}
TRANSLUCENT_HINTS = {"translucent", "frosted", "semi-transparent", "semi transparent"}
OPAQUE_HINTS = {"opaque", "solid color"}

FUZZY_HINTS = {
    "fuzzy",
    "plush",
    "wool",
    "fleece",
    "fur",
    "velvet",
    "fluffy",
}
ROUGH_HINTS = {
    "rough",
    "textured",
    "grain",
    "grained",
    "ribbed",
    "embossed",
    "matte",
    "hammered",
}
POROUS_HINTS = {"porous", "sponge", "mesh", "foam"}
SMOOTH_HINTS = {"smooth", "glossy", "polished", "sleek"}

FRAGILE_HINTS = {"fragile", "breakable", "delicate", "brittle"}
DURABLE_HINTS = {"durable", "unbreakable", "shatterproof", "heavy-duty", "heavy duty"}


PRIMARY_OBJECT_MAP = {
    "CELLULAR_PHONE_CASE": "phone case",
    "PORTABLE_ELECTRONIC_DEVICE_COVER": "device cover",
    "SHOES": "shoes",
    "SANDAL": "sandals",
    "BOOT": "boots",
    "CHAIR": "chair",
    "SOFA": "sofa",
    "TABLE": "table",
    "RUG": "rug",
    "LAMP": "lamp",
    "LIGHT_FIXTURE": "light fixture",
    "GLASSWARE": "glassware",
}


@dataclass
class Candidate:
    item_id: str
    main_image_id: str
    image_rel: str
    product_type: str
    primary_object: str
    properties: Dict[str, str]
    notes: str
    title: Optional[str]
    domain_name: Optional[str]
    known_count: int


# ----------------- Generic helpers -----------------

class ProgressPrinter:
    def __init__(self, label: str, every: int):
        self.label = label
        self.every = max(0, int(every))
        self.start = time.perf_counter()

    def emit(self, current: int, extra: str = "", force: bool = False) -> None:
        if self.every <= 0:
            return
        if not force and current % self.every != 0:
            return
        elapsed = max(1e-6, time.perf_counter() - self.start)
        rate = current / elapsed
        suffix = f" | {extra}" if extra else ""
        print(
            f"[{self.label}] {current} items processed | elapsed={elapsed:.1f}s | rate={rate:.1f}/s{suffix}",
            flush=True,
        )

def safe_strip(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return " ".join(s.split())


def pick_value(field) -> Optional[str]:
    if field is None:
        return None

    if isinstance(field, str):
        return safe_strip(field)

    if isinstance(field, dict):
        standardized = field.get("standardized_values")
        if isinstance(standardized, list) and standardized:
            return safe_strip(standardized[0])
        return safe_strip(field.get("value"))

    if isinstance(field, list):
        for entry in field:
            if isinstance(entry, dict):
                standardized = entry.get("standardized_values")
                if isinstance(standardized, list) and standardized:
                    v = safe_strip(standardized[0])
                    if v:
                        return v
        for entry in field:
            if isinstance(entry, dict):
                v = safe_strip(entry.get("value"))
                if v:
                    return v
            else:
                v = safe_strip(entry)
                if v:
                    return v

    return None


def normalize_product_type(value: Optional[str]) -> str:
    if not value:
        return "UNKNOWN"
    return re.sub(r"\s+", "_", value.strip().upper())


def normalize_space_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_str_set(raw: str) -> set[str]:
    values: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.add(normalize_product_type(part))
    return values


# ----------------- Data loading -----------------

def load_image_index(images_csv_gz: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    with gzip.open(images_csv_gz, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_id = row.get("image_id")
            rel = row.get("path")
            if image_id and rel:
                index[image_id] = rel
    return index


def iter_listing_records(listing_files: Iterable[Path]) -> Iterator[dict]:
    for fp in listing_files:
        with gzip.open(fp, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield rec


# ----------------- Property inference -----------------

def detect_material_tokens(text: str) -> List[str]:
    hits = []
    low = normalize_space_lower(text)
    padded = f" {low} "

    for material, synonyms in MATERIAL_SYNONYMS.items():
        for syn in synonyms:
            syn_low = syn.lower()
            if len(syn_low) <= 2:
                continue
            if syn_low.strip().isalnum():
                # Word-boundary style check for short alnum tokens.
                if re.search(rf"\b{re.escape(syn_low.strip())}\b", low):
                    hits.append(material)
                    break
            else:
                if syn_low in padded:
                    hits.append(material)
                    break

    # Normalize repeated labels.
    out = []
    seen = set()
    for mat in hits:
        if mat not in seen:
            seen.add(mat)
            out.append(mat)
    return out


def infer_material(rec: dict) -> Tuple[str, str]:
    material_field = pick_value(rec.get("material"))
    fabric_field = pick_value(rec.get("fabric_type"))
    finish_field = pick_value(rec.get("finish_type"))
    title = pick_value(rec.get("item_name"))
    product_type = pick_value(rec.get("product_type"))

    text_chunks = [
        material_field or "",
        fabric_field or "",
        finish_field or "",
        title or "",
        product_type or "",
    ]
    text_blob = " | ".join(text_chunks)

    direct_hits = detect_material_tokens(material_field or "")
    if direct_hits:
        if len(direct_hits) == 1:
            return direct_hits[0], "material_field"
        return "mixed", "material_field_multi"

    all_hits = detect_material_tokens(text_blob)
    if len(all_hits) == 1:
        return all_hits[0], "metadata_text"
    if len(all_hits) > 1:
        return "mixed", "metadata_text_multi"

    return "unknown", "none"


def has_any(text: str, hints: set[str]) -> bool:
    low = normalize_space_lower(text)
    return any(h in low for h in hints)


def infer_rigidity(material: str, text: str) -> str:
    if material in {"glass", "metal", "wood", "ceramic", "stone"}:
        return "rigid"
    if material == "fabric":
        return "soft"
    if material == "rubber":
        return "flexible"
    if material == "paper":
        return "flexible"
    if material == "mixed":
        return "mixed"
    if material == "plastic":
        if has_any(text, SOFT_HINTS):
            return "flexible"
        if has_any(text, HARD_HINTS):
            return "rigid"
        return "rigid"
    return "unknown"


def infer_transparency(material: str, text: str, color: Optional[str]) -> str:
    hay = " ".join([text, color or ""])
    if has_any(hay, TRANSLUCENT_HINTS):
        return "translucent"
    if has_any(hay, TRANSPARENT_HINTS):
        return "transparent"
    if has_any(hay, OPAQUE_HINTS):
        return "opaque"

    if material == "glass":
        return "transparent"
    if material in {"metal", "wood", "plastic", "rubber", "fabric", "paper", "ceramic", "stone"}:
        return "opaque"
    return "unknown"


def infer_surface(material: str, text: str, product_type: str) -> str:
    hay = f"{text} {product_type}"
    if has_any(hay, FUZZY_HINTS):
        return "fuzzy"
    if has_any(hay, POROUS_HINTS):
        return "porous"
    if has_any(hay, ROUGH_HINTS):
        return "rough"
    if has_any(hay, SMOOTH_HINTS):
        return "smooth"

    if material in {"glass", "metal", "plastic", "ceramic"}:
        return "smooth"
    if material in {"stone", "wood"}:
        return "rough"
    if material == "mixed":
        return "mixed"
    if material == "rubber":
        return "smooth"
    if material == "paper":
        return "rough"

    # For fabric, use weaker default only for obvious product groups.
    if material == "fabric" and any(tag in product_type for tag in ["RUG", "BLANKET", "PILLOW", "SHEET"]):
        return "fuzzy"
    if material == "fabric":
        # Fabric can look either smooth or textured; keep it explicit as mixed.
        return "mixed"

    return "unknown"


def infer_fragility(material: str, text: str) -> str:
    if has_any(text, FRAGILE_HINTS):
        return "fragile"
    if has_any(text, DURABLE_HINTS):
        return "durable"

    if material in {"glass", "ceramic"}:
        return "fragile"
    if material in {"metal", "wood", "stone", "plastic", "rubber", "fabric"}:
        return "durable"
    if material == "paper":
        return "fragile"
    return "unknown"


def product_type_to_primary_object(product_type: str) -> str:
    if product_type in PRIMARY_OBJECT_MAP:
        return PRIMARY_OBJECT_MAP[product_type]
    if product_type == "UNKNOWN":
        return "object"
    return product_type.replace("_", " ").lower()


def infer_properties(rec: dict) -> Tuple[Dict[str, str], str]:
    product_type = normalize_product_type(pick_value(rec.get("product_type")))
    title = pick_value(rec.get("item_name")) or ""
    bullet = pick_value(rec.get("bullet_point")) or ""
    description = pick_value(rec.get("product_description")) or ""
    color = pick_value(rec.get("color"))

    context = " | ".join([title, bullet, description, color or "", product_type])

    material, material_source = infer_material(rec)
    rigidity = infer_rigidity(material, context)
    transparency = infer_transparency(material, context, color)
    surface = infer_surface(material, context, product_type)
    fragility = infer_fragility(material, context)

    props = {
        "material": material,
        "rigidity": rigidity,
        "transparency": transparency,
        "surface": surface,
        "fragility": fragility,
    }

    # Ensure values in expected schema.
    if props["material"] not in CANONICAL_MATERIALS:
        props["material"] = "unknown"
    if props["rigidity"] not in {"rigid", "soft", "flexible", "mixed", "unknown"}:
        props["rigidity"] = "unknown"
    if props["transparency"] not in {"opaque", "transparent", "translucent", "unknown"}:
        props["transparency"] = "unknown"
    if props["surface"] not in {"smooth", "rough", "fuzzy", "porous", "mixed", "unknown"}:
        props["surface"] = "unknown"
    if props["fragility"] not in {"fragile", "durable", "unknown"}:
        props["fragility"] = "unknown"

    return props, material_source


# ----------------- Candidate building -----------------

def build_candidates(
    listing_files: List[Path],
    image_index: Dict[str, str],
    source_images_dir: Path,
    min_known: int,
    require_local_image: bool,
    excluded_product_types: set[str],
    progress_every_records: int = 5000,
) -> Tuple[List[Candidate], Dict[str, int]]:
    stats = defaultdict(int)
    out: List[Candidate] = []
    progress = ProgressPrinter("build_candidates", progress_every_records)

    for rec in iter_listing_records(listing_files):
        stats["records_total"] += 1

        item_id = safe_strip(rec.get("item_id"))
        main_image_id = safe_strip(rec.get("main_image_id"))
        if not item_id or not main_image_id:
            continue
        stats["records_with_ids"] += 1

        image_rel = image_index.get(main_image_id)
        if not image_rel:
            continue
        stats["records_with_image_rel"] += 1

        if require_local_image and not (source_images_dir / image_rel).exists():
            continue
        stats["records_with_local_image"] += 1

        props, _source = infer_properties(rec)
        known_count = sum(1 for k in PROPERTY_ORDER if props[k] != "unknown")
        if known_count < min_known:
            continue
        stats["records_with_min_known"] += 1

        product_type = normalize_product_type(pick_value(rec.get("product_type")))
        if product_type in excluded_product_types:
            stats["records_excluded_by_product_type"] += 1
            continue
        primary_object = product_type_to_primary_object(product_type)
        title = pick_value(rec.get("item_name"))

        candidate = Candidate(
            item_id=item_id,
            main_image_id=main_image_id,
            image_rel=image_rel,
            product_type=product_type,
            primary_object=primary_object,
            properties=props,
            notes="metadata-derived physical pseudo-label",
            title=title,
            domain_name=safe_strip(rec.get("domain_name")),
            known_count=known_count,
        )
        out.append(candidate)

        progress.emit(
            stats["records_total"],
            extra=(
                f"candidates={len(out)}"
                f", local_images={stats['records_with_local_image']}"
                f", min_known={stats['records_with_min_known']}"
            ),
        )

    progress.emit(
        stats["records_total"],
        extra=(
            f"candidates={len(out)}"
            f", local_images={stats['records_with_local_image']}"
            f", min_known={stats['records_with_min_known']}"
        ),
        force=True,
    )
    return out, dict(stats)


def dedup_candidates(candidates: List[Candidate]) -> List[Candidate]:
    by_key: Dict[str, Candidate] = {}
    for c in candidates:
        # Prefer one sample per visual scene (image) to avoid repeated duplicates.
        key = c.main_image_id or c.item_id
        prev = by_key.get(key)
        if prev is None or c.known_count > prev.known_count:
            by_key[key] = c
    return list(by_key.values())


def select_balanced(
    candidates: List[Candidate],
    max_samples: int,
    max_per_product_type: int,
    seed: int,
) -> List[Candidate]:
    if not candidates:
        return []

    rng = random.Random(seed)

    # Prefer high-confidence records first, then randomize within same confidence.
    grouped: Dict[int, List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped[c.known_count].append(c)

    ordered: List[Candidate] = []
    for known_count in sorted(grouped.keys(), reverse=True):
        bucket = grouped[known_count]
        rng.shuffle(bucket)
        ordered.extend(bucket)

    materials = sorted({c.properties["material"] for c in ordered if c.properties["material"] != "unknown"})
    per_material_quota = max(1, max_samples // max(1, len(materials)))

    selected: List[Candidate] = []
    selected_ids = set()
    material_counts = Counter()
    type_counts = Counter()

    # Pass 1: balanced by material.
    for c in ordered:
        if len(selected) >= max_samples:
            break
        key = c.item_id or c.main_image_id
        if key in selected_ids:
            continue

        if type_counts[c.product_type] >= max_per_product_type:
            continue

        mat = c.properties["material"]
        if mat == "unknown":
            continue
        if material_counts[mat] >= per_material_quota:
            continue

        selected.append(c)
        selected_ids.add(key)
        material_counts[mat] += 1
        type_counts[c.product_type] += 1

    # Pass 2: fill remainder while keeping product type cap.
    for c in ordered:
        if len(selected) >= max_samples:
            break
        key = c.item_id or c.main_image_id
        if key in selected_ids:
            continue
        if type_counts[c.product_type] >= max_per_product_type:
            continue

        selected.append(c)
        selected_ids.add(key)
        material_counts[c.properties["material"]] += 1
        type_counts[c.product_type] += 1

    return selected


# ----------------- Image materialization -----------------

def make_s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 is not installed but --download-missing was requested")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def is_s3_not_found_error(exc: Exception) -> bool:
    if ClientError is not None and isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}
    text = str(exc).lower()
    return "404" in text or "not found" in text or "nosuchkey" in text


def ensure_metadata_file(
    local_path: Path,
    s3_key: str,
    download_missing: bool,
    bucket: str,
    s3_client,
    required: bool = True,
) -> bool:
    if local_path.exists():
        return True
    if not download_missing:
        if required:
            raise FileNotFoundError(f"Missing required metadata file: {local_path}")
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, s3_key, str(local_path))
        return True
    except Exception as exc:
        if not required and is_s3_not_found_error(exc):
            print(f"Warning: metadata shard not found in bucket, skipping: {s3_key}")
            return False
        raise


def materialize_image(
    image_rel: str,
    source_images_dir: Path,
    output_images_dir: Path,
    download_missing: bool,
    bucket: str,
    s3_client,
) -> Tuple[bool, str]:
    src = source_images_dir / image_rel
    dst = output_images_dir / image_rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return True, "exists"

    if src.exists():
        shutil.copy2(src, dst)
        return True, "copied_local"

    if download_missing:
        key = f"images/small/{image_rel}"
        try:
            s3_client.download_file(bucket, key, str(dst))
            return True, "downloaded"
        except Exception:
            return False, "download_failed"

    return False, "missing_local"


def _make_border_mask(h: int, w: int, b: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True
    return mask


def _quantized_distribution(rgb_pixels: np.ndarray, quant_step: int) -> Tuple[float, float]:
    if rgb_pixels.size == 0:
        return 1.0, 0.0

    qstep = max(1, int(quant_step))
    bins = max(1, int(np.ceil(256 / qstep)))
    quantized = np.clip(rgb_pixels.astype(np.int32) // qstep, 0, bins - 1)
    code = quantized[:, 0] * (bins ** 2) + quantized[:, 1] * bins + quantized[:, 2]
    _, counts = np.unique(code, return_counts=True)
    dominant_bin_ratio = float(counts.max() / counts.sum()) if len(counts) else 1.0
    probs = counts.astype(np.float32) / counts.sum() if len(counts) else np.asarray([1.0], dtype=np.float32)
    entropy = float(
        0.0
        if len(probs) <= 1
        else (-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
    )
    return dominant_bin_ratio, entropy


def _estimate_background_mask(
    arr: np.ndarray,
    border_mask: np.ndarray,
    quant_step: int,
    color_tolerance: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w, _ = arr.shape
    border_pixels = arr[border_mask]
    if border_pixels.size == 0:
        return np.zeros((h, w), dtype=bool), {"background_area_ratio": 0.0}

    qstep = max(1, int(quant_step))
    bins = max(1, int(np.ceil(256 / qstep)))
    border_quantized = np.clip(border_pixels.astype(np.int32) // qstep, 0, bins - 1)
    border_code = (
        border_quantized[:, 0] * (bins ** 2)
        + border_quantized[:, 1] * bins
        + border_quantized[:, 2]
    )
    unique_codes, counts = np.unique(border_code, return_counts=True)
    dominant_code = int(unique_codes[int(np.argmax(counts))])
    dominant_border_pixels = border_pixels[border_code == dominant_code]
    dominant_rgb = (
        np.median(dominant_border_pixels, axis=0).astype(np.int16)
        if len(dominant_border_pixels)
        else np.median(border_pixels, axis=0).astype(np.int16)
    )

    diff = np.max(np.abs(arr.astype(np.int16) - dominant_rgb.reshape(1, 1, 3)), axis=2)
    bg_like = diff <= int(color_tolerance)
    seeds = border_mask & bg_like

    if not np.any(seeds):
        arr_quantized = np.clip(arr.astype(np.int32) // qstep, 0, bins - 1)
        arr_code = arr_quantized[:, :, 0] * (bins ** 2) + arr_quantized[:, :, 1] * bins + arr_quantized[:, :, 2]
        bg_like = arr_code == dominant_code
        seeds = border_mask & bg_like

    if not np.any(seeds):
        return np.zeros((h, w), dtype=bool), {"background_area_ratio": 0.0}

    flat_bg_like = bg_like.ravel()
    flat_visited = np.zeros(h * w, dtype=bool)
    queue = deque(np.flatnonzero(seeds))

    while queue:
        idx = queue.popleft()
        if flat_visited[idx] or not flat_bg_like[idx]:
            continue
        flat_visited[idx] = True
        y, x = divmod(int(idx), w)
        if y > 0:
            queue.append(idx - w)
        if y + 1 < h:
            queue.append(idx + w)
        if x > 0:
            queue.append(idx - 1)
        if x + 1 < w:
            queue.append(idx + 1)

    background_mask = flat_visited.reshape(h, w)
    return background_mask, {
        "background_area_ratio": float(background_mask.mean()),
    }


def compute_context_metrics(
    image_path: Path,
    border_frac: float = 0.08,
    white_threshold: int = 245,
    dark_threshold: int = 16,
    neutral_delta: int = 12,
    quant_step: int = 32,
    max_eval_side: int = 256,
    background_color_tolerance: int = 28,
) -> Dict[str, float]:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_eval_side:
        scale = float(max_eval_side) / float(max(img.size))
        new_size = (
            max(8, int(round(img.size[0] * scale))),
            max(8, int(round(img.size[1] * scale))),
        )
        img = img.resize(new_size, Image.BILINEAR)

    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape
    b = max(1, int(min(h, w) * border_frac))
    border_mask = _make_border_mask(h, w, b)

    border = arr[border_mask].reshape(-1, 3)

    border_white_ratio = float(np.mean(np.all(border >= white_threshold, axis=1)))
    border_dark_ratio = float(np.mean(np.all(border <= dark_threshold, axis=1)))
    global_white_ratio = float(np.mean(np.all(arr >= white_threshold, axis=2)))

    gray = (0.299 * border[:, 0] + 0.587 * border[:, 1] + 0.114 * border[:, 2]).astype(np.float32)
    border_std = float(np.std(gray) / 255.0)

    mx = border.max(axis=1).astype(np.float32)
    mn = border.min(axis=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx <= 1e-6, 0.0, (mx - mn) / mx)
    border_saturation = float(np.mean(sat))
    border_neutral_ratio = float(np.mean((mx - mn) <= neutral_delta))
    dominant_bin_ratio, border_color_entropy = _quantized_distribution(border, quant_step)

    background_mask, background_meta = _estimate_background_mask(
        arr=arr,
        border_mask=border_mask,
        quant_step=quant_step,
        color_tolerance=background_color_tolerance,
    )
    background_pixels = arr[background_mask].reshape(-1, 3)
    if background_pixels.size == 0:
        background_white_ratio = border_white_ratio
        background_dark_ratio = border_dark_ratio
        background_neutral_ratio = border_neutral_ratio
        background_dominant_bin_ratio = dominant_bin_ratio
        background_color_entropy = border_color_entropy
        background_std = border_std
        background_saturation = border_saturation
        background_area_ratio = float(background_meta.get("background_area_ratio", 0.0))
    else:
        background_white_ratio = float(np.mean(np.all(background_pixels >= white_threshold, axis=1)))
        background_dark_ratio = float(np.mean(np.all(background_pixels <= dark_threshold, axis=1)))
        bg_mx = background_pixels.max(axis=1).astype(np.float32)
        bg_mn = background_pixels.min(axis=1).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            bg_sat = np.where(bg_mx <= 1e-6, 0.0, (bg_mx - bg_mn) / bg_mx)
        background_saturation = float(np.mean(bg_sat))
        bg_gray = (
            0.299 * background_pixels[:, 0]
            + 0.587 * background_pixels[:, 1]
            + 0.114 * background_pixels[:, 2]
        ).astype(np.float32)
        background_std = float(np.std(bg_gray) / 255.0)
        background_neutral_ratio = float(np.mean((bg_mx - bg_mn) <= neutral_delta))
        background_dominant_bin_ratio, background_color_entropy = _quantized_distribution(background_pixels, quant_step)
        background_area_ratio = float(background_meta.get("background_area_ratio", 0.0))

    # Higher score => richer natural context at image borders and less studio-like uniform backdrop.
    context_score = float(
        np.clip(
            0.25 * (1.0 - background_dominant_bin_ratio)
            + 0.20 * background_color_entropy
            + 0.15 * background_std
            + 0.10 * background_saturation
            + 0.10 * (1.0 - background_white_ratio)
            + 0.10 * (1.0 - background_dark_ratio)
            + 0.10 * (1.0 - dominant_bin_ratio),
            0.0,
            1.0,
        )
    )

    return {
        "border_white_ratio": border_white_ratio,
        "border_dark_ratio": border_dark_ratio,
        "global_white_ratio": global_white_ratio,
        "border_std": border_std,
        "border_saturation": border_saturation,
        "border_neutral_ratio": border_neutral_ratio,
        "border_dominant_bin_ratio": dominant_bin_ratio,
        "border_color_entropy": border_color_entropy,
        "background_area_ratio": background_area_ratio,
        "background_white_ratio": background_white_ratio,
        "background_dark_ratio": background_dark_ratio,
        "background_std": background_std,
        "background_saturation": background_saturation,
        "background_neutral_ratio": background_neutral_ratio,
        "background_dominant_bin_ratio": background_dominant_bin_ratio,
        "background_color_entropy": background_color_entropy,
        "context_score": context_score,
    }


def simple_mask_from_image(image_path: Path, mask_path: Path, white_threshold: int = 245) -> Tuple[bool, str, float]:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False, "mask_image_open_failed", 0.0

    arr = np.asarray(img)
    # Foreground heuristic: not almost-white.
    fg = ~np.all(arr >= white_threshold, axis=2)

    # If too little detected, use a softer threshold fallback.
    if fg.mean() < 0.005:
        fg = ~np.all(arr >= 235, axis=2)

    mask = Image.fromarray((fg.astype(np.uint8) * 255), mode="L")
    # Light morphological smoothing.
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.MinFilter(5))

    ratio = float((np.asarray(mask) > 0).mean())
    mask.save(mask_path)
    return True, "generated_simple", ratio


def rembg_mask_from_image(image_path: Path, mask_path: Path, session=None) -> Tuple[bool, str, float]:
    try:
        from rembg import remove
    except Exception:
        return False, "rembg_not_installed", 0.0

    try:
        img_bytes = image_path.read_bytes()
        out_bytes = remove(img_bytes, session=session, only_mask=True)
        mask = Image.open(io.BytesIO(out_bytes)).convert("L")
        mask = mask.point(lambda p: 255 if p >= 127 else 0)
        ratio = float((np.asarray(mask) > 0).mean())
        mask.save(mask_path)
        return True, "generated_rembg", ratio
    except Exception:
        return False, "rembg_failed", 0.0


def make_rembg_session():
    try:
        from rembg import new_session
    except Exception:
        return None
    try:
        return new_session("u2net")
    except Exception:
        return None


def materialize_mask(
    image_rel: str,
    output_images_dir: Path,
    output_masks_dir: Path,
    backend: str,
    overwrite: bool,
    rembg_session,
) -> Tuple[bool, str, Optional[str], float]:
    image_path = output_images_dir / image_rel
    rel_mask = str(Path(image_rel).with_suffix(".png"))
    mask_path = output_masks_dir / rel_mask
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if mask_path.exists() and not overwrite:
        try:
            ratio = float((np.asarray(Image.open(mask_path).convert("L")) > 0).mean())
        except Exception:
            ratio = 0.0
        return True, "mask_exists", rel_mask, ratio

    if backend == "simple":
        ok, status, ratio = simple_mask_from_image(image_path, mask_path)
        return ok, status, rel_mask if ok else None, ratio

    if backend == "rembg":
        ok, status, ratio = rembg_mask_from_image(image_path, mask_path, session=rembg_session)
        if ok:
            return True, status, rel_mask, ratio
        # Fallback to simple segmentation to avoid empty dataset when rembg is unavailable.
        ok2, status2, ratio2 = simple_mask_from_image(image_path, mask_path)
        if ok2:
            return True, f"fallback_simple_after_{status}", rel_mask, ratio2
        return False, f"{status}|{status2}", None, 0.0

    return False, f"unknown_mask_backend:{backend}", None, 0.0


def prune_unused_files(output_dir: Path, used_files: set[Path]) -> int:
    removed = 0
    for fp in output_dir.rglob("*"):
        if fp.is_file() and fp not in used_files:
            fp.unlink()
            removed += 1
    # Cleanup empty directories (deepest first).
    for d in sorted([d for d in output_dir.rglob("*") if d.is_dir()], key=lambda x: len(x.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    return removed


# ----------------- Main -----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ABO physics subset")
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--output-name", type=str, default="abo_physics_val")
    p.add_argument("--cache-dir", type=Path, default=Path("dataset/abo_vlm_val/_cache"))
    p.add_argument("--source-images-dir", type=Path, default=Path("dataset/abo_vlm_val/images"))
    p.add_argument("--listing-shards", type=str, default="0,1")
    p.add_argument("--max-samples", type=int, default=240)
    p.add_argument("--selection-pool-multiplier", type=int, default=5)
    p.add_argument("--min-known-properties", type=int, default=4)
    p.add_argument("--max-per-product-type", type=int, default=25)
    p.add_argument(
        "--exclude-product-types",
        type=str,
        default="CELLULAR_PHONE_CASE,PORTABLE_ELECTRONIC_DEVICE_COVER",
        help="Comma-separated product types to skip (normalized to UPPER_SNAKE_CASE)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bucket-name", type=str, default="amazon-berkeley-objects")
    p.add_argument("--download-missing", action="store_true")
    p.add_argument("--generate-masks", action="store_true")
    p.add_argument("--mask-backend", type=str, default="simple", choices=["simple", "rembg"])
    p.add_argument("--overwrite-masks", action="store_true")
    # Keep meaningful foreground while preserving visible background context.
    p.add_argument("--min-mask-area-ratio", type=float, default=0.01)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    p.add_argument("--disable-context-filter", action="store_true")
    p.add_argument("--min-context-score", type=float, default=0.22)
    p.add_argument("--max-border-white-ratio", type=float, default=0.98)
    p.add_argument("--max-border-dark-ratio", type=float, default=0.98)
    p.add_argument("--max-border-dominant-bin-ratio", type=float, default=0.75)
    p.add_argument("--min-background-area-ratio", type=float, default=0.03)
    p.add_argument("--max-background-white-ratio", type=float, default=0.97)
    p.add_argument("--max-background-dark-ratio", type=float, default=0.97)
    p.add_argument("--max-background-dominant-bin-ratio", type=float, default=0.80)
    p.add_argument("--clear-output", action="store_true")
    p.add_argument("--progress-every-records", type=int, default=5000)
    p.add_argument("--progress-every-selected", type=int, default=25)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print("Phase 1/5: preparing metadata and configuration...", flush=True)
    s3_client = make_s3_client() if args.download_missing else None
    excluded_product_types = parse_str_set(args.exclude_product_types)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    shards = parse_int_list(args.listing_shards)
    images_csv_gz = args.cache_dir / "images.csv.gz"
    ensure_metadata_file(
        local_path=images_csv_gz,
        s3_key="images/metadata/images.csv.gz",
        download_missing=args.download_missing,
        bucket=args.bucket_name,
        s3_client=s3_client,
        required=True,
    )

    listing_files = []
    for shard in shards:
        fp = args.cache_dir / f"listings_{shard}.json.gz"
        ok = ensure_metadata_file(
            local_path=fp,
            s3_key=f"listings/metadata/listings_{shard}.json.gz",
            download_missing=args.download_missing,
            bucket=args.bucket_name,
            s3_client=s3_client,
            required=False,
        )
        if ok and fp.exists():
            listing_files.append(fp)

    if not listing_files:
        raise FileNotFoundError(
            "No listing metadata shards available. "
            "Check --listing-shards and dataset availability in S3."
        )

    dataset_root = args.dataset_root
    output_dir = dataset_root / args.output_name
    output_images_dir = output_dir / "images"
    output_masks_dir = output_dir / "masks"
    meta_path = output_dir / "meta.json"
    summary_path = output_dir / "summary.json"

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    if args.generate_masks:
        output_masks_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 2/5: loading ABO image index...", flush=True)
    image_index = load_image_index(images_csv_gz)
    print(f"Loaded image index: {len(image_index)} entries", flush=True)
    print(f"Phase 3/5: building candidate pool from {len(listing_files)} listing shard(s)...", flush=True)
    candidates, stats = build_candidates(
        listing_files=listing_files,
        image_index=image_index,
        source_images_dir=args.source_images_dir,
        min_known=args.min_known_properties,
        require_local_image=not args.download_missing,
        excluded_product_types=excluded_product_types,
        progress_every_records=args.progress_every_records,
    )

    print("Phase 4/5: deduplicating and selecting balanced candidate pool...", flush=True)
    deduped = dedup_candidates(candidates)
    pool_multiplier = max(1, int(args.selection_pool_multiplier))
    pool_max_samples = max(args.max_samples, args.max_samples * pool_multiplier)
    pool_max_per_product_type = max(args.max_per_product_type, args.max_per_product_type * pool_multiplier)
    selected = select_balanced(
        candidates=deduped,
        max_samples=pool_max_samples,
        max_per_product_type=pool_max_per_product_type,
        seed=args.seed,
    )

    materialized = []
    image_stats = Counter()
    mask_stats = Counter()
    context_stats = Counter()
    context_scores: List[float] = []
    rembg_session = make_rembg_session() if args.generate_masks and args.mask_backend == "rembg" else None
    selected_progress = ProgressPrinter("materialize_subset", args.progress_every_selected)

    print(
        f"Phase 5/5: materializing images and applying context filter over {len(selected)} candidate(s)...",
        flush=True,
    )

    for idx, c in enumerate(selected, start=1):
        if len(materialized) >= args.max_samples:
            break

        ok, status = materialize_image(
            image_rel=c.image_rel,
            source_images_dir=args.source_images_dir,
            output_images_dir=output_images_dir,
            download_missing=args.download_missing,
            bucket=args.bucket_name,
            s3_client=s3_client,
        )
        image_stats[status] += 1
        if not ok:
            selected_progress.emit(
                idx,
                extra=(
                    f"kept={len(materialized)}/{args.max_samples}"
                    f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                    f", image_failures={sum(v for k, v in image_stats.items() if k not in {'exists', 'copied_local', 'downloaded'})}"
                    f", mask_failures={sum(v for k, v in mask_stats.items() if 'failed' in k or 'out_of_range' in k)}"
                ),
            )
            continue

        image_path = output_images_dir / c.image_rel
        try:
            context = compute_context_metrics(image_path=image_path)
        except Exception:
            context_stats["context_metrics_failed"] += 1
            selected_progress.emit(
                idx,
                extra=(
                    f"kept={len(materialized)}/{args.max_samples}"
                    f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                    f", context_failed={context_stats.get('context_metrics_failed', 0)}"
                ),
            )
            continue

        context_scores.append(context["context_score"])
        background_filter_ok = True
        if context["background_area_ratio"] >= args.min_background_area_ratio:
            background_filter_ok = (
                context["background_white_ratio"] <= args.max_background_white_ratio
                and context["background_dark_ratio"] <= args.max_background_dark_ratio
                and context["background_dominant_bin_ratio"] <= args.max_background_dominant_bin_ratio
            )
        context_ok = (
            context["context_score"] >= args.min_context_score
            and context["border_white_ratio"] <= args.max_border_white_ratio
            and context["border_dark_ratio"] <= args.max_border_dark_ratio
            and context["border_dominant_bin_ratio"] <= args.max_border_dominant_bin_ratio
            and background_filter_ok
        )
        if not args.disable_context_filter and not context_ok:
            context_stats["filtered_by_context"] += 1
            selected_progress.emit(
                idx,
                extra=(
                    f"kept={len(materialized)}/{args.max_samples}"
                    f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                    f", image_failures={sum(v for k, v in image_stats.items() if k not in {'exists', 'copied_local', 'downloaded'})}"
                    f", mask_failures={sum(v for k, v in mask_stats.items() if 'failed' in k or 'out_of_range' in k)}"
                ),
            )
            continue
        context_stats["kept_by_context"] += 1

        mask_rel = None
        if args.generate_masks:
            ok_mask, mask_status, mask_rel, mask_ratio = materialize_mask(
                image_rel=c.image_rel,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                backend=args.mask_backend,
                overwrite=args.overwrite_masks,
                rembg_session=rembg_session,
            )
            mask_stats[mask_status] += 1
            if not ok_mask:
                selected_progress.emit(
                    idx,
                    extra=(
                        f"kept={len(materialized)}/{args.max_samples}"
                        f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                        f", image_failures={sum(v for k, v in image_stats.items() if k not in {'exists', 'copied_local', 'downloaded'})}"
                        f", mask_failures={sum(v for k, v in mask_stats.items() if 'failed' in k or 'out_of_range' in k)}"
                    ),
                )
                continue
            if not (args.min_mask_area_ratio <= mask_ratio <= args.max_mask_area_ratio):
                mask_stats["mask_area_out_of_range"] += 1
                selected_progress.emit(
                    idx,
                    extra=(
                        f"kept={len(materialized)}/{args.max_samples}"
                        f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                        f", image_failures={sum(v for k, v in image_stats.items() if k not in {'exists', 'copied_local', 'downloaded'})}"
                        f", mask_failures={sum(v for k, v in mask_stats.items() if 'failed' in k or 'out_of_range' in k)}"
                    ),
                )
                continue

        sample = {
            "image_id": c.main_image_id,
            "path": f"{args.output_name}/images/{c.image_rel}",
            "primary_object": c.primary_object,
            "properties": c.properties,
            "notes": c.notes,
            "abo_meta": {
                "item_id": c.item_id,
                "product_type": c.product_type,
                "domain_name": c.domain_name,
                "title": c.title,
            },
            "background_metrics": context,
        }
        if mask_rel is not None:
            sample["mask_path"] = f"{args.output_name}/masks/{mask_rel}"
            sample["mask_source"] = args.mask_backend
        materialized.append(sample)

        selected_progress.emit(
            idx,
            extra=(
                f"kept={len(materialized)}/{args.max_samples}"
                f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                f", image_failures={sum(v for k, v in image_stats.items() if k not in {'exists', 'copied_local', 'downloaded'})}"
                f", mask_failures={sum(v for k, v in mask_stats.items() if 'failed' in k or 'out_of_range' in k)}"
            ),
        )

    # Stable order for reproducibility.
    materialized.sort(key=lambda x: x["image_id"])

    # Keep output directories strictly aligned with meta.json.
    image_prefix = Path(args.output_name) / "images"
    mask_prefix = Path(args.output_name) / "masks"

    used_images: set[Path] = set()
    used_masks: set[Path] = set()
    for s in materialized:
        img_rel = Path(s["path"]).relative_to(image_prefix)
        used_images.add(output_images_dir / img_rel)
        if "mask_path" in s:
            mask_rel = Path(s["mask_path"]).relative_to(mask_prefix)
            used_masks.add(output_masks_dir / mask_rel)

    pruned_images = prune_unused_files(output_images_dir, used_images)
    pruned_masks = prune_unused_files(output_masks_dir, used_masks) if args.generate_masks else 0

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(materialized, fh, ensure_ascii=False, indent=2)

    prop_value_counts = {prop: Counter() for prop in PROPERTY_ORDER}
    for s in materialized:
        props = s["properties"]
        for prop in PROPERTY_ORDER:
            prop_value_counts[prop][props[prop]] += 1

    summary = {
        "output_dir": str(output_dir),
        "listing_shards": shards,
        "max_samples": args.max_samples,
        "selection_pool_multiplier": pool_multiplier,
        "min_known_properties": args.min_known_properties,
        "max_per_product_type": args.max_per_product_type,
        "exclude_product_types": sorted(excluded_product_types),
        "seed": args.seed,
        "records_stats": stats,
        "num_candidates": len(candidates),
        "num_candidates_dedup": len(deduped),
        "num_selected_before_images": len(selected),
        "num_written": len(materialized),
        "image_stats": dict(image_stats),
        "context_filter": {
            "enabled": not args.disable_context_filter,
            "min_context_score": args.min_context_score,
            "max_border_white_ratio": args.max_border_white_ratio,
            "max_border_dark_ratio": args.max_border_dark_ratio,
            "max_border_dominant_bin_ratio": args.max_border_dominant_bin_ratio,
            "min_background_area_ratio": args.min_background_area_ratio,
            "max_background_white_ratio": args.max_background_white_ratio,
            "max_background_dark_ratio": args.max_background_dark_ratio,
            "max_background_dominant_bin_ratio": args.max_background_dominant_bin_ratio,
        },
        "context_stats": dict(context_stats),
        "context_score_avg": float(np.mean(context_scores)) if context_scores else None,
        "context_score_min": float(np.min(context_scores)) if context_scores else None,
        "context_score_max": float(np.max(context_scores)) if context_scores else None,
        "mask_config": {
            "generate_masks": args.generate_masks,
            "mask_backend": args.mask_backend if args.generate_masks else None,
            "min_mask_area_ratio": args.min_mask_area_ratio if args.generate_masks else None,
            "max_mask_area_ratio": args.max_mask_area_ratio if args.generate_masks else None,
        },
        "mask_stats": dict(mask_stats),
        "pruned_files": {
            "images_removed": pruned_images,
            "masks_removed": pruned_masks,
        },
        "property_distributions": {k: dict(v) for k, v in prop_value_counts.items()},
        "top_product_types": dict(
            Counter(s["abo_meta"]["product_type"] for s in materialized).most_common(20)
        ),
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if len(selected) > 0:
        selected_progress.emit(
            len(selected),
            extra=(
                f"kept={len(materialized)}/{args.max_samples}"
                f", context_filtered={context_stats.get('filtered_by_context', 0)}"
                f", image_stats={dict(image_stats)}"
                f", mask_stats={dict(mask_stats)}"
            ),
            force=True,
        )

    if len(materialized) == 0:
        raise RuntimeError(
            "Dataset is empty after filtering. "
            f"records_total={stats.get('records_total', 0)}, "
            f"candidates={len(candidates)}, selected={len(selected)}, "
            f"image_stats={dict(image_stats)}, mask_stats={dict(mask_stats)}. "
            "Try: --download-missing, lower --min-known-properties, "
            "or relax mask thresholds (--min-mask-area-ratio/--max-mask-area-ratio)."
        )

    print(f"Wrote: {meta_path}")
    print(f"Wrote: {summary_path}")
    print(f"Samples: {len(materialized)}")


if __name__ == "__main__":
    main()
