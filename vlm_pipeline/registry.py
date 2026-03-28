from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parents[1]


MODEL_REGISTRY = {
    "qwen2_vl_2b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "qwen2_5_vl_3b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "qwen3_vl_8b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "qwen2_5_vl_7b": {
        "backend": "hf_chat",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "llava_onevision_1_5_8b": {
        "backend": "hf_chat",
        "model_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "use_4bit": True,
        "max_new_tokens": 128,
        "do_sample": False,
        "max_prompt_batch_size": 32,
    },
}


@dataclass
class DatasetContext:
    dataset_name: str
    dataset_dir: Path
    dataset_type: str  # abo150_annotations | meta_json
    annotations_path: Optional[Path] = None
    schema_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    reports_dir: Optional[Path] = None


def get_dataset_context(dataset_name: str, dataset_root: Optional[Path] = None) -> DatasetContext:
    dataset_root = (dataset_root or (ROOT_DIR / "dataset")).resolve()
    dataset_dir = (dataset_root / dataset_name).resolve()

    if dataset_name == "abo_150_expanded":
        return DatasetContext(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            dataset_type="abo150_annotations",
            annotations_path=dataset_dir / "selected_150_annotations.jsonl",
            schema_path=dataset_dir / "physics_properties.yaml",
            reports_dir=ROOT_DIR / "reports_abo150_expanded",
        )

    if dataset_name == "abo_physics_natural_bg_v2":
        return DatasetContext(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            dataset_type="meta_json",
            meta_path=dataset_dir / "meta.json",
            reports_dir=ROOT_DIR / "reports_natural_bg_v2",
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
