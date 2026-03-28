from __future__ import annotations

import argparse
import json
import math
import os
from inspect import signature
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

try:
    import comet_ml  # noqa: F401
except Exception:  # pragma: no cover
    comet_ml = None

import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_pipeline import MODEL_REGISTRY
from vlm_pipeline.comet import get_secret_value, init_comet_experiment
from vlm_pipeline.runtime import make_bnb_config

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    Trainer = None
    TrainerCallback = None
    TrainingArguments = None

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for ABO150 per-property multimodal SFT dataset.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--model-key", type=str, default="qwen3_vl_8b", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-comet", action="store_true")
    parser.add_argument("--comet-project-name", type=str, default="vlm-physics-training")
    parser.add_argument("--comet-run-name", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-namespace", type=str, default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-token", type=str, default=None)
    parser.add_argument("--hub-commit-message", type=str, default="Upload QLoRA adapter")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


class ABO150SFTDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def _fallback_chat_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                lines.append(f"{role}: {text}")
            continue
        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text = str(chunk.get("text") or "").strip()
                    if text:
                        parts.append(text)
            if parts:
                lines.append(f"{role}: " + "\n".join(parts))
    return "\n\n".join(lines).strip()


def _apply_chat_template(processor: Any, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            text = processor.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
    return _fallback_chat_text(messages)


def _build_messages(prompt: str, response: str) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    user_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    full_messages = user_messages + [{"role": "assistant", "content": response}]
    return user_messages, full_messages


class VLMTrainCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        prompt_texts: List[str] = []
        full_texts: List[str] = []

        for item in features:
            image = Image.open(Path(item["image_path"])).convert("RGB")
            user_messages, full_messages = _build_messages(item["prompt"], item["response"])
            prompt_texts.append(_apply_chat_template(self.processor, user_messages, add_generation_prompt=True))
            full_texts.append(_apply_chat_template(self.processor, full_messages, add_generation_prompt=False))
            images.append(image)

        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lens):
            labels[idx, : int(prompt_len)] = -100
        full_batch["labels"] = labels
        return full_batch


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_metric_value(key: str, value: float) -> str:
    if key in {"learning_rate"}:
        return f"{value:.2e}"
    if key in {"epoch"}:
        return f"{value:.2f}"
    return f"{value:.4f}"


class RealtimeMetricsCallback(TrainerCallback):
    def __init__(self, comet_experiment: Any | None = None) -> None:
        self.comet_experiment = comet_experiment

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        logs = dict(logs or {})
        numeric_logs = {k: float(v) for k, v in logs.items() if _is_numeric_metric(v)}
        if not numeric_logs:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        epoch = numeric_logs.get("epoch")

        preferred_order = ["loss", "eval_loss", "grad_norm", "learning_rate", "epoch"]
        rendered: List[str] = []
        for key in preferred_order:
            if key in numeric_logs:
                rendered.append(f"{key}={_format_metric_value(key, numeric_logs[key])}")
        for key in sorted(numeric_logs.keys()):
            if key not in preferred_order:
                rendered.append(f"{key}={_format_metric_value(key, numeric_logs[key])}")
        print(f"[train step {step}] " + " | ".join(rendered))

        if self.comet_experiment is not None:
            self.comet_experiment.log_metrics(
                numeric_logs,
                step=step,
                epoch=epoch,
            )


def guess_lora_target_modules(model: torch.nn.Module) -> List[str]:
    candidate_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "out_proj",
    ]
    found = set()

    def _is_supported_linear_like(module: torch.nn.Module) -> bool:
        if isinstance(module, torch.nn.Linear):
            return True
        cls_name = module.__class__.__name__.lower()
        if cls_name in {"linear4bit", "linear8bitlt"}:
            return True
        return hasattr(module, "weight") and callable(getattr(module, "forward", None))

    for module_name, module in model.named_modules():
        if not _is_supported_linear_like(module):
            continue
        leaf = module_name.split(".")[-1]
        if leaf in candidate_suffixes:
            found.add(leaf)
    if found:
        return sorted(found)
    raise RuntimeError("Could not infer LoRA target modules from the model.")


def load_base_model_and_processor(model_key: str, use_4bit: bool) -> tuple[Any, Any, str]:
    if AutoProcessor is None or AutoModelForImageTextToText is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is not installed. Install training dependencies first.")
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "right"
        except Exception:
            pass
        try:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    bnb = make_bnb_config(use_4bit)
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        print(f"AutoModelForImageTextToText failed: {exc}")
        print("Falling back to AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, processor, model_id


def ensure_input_require_grads(model: torch.nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            return
        except Exception:
            pass

    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        try:
            input_embeddings = model.get_input_embeddings()
        except Exception:
            input_embeddings = None
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(_module, _inputs, output):
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    return (
        get_secret_value("HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def resolve_hf_namespace(explicit_namespace: str | None) -> str | None:
    if explicit_namespace:
        return explicit_namespace.strip() or None
    return (
        get_secret_value("HF_USERNAME")
        or os.getenv("HF_USERNAME")
        or os.getenv("HUGGINGFACE_USERNAME")
    )


def resolve_hub_model_id(
    explicit_model_id: str | None,
    output_dir: Path,
    token: str | None,
    explicit_namespace: str | None,
) -> str:
    if explicit_model_id:
        return explicit_model_id.strip()

    namespace = resolve_hf_namespace(explicit_namespace)
    if namespace:
        return f"{namespace.rstrip('/')}/{output_dir.name}"

    if HfApi is not None and token:
        try:
            whoami = HfApi(token=token).whoami()
            inferred_namespace = str(whoami.get("name") or "").strip()
            if inferred_namespace:
                return f"{inferred_namespace}/{output_dir.name}"
        except Exception:
            pass

    raise ValueError(
        "Could not resolve hub model id. Pass --hub-model-id explicitly or set "
        "HF_USERNAME / HUGGINGFACE_USERNAME (or Colab secret HF_USERNAME)."
    )


def push_adapter_to_hub(
    output_dir: Path,
    hub_model_id: str,
    token: str,
    private: bool,
    commit_message: str,
) -> None:
    if HfApi is None:
        raise ImportError("huggingface_hub is required for push_to_hub.")
    api = HfApi(token=token)
    api.create_repo(repo_id=hub_model_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=hub_model_id,
        repo_type="model",
        folder_path=str(output_dir),
        path_in_repo="",
        commit_message=commit_message,
        ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
    )


def estimate_training_schedule(
    train_rows: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
) -> Dict[str, int]:
    micro_batch = max(1, int(per_device_train_batch_size))
    grad_accum = max(1, int(gradient_accumulation_steps))
    steps_per_epoch = max(1, math.ceil(train_rows / (micro_batch * grad_accum)))
    if int(max_steps) > 0:
        total_steps = int(max_steps)
        estimated_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    else:
        estimated_epochs = max(1, math.ceil(float(num_train_epochs)))
        total_steps = max(1, math.ceil(steps_per_epoch * float(num_train_epochs)))
    return {
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "estimated_epochs": int(estimated_epochs),
    }


def build_training_arguments_kwargs(
    output_dir: Path,
    args: argparse.Namespace,
    has_eval_dataset: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "num_train_epochs": float(args.num_train_epochs),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "logging_steps": int(args.logging_steps),
        "logging_first_step": True,
        "save_steps": int(args.save_steps),
        "eval_steps": int(args.eval_steps),
        "max_steps": int(args.max_steps),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
        "report_to": args.report_to,
        "load_best_model_at_end": False,
        "dataloader_num_workers": 0,
        "seed": int(args.seed),
        "disable_tqdm": bool(args.disable_tqdm),
        "run_name": output_dir.name,
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }

    eval_strategy_value = "steps" if has_eval_dataset else "no"
    supported = set(signature(TrainingArguments.__init__).parameters)
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = eval_strategy_value

    return {key: value for key, value in kwargs.items() if key in supported}


def main() -> None:
    args = parse_args()
    if LoraConfig is None or get_peft_model is None or prepare_model_for_kbit_training is None:
        raise ImportError("peft is required for QLoRA training.")
    if Trainer is None or TrainingArguments is None or TrainerCallback is None:
        raise ImportError("transformers Trainer is not available. Install notebook dependencies first.")

    train_rows = read_jsonl(args.train_jsonl)
    eval_rows = read_jsonl(args.val_jsonl) if args.val_jsonl is not None else None

    model, processor, base_model_id = load_base_model_and_processor(
        model_key=args.model_key,
        use_4bit=bool(args.use_4bit),
    )
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    ensure_input_require_grads(model)

    target_modules = guess_lora_target_modules(model)
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ABO150SFTDataset(train_rows)
    eval_dataset = ABO150SFTDataset(eval_rows) if eval_rows is not None else None
    collator = VLMTrainCollator(processor)
    schedule = estimate_training_schedule(
        train_rows=len(train_rows),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
    )
    print("Training schedule:")
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows) if eval_rows is not None else 0,
                "per_device_train_batch_size": int(args.per_device_train_batch_size),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "num_train_epochs": float(args.num_train_epochs),
                "max_steps": int(args.max_steps),
                **schedule,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    comet_run_name = args.comet_run_name or output_dir.name
    comet_run_params = {
        "model_key": args.model_key,
        "base_model_id": base_model_id,
        "train_jsonl": str(args.train_jsonl.resolve()),
        "val_jsonl": str(args.val_jsonl.resolve()) if args.val_jsonl is not None else None,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows) if eval_rows is not None else 0,
        "use_4bit": bool(args.use_4bit),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": float(args.num_train_epochs),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "eval_steps": int(args.eval_steps),
        "max_steps": int(args.max_steps),
        "steps_per_epoch": schedule["steps_per_epoch"],
        "total_steps": schedule["total_steps"],
        "estimated_epochs": schedule["estimated_epochs"],
        "push_to_hub": bool(args.push_to_hub),
        "hub_model_id": args.hub_model_id,
        "report_to": args.report_to,
    }
    comet_experiment = init_comet_experiment(
        run_tag=comet_run_name,
        run_params=comet_run_params,
        enabled=not bool(args.disable_comet),
        default_project=args.comet_project_name,
    )
    if comet_experiment is not None:
        print(f"Comet run name: {comet_run_name}")
        try:
            run_url = getattr(comet_experiment, "url", None)
            if run_url:
                print(f"Comet URL: {run_url}")
        except Exception:
            pass

    training_args = TrainingArguments(
        **build_training_arguments_kwargs(
            output_dir=output_dir,
            args=args,
            has_eval_dataset=eval_dataset is not None,
        )
    )

    callbacks: List[Any] = [RealtimeMetricsCallback(comet_experiment=comet_experiment)]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    try:
        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        manifest = {
            "model_key": args.model_key,
            "base_model_id": base_model_id,
            "train_jsonl": str(args.train_jsonl.resolve()),
            "val_jsonl": str(args.val_jsonl.resolve()) if args.val_jsonl is not None else None,
            "output_dir": str(output_dir),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows) if eval_rows is not None else 0,
            "use_4bit": bool(args.use_4bit),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": target_modules,
            "num_train_epochs": float(args.num_train_epochs),
            "learning_rate": float(args.learning_rate),
            "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
            "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
            "push_to_hub": bool(args.push_to_hub),
            "hub_model_id": None,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_model_config = {
            "backend": "hf_chat",
            "model_id": f"{base_model_id}+qlora",
            "base_model_id": base_model_id,
            "adapter_path": str(output_dir),
            "use_4bit": bool(args.use_4bit),
            "max_new_tokens": int(MODEL_REGISTRY[args.model_key].get("max_new_tokens", 128)),
            "do_sample": bool(MODEL_REGISTRY[args.model_key].get("do_sample", False)),
        }
        local_runtime_config_path = output_dir / "runtime_model_config.json"
        local_runtime_config_path.write_text(
            json.dumps(runtime_model_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        hub_runtime_config_path = None
        if args.push_to_hub:
            token = resolve_hf_token(args.hub_token)
            if not token:
                raise ValueError("HF_TOKEN is missing. Pass --hub-token or set HF_TOKEN / Colab secret HF_TOKEN.")
            resolved_hub_model_id = resolve_hub_model_id(
                explicit_model_id=args.hub_model_id,
                output_dir=output_dir,
                token=token,
                explicit_namespace=args.hub_namespace,
            )
            push_adapter_to_hub(
                output_dir=output_dir,
                hub_model_id=resolved_hub_model_id,
                token=token,
                private=bool(args.hub_private),
                commit_message=args.hub_commit_message,
            )
            manifest["hub_model_id"] = resolved_hub_model_id
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            hub_runtime_config = dict(runtime_model_config)
            hub_runtime_config["adapter_path"] = resolved_hub_model_id
            hub_runtime_config["model_id"] = f"{base_model_id}+qlora@{resolved_hub_model_id}"
            hub_runtime_config_path = output_dir / "runtime_model_config.hub.json"
            hub_runtime_config_path.write_text(
                json.dumps(hub_runtime_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if comet_experiment is not None:
            comet_experiment.log_metrics(
                {
                    "final_train_loss": float(train_result.metrics.get("train_loss", 0.0)),
                    "final_train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
                },
                step=int(getattr(trainer.state, "global_step", 0) or 0),
                epoch=getattr(trainer.state, "epoch", None),
            )
            comet_experiment.log_parameters(
                {
                    "resolved_hub_model_id": manifest.get("hub_model_id"),
                    "output_dir": str(output_dir),
                }
            )

        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("Local runtime config saved to:", local_runtime_config_path)
        if hub_runtime_config_path is not None:
            print("Adapter pushed to Hugging Face Hub:", manifest["hub_model_id"])
            print("Hub runtime config saved to:", hub_runtime_config_path)
    finally:
        if comet_experiment is not None:
            try:
                comet_experiment.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
