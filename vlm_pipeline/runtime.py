from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
    BitsAndBytesConfig,
    )
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


@dataclass
class VLMRuntime:
    name: str
    backend: str
    model_id: str
    processor: Any
    model: Any
    gen_kwargs: Dict[str, Any]
    prompt_batch_cap: Optional[int] = None
    stats: Dict[str, int] = field(default_factory=dict)


def _bump_runtime_stat(runtime: VLMRuntime, key: str, amount: int = 1) -> None:
    runtime.stats[key] = int(runtime.stats.get(key, 0)) + int(amount)


def make_bnb_config(use_4bit: bool):
    if BitsAndBytesConfig is None or not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_hf_chat_runtime(name: str, cfg: Dict[str, Any]) -> VLMRuntime:
    if AutoProcessor is None or AutoModelForImageTextToText is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is not installed. Install notebook dependencies first.")

    model_id = cfg["model_id"]
    base_model_id = cfg.get("base_model_id", model_id)
    adapter_path = cfg.get("adapter_path")
    if adapter_path is not None:
        adapter_candidate = Path(str(adapter_path)).expanduser()
        if adapter_candidate.exists():
            adapter_path = str(adapter_candidate.resolve())
        else:
            adapter_path = str(adapter_path).strip()
        if not adapter_path:
            raise ValueError("adapter_path is empty")
        if PeftModel is None:
            raise ImportError("peft is required to load LoRA/QLoRA adapters.")
    use_4bit = bool(cfg.get("use_4bit", True))
    gen_kwargs = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "do_sample": bool(cfg.get("do_sample", False)),
    }

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "left"
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
        model = AutoModelForImageTextToText.from_pretrained(base_model_id, **model_kwargs)
    except Exception as exc:
        print(f"AutoModelForImageTextToText failed: {exc}")
        print("Falling back to AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        try:
            generation_config.do_sample = gen_kwargs["do_sample"]
        except Exception:
            pass
        if not gen_kwargs["do_sample"]:
            for attr in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                if hasattr(generation_config, attr):
                    try:
                        setattr(generation_config, attr, None)
                    except Exception:
                        pass

    model.eval()
    return VLMRuntime(
        name=name,
        backend="hf_chat",
        model_id=model_id,
        processor=processor,
        model=model,
        gen_kwargs=gen_kwargs,
        prompt_batch_cap=cfg.get("max_prompt_batch_size"),
        stats={},
    )


def _simple_user_message(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _messages_to_fallback_text(messages: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = message.get("content")
        parts: List[str] = []
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text = str(chunk.get("text") or "").strip()
                    if text:
                        parts.append(text)
        if parts:
            lines.append(f"{role}: " + "\n".join(parts))
    return "\n\n".join(lines).strip()


def _apply_chat_template_text(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]]) -> str:
    if hasattr(runtime.processor, "apply_chat_template"):
        try:
            text = runtime.processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            if isinstance(text, str) and text.strip():
                return text
        except TypeError:
            text = runtime.processor.apply_chat_template(list(messages), add_generation_prompt=True)
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
    return _messages_to_fallback_text(messages)


def infer_hf_chat_messages(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]], images: Sequence[Image.Image]) -> str:
    if not images:
        raise ValueError("images must not be empty")
    _bump_runtime_stat(runtime, "single_message_calls")

    image_input: Any = images[0] if len(images) == 1 else list(images)
    text = _apply_chat_template_text(runtime, messages)

    inputs = runtime.processor(text=text, images=image_input, return_tensors="pt", truncation=False)
    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out = runtime.model.generate(**inputs, **runtime.gen_kwargs)
    except Exception:
        _bump_runtime_stat(runtime, "single_message_generate_exception")
        raise

    input_len = inputs["input_ids"].shape[1]
    if out.shape[1] > input_len:
        gen_tokens = out[:, input_len:]
    else:
        gen_tokens = out
    try:
        decoded = runtime.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    except Exception:
        _bump_runtime_stat(runtime, "single_message_decode_exception")
        raise
    return decoded


def infer_hf_chat(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    return infer_hf_chat_messages(runtime, _simple_user_message(prompt), [image])


def infer_hf_chat_messages_batch(
    runtime: VLMRuntime,
    messages_batch: Sequence[Sequence[Dict[str, Any]]],
    images_batch: Sequence[Sequence[Image.Image]],
) -> List[str]:
    if not messages_batch:
        return []
    _bump_runtime_stat(runtime, "messages_batch_calls")
    _bump_runtime_stat(runtime, "messages_batch_items", len(messages_batch))
    if len(messages_batch) == 1:
        _bump_runtime_stat(runtime, "messages_batch_degenerate_to_single")
        return [infer_hf_chat_messages(runtime, messages_batch[0], images_batch[0])]
    if "llava" in runtime.model_id.lower() and any(
        len(images) > 1 or len(messages) > 1
        for messages, images in zip(messages_batch, images_batch)
    ):
        _bump_runtime_stat(runtime, "llava_multi_image_sequential_fallback")
        return [infer_hf_chat_messages(runtime, messages, images) for messages, images in zip(messages_batch, images_batch)]

    texts: List[str] = []
    image_inputs: List[Any] = []
    for messages, images in zip(messages_batch, images_batch):
        text = _apply_chat_template_text(runtime, messages)
        if not text.strip():
            _bump_runtime_stat(runtime, "messages_batch_blank_template_fallback")
            return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]
        texts.append(text)
        image_inputs.append(images[0] if len(images) == 1 else list(images))

    try:
        inputs = runtime.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=False)
    except Exception:
        _bump_runtime_stat(runtime, "messages_batch_processor_exception_fallback")
        return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]

    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out = runtime.model.generate(**inputs, **runtime.gen_kwargs)
    except Exception:
        _bump_runtime_stat(runtime, "messages_batch_generate_exception")
        raise

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lens = [inputs["input_ids"].shape[1]] * len(texts)

    generated = []
    for i, input_len in enumerate(input_lens):
        if out.shape[1] > int(input_len):
            generated.append(out[i, int(input_len):].detach().cpu())
        else:
            generated.append(out[i].detach().cpu())
    try:
        decoded = runtime.processor.batch_decode(generated, skip_special_tokens=True)
    except Exception:
        _bump_runtime_stat(runtime, "messages_batch_decode_exception")
        raise
    if any((not isinstance(item, str)) or (not item.strip()) for item in decoded):
        _bump_runtime_stat(runtime, "messages_batch_empty_decode_fallback")
        return [infer_hf_chat_messages(runtime, m, i) for m, i in zip(messages_batch, images_batch)]
    _bump_runtime_stat(runtime, "messages_batch_success")
    return decoded


def infer_hf_chat_batch(runtime: VLMRuntime, image: Image.Image, prompts: Sequence[str]) -> List[str]:
    if not prompts:
        return []
    _bump_runtime_stat(runtime, "simple_batch_calls")
    _bump_runtime_stat(runtime, "simple_batch_items", len(prompts))
    if len(prompts) == 1:
        _bump_runtime_stat(runtime, "simple_batch_degenerate_to_single")
        return [infer_hf_chat(runtime, image, prompts[0])]

    texts = [_apply_chat_template_text(runtime, _simple_user_message(prompt)) for prompt in prompts]
    if any(not text.strip() for text in texts):
        _bump_runtime_stat(runtime, "simple_batch_blank_template_fallback")
        return [infer_hf_chat(runtime, image, prompt) for prompt in prompts]

    try:
        inputs = runtime.processor(
            text=texts,
            images=[image for _ in prompts],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    except Exception:
        _bump_runtime_stat(runtime, "simple_batch_processor_exception_fallback")
        return [infer_hf_chat(runtime, image, prompt) for prompt in prompts]

    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out = runtime.model.generate(**inputs, **runtime.gen_kwargs)
    except Exception:
        _bump_runtime_stat(runtime, "simple_batch_generate_exception")
        raise

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lens = [inputs["input_ids"].shape[1]] * len(texts)

    generated = []
    for i, input_len in enumerate(input_lens):
        if out.shape[1] > int(input_len):
            generated.append(out[i, int(input_len):].detach().cpu())
        else:
            generated.append(out[i].detach().cpu())
    try:
        decoded = runtime.processor.batch_decode(generated, skip_special_tokens=True)
    except Exception:
        _bump_runtime_stat(runtime, "simple_batch_decode_exception")
        raise
    if any((not isinstance(item, str)) or (not item.strip()) for item in decoded):
        _bump_runtime_stat(runtime, "simple_batch_empty_decode_fallback")
        return [infer_hf_chat(runtime, image, prompt) for prompt in prompts]
    _bump_runtime_stat(runtime, "simple_batch_success")
    return decoded


BACKEND_LOADERS = {"hf_chat": load_hf_chat_runtime}
BACKEND_INFER = {"hf_chat": infer_hf_chat}
BACKEND_INFER_BATCH = {"hf_chat": infer_hf_chat_batch}
BACKEND_INFER_MESSAGES = {"hf_chat": infer_hf_chat_messages}
BACKEND_INFER_MESSAGES_BATCH = {"hf_chat": infer_hf_chat_messages_batch}


def load_runtime(model_key: str, model_registry: Dict[str, Dict[str, Any]]) -> VLMRuntime:
    cfg = model_registry[model_key]
    runtime = BACKEND_LOADERS[cfg["backend"]](model_key, cfg)
    print(f"Loaded model: {runtime.name} -> {runtime.model_id}")
    return runtime


def infer_runtime(runtime: VLMRuntime, image: Image.Image, prompt: str) -> str:
    return BACKEND_INFER[runtime.backend](runtime, image, prompt)


def infer_runtime_batch(runtime: VLMRuntime, image: Image.Image, prompts: Sequence[str]) -> List[str]:
    fn = BACKEND_INFER_BATCH.get(runtime.backend)
    if fn is None:
        return [infer_runtime(runtime, image, prompt) for prompt in prompts]
    return fn(runtime, image, prompts)


def infer_runtime_messages(runtime: VLMRuntime, messages: Sequence[Dict[str, Any]], images: Sequence[Image.Image]) -> str:
    fn = BACKEND_INFER_MESSAGES.get(runtime.backend)
    if fn is not None:
        return fn(runtime, messages, images)
    if not images:
        raise ValueError("images must not be empty")
    last_prompt = ""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        for chunk in message.get("content", []):
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                last_prompt = str(chunk.get("text") or "")
                break
        if last_prompt:
            break
    return infer_runtime(runtime, images[-1], last_prompt)


def infer_runtime_messages_batch(
    runtime: VLMRuntime,
    messages_batch: Sequence[Sequence[Dict[str, Any]]],
    images_batch: Sequence[Sequence[Image.Image]],
) -> List[str]:
    fn = BACKEND_INFER_MESSAGES_BATCH.get(runtime.backend)
    if fn is not None:
        return fn(runtime, messages_batch, images_batch)
    return [infer_runtime_messages(runtime, messages, images) for messages, images in zip(messages_batch, images_batch)]


def unload_runtime(runtime: Optional[VLMRuntime]) -> None:
    if runtime is None:
        return
    try:
        del runtime.model
    except Exception:
        pass
    try:
        del runtime.processor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
