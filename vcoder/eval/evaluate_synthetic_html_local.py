"""Evaluate local HF models/adapters on synthetic screenshot->HTML data.

This is a no-server path for quick checkpoint verification:
    python -m vcoder.eval.evaluate_synthetic_html_local \
        --dataset_dir outputs/synth_sft/val \
        --model_id Qwen/Qwen3-VL-2B-Instruct \
        --output_dir outputs/synth_eval/base \
        --limit 20
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from vcoder.data.synth_html.sft_dataset import SYSTEM_PROMPT, USER_PROMPT
from vcoder.eval.synthetic_eval_common import (
    REPORT_FIELDS,
    cleanup_response,
    load_samples,
    numeric_exact_match_rate,
    write_csv,
    write_gallery,
    write_summary,
)
from vcoder.rendering.html_renderer import render_html_to_image
from vcoder.utils.image_utils import compute_clip_similarity


ATTN_BACKEND_CHOICES = ("auto", "flash_attention_2", "sdpa", "eager")
DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")

DEFAULT_DATASET_DIR = Path("outputs/synth_html")
DEFAULT_OUTPUT_DIR = Path("outputs/synth_eval_local")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_GALLERY_SIZE = 16
DEFAULT_SEED = 42


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested}' but CUDA is not available.")
    return requested


def _resolve_dtype(requested: str, has_cuda: bool) -> torch.dtype:
    if requested == "auto":
        if has_cuda:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {requested}")


def _resolve_attn_backend(requested: str, has_cuda: bool) -> str | None:
    if not has_cuda:
        return None
    if requested == "auto":
        return "flash_attention_2" if _has_flash_attn() else "sdpa"
    return requested


def _load_model_with_attn_fallback(
    model_id: str,
    dtype: torch.dtype,
    attn_backend: str | None,
    use_cuda: bool,
):
    model_kwargs: dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True}
    if attn_backend is not None:
        model_kwargs["attn_implementation"] = attn_backend
        print(f"Loading model with attention backend: {attn_backend}")
    else:
        print("Loading model with default CPU attention backend")
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        return model, attn_backend
    except ImportError as exc:
        msg = str(exc).lower()
        needs_fallback = attn_backend == "flash_attention_2" and (
            "flashattention2" in msg or "flash_attn" in msg
        )
        if not needs_fallback:
            raise
        print(
            "Warning: flash_attention_2 requested but flash_attn is unavailable. "
            "Retrying with sdpa backend."
        )
        retry_kwargs = dict(model_kwargs)
        retry_kwargs["attn_implementation"] = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **retry_kwargs)
        return model, "sdpa"


def _model_input_device(model: torch.nn.Module) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device) and model_device.type != "meta":
        return model_device

    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for mapped_device in hf_map.values():
            if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                return torch.device(mapped_device)
        for mapped_device in hf_map.values():
            if isinstance(mapped_device, str):
                return torch.device(mapped_device)
    return next(model.parameters()).device


def _apply_chat_template(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    disable_thinking: bool,
) -> str:
    kwargs: dict[str, Any] = {}
    if disable_thinking:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate_html(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt_text = _apply_chat_template(
        processor=processor,
        messages=messages,
        disable_thinking=disable_thinking,
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    input_device = _model_input_device(model)
    inputs = {k: v.to(input_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0.0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = generated[:, prompt_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return cleanup_response(decoded)


def _evaluate_sample(
    sample_dir: Path,
    predictions_dir: Path,
    model: torch.nn.Module,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    clip_device: str,
) -> tuple[dict[str, str], dict[str, object] | None]:
    sample_id = sample_dir.name
    source_html_path = sample_dir / "source.html"
    source_png_path = sample_dir / "render.png"

    pred_sample_dir = predictions_dir / sample_id
    pred_sample_dir.mkdir(parents=True, exist_ok=True)
    pred_html_path = pred_sample_dir / "predicted.html"
    pred_png_path = pred_sample_dir / "predicted.png"

    row = {
        "sample_id": sample_id,
        "status": "error",
        "clip_similarity": "",
        "numeric_exact_match_rate": "",
        "source_numeric_count": "0",
        "matched_numeric_count": "0",
        "source_html_path": str(source_html_path),
        "source_render_path": str(source_png_path),
        "predicted_html_path": str(pred_html_path),
        "predicted_render_path": str(pred_png_path),
        "error": "",
    }

    source_html = source_html_path.read_text(encoding="utf-8")
    with Image.open(source_png_path) as source_image_opened:
        source_image = source_image_opened.convert("RGB")

    try:
        predicted_html = _generate_html(
            model=model,
            processor=processor,
            image=source_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_thinking=disable_thinking,
        )
        pred_html_path.write_text(predicted_html, encoding="utf-8")

        pred_image = render_html_to_image(predicted_html, full_page=True)
        pred_image.save(pred_png_path)

        clip_score = compute_clip_similarity(pred_image, source_image, device=clip_device)
        numeric_rate, source_num_count, matched_num_count = numeric_exact_match_rate(source_html, predicted_html)

        row["status"] = "ok"
        row["clip_similarity"] = f"{clip_score:.6f}"
        row["numeric_exact_match_rate"] = f"{numeric_rate:.6f}"
        row["source_numeric_count"] = str(source_num_count)
        row["matched_numeric_count"] = str(matched_num_count)

        ok_row = {
            "sample_id": sample_id,
            "clip": clip_score,
            "numeric": numeric_rate,
            "source_image": source_image.copy(),
            "pred_image": pred_image.copy(),
            "source_html_path": source_html_path,
            "pred_html_path": pred_html_path,
        }
        return row, ok_row
    except Exception as exc:
        row["error"] = str(exc).replace("\n", " ")[:500]
        return row, None


def evaluate_local(
    dataset_dir: Path,
    output_dir: Path,
    model_id: str,
    adapter_path: str | None,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    dtype: str,
    attn_backend: str,
    clip_device: str,
    gallery_size: int,
    limit: int | None,
    disable_thinking: bool,
    seed: int,
) -> None:
    if max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")
    if gallery_size < 0:
        raise ValueError("--gallery_size must be >= 0")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0 when set")

    samples = load_samples(dataset_dir)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_dir}")

    resolved_device = _resolve_device(device)
    use_cuda = resolved_device.startswith("cuda")
    resolved_dtype = _resolve_dtype(dtype, has_cuda=use_cuda)
    resolved_attn = _resolve_attn_backend(attn_backend, has_cuda=use_cuda)

    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_path or 'none'}")
    print(f"Device: {resolved_device} | DType: {resolved_dtype}")
    print(f"Attention backend (requested/resolved): {attn_backend}/{resolved_attn or 'cpu-default'}")

    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    model, loaded_attn_backend = _load_model_with_attn_fallback(
        model_id=model_id,
        dtype=resolved_dtype,
        attn_backend=resolved_attn,
        use_cuda=use_cuda,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("peft is required to load --adapter_path.") from exc
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    if not use_cuda:
        model = model.to(resolved_device)
    model.eval()
    print(f"Loaded model attention backend: {loaded_attn_backend or 'cpu-default'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    ok_rows: list[dict[str, object]] = []

    print(f"Evaluating {len(samples)} samples")
    for idx, sample_dir in enumerate(samples, start=1):
        row, ok_row = _evaluate_sample(
            sample_dir=sample_dir,
            predictions_dir=predictions_dir,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_thinking=disable_thinking,
            clip_device=clip_device,
        )
        rows.append(row)
        if ok_row is not None:
            ok_rows.append(ok_row)
        if idx % 5 == 0 or idx == len(samples):
            print(f"[eval-local] {idx}/{len(samples)}")

    report_path = output_dir / "report.csv"
    write_csv(report_path, REPORT_FIELDS, rows)

    successful_rows = [row for row in rows if row["status"] == "ok"]
    model_label = model_id if not adapter_path else f"{model_id} + {adapter_path}"
    summary_path = write_summary(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        model_id=model_label,
        num_samples=len(samples),
        successful_rows=successful_rows,
    )
    write_gallery(gallery_dir, ok_rows, gallery_size)

    print(f"Report:  {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Gallery: {gallery_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local model/checkpoint on synthetic screenshot->HTML.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--user_prompt", type=str, default=USER_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    parser.add_argument("--attn_backend", choices=ATTN_BACKEND_CHOICES, default="auto")
    parser.add_argument("--gallery_size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--disable_thinking", dest="disable_thinking", action="store_true")
    parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false")
    parser.set_defaults(disable_thinking=True)
    args = parser.parse_args()

    evaluate_local(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        dtype=args.dtype,
        attn_backend=args.attn_backend,
        clip_device=args.clip_device,
        gallery_size=args.gallery_size,
        limit=args.limit,
        disable_thinking=args.disable_thinking,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
