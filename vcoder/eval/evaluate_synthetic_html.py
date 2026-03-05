"""Evaluate a model server on the synthetic screenshot->HTML dataset.

Inputs (per sample):
    <dataset_dir>/sample_XXXX/source.html
    <dataset_dir>/sample_XXXX/render.png

Outputs:
    <output_dir>/report.csv
    <output_dir>/summary.json
    <output_dir>/predictions/sample_XXXX/predicted.html
    <output_dir>/predictions/sample_XXXX/predicted.png
    <output_dir>/gallery/*.png
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any

import requests
from PIL import Image

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


DIRECT_PROMPT = (
    "You are an expert web developer. "
    "Given a screenshot, return one self-contained HTML file with inline CSS that reproduces it. "
    "Output only HTML code."
)

DEFAULT_DATASET_DIR = Path("outputs/synth_html")
DEFAULT_OUTPUT_DIR = Path("outputs/synth_eval")
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 180
DEFAULT_DEVICE = "cpu"
DEFAULT_GALLERY_SIZE = 16


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=95)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def discover_model_id(host: str, port: int, timeout: int) -> str:
    url = f"http://{host}:{port}/v1/models"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    items = payload.get("data", [])
    if not items:
        raise RuntimeError(f"No models returned by {url}")

    model_id = items[0].get("id")
    if not model_id:
        raise RuntimeError(f"Model list from {url} had no id field")
    return model_id


def query_model(
    host: str,
    port: int,
    model_id: str,
    image_path: Path,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _evaluate_sample(
    sample_dir: Path,
    predictions_dir: Path,
    host: str,
    port: int,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device: str,
    disable_thinking: bool,
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
        raw_output = query_model(
            host=host,
            port=port,
            model_id=model_id,
            image_path=source_png_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            disable_thinking=disable_thinking,
        )
        predicted_html = cleanup_response(raw_output)
        pred_html_path.write_text(predicted_html, encoding="utf-8")

        pred_image = render_html_to_image(predicted_html, full_page=True)
        pred_image.save(pred_png_path)

        clip_score = compute_clip_similarity(pred_image, source_image, device=device)
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


def evaluate(
    dataset_dir: Path,
    output_dir: Path,
    host: str,
    port: int,
    model_id: str | None,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device: str,
    gallery_size: int,
    limit: int | None,
    disable_thinking: bool,
) -> None:
    samples = load_samples(dataset_dir)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"No valid samples found in {dataset_dir}")
    if gallery_size < 0:
        raise ValueError("--gallery_size must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_id = model_id or discover_model_id(host, port, timeout)
    print(f"Using model: {resolved_model_id}")
    print(f"Evaluating {len(samples)} samples")

    rows: list[dict[str, str]] = []
    ok_rows: list[dict[str, object]] = []

    for idx, sample_dir in enumerate(samples, start=1):
        row, ok_row = _evaluate_sample(
            sample_dir=sample_dir,
            predictions_dir=predictions_dir,
            host=host,
            port=port,
            model_id=resolved_model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            device=device,
            disable_thinking=disable_thinking,
        )
        rows.append(row)
        if ok_row is not None:
            ok_rows.append(ok_row)

        if idx % 10 == 0 or idx == len(samples):
            print(f"[eval] {idx}/{len(samples)}")

    report_path = output_dir / "report.csv"
    write_csv(report_path, REPORT_FIELDS, rows)

    successful_rows = [row for row in rows if row["status"] == "ok"]
    summary_path = write_summary(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        model_id=resolved_model_id,
        num_samples=len(samples),
        successful_rows=successful_rows,
    )
    write_gallery(gallery_dir, ok_rows, gallery_size)

    print(f"Report:  {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Gallery: {gallery_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate screenshot->HTML model on synthetic dataset.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DIRECT_PROMPT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--gallery_size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", dest="disable_thinking", action="store_true")
    parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false")
    parser.set_defaults(disable_thinking=True)
    args = parser.parse_args()

    evaluate(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        prompt=args.prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        device=args.device,
        gallery_size=args.gallery_size,
        limit=args.limit,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
