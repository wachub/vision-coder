"""Shared utilities for synthetic screenshot->HTML evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from shutil import copyfile
from typing import Any

from PIL import Image, ImageDraw, ImageOps


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

REPORT_FIELDS = [
    "sample_id",
    "status",
    "clip_similarity",
    "numeric_exact_match_rate",
    "source_numeric_count",
    "matched_numeric_count",
    "source_html_path",
    "source_render_path",
    "predicted_html_path",
    "predicted_render_path",
    "error",
]
GALLERY_FIELDS = [
    "sample_id",
    "clip_similarity",
    "numeric_exact_match_rate",
    "tile",
    "source_html",
    "predicted_html",
]


def cleanup_response(text: str) -> str:
    fenced = re.search(r"```html\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    fenced_any = re.search(r"```\s*(<!DOCTYPE|<html)(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced_any:
        return (fenced_any.group(1) + fenced_any.group(2)).strip()

    unclosed = re.match(r"```(?:html)?\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

    return text.strip()


def numeric_exact_match_rate(source_html: str, predicted_html: str) -> tuple[float, int, int]:
    source_numbers = NUMBER_PATTERN.findall(source_html)
    predicted_numbers = NUMBER_PATTERN.findall(predicted_html)
    if not source_numbers:
        return 1.0, 0, 0

    source_counts = Counter(source_numbers)
    predicted_counts = Counter(predicted_numbers)
    matched = sum(min(count, predicted_counts.get(value, 0)) for value, count in source_counts.items())
    return matched / len(source_numbers), len(source_numbers), matched


def sample_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.split("_")[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, path.name)


def load_samples(dataset_dir: Path) -> list[Path]:
    sample_dirs: list[Path] = []
    for path in sorted(dataset_dir.glob("sample_*"), key=sample_sort_key):
        if not path.is_dir():
            continue
        if (path / "source.html").exists() and (path / "render.png").exists():
            sample_dirs.append(path)
    return sample_dirs


def save_gallery_tile(
    sample_id: str,
    source_image: Image.Image,
    pred_image: Image.Image,
    clip_score: float,
    numeric_rate: float,
    out_path: Path,
) -> None:
    tile_w, tile_h = 420, 290
    source_fit = ImageOps.fit(source_image.convert("RGB"), (tile_w, tile_h), Image.Resampling.LANCZOS)
    pred_fit = ImageOps.fit(pred_image.convert("RGB"), (tile_w, tile_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (tile_w * 2, tile_h + 72), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{sample_id} | CLIP={clip_score:.4f} | numeric_exact={numeric_rate:.4f}", fill=(26, 31, 37))
    draw.text((12, 34), "source", fill=(85, 93, 104))
    draw.text((tile_w + 12, 34), "predicted", fill=(85, 93, 104))
    canvas.paste(source_fit, (0, 54))
    canvas.paste(pred_fit, (tile_w, 54))
    canvas.save(out_path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    dataset_dir: Path,
    model_id: str,
    num_samples: int,
    successful_rows: list[dict[str, str]],
) -> Path:
    if successful_rows:
        mean_clip = sum(float(row["clip_similarity"]) for row in successful_rows) / len(successful_rows)
        mean_numeric = sum(float(row["numeric_exact_match_rate"]) for row in successful_rows) / len(successful_rows)
    else:
        mean_clip = 0.0
        mean_numeric = 0.0

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "num_samples": num_samples,
        "num_success": len(successful_rows),
        "num_failed": num_samples - len(successful_rows),
        "mean_clip_similarity": mean_clip,
        "mean_numeric_exact_match_rate": mean_numeric,
        "model_id": model_id,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def write_gallery(
    gallery_dir: Path,
    ok_rows: list[dict[str, Any]],
    gallery_size: int,
) -> None:
    selected = sorted(ok_rows, key=lambda item: float(item["clip"]), reverse=True)[: max(gallery_size, 0)]
    gallery_rows: list[dict[str, str]] = []

    for entry in selected:
        sample_id = str(entry["sample_id"])
        clip_value = float(entry["clip"])
        numeric_value = float(entry["numeric"])

        tile_path = gallery_dir / f"{sample_id}.png"
        save_gallery_tile(
            sample_id=sample_id,
            source_image=entry["source_image"],
            pred_image=entry["pred_image"],
            clip_score=clip_value,
            numeric_rate=numeric_value,
            out_path=tile_path,
        )

        source_copy = gallery_dir / f"{sample_id}_source.html"
        pred_copy = gallery_dir / f"{sample_id}_predicted.html"
        copyfile(entry["source_html_path"], source_copy)
        copyfile(entry["pred_html_path"], pred_copy)

        gallery_rows.append(
            {
                "sample_id": sample_id,
                "clip_similarity": f"{clip_value:.6f}",
                "numeric_exact_match_rate": f"{numeric_value:.6f}",
                "tile": str(tile_path),
                "source_html": str(source_copy),
                "predicted_html": str(pred_copy),
            }
        )

    write_csv(gallery_dir / "index.csv", GALLERY_FIELDS, gallery_rows)
