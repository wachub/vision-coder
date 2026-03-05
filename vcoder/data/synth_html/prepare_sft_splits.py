"""Prepare train/val splits for synthetic SFT datasets.

Input layout:
    <source_dir>/sample_XXXX/source.html
    <source_dir>/sample_XXXX/render.png

Output layout:
    <output_root>/train/sample_XXXX/...
    <output_root>/val/sample_XXXX/...
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


INDEX_FIELDS = ("sample_id", "source_html", "render_png")


def _sample_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.split("_")[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, path.name)


def _load_samples(source_dir: Path) -> list[Path]:
    samples: list[Path] = []
    for path in sorted(source_dir.glob("sample_*"), key=_sample_sort_key):
        if not path.is_dir():
            continue
        if (path / "source.html").exists() and (path / "render.png").exists():
            samples.append(path)
    return samples


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _write_split(split_dir: Path, samples: list[Path]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for src_sample in samples:
        sample_id = src_sample.name
        out_sample = split_dir / sample_id
        out_sample.mkdir(parents=True, exist_ok=True)

        src_html = src_sample / "source.html"
        src_png = src_sample / "render.png"
        dst_html = out_sample / "source.html"
        dst_png = out_sample / "render.png"

        _safe_link_or_copy(src_html, dst_html)
        _safe_link_or_copy(src_png, dst_png)

        rows.append(
            {
                "sample_id": sample_id,
                "source_html": str(dst_html),
                "render_png": str(dst_png),
            }
        )

    with (split_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _clear_existing_split(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for sample_dir in split_dir.glob("sample_*"):
        if sample_dir.is_dir():
            shutil.rmtree(sample_dir)
    index_path = split_dir / "index.csv"
    if index_path.exists():
        index_path.unlink()


def prepare_splits(
    source_dir: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
) -> None:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1 (exclusive)")

    samples = _load_samples(source_dir)
    if len(samples) < 2:
        raise RuntimeError(f"Need at least 2 samples in {source_dir}, found {len(samples)}")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples:
        raise RuntimeError("Split produced 0 train samples; lower --val_ratio or add more samples")

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    _clear_existing_split(train_dir)
    _clear_existing_split(val_dir)

    _write_split(train_dir, train_samples)
    _write_split(val_dir, val_samples)

    print(f"Source: {source_dir}")
    print(f"Output root: {output_root}")
    print(f"Train samples: {len(train_samples)} -> {train_dir}")
    print(f"Val samples:   {len(val_samples)} -> {val_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val splits for synthetic SFT.")
    parser.add_argument("--source_dir", type=Path, default=Path("assets/synth_samples"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/synth_sft"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_splits(
        source_dir=args.source_dir,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
