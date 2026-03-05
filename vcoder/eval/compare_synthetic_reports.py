"""Compare synthetic evaluation outputs and write JSON/Markdown/graph reports."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


KEYS = (
    "mean_clip_similarity",
    "mean_numeric_exact_match_rate",
    "num_samples",
    "num_success",
    "num_failed",
    "success_rate",
)


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_report_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _delta(after: float | int, before: float | int) -> float:
    return float(after) - float(before)


def _to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | float | int | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _compute_success_rate(summary: dict[str, Any]) -> float:
    num_samples = max(_to_int(summary.get("num_samples")), 0)
    if num_samples == 0:
        return 0.0
    num_success = max(_to_int(summary.get("num_success")), 0)
    return num_success / num_samples


def compare_summaries(baseline_path: Path, candidate_path: Path, output_path: Path | None) -> dict:
    baseline = _load_summary(baseline_path)
    candidate = _load_summary(candidate_path)

    baseline["success_rate"] = _compute_success_rate(baseline)
    candidate["success_rate"] = _compute_success_rate(candidate)

    result = {
        "baseline_summary": str(baseline_path),
        "candidate_summary": str(candidate_path),
        "baseline_model_id": baseline.get("model_id"),
        "candidate_model_id": candidate.get("model_id"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
    }
    for key in KEYS:
        before = baseline.get(key, 0)
        after = candidate.get(key, 0)
        result["metrics"][key] = {
            "baseline": before,
            "candidate": after,
            "delta": _delta(after, before),
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _is_better(metric: str, delta: float) -> bool:
    if metric == "num_failed":
        return delta <= 0.0
    return delta >= 0.0


def _sort_candidate_rows(report_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ok_rows = [row for row in report_rows if row.get("status") == "ok"]
    return sorted(ok_rows, key=lambda row: _to_float(row.get("clip_similarity"), -1.0), reverse=True)


def _build_sample_delta_table(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not baseline_rows or not candidate_rows:
        return []

    base_by_id = {row.get("sample_id", ""): row for row in baseline_rows}
    cand_by_id = {row.get("sample_id", ""): row for row in candidate_rows}
    shared_ids = sorted(set(base_by_id) & set(cand_by_id))
    deltas: list[dict[str, Any]] = []

    for sample_id in shared_ids:
        b = base_by_id[sample_id]
        c = cand_by_id[sample_id]
        if b.get("status") != "ok" or c.get("status") != "ok":
            continue
        b_clip = _to_float(b.get("clip_similarity"))
        c_clip = _to_float(c.get("clip_similarity"))
        b_num = _to_float(b.get("numeric_exact_match_rate"))
        c_num = _to_float(c.get("numeric_exact_match_rate"))
        deltas.append(
            {
                "sample_id": sample_id,
                "baseline_clip": b_clip,
                "candidate_clip": c_clip,
                "delta_clip": c_clip - b_clip,
                "baseline_numeric": b_num,
                "candidate_numeric": c_num,
                "delta_numeric": c_num - b_num,
            }
        )

    return sorted(deltas, key=lambda row: row["delta_clip"], reverse=True)[: max(limit, 0)]


def _write_markdown_report(
    result: dict[str, Any],
    md_output: Path,
    graph_output: Path | None,
    baseline_report_path: Path | None,
    candidate_report_path: Path | None,
    sample_limit: int = 8,
) -> None:
    baseline_rows = _load_report_rows(baseline_report_path)
    candidate_rows = _load_report_rows(candidate_report_path)
    top_candidate = _sort_candidate_rows(candidate_rows)[: max(sample_limit, 0)]
    sample_deltas = _build_sample_delta_table(baseline_rows, candidate_rows, limit=sample_limit)

    lines: list[str] = []
    lines.append("# Synthetic Evaluation Comparison")
    lines.append("")
    lines.append(f"- Generated (UTC): `{result.get('generated_at_utc', 'n/a')}`")
    lines.append(f"- Baseline model: `{result.get('baseline_model_id', 'n/a')}`")
    lines.append(f"- Candidate model: `{result.get('candidate_model_id', 'n/a')}`")
    lines.append(f"- Baseline summary: `{result.get('baseline_summary', 'n/a')}`")
    lines.append(f"- Candidate summary: `{result.get('candidate_summary', 'n/a')}`")
    if graph_output is not None:
        lines.append(f"- Graph: `{graph_output}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|---|---:|---:|---:|")
    for metric, payload in result["metrics"].items():
        delta = _to_float(payload.get("delta"))
        delta_str = f"+{delta:.6f}" if delta >= 0 else f"{delta:.6f}"
        marker = "improved" if _is_better(metric, delta) else "worse"
        lines.append(
            f"| {metric} | {_fmt(payload.get('baseline', 0))} | {_fmt(payload.get('candidate', 0))} | {delta_str} ({marker}) |"
        )
    lines.append("")

    if top_candidate:
        lines.append("## Top Candidate Samples (by CLIP)")
        lines.append("")
        lines.append("| sample_id | clip_similarity | numeric_exact_match_rate |")
        lines.append("|---|---:|---:|")
        for row in top_candidate:
            lines.append(
                f"| {row.get('sample_id', '')} | {_to_float(row.get('clip_similarity')):.6f} | "
                f"{_to_float(row.get('numeric_exact_match_rate')):.6f} |"
            )
        lines.append("")

    if sample_deltas:
        lines.append("## Biggest CLIP Gains (Candidate - Baseline)")
        lines.append("")
        lines.append("| sample_id | baseline_clip | candidate_clip | delta_clip | baseline_numeric | candidate_numeric | delta_numeric |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sample_deltas:
            lines.append(
                f"| {row['sample_id']} | {row['baseline_clip']:.6f} | {row['candidate_clip']:.6f} | "
                f"{row['delta_clip']:+.6f} | {row['baseline_numeric']:.6f} | {row['candidate_numeric']:.6f} | "
                f"{row['delta_numeric']:+.6f} |"
            )
        lines.append("")

    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text("\n".join(lines), encoding="utf-8")


def _draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    max_value: float,
    fill: tuple[int, int, int],
) -> None:
    value = max(0.0, min(value, max_value))
    bar_w = int((value / max_value) * width) if max_value > 0 else 0
    draw.rectangle((x, y, x + width, y + height), outline=(180, 186, 197), width=1, fill=(245, 247, 250))
    draw.rectangle((x, y, x + bar_w, y + height), fill=fill)


def _write_graph(result: dict[str, Any], graph_output: Path) -> None:
    metrics = result["metrics"]
    baseline_success = _to_float(metrics["success_rate"]["baseline"])
    candidate_success = _to_float(metrics["success_rate"]["candidate"])

    graph_specs = [
        ("mean_clip_similarity", _to_float(metrics["mean_clip_similarity"]["baseline"]), _to_float(metrics["mean_clip_similarity"]["candidate"]), 1.0),
        (
            "mean_numeric_exact_match_rate",
            _to_float(metrics["mean_numeric_exact_match_rate"]["baseline"]),
            _to_float(metrics["mean_numeric_exact_match_rate"]["candidate"]),
            1.0,
        ),
        ("success_rate", baseline_success, candidate_success, 1.0),
    ]

    width, height = 1100, 620
    canvas = Image.new("RGB", (width, height), (251, 252, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((24, 18), "Synthetic Evaluation Comparison", fill=(22, 31, 48))
    draw.text(
        (24, 42),
        f"baseline={result.get('baseline_model_id', 'n/a')} | candidate={result.get('candidate_model_id', 'n/a')}",
        fill=(80, 90, 106),
    )

    left_x = 220
    right_x = 610
    bar_w = 320
    bar_h = 28
    y0 = 110
    row_gap = 150
    baseline_color = (88, 131, 235)
    candidate_color = (56, 182, 113)

    for idx, (name, baseline_v, candidate_v, max_v) in enumerate(graph_specs):
        y = y0 + idx * row_gap
        draw.text((24, y + 2), name, fill=(33, 43, 58))
        draw.text((24, y + 28), "baseline", fill=(88, 131, 235))
        draw.text((24, y + 67), "candidate", fill=(56, 182, 113))

        _draw_bar(draw, left_x, y + 20, bar_w, bar_h, baseline_v, max_v, baseline_color)
        _draw_bar(draw, right_x, y + 20, bar_w, bar_h, candidate_v, max_v, candidate_color)
        draw.text((left_x + bar_w + 10, y + 24), f"{baseline_v:.4f}", fill=(33, 43, 58))
        draw.text((right_x + bar_w + 10, y + 24), f"{candidate_v:.4f}", fill=(33, 43, 58))

    metrics_bottom = y0 + len(graph_specs) * row_gap + 20
    draw.text((24, metrics_bottom), "Counts", fill=(33, 43, 58))
    draw.text(
        (24, metrics_bottom + 26),
        (
            f"num_samples: {metrics['num_samples']['baseline']} -> {metrics['num_samples']['candidate']} | "
            f"num_success: {metrics['num_success']['baseline']} -> {metrics['num_success']['candidate']} | "
            f"num_failed: {metrics['num_failed']['baseline']} -> {metrics['num_failed']['candidate']}"
        ),
        fill=(70, 80, 95),
    )

    graph_output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(graph_output)


def _resolve_default_path(path: Path | None, json_output: Path | None, suffix: str) -> Path | None:
    if path is not None:
        return path
    if json_output is None:
        return None
    return json_output.with_suffix(suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic eval summary metrics.")
    parser.add_argument("--baseline_summary", type=Path, required=True)
    parser.add_argument("--candidate_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="JSON comparison output path.")
    parser.add_argument("--baseline_report", type=Path, default=None, help="Optional baseline report.csv path.")
    parser.add_argument("--candidate_report", type=Path, default=None, help="Optional candidate report.csv path.")
    parser.add_argument("--md_output", type=Path, default=None, help="Optional Markdown report output path.")
    parser.add_argument("--graph_output", type=Path, default=None, help="Optional PNG graph output path.")
    parser.add_argument("--sample_limit", type=int, default=8, help="Max rows in sample tables.")
    args = parser.parse_args()

    if args.sample_limit < 0:
        raise ValueError("--sample_limit must be >= 0")

    result = compare_summaries(
        baseline_path=args.baseline_summary,
        candidate_path=args.candidate_summary,
        output_path=args.output,
    )

    md_output = _resolve_default_path(args.md_output, args.output, ".md")
    graph_output = _resolve_default_path(args.graph_output, args.output, ".png")
    if graph_output is not None:
        _write_graph(result, graph_output)
    if md_output is not None:
        _write_markdown_report(
            result=result,
            md_output=md_output,
            graph_output=graph_output,
            baseline_report_path=args.baseline_report,
            candidate_report_path=args.candidate_report,
            sample_limit=args.sample_limit,
        )

    print(f"Baseline:  {result['baseline_model_id']}")
    print(f"Candidate: {result['candidate_model_id']}")
    for key, payload in result["metrics"].items():
        print(
            f"{key}: baseline={_fmt(payload['baseline'])} "
            f"candidate={_fmt(payload['candidate'])} "
            f"delta={_fmt(payload['delta'])}"
        )
    if args.output is not None:
        print(f"Saved comparison JSON: {args.output}")
    if graph_output is not None:
        print(f"Saved graph PNG:       {graph_output}")
    if md_output is not None:
        print(f"Saved markdown report: {md_output}")


if __name__ == "__main__":
    main()
