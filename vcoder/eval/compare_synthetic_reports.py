"""Compare two synthetic evaluation summaries and write metric deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = (
    "mean_clip_similarity",
    "mean_numeric_exact_match_rate",
    "num_samples",
    "num_success",
    "num_failed",
)


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(after: float | int, before: float | int) -> float:
    return float(after) - float(before)


def compare_summaries(baseline_path: Path, candidate_path: Path, output_path: Path | None) -> dict:
    baseline = _load_summary(baseline_path)
    candidate = _load_summary(candidate_path)

    result = {
        "baseline_summary": str(baseline_path),
        "candidate_summary": str(candidate_path),
        "baseline_model_id": baseline.get("model_id"),
        "candidate_model_id": candidate.get("model_id"),
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


def _fmt(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic eval summary metrics.")
    parser.add_argument("--baseline_summary", type=Path, required=True)
    parser.add_argument("--candidate_summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = compare_summaries(
        baseline_path=args.baseline_summary,
        candidate_path=args.candidate_summary,
        output_path=args.output,
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
        print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
