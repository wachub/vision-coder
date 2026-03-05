"""Verify synthetic SFT updates LoRA weights and advances optimizer steps.

This script is designed as a fast integrity check, not a full model-quality eval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from vcoder.data.synth_html.dataset import generate_dataset
from vcoder.data.synth_html.prepare_sft_splits import prepare_splits


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_WORK_DIR = Path("outputs/sft_verify")
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_SAMPLES = 12
DEFAULT_MAX_EVAL_SAMPLES = 2
DEFAULT_MAX_LENGTH = 2048
DEFAULT_ATTN_BACKEND = "auto"
DEFAULT_VAL_RATIO = 0.2
REQUIRED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Any = None


def _discover_samples(directory: Path) -> list[Path]:
    samples: list[Path] = []
    for path in sorted(directory.glob("sample_*")):
        if not path.is_dir():
            continue
        if (path / "source.html").exists() and (path / "render.png").exists():
            samples.append(path)
    return samples


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_train_losses(log_history: list[dict[str, Any]]) -> list[float]:
    losses: list[float] = []
    for item in log_history:
        if not isinstance(item, dict):
            continue
        if "loss" not in item:
            continue
        value = item["loss"]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            losses.append(float(value))
    return losses


def _print_checks(checks: list[Check]) -> None:
    print("\nSFT Integrity Checks")
    print("-" * 96)
    print(f"{'Status':<8} {'Check':<34} Detail")
    print("-" * 96)
    for item in checks:
        label = "PASS" if item.status == "pass" else "FAIL"
        print(f"{label:<8} {item.name:<34} {item.detail}")
    print("-" * 96)


def _ensure_source_dataset(source_dir: Path, num_samples: int, seed: int) -> tuple[Path, bool]:
    existing = _discover_samples(source_dir) if source_dir.exists() else []
    if len(existing) >= 2:
        return source_dir, False

    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic source dataset in {source_dir} ({num_samples} samples)")
    generate_dataset(output_dir=source_dir, num_samples=num_samples, seed=seed)

    created = _discover_samples(source_dir)
    if len(created) < 2:
        raise RuntimeError(f"Source dataset generation failed: found {len(created)} samples in {source_dir}")
    return source_dir, True


def _run_training_subprocess(
    model_id: str,
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    summary_path: Path,
    max_steps: int,
    max_samples: int,
    max_eval_samples: int,
    max_length: int,
    seed: int,
    attn_backend: str,
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "vcoder.pipelines.sft_training",
        "--model_id",
        model_id,
        "--train_dataset_dir",
        str(train_dir),
        "--eval_dataset_dir",
        str(val_dir),
        "--output_dir",
        str(output_dir),
        "--max_samples",
        str(max_samples),
        "--max_eval_samples",
        str(max_eval_samples),
        "--max_steps",
        str(max_steps),
        "--num_train_epochs",
        "1",
        "--per_device_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--max_length",
        str(max_length),
        "--logging_steps",
        "1",
        "--save_steps",
        str(max(max_steps, 1)),
        "--eval_steps",
        "1",
        "--seed",
        str(seed),
        "--attn_backend",
        attn_backend,
        "--training_summary_path",
        str(summary_path),
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("\nRunning training command:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(repo_root), env=env, text=True)


def verify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.max_steps < 1:
        raise ValueError("--max_steps must be >= 1")
    if args.max_samples < 2:
        raise ValueError("--max_samples must be >= 2")
    if args.max_eval_samples < 1:
        raise ValueError("--max_eval_samples must be >= 1")

    work_dir = Path(args.work_dir)
    source_dir = Path(args.source_dir) if args.source_dir else (work_dir / "synth_source")
    split_root = work_dir / "synth_split"
    output_dir = Path(args.output_dir) if args.output_dir else (work_dir / "sft_output")
    verification_path = work_dir / "verification.json"
    training_summary_path = work_dir / "training_summary.json"

    repo_root = Path(__file__).resolve().parents[2]
    checks: list[Check] = []
    metrics: dict[str, Any] = {
        "global_step": 0,
        "last_train_loss": None,
        "num_loss_points": 0,
        "nonzero_lora_b_tensors": 0,
        "total_lora_b_tensors": 0,
    }
    failure_reasons: list[str] = []

    source_dir, source_was_generated = _ensure_source_dataset(
        source_dir=source_dir,
        num_samples=args.max_samples,
        seed=args.seed,
    )
    checks.append(
        Check(
            name="source_dataset_ready",
            status="pass",
            detail=f"{len(_discover_samples(source_dir))} samples available at {source_dir}",
            value=str(source_dir),
        )
    )

    if split_root.exists():
        shutil.rmtree(split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    val_ratio = max(1.0 / args.max_samples, min(0.5, args.max_eval_samples / args.max_samples))
    prepare_splits(source_dir=source_dir, output_root=split_root, val_ratio=val_ratio, seed=args.seed)
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    train_count = len(_discover_samples(train_dir))
    val_count = len(_discover_samples(val_dir))
    checks.append(
        Check(
            name="split_ready",
            status="pass" if train_count > 0 and val_count > 0 else "fail",
            detail=f"train={train_count}, val={val_count}",
            value={"train_dir": str(train_dir), "val_dir": str(val_dir)},
        )
    )

    checks.append(
        Check(
            name="lora_mode_enabled",
            status="pass",
            detail="Verifier always runs SFT with LoRA enabled",
            value=True,
        )
    )

    train_returncode = None
    if not args.skip_training:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if training_summary_path.exists():
            training_summary_path.unlink()
        completed = _run_training_subprocess(
            model_id=args.model_id,
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=output_dir,
            summary_path=training_summary_path,
            max_steps=args.max_steps,
            max_samples=args.max_samples,
            max_eval_samples=args.max_eval_samples,
            max_length=args.max_length,
            seed=args.seed,
            attn_backend=args.attn_backend,
            repo_root=repo_root,
        )
        train_returncode = int(completed.returncode)
        checks.append(
            Check(
                name="training_subprocess_exit",
                status="pass" if completed.returncode == 0 else "fail",
                detail=f"exit_code={completed.returncode}",
                value=completed.returncode,
            )
        )
    else:
        checks.append(
            Check(
                name="training_subprocess_exit",
                status="pass",
                detail="Skipped training run (--skip_training enabled)",
                value=None,
            )
        )

    trainer_state_path = output_dir / "trainer_state.json"
    trainer_state = _load_json(trainer_state_path) if trainer_state_path.exists() else {}
    checks.append(
        Check(
            name="trainer_state_exists",
            status="pass" if trainer_state_path.exists() else "fail",
            detail=f"path={trainer_state_path}",
            value=str(trainer_state_path),
        )
    )

    training_summary = _load_json(training_summary_path) if training_summary_path.exists() else {}
    checks.append(
        Check(
            name="training_summary_exists",
            status="pass" if training_summary_path.exists() else "fail",
            detail=f"path={training_summary_path}",
            value=str(training_summary_path),
        )
    )

    global_step = int(training_summary.get("global_step", trainer_state.get("global_step", 0)))
    log_history = trainer_state.get("log_history", [])
    train_losses = _extract_train_losses(log_history)
    if not train_losses and "train_loss_points" in training_summary:
        train_losses = [float(v) for v in training_summary["train_loss_points"] if isinstance(v, (int, float))]

    metrics["global_step"] = global_step
    metrics["num_loss_points"] = len(train_losses)
    metrics["last_train_loss"] = train_losses[-1] if train_losses else None

    checks.append(
        Check(
            name="global_step_reached",
            status="pass" if global_step >= args.max_steps else "fail",
            detail=f"global_step={global_step}, required>={args.max_steps}",
            value=global_step,
        )
    )
    checks.append(
        Check(
            name="train_loss_logged",
            status="pass" if len(train_losses) > 0 else "fail",
            detail=f"loss_points={len(train_losses)}",
            value=train_losses[-1] if train_losses else None,
        )
    )

    adapter_path = output_dir / "adapter_model.safetensors"
    adapter_loaded: dict[str, Any] | None = None
    if adapter_path.exists():
        try:
            adapter_loaded = load_file(str(adapter_path))
            checks.append(
                Check(
                    name="adapter_safetensors_load",
                    status="pass",
                    detail=f"loaded {len(adapter_loaded)} tensors",
                    value=len(adapter_loaded),
                )
            )
        except Exception as exc:
            checks.append(
                Check(
                    name="adapter_safetensors_load",
                    status="fail",
                    detail=f"failed to load: {exc}",
                    value=None,
                )
            )
    else:
        checks.append(
            Check(
                name="adapter_safetensors_load",
                status="fail",
                detail=f"missing file: {adapter_path}",
                value=None,
            )
        )

    if adapter_loaded is not None:
        lora_b_keys = [k for k in adapter_loaded.keys() if "lora_B" in k]
        nonzero = 0
        for key in lora_b_keys:
            tensor = adapter_loaded[key]
            if tensor.numel() > 0 and float(tensor.abs().max().item()) > 0.0:
                nonzero += 1
        metrics["total_lora_b_tensors"] = len(lora_b_keys)
        metrics["nonzero_lora_b_tensors"] = nonzero

        checks.append(
            Check(
                name="lora_b_tensors_present",
                status="pass" if len(lora_b_keys) > 0 else "fail",
                detail=f"count={len(lora_b_keys)}",
                value=len(lora_b_keys),
            )
        )
        checks.append(
            Check(
                name="lora_b_nonzero_after_train",
                status="pass" if nonzero > 0 else "fail",
                detail=f"nonzero={nonzero}/{len(lora_b_keys)}",
                value=nonzero,
            )
        )

    adapter_config_path = output_dir / "adapter_config.json"
    adapter_config = _load_json(adapter_config_path) if adapter_config_path.exists() else {}
    checks.append(
        Check(
            name="adapter_config_exists",
            status="pass" if adapter_config_path.exists() else "fail",
            detail=f"path={adapter_config_path}",
            value=str(adapter_config_path),
        )
    )
    if adapter_config:
        peft_type = str(adapter_config.get("peft_type", "")).upper()
        target_modules = set(adapter_config.get("target_modules", []))
        missing_modules = sorted(REQUIRED_TARGET_MODULES - target_modules)
        checks.append(
            Check(
                name="adapter_config_lora_type",
                status="pass" if peft_type == "LORA" else "fail",
                detail=f"peft_type={peft_type or 'missing'}",
                value=peft_type,
            )
        )
        checks.append(
            Check(
                name="adapter_target_modules",
                status="pass" if not missing_modules else "fail",
                detail="missing=" + (",".join(missing_modules) if missing_modules else "none"),
                value=sorted(target_modules),
            )
        )

    for item in checks:
        if item.status != "pass":
            failure_reasons.append(f"{item.name}: {item.detail}")

    verdict = "pass" if not failure_reasons else "fail"
    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "source_dir": str(source_dir),
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "output_dir": str(output_dir),
            "max_steps": args.max_steps,
            "max_samples": args.max_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
            "skip_training": bool(args.skip_training),
            "keep_artifacts": bool(args.keep_artifacts),
            "training_returncode": train_returncode,
        },
        "checks": [asdict(item) for item in checks],
        "metrics": metrics,
        "verdict": verdict,
        "failure_reasons": failure_reasons,
    }
    _write_json(verification_path, payload)
    _print_checks(checks)
    print(f"Verification JSON: {verification_path}")
    print(f"Verdict: {verdict.upper()}")

    if not args.keep_artifacts:
        if source_was_generated and source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
        if split_root.exists():
            shutil.rmtree(split_root, ignore_errors=True)

    return payload, (0 if verdict == "pass" else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify synthetic SFT updates LoRA weights.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--max_eval_samples", type=int, default=DEFAULT_MAX_EVAL_SAMPLES)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_backend", type=str, default=DEFAULT_ATTN_BACKEND)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--keep_artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, exit_code = verify(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
