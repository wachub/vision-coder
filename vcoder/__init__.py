"""Top-level package exports.

This module keeps imports lazy so light-weight commands (for example synthetic
data generation) do not require all optional training/reward dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = (
    "clip_visual_reward",
    "format_reward",
    "html_validity_reward",
    "visual_fidelity_reward",
    "structural_similarity_reward",
    "load_websight_dataset",
    "load_synthetic_sft_dataset",
)


def _load_reward(name: str) -> Callable[..., Any]:
    from vcoder import rewards

    return getattr(rewards, name)


def clip_visual_reward(*args: Any, **kwargs: Any) -> Any:
    return _load_reward("clip_visual_reward")(*args, **kwargs)


def format_reward(*args: Any, **kwargs: Any) -> Any:
    return _load_reward("format_reward")(*args, **kwargs)


def html_validity_reward(*args: Any, **kwargs: Any) -> Any:
    return _load_reward("html_validity_reward")(*args, **kwargs)


def visual_fidelity_reward(*args: Any, **kwargs: Any) -> Any:
    return _load_reward("visual_fidelity_reward")(*args, **kwargs)


def structural_similarity_reward(*args: Any, **kwargs: Any) -> Any:
    return _load_reward("structural_similarity_reward")(*args, **kwargs)


def load_websight_dataset(*args: Any, **kwargs: Any) -> Any:
    from vcoder.data.websight import load_websight_dataset as _load

    return _load(*args, **kwargs)


def load_synthetic_sft_dataset(*args: Any, **kwargs: Any) -> Any:
    from vcoder.data.synth_html import load_synthetic_sft_dataset as _load

    return _load(*args, **kwargs)
