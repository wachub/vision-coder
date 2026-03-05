"""Data loaders with lazy imports to avoid optional dependency failures."""

from __future__ import annotations

from typing import Any

__all__ = ["load_websight_dataset", "load_synthetic_sft_dataset"]


def load_websight_dataset(*args: Any, **kwargs: Any) -> Any:
    from vcoder.data.websight import load_websight_dataset as _load

    return _load(*args, **kwargs)


def load_synthetic_sft_dataset(*args: Any, **kwargs: Any) -> Any:
    from vcoder.data.synth_html import load_synthetic_sft_dataset as _load

    return _load(*args, **kwargs)
