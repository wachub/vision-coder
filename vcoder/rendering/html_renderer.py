from __future__ import annotations

import asyncio
import io
import threading

from PIL import Image

from vcoder.rendering.browser_pool import BrowserPool

_sync_loop: asyncio.AbstractEventLoop | None = None
_sync_loop_lock = threading.Lock()


def _blank_white_image(width: int = 512, height: int = 512) -> Image.Image:
    """Return a blank white PIL image (used as fallback on render failure)."""
    return Image.new("RGB", (width, height), (255, 255, 255))


async def _render_html_async(
    html: str,
    width: int = 1280,
    height: int = 1024,
    full_page: bool = False,
) -> Image.Image:
    """Render HTML to a PIL Image using the browser pool."""
    pool = await BrowserPool.get_instance()
    png_bytes = await pool.render(html, width=width, height=height, full_page=full_page)
    if png_bytes is None:
        return _blank_white_image(width=width, height=height)
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """Return a process-wide event loop running on a daemon thread."""
    global _sync_loop
    with _sync_loop_lock:
        if _sync_loop is not None and _sync_loop.is_running():
            return _sync_loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_runner, name="vcoder-render-loop", daemon=True)
        thread.start()
        _sync_loop = loop
        return loop


def render_html_to_image(
    html: str,
    width: int = 1280,
    height: int = 1024,
    full_page: bool = False,
) -> Image.Image:
    """Synchronous wrapper: render an HTML string to a PIL Image.

    Returns a blank white image on failure.
    """
    try:
        loop = _get_sync_loop()
        future = asyncio.run_coroutine_threadsafe(
            _render_html_async(html, width=width, height=height, full_page=full_page),
            loop,
        )
        return future.result(timeout=20)
    except Exception:
        return _blank_white_image(width=width, height=height)
