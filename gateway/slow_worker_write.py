"""Names what a slow managed-worker write waited on. Observation only.

A worker's ``worker.persist`` / ``register`` / ``adopt`` runs off the loop (``asyncio.to_thread``)
and the worker gives up on the reply after 20 s. On Lakpa (28 and 29 Oct 2026) these writes all
answered 20-50 s late at once, while the loop itself kept running, so the loop lag watchdog had
nothing to report. This runs the same call, and once it has taken longer than the threshold logs
how long it waited for a pool thread, how long it has run, how busy the default pool is and the
running thread's stack (for example inside SQLite's busy wait), then how long it took in all.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)

SLOW_WRITE_S = 5.0
_STACK_FRAMES = 40
_MIN_STACK_INTERVAL_S = 30.0
# One stack per stall is enough: every write stuck behind the same holder shows the same frames.
_last_stack = -float("inf")
_stack_lock = threading.Lock()


def _pool_state(loop: asyncio.AbstractEventLoop) -> str:
    executor = getattr(loop, "_default_executor", None)
    threads, queued = getattr(executor, "_threads", None), getattr(executor, "_work_queue", None)
    if threads is None or queued is None:
        return "pool unknown"
    return f"pool threads={len(threads)}/{getattr(executor, '_max_workers', '?')} queued={queued.qsize()}"


def _take_stack_slot(now: float) -> bool:
    global _last_stack
    with _stack_lock:
        if now - _last_stack < _MIN_STACK_INTERVAL_S:
            return False
        _last_stack = now
        return True


async def timed_worker_write(label: str, fn: Callable[..., Any], /, *args: Any,
                             threshold_s: float = SLOW_WRITE_S, **kwargs: Any) -> Any:
    """``asyncio.to_thread(fn, *args, **kwargs)`` that reports itself past ``threshold_s``."""
    loop = asyncio.get_running_loop()
    submitted = time.monotonic()
    state: dict[str, Any] = {}

    def run() -> Any:
        state["started"], state["ident"] = time.monotonic(), threading.get_ident()
        return fn(*args, **kwargs)

    task = asyncio.ensure_future(asyncio.to_thread(run))
    try:
        done, _ = await asyncio.wait({task}, timeout=threshold_s)
    except asyncio.CancelledError:
        task.cancel()  # as a cancelled ``await asyncio.to_thread(...)`` would be; the thread runs on
        raise
    if done:
        return task.result()
    now = time.monotonic()
    started = state.get("started")
    waited = (started if started is not None else now) - submitted
    ran = now - started if started is not None else 0.0
    stack = ""
    if started is not None and _take_stack_slot(now):
        frame = sys._current_frames().get(state["ident"])
        if frame is not None:
            stack = ("; running thread stack (most recent call last):\n"
                     + "".join(traceback.format_stack(frame)[-_STACK_FRAMES:]).rstrip("\n"))
    logger.warning("Slow worker write %s: %.1fs so far (waited %.1fs for a pool thread%s, ran %.1fs; %s)%s",
                   label, now - submitted, waited, "" if started is not None else ", still waiting",
                   ran, _pool_state(loop), stack)
    try:
        return await task
    finally:
        started = state.get("started")
        logger.warning("Slow worker write %s finished after %.1fs (waited %.1fs for a pool thread)",
                       label, time.monotonic() - submitted,
                       (started if started is not None else time.monotonic()) - submitted)
