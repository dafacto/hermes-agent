"""Names what a slow managed-worker write waited on. Observation only.

A worker's ``worker.persist`` / ``register`` / ``adopt`` runs off the loop (``asyncio.to_thread``)
and the worker gives up on the reply after 20 s. On Lakpa (28 and 29 Oct 2026) these writes all
answered 20-50 s late at once, while the loop itself kept running, so the loop lag watchdog had
nothing to report. This runs the same call, and once it has taken longer than the threshold logs
how long it waited for a pool thread, how long it has run, how busy the default pool is and the
running thread's stack (for example inside SQLite's busy wait), then how it ended. A failure in
the report itself never touches the write's result, error or cancellation.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional

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


def _report_slow(label: str, loop: asyncio.AbstractEventLoop, submitted: float,
                 running: Optional[tuple[float, int]]) -> None:
    now = time.monotonic()
    waited = (running[0] if running else now) - submitted
    ran = now - running[0] if running else 0.0
    stack = ""
    if running and _take_stack_slot(now):
        frame = sys._current_frames().get(running[1])
        if frame is not None:
            stack = ("; running thread stack (most recent call last):\n"
                     + "".join(traceback.format_stack(frame)[-_STACK_FRAMES:]).rstrip("\n"))
    logger.warning("Slow worker write %s: %.1fs so far (waited %.1fs for a pool thread%s, ran %.1fs; %s)%s",
                   label, now - submitted, waited, "" if running else ", still waiting", ran,
                   _pool_state(loop), stack)


async def timed_worker_write(label: str, fn: Callable[..., Any], /, *args: Any,
                             threshold_s: float = SLOW_WRITE_S, **kwargs: Any) -> Any:
    """``asyncio.to_thread(fn, *args, **kwargs)`` that reports itself past ``threshold_s``."""
    loop = asyncio.get_running_loop()
    submitted = time.monotonic()
    running: list[tuple[float, int]] = []  # one (started, thread id), published in one append

    def run() -> Any:
        running.append((time.monotonic(), threading.get_ident()))
        return fn(*args, **kwargs)

    task = asyncio.ensure_future(asyncio.to_thread(run))
    try:
        done, _ = await asyncio.wait({task}, timeout=threshold_s)
    except asyncio.CancelledError:
        task.cancel()  # as a cancelled ``await asyncio.to_thread(...)`` would be; the thread runs on
        raise
    if done:
        return task.result()
    try:
        _report_slow(label, loop, submitted, running[0] if running else None)
    except Exception:  # health: allow BLE001 -- observation must never change the write's outcome
        logger.debug("Slow worker write report failed for %s", label, exc_info=True)
    ending = "finished"
    try:
        return await task
    except asyncio.CancelledError:
        ending = "abandoned by its caller (a started write runs on)"
        task.cancel()
        raise
    finally:
        logger.warning("Slow worker write %s %s after %.1fs (%s)", label, ending,
                       time.monotonic() - submitted,
                       "started after %.1fs" % (running[0][0] - submitted) if running else "never started")
