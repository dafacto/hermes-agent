"""Off-loop reporter that names what blocked the gateway event loop. Observation only.

The liveness watchdog (``shutdown_watchdog.py``) probes every 30 s and stays silent until three
probes miss, so a block of a few tens of seconds leaves no trace, yet one that long already times
out every managed worker's 20 s receipt wait. This thread posts a no-op to the loop every second;
once one has waited past the threshold it logs the loop thread's current stack, once per block and
at most once a minute, then logs how long the block lasted when the loop answers.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

LAG_THRESHOLD_S = 5.0
_PROBE_INTERVAL_S = 1.0
_POLL_S = 0.25
_MIN_REPORT_INTERVAL_S = 60.0
_STACK_FRAMES = 40


class LoopLagWatchdog:
    def __init__(self, loop: asyncio.AbstractEventLoop, loop_thread_id: int, *,
                 threshold_s: float = LAG_THRESHOLD_S, probe_interval_s: float = _PROBE_INTERVAL_S,
                 min_report_interval_s: float = _MIN_REPORT_INTERVAL_S) -> None:
        self._loop, self._loop_thread_id = loop, loop_thread_id
        self._threshold_s, self._probe_interval_s = threshold_s, probe_interval_s
        self._min_report_interval_s = min_report_interval_s
        self._stop = threading.Event()
        # health: allow HX012 -- process-wide loop observer; it reads no profile-scoped state
        self._thread = threading.Thread(target=self._run, daemon=True, name="gateway-loop-lag-watchdog")

    def start(self) -> LoopLagWatchdog:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        last_report = -float("inf")
        while not self._stop.wait(self._probe_interval_s):
            answered = threading.Event()
            posted = time.monotonic()
            try:
                self._loop.call_soon_threadsafe(answered.set)
            except RuntimeError:  # the loop closed: nothing left to watch
                return
            reported = False
            while not answered.wait(min(_POLL_S, self._threshold_s)):
                if self._stop.is_set():
                    return
                now = time.monotonic()
                if (not reported and now - posted >= self._threshold_s
                        and now - last_report >= self._min_report_interval_s):
                    self._report(now - posted)
                    reported, last_report = True, now
            if reported:
                logger.warning("Gateway event loop answered again after %.1fs blocked",
                               time.monotonic() - posted)

    def _report(self, blocked_s: float) -> None:
        frame = sys._current_frames().get(self._loop_thread_id)
        stack = ("".join(traceback.format_stack(frame)[-_STACK_FRAMES:]) if frame is not None
                 else "(loop thread not found)\n")
        logger.warning("Gateway event loop blocked for %.1fs; loop thread stack (most recent call last):"
                       "\n%s", blocked_s, stack.rstrip("\n"))


def start_loop_lag_watchdog(loop: asyncio.AbstractEventLoop) -> Optional[LoopLagWatchdog]:
    """Start watching ``loop``; call from the loop's own thread, whose stack it will report."""
    try:
        return LoopLagWatchdog(loop, threading.get_ident()).start()
    except RuntimeError:  # thread start refused (interpreter shutting down)
        logger.debug("Failed to start gateway loop lag watchdog", exc_info=True)
        return None
