"""A blocked gateway loop is named in the log while it is blocked, once per block, rate-limited."""
import asyncio
import logging
import threading
import time

import pytest

from gateway.loop_lag_watchdog import LoopLagWatchdog
from tests.waiting import wait_until


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder = {}

    def run():
        holder['ident'] = threading.get_ident()
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5)
    try:
        yield loop, holder['ident']
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def _block_loop_in_a_named_frame(loop, seconds):
    done = threading.Event()

    def wedged_gateway_callback():
        time.sleep(seconds)  # health: allow HX013 -- the block under test
        done.set()
    loop.call_soon_threadsafe(wedged_gateway_callback)
    return done


def _lag_records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'gateway.loop_lag_watchdog']


def test_a_block_past_the_threshold_logs_the_loop_threads_stack(loop_thread, caplog):
    loop, ident = loop_thread
    caplog.set_level(logging.WARNING, logger='gateway.loop_lag_watchdog')
    watchdog = LoopLagWatchdog(loop, ident, threshold_s=0.3, probe_interval_s=0.05).start()
    try:
        assert _block_loop_in_a_named_frame(loop, 0.9).wait(5)
        wait_until(lambda: len(_lag_records(caplog)) >= 2, timeout=5)
    finally:
        watchdog.stop()
    blocked, resumed = _lag_records(caplog)
    assert blocked.startswith('Gateway event loop blocked for')
    assert 'wedged_gateway_callback' in blocked
    assert resumed.startswith('Gateway event loop answered again after')


def test_a_second_block_inside_the_report_interval_is_not_reported_again(loop_thread, caplog):
    loop, ident = loop_thread
    caplog.set_level(logging.WARNING, logger='gateway.loop_lag_watchdog')
    watchdog = LoopLagWatchdog(loop, ident, threshold_s=0.2, probe_interval_s=0.05,
                               min_report_interval_s=60).start()
    try:
        assert _block_loop_in_a_named_frame(loop, 0.6).wait(5)
        wait_until(lambda: any(m.startswith('Gateway event loop answered again')
                               for m in _lag_records(caplog)), timeout=5)
        # Three times the threshold: an unthrottled watchdog reports while this block lasts.
        assert _block_loop_in_a_named_frame(loop, 0.6).wait(5)
    finally:
        watchdog.stop()
    assert sum(m.startswith('Gateway event loop blocked for') for m in _lag_records(caplog)) == 1


def test_a_responsive_loop_logs_nothing(loop_thread, caplog):
    loop, ident = loop_thread
    caplog.set_level(logging.WARNING, logger='gateway.loop_lag_watchdog')
    watchdog = LoopLagWatchdog(loop, ident, threshold_s=0.3, probe_interval_s=0.02).start()
    try:
        for _ in range(10):  # ordinary awaiting work, well over the threshold in total
            # health: allow HX013 -- the non-blocking loop activity under test
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), loop).result(5)
    finally:
        watchdog.stop()
    assert _lag_records(caplog) == []


def test_the_watchdog_ends_when_its_loop_closes():
    loop = asyncio.new_event_loop()
    watchdog = LoopLagWatchdog(loop, threading.get_ident(), probe_interval_s=0.02).start()
    loop.close()
    watchdog._thread.join(5)
    assert not watchdog.is_alive()
