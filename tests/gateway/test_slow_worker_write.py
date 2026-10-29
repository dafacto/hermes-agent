"""A slow managed-worker write is named in the gateway log while it waits, with what it waits on.

Replays Lakpa, 29 Oct 2026 11:05: workers' ``worker.persist`` calls answered about 50 s late, all at
once, while the gateway loop kept running, so nothing said what they were waiting for.
"""
import asyncio
import concurrent.futures
import logging
import sqlite3
import threading

import pytest

from gateway import slow_worker_write
from gateway.session_worker import _write_label
from gateway.slow_worker_write import timed_worker_write

LOGGER = "gateway.slow_worker_write"


@pytest.fixture(autouse=True)
def _fresh_stack_budget(monkeypatch):
    monkeypatch.setattr(slow_worker_write, "_last_stack", -float("inf"))


class _Reports(logging.Handler):
    """Signals the test once ``count`` in-progress reports were logged: the condition, not a clock."""

    def __init__(self, count=1):
        super().__init__(logging.WARNING)
        self.count, self.messages, self.seen = count, [], asyncio.Event()

    def emit(self, record):
        self.messages.append(record.getMessage())
        if sum("so far" in m for m in self.messages) >= self.count:
            self.seen.set()


@pytest.fixture
def reports():
    handlers = []

    def attach(count=1):
        handler = _Reports(count)
        logging.getLogger(LOGGER).addHandler(handler)
        handlers.append(handler)
        return handler

    logging.getLogger(LOGGER).setLevel(logging.WARNING)
    yield attach
    for handler in handlers:
        logging.getLogger(LOGGER).removeHandler(handler)


def test_a_write_held_by_another_connections_lock_names_the_sqlite_wait(tmp_path, reports):
    path = tmp_path / "state.db"
    setup = sqlite3.connect(path)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE receipts (id INTEGER PRIMARY KEY, body TEXT)")
    setup.commit()
    setup.close()

    def mutate_worker_execution_under_test():
        writer = sqlite3.connect(path, timeout=30, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO receipts (body) VALUES ('turn.renew')")
            writer.execute("COMMIT")
            return "stored"
        finally:
            writer.close()

    async def scenario(log):
        holder = sqlite3.connect(path, timeout=0, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")  # another process holding the write lock
        try:
            write = asyncio.ensure_future(timed_worker_write(
                "persist:turn.renew seq=7", mutate_worker_execution_under_test, threshold_s=0.2))
            await asyncio.wait_for(log.seen.wait(), timeout=20)
        finally:
            holder.execute("COMMIT")
            holder.close()
        return await asyncio.wait_for(write, timeout=20)

    log = reports()
    assert asyncio.run(scenario(log)) == "stored"
    report, finished = log.messages
    assert report.startswith("Slow worker write persist:turn.renew seq=7:")
    assert "mutate_worker_execution_under_test" in report and 'writer.execute("BEGIN IMMEDIATE")' in report
    assert finished.startswith("Slow worker write persist:turn.renew seq=7 finished after")
    assert "started after" in finished


def test_a_write_queued_behind_a_full_pool_says_it_is_still_waiting_for_a_thread(reports):
    gate = threading.Event()

    async def scenario(log):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        occupied = asyncio.Event()

        def occupy():
            loop.call_soon_threadsafe(occupied.set)
            gate.wait(20)

        busy = asyncio.ensure_future(asyncio.to_thread(occupy))
        await asyncio.wait_for(occupied.wait(), timeout=20)
        write = asyncio.ensure_future(timed_worker_write("persist:message.append seq=3", lambda: "stored",
                                                         threshold_s=0.2))
        try:
            await asyncio.wait_for(log.seen.wait(), timeout=20)
        finally:
            gate.set()
        await busy
        return await write

    log = reports()
    assert asyncio.run(scenario(log)) == "stored"
    report = log.messages[0]
    assert "still waiting" in report and "pool threads=1/1 queued=1" in report
    assert "stack" not in report  # nothing is running for it yet


def test_a_quick_write_logs_nothing_and_errors_pass_through(reports):
    def refused():
        raise ValueError("stale_generation")

    log = reports()
    assert asyncio.run(timed_worker_write("adopt", lambda: 42, threshold_s=5.0)) == 42
    with pytest.raises(ValueError, match="stale_generation"):
        asyncio.run(timed_worker_write("adopt", refused, threshold_s=5.0))
    assert log.messages == []


def test_only_one_stack_per_stall(reports):
    gate = threading.Event()

    async def scenario(log):
        writes = [asyncio.ensure_future(timed_worker_write(f"persist seq={n}", gate.wait, 20, threshold_s=0.2))
                  for n in range(3)]
        try:
            await asyncio.wait_for(log.seen.wait(), timeout=20)
        finally:
            gate.set()
        await asyncio.gather(*writes)

    log = reports(count=3)
    asyncio.run(scenario(log))
    in_progress = [m for m in log.messages if "so far" in m]
    assert len(in_progress) == 3
    assert sum("stack (most recent call last)" in m for m in in_progress) == 1


def test_a_caller_that_stops_waiting_is_not_reported_as_finished(reports):
    gate = threading.Event()

    async def scenario(log):
        write = asyncio.ensure_future(timed_worker_write("persist seq=9", gate.wait, 20, threshold_s=0.2))
        try:
            await asyncio.wait_for(log.seen.wait(), timeout=20)
            write.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write
        finally:
            gate.set()

    log = reports()
    asyncio.run(scenario(log))
    ending = log.messages[-1]
    assert "abandoned by its caller" in ending and "finished" not in ending


def test_a_report_that_fails_leaves_the_write_untouched(monkeypatch, reports):
    gate = threading.Event()

    def broken_report(*_):
        gate.set()
        raise RuntimeError("report failed")

    monkeypatch.setattr(slow_worker_write, "_report_slow", broken_report)
    assert asyncio.run(timed_worker_write("persist seq=1", lambda: gate.wait(20) and "stored", threshold_s=0.2)) == "stored"


def test_labels_never_carry_unvalidated_text():
    params = {"operation": "turn.renew\nsecret=1", "sequence": "7 extra", "execution_id": "x" * 40,
              "session_id": "local-abc", "payload": {"text": "never logged"}, "secret": "s3cr3t"}
    label = _write_label("persist", params)
    assert label == f"persist:? seq=? execution={'x' * 16} session=local-abc"
    assert "\n" not in label and "s3cr3t" not in label and "never logged" not in label
