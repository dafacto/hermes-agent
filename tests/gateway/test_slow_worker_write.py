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
from gateway.slow_worker_write import timed_worker_write


@pytest.fixture(autouse=True)
def _fresh_stack_budget(monkeypatch):
    monkeypatch.setattr(slow_worker_write, "_last_stack", -float("inf"))


def _messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "gateway.slow_worker_write"]


def test_a_write_held_by_another_connections_lock_names_the_sqlite_wait(tmp_path, caplog):
    path = tmp_path / "state.db"
    setup = sqlite3.connect(path)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE receipts (id INTEGER PRIMARY KEY, body TEXT)")
    setup.commit()
    setup.close()
    holder = sqlite3.connect(path, timeout=0, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")  # another process holding the write lock

    def mutate_worker_execution_under_test():
        writer = sqlite3.connect(path, timeout=10, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO receipts (body) VALUES ('turn.renew')")
            writer.execute("COMMIT")
            return "stored"
        finally:
            writer.close()

    release = threading.Timer(0.6, lambda: holder.execute("COMMIT"))
    release.start()
    with caplog.at_level(logging.WARNING, logger="gateway.slow_worker_write"):
        result = asyncio.run(timed_worker_write("persist:turn.renew seq=7", mutate_worker_execution_under_test,
                                                threshold_s=0.2))
    release.join()
    holder.close()

    assert result == "stored"
    report, finished = _messages(caplog)
    assert report.startswith("Slow worker write persist:turn.renew seq=7:")
    assert "waited 0.0s for a pool thread" in report
    assert "mutate_worker_execution_under_test" in report and 'writer.execute("BEGIN IMMEDIATE")' in report
    assert finished.startswith("Slow worker write persist:turn.renew seq=7 finished after")


def test_a_write_queued_behind_a_full_pool_says_it_is_still_waiting_for_a_thread(caplog):
    gate = threading.Event()

    async def scenario():
        asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        busy = asyncio.ensure_future(asyncio.to_thread(gate.wait, 5))
        await asyncio.sleep(0.05)
        threading.Timer(0.5, gate.set).start()
        result = await timed_worker_write("persist:message.append seq=3", lambda: "stored", threshold_s=0.2)
        await busy
        return result

    with caplog.at_level(logging.WARNING, logger="gateway.slow_worker_write"):
        assert asyncio.run(scenario()) == "stored"
    report = _messages(caplog)[0]
    assert "still waiting" in report and "pool threads=1/1 queued=1" in report
    assert "stack" not in report  # nothing is running for it yet


def test_a_quick_write_logs_nothing_and_errors_pass_through(caplog):
    def refused():
        raise ValueError("stale_generation")

    with caplog.at_level(logging.WARNING, logger="gateway.slow_worker_write"):
        assert asyncio.run(timed_worker_write("adopt", lambda: 42, threshold_s=1.0)) == 42
        with pytest.raises(ValueError, match="stale_generation"):
            asyncio.run(timed_worker_write("adopt", refused, threshold_s=1.0))
    assert _messages(caplog) == []


def test_only_one_stack_per_stall(caplog):
    gate = threading.Event()

    async def scenario():
        threading.Timer(0.5, gate.set).start()
        return await asyncio.gather(*(timed_worker_write(f"persist seq={n}", gate.wait, 5, threshold_s=0.2)
                                      for n in range(3)))

    with caplog.at_level(logging.WARNING, logger="gateway.slow_worker_write"):
        asyncio.run(scenario())
    reports = [m for m in _messages(caplog) if "so far" in m]
    assert len(reports) == 3
    assert sum("stack (most recent call last)" in m for m in reports) == 1
