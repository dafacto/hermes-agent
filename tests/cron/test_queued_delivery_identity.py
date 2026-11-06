"""A queued cron send carries its execution's identity into the gateway drain.

Replays the Lakpa incident (26 Oct 2026): every agent fire hands its send to the durable delivery
queue, and the gateway drains it later from its housekeeping thread. The drain ran outside any cron
identity, so a platform that admits a report only for its own job (Oi scheduled reports:
``ctx.current_cron_execution()``) refused every queued run with "lacks exact authorized job
identity" — both the live lane and the standalone fallback.
"""
import asyncio
import threading
from unittest.mock import patch

import pytest

from cron import delivery_queue, scheduler
from cron.execution_identity import current_cron_execution
from cron.executions import create_execution, finish_execution, get_execution, mark_execution_running
from gateway.config import Platform
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture()
def home(tmp_path):
    root = tmp_path / "home"
    (root / "cron").mkdir(parents=True)
    token = set_hermes_home_override(str(root))
    try:
        yield root
    finally:
        reset_hermes_home_override(token)


def _start_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


@pytest.fixture()
def gateway_loop():
    loop, thread = _start_loop()
    try:
        yield loop
    finally:
        _stop_loop(loop, thread)


class _IdentityProbeAdapter:
    """A live adapter that, like the Oi reports platform, reads the cron identity at send time."""

    def __init__(self):
        self.seen = []

    async def send(self, chat_id, content, metadata=None):
        self.seen.append((chat_id, current_cron_execution()))
        return {"success": True, "message_id": "m1"}


def _running_execution(job_id: str) -> str:
    execution = create_execution(job_id, source="builtin")
    mark_execution_running(execution["id"])
    return execution["id"]


def _queue(execution_job_id: str, queued_job_id: str) -> str:
    execution_id = _running_execution(execution_job_id)
    delivery_queue.enqueue(
        execution_id,
        {"id": queued_job_id, "name": "probe job", "deliver": "discord:C1",
         "execution_id": execution_id},
        "report body")
    return execution_id


def _drain(adapter, loop):
    with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
        return scheduler.drain_delivery_queue({Platform.DISCORD: adapter}, loop)


def test_queued_send_sees_its_own_execution_identity(home, gateway_loop):
    execution_id = _queue("job1", "job1")
    adapter = _IdentityProbeAdapter()

    assert _drain(adapter, gateway_loop) == 1

    ((chat_id, ident),) = adapter.seen
    assert chat_id == "C1"
    assert ident is not None, "queued cron send was drained without its cron identity"
    assert ident.job_id == "job1"
    assert ident.job_name == "probe job"
    assert ident.execution_id == execution_id
    assert ident.source == "builtin"
    # Bound only for that send: the housekeeping thread is not left inside the fire.
    assert current_cron_execution() is None
    assert delivery_queue.get_status(execution_id)["status"] == "delivered"


def test_queued_send_for_another_jobs_execution_binds_nothing(home, gateway_loop):
    """The ledger, not the queued payload, decides: a row naming an execution of another job
    sends with no identity, so identity-gated platforms fail closed."""
    _queue("other-job", "job1")
    adapter = _IdentityProbeAdapter()

    _drain(adapter, gateway_loop)

    ((_, ident),) = adapter.seen
    assert ident is None


def test_queued_send_without_a_ledger_row_binds_nothing(home, gateway_loop):
    delivery_queue.enqueue(
        "missing-exec", {"id": "job1", "name": "probe job", "deliver": "discord:C1",
                         "execution_id": "missing-exec"}, "report body")
    adapter = _IdentityProbeAdapter()

    _drain(adapter, gateway_loop)

    ((_, ident),) = adapter.seen
    assert ident is None


def test_identity_survives_a_replayed_delivery_after_the_worker_gave_up(home, monkeypatch):
    """Delivery replay: the restart-safe worker queues its send through the real worker lane
    (``_deliver_result`` with no adapters), no gateway drains it inside the worker's wait (the
    owner is restarting), the worker leaves the row pending and finishes its execution. The
    replacement gateway (a fresh loop) drains it later and the send still carries the fire's
    identity, read from the ledger rather than from anything the first process held."""
    execution_id = _running_execution("job1")
    job = {"id": "job1", "name": "probe job", "deliver": "discord:C1", "execution_id": execution_id}
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution_id)
    monkeypatch.setattr(delivery_queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0.2)
    with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
        # The worker's own wait times out with the row still pending: a deferral, not a failure.
        assert scheduler._deliver_result(dict(job), "report body", adapters=None, loop=None) is None
    assert delivery_queue.get_status(execution_id)["status"] == "pending"
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER")
    finish_execution(execution_id, success=True)
    assert get_execution(execution_id)["status"] == "completed"

    replacement_loop, replacement_thread = _start_loop()
    adapter = _IdentityProbeAdapter()
    try:
        assert _drain(adapter, replacement_loop) == 1
    finally:
        _stop_loop(replacement_loop, replacement_thread)

    ((chat_id, ident),) = adapter.seen
    assert chat_id == "C1"
    assert ident is not None, "replayed queued send lost its cron identity"
    assert (ident.job_id, ident.execution_id) == ("job1", execution_id)
    assert delivery_queue.get_status(execution_id)["status"] == "delivered"
