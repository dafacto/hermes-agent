"""After the owner admits a claim, a lost connection is not the run's outcome.

Replays Lakpa, 29 Oct 2026 12:42: the gateway stalled about 10 s, the submitter's socket closed
while it polled ``prompt.receipt``, and it exited "Kanban owner refused: Gateway connection failed;
no local fallback", booking a crash for a card whose run was still going in the owner. The
submitter now reconnects and asks for the same receipt; the claim itself is never sent twice.
"""
import asyncio
import contextlib
import json

import pytest

from hermes_cli import gateway_client, kanban_worker_client
from hermes_cli.gateway_client import GatewayClient, GatewayClientError

PARAMS = {'board': 'b', 'task_id': 't1', 'run_id': 1, 'claim_lock': 'lock'}


class _Owner:
    """One connection to the owner: answers each request from ``script`` (method -> list of
    results or ``'close'``), recording what was asked; ``'close'`` hangs up without answering."""

    def __init__(self, sent, script):
        self.sent, self.script, self.inbox = sent, script, asyncio.Queue()

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append((req['method'], req['params'].get('admission_id')))
        answer = self.script[req['method']].pop(0)
        if answer == 'close':
            await self.inbox.put(None)
        elif isinstance(answer, dict) and 'error' in answer:
            await self.inbox.put(json.dumps({'id': req['id'], 'error': answer['error']}))
        else:
            await self.inbox.put(json.dumps({'id': req['id'], 'result': answer}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.inbox.get()
        if frame is None:
            raise StopAsyncIteration  # the socket closed: pending requests fail as disconnected
        return frame


def _running(status='running'):
    return {'status': status, 'admission_id': 'a1'}


@pytest.fixture
def owner_connections(monkeypatch, tmp_path):
    """Each ``connect_gateway()`` takes the next step: an ``_Owner`` script, or an exception."""
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda db, params: {'exit_code': 0, 'last_output': 'done'})
    monkeypatch.setattr(kanban_worker_client, '_RECONNECT_BACKOFF_S', (0,))
    sent, steps = [], []

    @contextlib.asynccontextmanager
    async def connect_gateway():
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        async with GatewayClient(_Owner(sent, step)) as client:
            yield client

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect_gateway)

    def run(*connections):
        steps.extend(connections)
        return asyncio.run(kanban_worker_client.run(PARAMS, str(tmp_path / 'kanban.db')))

    return run, sent


def test_a_connection_lost_after_admission_is_reconnected_and_the_same_receipt_is_asked_for(owner_connections, capsys):
    run, sent = owner_connections
    code = run(
        {'kanban.run': [{'session_id': 's1', 'receipt': _running()}], 'prompt.receipt': [_running(), 'close']},
        GatewayClientError('Gateway connection failed; no local fallback'),  # the owner still stalled
        {'prompt.receipt': [_running(), _running('terminal')]},
    )
    assert code == 0
    assert [m for m, _ in sent].count('kanban.run') == 1  # the claim is never sent again
    assert all(a == 'a1' for m, a in sent if m == 'prompt.receipt')
    err = capsys.readouterr().err
    assert err.count('Gateway connection lost while waiting for the result') == 2


def test_a_lost_answer_to_the_claim_itself_still_fails_without_resending_it(owner_connections):
    run, sent = owner_connections
    with pytest.raises(GatewayClientError, match='Gateway disconnected'):
        run({'kanban.run': ['close']}, {'kanban.run': [{'session_id': 's1', 'receipt': _running()}]})
    assert sent == [('kanban.run', None)]


def test_an_owner_refusal_or_an_unknown_run_is_an_answer_not_a_lost_connection(owner_connections):
    run, sent = owner_connections
    with pytest.raises(gateway_client.GatewayRPCError):
        run({'kanban.run': [{'session_id': 's1', 'receipt': _running()}],
             'prompt.receipt': [{'error': {'code': -32003, 'message': 'permission_denied'}}]})
    assert [m for m, _ in sent] == ['kanban.run', 'prompt.receipt']
    sent.clear()
    with pytest.raises(GatewayClientError, match='Kanban execution unknown; retained by owner'):
        run({'kanban.run': [{'session_id': 's1', 'receipt': _running()}], 'prompt.receipt': [_running('unknown')]})
    assert [m for m, _ in sent] == ['kanban.run', 'prompt.receipt']


def test_it_gives_up_once_no_answer_has_come_for_the_reconnect_window(owner_connections, monkeypatch):
    run, sent = owner_connections
    monkeypatch.setattr(kanban_worker_client, 'RESULT_RECONNECT_S', 0.0)
    with pytest.raises(GatewayClientError, match='Gateway disconnected'):
        run({'kanban.run': [{'session_id': 's1', 'receipt': _running()}], 'prompt.receipt': ['close']})
    assert [m for m, _ in sent] == ['kanban.run', 'prompt.receipt']  # no reconnect once the window is spent
