"""A managed worker's turn survives an owner stall that loses one ``worker.persist`` reply.

Incident replay: the owner process stopped answering for ~22 s, both workers' turn-lease renewals
hit the 20 s receipt wait, and that single lost reply was treated as a lost lease: the turn was
hard-interrupted and the store latched the failure, so the worker died with work still assigned.
The lease had about four minutes left. A lost receipt is now resent as the same journal entry; the
owner answers an exact retry with its stored receipt, so it applies once.
"""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent import runtime_session_store as store_module
from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError, WorkerRPC
from agent.turn_facade_lease import LEASE_TTL_SECONDS, DurableTurnLease
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution, register_worker_execution,
)

HOLDER = 'pid=1:turn=t:platform=kanban'


@pytest.fixture
def fast_receipts(monkeypatch):
    """Scale the receipt wait, redial allowance and backoff from seconds to tenths."""
    monkeypatch.setattr(store_module, '_RECEIPT_TIMEOUT_S', 0.4)
    monkeypatch.setattr(store_module, '_REDIAL_ALLOWANCE_S', 0.1)
    monkeypatch.setattr(store_module, '_LOST_RECEIPT_BACKOFF_S', (0.05,))


@pytest.fixture
def owner_db(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli', system_prompt='prefix')
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
    try:
        yield db, scope
    finally:
        db.close()


def _receipts(db, execution_id='worker'):
    rows = db._read_all('SELECT sequence FROM worker_receipts WHERE execution_id=? ORDER BY sequence',
                        (execution_id,))
    return [row[0] for row in rows]


def _lease(store, interrupts, *, expires_in):
    agent = SimpleNamespace(session_id='owned',
                            interrupt=lambda message, **kwargs: interrupts.append(message))
    lease = DurableTurnLease(agent, store, 'owned', HOLDER, expires_at=time.time() + expires_in)
    lease.turn_active = True
    return lease


@pytest.fixture
def owner(monkeypatch, owner_db):
    """A WebSocket owner that commits each ``worker.persist`` in the real receipt store. ``stall``
    holds the reply to the next ``turn.renew`` (``after_commit``) or the request itself
    (``before_commit``) past the worker's receipt wait, as the stalled owner did."""
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    db, scope = owner_db
    state = SimpleNamespace(db=db, scope=scope, sent=[], connections=0, stall=None,
                            stall_s=1.0, stalled_done=threading.Event())

    def handler(ws):
        state.connections += 1
        for raw in ws:
            request = json.loads(raw)
            params = request['params']
            state.sent.append((params['sequence'], params['operation'], json.dumps(params['payload'])))
            stall, delayed = state.stall, params['operation'] == 'turn.renew' and state.stall
            if delayed:
                state.stall = None
                if stall == 'before_commit':
                    time.sleep(state.stall_s)  # health: allow HX013 -- the owner stall under test
            try:
                reply = {'result': mutate_worker_execution(db, **params)}
            except RuntimeStoreError as exc:
                reply = {'error': {'code': -32000, 'message': str(exc)}}
            if delayed and stall == 'after_commit':
                time.sleep(state.stall_s)  # health: allow HX013 -- the owner stall under test
            try:
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], **reply}))
            except Exception:  # the worker hung up on this connection: the late reply is lost
                pass
            if delayed:
                state.stalled_done.set()

    with serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}',
                                   control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'ticket')
        try:
            yield state
        finally:
            server.shutdown()


@pytest.mark.parametrize('stall', ['after_commit', 'before_commit'])
def test_owner_stall_past_the_receipt_wait_keeps_the_lease_and_the_turn(
        owner, fast_receipts, tmp_path, stall):
    rpc = WorkerRPC(tmp_path)
    store = RuntimeSessionStore(rpc, owner.scope, tmp_path / 'outbox')
    interrupts = []
    try:
        assert store.try_acquire_session_turn_lease('owned', HOLDER) is True
        lease = _lease(store, interrupts, expires_in=LEASE_TTL_SECONDS)
        before = lease._authority_deadline
        owner.stall = stall
        assert lease.refresh_tick() is None
        assert interrupts == [] and lease.interrupt_message is None
        assert lease._authority_deadline > before
        assert store.failure is None and store.journal['pending'] == []
        # The renewal (sequence 2) went out twice, byte-identical, and applied once.
        renewals = [sent for sent in owner.sent if sent[1] == 'turn.renew']
        assert len(renewals) == 2 and renewals[0] == renewals[1] and renewals[0][0] == 2
        assert owner.connections == 2
        assert owner.db.session_turn_lease_expires_at('owned', HOLDER) > time.time() + LEASE_TTL_SECONDS - 5
        # The worker carries on and finishes normally.
        assert store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'done'}]) == 1
        store.release_session_turn_lease('owned', HOLDER)
        assert store.finish()['status'] == 'terminal'
        assert owner.stalled_done.wait(5)
        assert [m['content'] for m in owner.db.get_messages('owned')] == ['done']
        assert _receipts(owner.db) == [1, 2, 3, 4, 5]
    finally:
        store.close()
        rpc.close()


class _OwnerCalls:
    """In-process owner over the real receipt store; ``faults`` decides each call's outcome."""
    def __init__(self, db, faults):
        self.db, self.faults, self.sent = db, list(faults), []

    def __call__(self, method, **params):
        self.sent.append((params['sequence'], params['operation']))
        fault = self.faults.pop(0) if self.faults else None
        if fault == 'lost_reply':  # applied, but the answer never reached the worker
            mutate_worker_execution(self.db, **params)
            raise TimeoutError('worker_receipt_timeout')
        if isinstance(fault, BaseException):
            raise fault
        return mutate_worker_execution(self.db, **params)


def test_a_lost_append_receipt_is_resent_and_lands_once(owner_db, fast_receipts, tmp_path):
    db, scope = owner_db
    owner = _OwnerCalls(db, ['lost_reply'])
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    try:
        rows = [{'role': 'user', 'content': 'once'}]
        assert store.append_messages_batch('owned', rows) == 1
        assert '_row_id' in rows[0]
        assert owner.sent == [(1, 'transcript.append'), (1, 'transcript.append')]
        assert [m['content'] for m in db.get_messages('owned')] == ['once']
        assert _receipts(db) == [1]
    finally:
        store.close()


@pytest.mark.parametrize('refusal', ['stale_epoch', 'stale_generation', 'permission_denied'])
def test_a_definitive_refusal_still_ends_the_lease_without_a_resend(
        owner_db, fast_receipts, tmp_path, refusal):
    db, scope = owner_db
    owner = _OwnerCalls(db, [None, WorkerPersistenceError(refusal)])
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    interrupts = []
    try:
        assert store.try_acquire_session_turn_lease('owned', HOLDER) is True
        lease = _lease(store, interrupts, expires_in=LEASE_TTL_SECONDS)
        assert lease.refresh_tick() is False
        assert len(interrupts) == 1
        assert owner.sent == [(1, 'turn.acquire'), (2, 'turn.renew')]
        assert store.failure == refusal  # latched: the entry stays pending for adoption
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_a_renewal_whose_holder_lost_the_row_still_ends_the_lease(owner_db, fast_receipts, tmp_path):
    db, scope = owner_db
    store = RuntimeSessionStore(_OwnerCalls(db, []), scope, tmp_path / 'outbox')
    interrupts = []
    try:
        lease = _lease(store, interrupts, expires_in=LEASE_TTL_SECONDS)  # never acquired: rowcount 0
        assert lease.refresh_tick() is False
        assert len(interrupts) == 1
    finally:
        store.close()


def test_a_stall_that_outlasts_the_lease_window_still_ends_the_lease(
        owner_db, fast_receipts, tmp_path):
    db, scope = owner_db
    owner = _OwnerCalls(db, [None] + [TimeoutError('worker_receipt_timeout')] * 100)
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    interrupts = []
    try:
        assert store.try_acquire_session_turn_lease('owned', HOLDER) is True
        # 2 s of margin before expiry leaves the renewal ~1.5 s of resends.
        expires_in = 3.5
        lease = _lease(store, interrupts, expires_in=expires_in)
        expiry = time.time() + expires_in
        assert lease.refresh_tick() is False
        assert len(interrupts) == 1
        assert time.time() < expiry  # gave up before the lease could expire under it
        renewals = owner.sent[1:]
        assert len(renewals) >= 3 and set(renewals) == {(2, 'turn.renew')}
        assert store.failure == 'worker_receipt_timeout'
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_an_unreachable_owner_on_the_first_send_is_not_resent(owner_db, fast_receipts, tmp_path):
    """Only a lost receipt starts resending; an owner that never took the request fails as before."""
    db, scope = owner_db
    owner = _OwnerCalls(db, [WorkerPersistenceError('owner_unavailable')])
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    try:
        with pytest.raises(WorkerPersistenceError, match='owner_unavailable'):
            store.append_messages_batch('owned', [{'role': 'user', 'content': 'x'}])
        assert owner.sent == [(1, 'transcript.append')]
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_a_redial_that_finds_the_owner_still_stalled_keeps_resending(owner_db, fast_receipts, tmp_path):
    db, scope = owner_db
    owner = _OwnerCalls(db, ['lost_reply', WorkerPersistenceError('owner_unavailable'),
                             ConnectionRefusedError()])
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    try:
        assert store.append_messages_batch('owned', [{'role': 'user', 'content': 'x'}]) == 1
        assert owner.sent == [(1, 'transcript.append')] * 4
        assert _receipts(db) == [1]
    finally:
        store.close()


def test_a_closed_renewal_window_sends_nothing_and_latches_nothing(owner_db, tmp_path):
    db, scope = owner_db
    owner = _OwnerCalls(db, [])
    store = RuntimeSessionStore(owner, scope, tmp_path / 'outbox')
    try:
        with pytest.raises(WorkerPersistenceError, match='receipt_window_closed'):
            store.refresh_session_turn_lease('owned', HOLDER, patience_s=0)
        assert owner.sent == [] and store.failure is None and store.journal['pending'] == []
    finally:
        store.close()
