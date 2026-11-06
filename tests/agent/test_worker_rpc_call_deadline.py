"""``WorkerRPC.call_by``'s absolute deadline covers every stage of a call, the session-ticket
request and the send included (review finding M5-R3-1 on dafacto/hermes-agent#5).

The ticket wait used to start its own 5 s clock after the caller's budget was checked, and a
send to an owner that stopped reading had no bound at all. Both run here against real sockets:
a control-socket peer that never answers, and a WebSocket peer that completes the upgrade and
then never reads. No gateway or operator socket is contacted.
"""
import base64
import contextlib
import hashlib
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from agent.runtime_session_store import WorkerRPC

# Well under the 5 s the ticket used to wait regardless of the caller, with room for a busy runner.
_SLACK_S = 2.0


def test_ticket_wait_ends_at_the_call_deadline(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    peer, silent = socket.socketpair()  # The owner's control socket: takes the request, never answers.
    endpoint = SimpleNamespace(profile_id='fixture', instance_id='fixture', control_home=None)
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
    monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(discovery, 'connect_private', lambda home, timeout: peer)
    rpc = WorkerRPC(tmp_path)
    budget = 0.2
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as raised:
            rpc.call_by(started + budget, 'worker.persist', fixture=True)
        elapsed = time.monotonic() - started
    finally:
        silent.close()
        peer.close()
        rpc.close()
    assert raised.value.args == ('worker_call_deadline',)  # Nothing was sent: no receipt can be lost.
    assert elapsed < budget + _SLACK_S


class _DeafOwner:
    """A WebSocket peer that completes the upgrade and then reads nothing more, as an owner whose
    loop has stalled: the worker's send fills the socket buffers and blocks."""

    def __init__(self):
        self.listener = socket.create_server(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.accepted = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        with contextlib.suppress(OSError):
            conn, _ = self.listener.accept()
            self.accepted.append(conn)
            request = b''
            while b'\r\n\r\n' not in request:
                request += conn.recv(4096)
            key = next(line.split(b':', 1)[1].strip() for line in request.split(b'\r\n')
                       if line.lower().startswith(b'sec-websocket-key:'))
            accept = base64.b64encode(hashlib.sha1(key + b'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest())
            conn.sendall(b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                         b'Sec-WebSocket-Accept: ' + accept + b'\r\n\r\n')

    def close(self):
        for sock in (*self.accepted, self.listener):
            with contextlib.suppress(OSError):
                sock.close()


def test_a_send_the_owner_never_takes_ends_at_the_call_deadline(tmp_path, monkeypatch):
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    owner = _DeafOwner()
    endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{owner.port}', control_home=None)
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
    monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose, deadline=None: 'ticket')
    for name in ('NO_PROXY', 'no_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy',
                 'ALL_PROXY', 'all_proxy'):
        monkeypatch.delenv(name, raising=False)
    rpc = WorkerRPC(tmp_path)
    budget = 0.5
    outcome = {}

    def call():
        started = time.monotonic()
        try:
            # Far more than loopback buffers hold, so the send blocks until the owner reads.
            rpc.call_by(started + budget, 'worker.persist', payload='x' * (64 * 1024 * 1024))
        except BaseException as exc:  # recorded for the assertions below
            outcome['error'] = exc
        outcome['elapsed'] = time.monotonic() - started

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        caller.join(budget + _SLACK_S + 3)
        finished = not caller.is_alive()
    finally:
        owner.close()  # Frees a send left blocked (the unfixed behaviour), so the thread ends.
        caller.join(10)
        rpc.close()
    assert finished, 'the send outlived the call deadline'
    assert outcome['elapsed'] < budget + _SLACK_S
    # Part of the frame may have gone: the outcome is unknown, so only the journal's exact retry
    # may send it again (a lost receipt), and the connection is dropped for the next call.
    error = outcome['error']
    assert isinstance(error, TimeoutError) and error.args == ('worker_receipt_timeout',)
    assert rpc.link is None
