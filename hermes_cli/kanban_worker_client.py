"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import os
from pathlib import Path
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0
# Once the owner has admitted the claim, the run lives in the owner: a dropped connection or a late
# answer while this waits for its receipt is not the run's outcome. Lakpa, 29 Oct 2026 12:42: a
# 10 s gateway stall closed this socket and the submitter exited "Gateway connection failed",
# booking a crash for a card whose run was still going. It reconnects and asks for the same
# receipt (a read; the claim is never sent again) until an answer arrives, or none has for this long.
# A soft bound, checked after each failure: counted from the last answer (or the first failure, before
# any), and one back-off plus one reconnect attempt can run past it.
RESULT_RECONNECT_S = 300.0
_RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0)


class _Settled(Exception):
    """The owner's own answer about the admitted run: an outcome, never retried."""


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


async def run(params, board_db):
    from hermes_cli.gateway_client import (
        connect_gateway, GatewayClientError, GatewayRPCError, GatewayUnavailableError,
    )
    from websockets.exceptions import ConnectionClosed
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    deadline = time.monotonic() + OWNER_WAIT_S
    waiting = _ResultWait()
    while True:
        try:
            async with connect_gateway() as client:
                # kanban.run subscribes this socket to the card's session, but only receipts matter
                # here: discard the turn's events, or they fill the client's bounded queue and its
                # reader stops, with the next receipt behind it.
                drain = asyncio.create_task(_discard(client.events))
                try:
                    if waiting.admitted is None:
                        waiting.admitted = await _admit_claim(client, params, board_db)
                    return await _await_result(client, waiting, params, board_db)
                finally:
                    drain.cancel()
        except _Settled as exc:
            raise GatewayClientError(str(exc)) from None
        except GatewayRPCError:
            raise  # the owner refused: an answer, not a lost one
        except GatewayUnavailableError as exc:
            if waiting.admitted is None:
                if exc.state not in _TRANSIENT_OWNER_STATES or time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(1)
                continue
            await waiting.lost(exc)
        except (GatewayClientError, ConnectionClosed, OSError) as exc:  # TimeoutError is an OSError
            if waiting.admitted is None:
                raise  # kanban.run's own answer was lost: its outcome is unknown, never resend it
            await waiting.lost(exc)


class _ResultWait:
    """The admitted run (``(session_id, admission_id)``) and how long no answer has come for it."""

    def __init__(self):
        self.admitted, self.last_answer, self.attempt = None, None, 0

    def answered(self):
        self.last_answer, self.attempt = time.monotonic(), 0

    async def lost(self, exc):
        if self.last_answer is None:
            self.last_answer = time.monotonic()
        if time.monotonic() - self.last_answer >= RESULT_RECONNECT_S:
            raise exc
        delay = _RECONNECT_BACKOFF_S[min(self.attempt, len(_RECONNECT_BACKOFF_S) - 1)]
        self.attempt += 1
        print(f'Gateway connection lost while waiting for the result ({type(exc).__name__}); '
              f'asking again in {delay:g}s', file=sys.stderr, flush=True)
        await asyncio.sleep(delay)


async def _discard(events):
    while True:
        await events.get()


async def _admit_claim(client, params, board_db):
    db = str(await asyncio.to_thread(Path(board_db).resolve))
    # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
    # in one board write transaction. The board (the cross-profile dispatch bus) can make that
    # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
    # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
    # admission was left ``unknown``). The wait is still finite: once the marker commits,
    # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
    # that stalls after admitting is bounded only here. A hang-up fails the wait sooner.
    accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
    sid, receipt = accepted['session_id'], accepted['receipt']
    print(f'Session: {sid}', file=sys.stderr)
    return sid, receipt


async def _await_result(client, waiting, params, board_db):
    sid, receipt = waiting.admitted
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await client.rpc('prompt.receipt', session_id=sid, admission_id=receipt['admission_id'])
        waiting.admitted = sid, receipt
        waiting.answered()
    if receipt['status'] == 'unknown':
        raise _Settled('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


async def _submit_claim(client, params, board_db):
    """Admit the claim and wait for its result on one connection (``run`` reconnects across both)."""
    waiting = _ResultWait()
    waiting.admitted = await _admit_claim(client, params, board_db)
    try:
        return await _await_result(client, waiting, params, board_db)
    except _Settled as exc:
        from hermes_cli.gateway_client import GatewayClientError
        raise GatewayClientError(str(exc)) from None


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
