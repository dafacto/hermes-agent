"""Execution-scoped worker operations on the ordinary authenticated authority.

Only explicit local compute registrations are enabled here. A delegated child is reserved by
the owner on its parent's receipt stream (gateway/session_worker_child.py), never registered.
"""
import hmac
import os
import re

from gateway.session_admission import admission_fingerprint
from gateway.slow_worker_write import timed_worker_write
from hermes_state_runtime import (
    RuntimeStoreError, _secret_digest, adopt_worker_execution,
    mutate_worker_execution, register_worker_execution,
)

_SCOPE = {'profile_id', 'session_id', 'execution_id', 'generation', 'pid', 'birth', 'secret'}


def _claim(connection, ref, params):
    import psutil
    authority = connection.authority
    if params.get('profile_id') != authority.profile_id or connection.actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if not connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    pid, birth = params.get('pid'), params.get('birth')
    secret = params.get('secret')
    if (type(pid) is not int or pid <= 0 or pid == os.getpid()
            or type(birth) not in (int, float) or not isinstance(secret, str) or not 16 <= len(secret) <= 256):
        raise RuntimeStoreError('invalid_params')
    try:
        process = psutil.Process(pid)
        from hermes_cli.process_identity import process_birth
        if (process_birth(process) != birth or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
                or process.username() != psutil.Process().username()):
            raise RuntimeStoreError('permission_denied')
    except psutil.Error as exc:
        raise RuntimeStoreError('worker_not_live') from exc
    # Reuse the durable adoption digest as the producer claim. Changing any
    # assignment, principal, profile, PID birth or secret cannot adopt it.
    from hermes_state_worker_compression import worker_claim_origin
    origin = worker_claim_origin(authority.db, params)
    return admission_fingerprint(canonical_target=origin, payload={
        key: params[key] for key in _SCOPE} | {'session_id': origin, 'principal': connection.actor.subject})


def _verify(connection, ref, params, claim):
    from hermes_state_worker_compression import worker_retry_target
    authority = connection.authority
    target = worker_retry_target(authority.db, params)
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                           (params['execution_id'],)).fetchone()
        if row is None:
            raise RuntimeStoreError('not_found')
        if row['session_id'] != target:
            raise RuntimeStoreError('permission_denied')
        if type(params['generation']) is not int or row['generation'] != params['generation']:
            raise RuntimeStoreError('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], _secret_digest(claim)):
            raise RuntimeStoreError('permission_denied')


_LABEL_PART = re.compile(r'[A-Za-z0-9_.:-]{1,64}')


def _label_part(value):
    """A label field as given when it is a short plain token, else '?': labels are formatted before
    the write validates its fields, so they never carry arbitrary text or a newline into the log."""
    text = str(value)
    return text if _LABEL_PART.fullmatch(text) else '?'


def _write_label(operation, params):
    """What a slow-write report names: never the claim, the payload or a secret."""
    detail = (f":{_label_part(params.get('operation'))} seq={_label_part(params.get('sequence'))}"
              if operation == 'persist' else '')
    return (f"{operation}{detail} execution={_label_part(params.get('execution_id'))[:16]}"
            f" session={_label_part(params.get('session_id'))[:24]}")


async def worker_request(connection, ref, params, *, operation):
    from gateway.session_authorities import require_serving
    require_serving(connection.authority)  # adopt/persist would write with the released epoch
    if operation == "persist" and isinstance(params.get("operation"), str) and params["operation"].startswith("delegation."):
        from gateway.session_worker_delegation import worker_delegation_request
        return await worker_delegation_request(connection, ref, params)
    if operation == "persist" and params.get("operation") == "child.reserve":
        from gateway.session_worker_child import worker_child_request
        return await worker_child_request(connection, ref, params)
    extra = {'register': {'kind'}, 'adopt': set(), 'persist': {'epoch', 'sequence', 'operation', 'payload'}}[operation]
    if set(params) != _SCOPE | extra:
        raise RuntimeStoreError('invalid_params')
    claim = _claim(connection, ref, params)
    authority = connection.authority
    if operation == 'register':
        authority._require_admission_open()
    scope = {key: params[key] for key in ('execution_id', 'session_id', 'generation')}
    if operation == 'register':
        # Only a worker-adoption ticket (what a launched worker carries): an interactive client
        # must not park a 'registered' row that nothing reaps (the FIFO would pause until restart).
        authority.authorize(connection.actor, ref, 'worker:adopt')
        if params['kind'] != 'compute':
            raise RuntimeStoreError('unsupported_producer')
        return await timed_worker_write(_write_label('register', params), register_worker_execution,
            authority.db, epoch=authority.epoch, **scope, kind='compute', adoption_secret=claim, require_idle=True)
    if 'worker:adopt' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    _verify(connection, ref, params, claim)
    if operation == 'adopt':
        return await timed_worker_write(_write_label('adopt', params), adopt_worker_execution, authority.db,
                                        epoch=authority.epoch, **scope, adoption_secret=claim)
    result = await timed_worker_write(_write_label('persist', params), mutate_worker_execution, authority.db,
        epoch=params['epoch'], **scope, sequence=params['sequence'], operation=params['operation'],
        payload=params['payload'])
    if params['operation'] == 'execution.finish':
        # Queued input parked behind this worker; nothing else would restart its FIFO.
        authority.wake_after_worker(params['session_id'])
    return result
