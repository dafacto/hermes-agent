"""Which cron execution the current code runs inside, for plugins (``ctx.current_cron_execution()``).

The scheduler binds it once it has won the execution's ``claimed`` → ``running`` transition (or a
restart-safe worker adopted it) and resets it when the fire ends. A fire's send that waits in the
durable delivery queue is drained by the gateway inside that same identity, bound from the
execution ledger row (``queued_delivery_identity``). It is a ContextVar, so it follows
the agent onto the threads the scheduler already runs it on via ``contextvars.copy_context()`` and
never leaks to another concurrently firing job or profile. Model output cannot reach it: tool
subprocesses do not share the interpreter. Delegated children read ``None`` — they are not the
scheduled occurrence.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import logging
import sqlite3
from typing import Any, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CronExecution:
    job_id: str
    job_name: str
    execution_id: str
    source: str  # "builtin" (ticker), "direct" (manual run / dashboard Run now), provider name...
    scheduled_instant: Optional[str]  # the occurrence fired; None for an off-schedule run
    started_at: Optional[str]
    profile: str


_CURRENT: ContextVar[Optional[CronExecution]] = ContextVar("hermes_cron_execution", default=None)


def enter_cron_execution(job: Mapping[str, Any], execution_id: str, record: Mapping[str, Any]) -> Token:
    """Bind this fire's identity from *record*, the ledger row of the running transition."""
    from hermes_cli.profiles import current_profile_name

    return _CURRENT.set(CronExecution(
        job_id=str(job["id"]),
        job_name=str(job.get("name") or job["id"]),
        execution_id=str(execution_id),
        source=str(record.get("source") or ""),
        scheduled_instant=record.get("scheduled_instant"),
        started_at=record.get("started_at"),
        profile=current_profile_name("default") or "default",
    ))


def exit_cron_execution(token: Optional[Token]) -> None:
    if token is not None:
        _CURRENT.reset(token)


@contextmanager
def queued_delivery_identity(queued_job: Mapping[str, Any]) -> Iterator[None]:
    """Bind, for one queued send, the cron identity of the execution it belongs to.

    The queued send IS that fire's delivery, so platforms that admit a report only for its own job
    (``ctx.current_cron_execution()``) must see the fire, not the gateway's housekeeping thread.
    The identity comes from the execution ledger, never from the queued payload alone: a row whose
    execution is absent or belongs to another job binds nothing, so identity-gated platforms fail
    closed.
    """
    token = None
    execution_id = str(queued_job.get("execution_id") or "")
    job_id = str(queued_job.get("id") or "")
    if execution_id and job_id:
        from cron.executions import get_execution

        try:
            record = get_execution(execution_id)
        except (OSError, sqlite3.Error):
            logger.warning("Queued delivery %s: execution lookup failed; sending without cron identity",
                           execution_id, exc_info=True)
            record = None
        if record and str(record.get("job_id") or "") == job_id:
            token = enter_cron_execution(queued_job, execution_id, record)
        else:
            logger.warning("Queued delivery %s: no matching execution for job %s; sending without cron "
                           "identity", execution_id, job_id)
    try:
        yield
    finally:
        exit_cron_execution(token)


def current_cron_execution() -> Optional[CronExecution]:
    from agent.delegation_context import is_delegated_child_context

    return None if is_delegated_child_context() else _CURRENT.get()
