"""Request-scoped upstream reads, with end-of-request revision validation."""

import copy
from contextvars import ContextVar

read_context: ContextVar[dict | None] = ContextVar("fpl_read_context", default=None)


def remember(key, data, stale, timestamp):
    context = read_context.get()
    if context is not None:
        context[key] = (copy.deepcopy(data), stale, timestamp)


def recalled(key):
    context = read_context.get()
    if context is not None and key in context:
        data, stale, _ = context[key]
        return copy.deepcopy(data), stale
    return None
