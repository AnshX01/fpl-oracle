"""Request-scoped search progress. Context follows asyncio.to_thread workers."""

from contextvars import ContextVar

search_progress = ContextVar("search_progress", default=None)


def report_search_progress(stage: str):
    callback = search_progress.get()
    if callback is not None:
        callback(stage)
