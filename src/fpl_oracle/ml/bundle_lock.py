"""In-process bundle exclusion for inference, load, promotion and rollback."""

from functools import wraps
from threading import RLock

BUNDLE_LOCK = RLock()


def bundle_locked(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with BUNDLE_LOCK:
            return method(*args, **kwargs)

    return wrapped
