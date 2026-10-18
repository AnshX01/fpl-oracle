"""Cross-process and in-process bundle exclusion for inference, load, promotion and rollback."""

from functools import wraps
from pathlib import Path
from threading import RLock

from filelock import FileLock

from fpl_oracle.config import MODELS_DIR

BUNDLE_THREAD_LOCK = RLock()


_LOCK_CACHE: dict[Path, FileLock] = {}


def get_bundle_file_lock(target_dir: Path | str | None = None, timeout: float = 30.0) -> FileLock:
    """Return an OS-level file lock anchored to target_dir or MODELS_DIR."""
    base_dir = (Path(target_dir) if target_dir else MODELS_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_file = base_dir / "bundle.lock"
    if lock_file not in _LOCK_CACHE:
        _LOCK_CACHE[lock_file] = FileLock(str(lock_file), timeout=timeout)
    return _LOCK_CACHE[lock_file]


def bundle_locked(method):
    """
    Decorator acquiring both in-process thread lock and cross-process file lock.
    Safely serializes promotion, loading, and rollback across threads and OS processes.
    """

    @wraps(method)
    def wrapped(*args, **kwargs):
        target_dir = None
        if args and hasattr(args[0], "models_dir"):
            target_dir = args[0].models_dir

        file_lock = get_bundle_file_lock(target_dir)
        with BUNDLE_THREAD_LOCK:
            with file_lock:
                return method(*args, **kwargs)

    return wrapped
