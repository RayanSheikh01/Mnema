from __future__ import annotations

from pathlib import Path
import os
import tempfile
import time
from contextlib import contextmanager


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".md", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@contextmanager
def file_lock(lock_path: Path, timeout_seconds: float = 5.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd = -1
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"timed out acquiring lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd >= 0:
            os.close(fd)
        if lock_path.exists():
            lock_path.unlink()
