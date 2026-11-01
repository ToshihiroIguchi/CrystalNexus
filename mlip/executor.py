"""Shared inference executor for every MLIP backend.

A single worker thread serializes model access (models are not thread-safe)
while keeping inference off the asyncio event loop. All backends and
``main.py`` share this one executor so concurrency semantics do not depend
on which model is selected.
"""
import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor

class _RestartableExecutor(ThreadPoolExecutor):
    """Single-worker pool that can be revived after ``shutdown``.

    Application shutdown closes the pool; a later startup in the same process
    (reload, repeated TestClient lifespans) calls ``ensure_running``.
    """

    def __init__(self):
        super().__init__(max_workers=1, thread_name_prefix="mlip")

    def ensure_running(self) -> None:
        if self._shutdown:
            self.__init__()


inference_executor = _RestartableExecutor()

# Set at application shutdown so a running relaxation / elastic job stops at
# its next observer call instead of keeping the non-daemon worker alive.
_abort_event = threading.Event()


def request_abort() -> None:
    """Ask running inference jobs to stop cooperatively (idempotent)."""
    _abort_event.set()


def clear_abort() -> None:
    _abort_event.clear()


def abort_requested() -> bool:
    return _abort_event.is_set()


def is_aborted(cancel_event=None) -> bool:
    """True if the global shutdown flag or this run's own cancel event is set.

    ``cancel_event`` is a per-request ``threading.Event`` (e.g. set when the
    HTTP client disconnected); it never touches the global shutdown flag.
    """
    return _abort_event.is_set() or (cancel_event is not None and cancel_event.is_set())


def abort_reason(cancel_event=None, default="timeout") -> str:
    """Why a run stopped: "shutdown" > "client_disconnected" > ``default``."""
    if _abort_event.is_set():
        return "shutdown"
    if cancel_event is not None and cancel_event.is_set():
        return "client_disconnected"
    return default


async def run_in_inference_executor(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` on the shared single-worker executor."""
    return await asyncio.get_running_loop().run_in_executor(
        inference_executor, functools.partial(func, *args, **kwargs))
