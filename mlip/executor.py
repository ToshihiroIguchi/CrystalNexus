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

inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlip")

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


async def run_in_inference_executor(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` on the shared single-worker executor."""
    return await asyncio.get_running_loop().run_in_executor(
        inference_executor, functools.partial(func, *args, **kwargs))
