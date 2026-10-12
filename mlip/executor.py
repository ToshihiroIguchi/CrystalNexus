"""Shared inference executor for every MLIP backend.

A single worker thread serializes model access (models are not thread-safe)
while keeping inference off the asyncio event loop. All backends and
``main.py`` share this one executor so concurrency semantics do not depend
on which model is selected.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlip")


async def run_in_inference_executor(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` on the shared single-worker executor."""
    return await asyncio.get_running_loop().run_in_executor(
        inference_executor, functools.partial(func, *args, **kwargs))
