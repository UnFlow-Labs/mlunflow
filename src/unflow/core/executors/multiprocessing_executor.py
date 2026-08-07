from __future__ import annotations

import os
import sys
import time
from collections import deque

import cloudpickle

from ...core.executors.executor import Executor
from ...core.unflow_types import Job, RState


def _resolve_context():
    """Use fork on macOS to avoid spawn re-import issues."""
    import multiprocessing as mp

    if os.name == "posix" and sys.platform == "darwin":
        return mp.get_context("fork")
    return mp.get_context()


def _run_state(pickled_state: bytes):
    """Target function run in worker processes."""
    state: RState = cloudpickle.loads(pickled_state)
    return state.procedure(**state.args)


class MultiprocessingExecutor(Executor):
    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers
        self._pool = None
        self._pending: dict[object, Job] = {}
        self._completed: deque[Job] = deque()

    def _ensure_pool(self):
        if self._pool is None:
            ctx = _resolve_context()
            self._pool = ctx.Pool(self._max_workers) if self._max_workers is not None else ctx.Pool()

    def submit(self, state: RState):
        self._ensure_pool()
        pickled = cloudpickle.dumps(state)
        async_result = self._pool.apply_async(_run_state, (pickled,))  # type: ignore[attr-defined]
        job = Job(state)
        job.start_time = time.time()
        self._pending[async_result] = job

        return job

    def wait(self):
        if not self._pending and not self._completed:
            return None
        while not self._completed:
            ready = [ar for ar in self._pending if ar.ready()]
            if not ready:
                time.sleep(0.01)
                continue
            for ar in ready:
                job = self._pending.pop(ar)
                try:
                    job.output = ar.get()
                except Exception as e:
                    job.error = e
                job.completed = True
                job.end_time = time.time()
                self._completed.append(job)
        return self._completed.popleft()

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
