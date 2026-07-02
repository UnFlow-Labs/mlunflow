import time

import pytest

from unflow.core.executors.multiprocessing_executor import MultiprocessingExecutor
from unflow.core.unflow_types import RState


def _simple(a, b=0):
    return a + b


def _failing(x):
    msg = "oops"
    raise ValueError(msg)


def _slow(x, delay=0.2):
    time.sleep(delay)
    return x * 2


class TestMultiprocessingExecutor:
    def test_submit__successful_job(self):
        state = RState(name="test", procedure=_simple, args={"a": 1, "b": 2})
        executor = MultiprocessingExecutor(max_workers=1)

        job = executor.submit(state)

        # Job is returned immediately, not yet completed
        assert job.completed is False

        # Wait for completion
        done = executor.wait()
        assert done is job
        assert job.completed is True
        assert job.output == 3
        assert job.error is None

        executor.close()

    def test_submit__failing_job(self):
        state = RState(name="fail", procedure=_failing, args={"x": 1})
        executor = MultiprocessingExecutor(max_workers=1)

        executor.submit(state)
        done = executor.wait()
        assert done.completed is True
        assert done.output is None
        assert isinstance(done.error, ValueError)
        executor.close()

    def test_multiple_jobs_execute_in_parallel(self):
        s1 = RState(name="s1", procedure=_slow, args={"x": 1, "delay": 0.4})
        s2 = RState(name="s2", procedure=_slow, args={"x": 2, "delay": 0.2})
        executor = MultiprocessingExecutor(max_workers=2)

        executor.submit(s1)
        executor.submit(s2)

        j1 = executor.wait()
        assert j1.output == 4

        j2 = executor.wait()
        assert j2.output == 2

        executor.close()

    def test_wait__empty_queue(self):
        executor = MultiprocessingExecutor(max_workers=1)
        assert executor.wait() is None
        executor.close()

    def test_submit_and_wait_multiple_times(self):
        executor = MultiprocessingExecutor(max_workers=1)
        s1 = RState(name="s1", procedure=_simple, args={"a": 10})
        s2 = RState(name="s2", procedure=_simple, args={"a": 20})

        executor.submit(s1)
        assert executor.wait().output == 10

        executor.submit(s2)
        assert executor.wait().output == 20

        executor.close()
