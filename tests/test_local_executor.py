from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.unflow_types import RState


def _simple(a, b=0):
    return a + b


def _failing(x):
    msg = "oops"
    raise ValueError(msg)


class TestLocalExecutor:
    def test_submit__successful_job(self):
        state = RState(name="test", procedure=_simple, args={"a": 1, "b": 2})
        executor = LocalExecutor()
        job = executor.submit(state)

        assert job.completed is True
        assert job.output == 3
        assert job.error is None

    def test_submit__failing_job(self):
        state = RState(name="fail", procedure=_failing, args={"x": 1})
        executor = LocalExecutor()
        job = executor.submit(state)

        assert job.completed is True
        assert job.output is None
        assert isinstance(job.error, ValueError)
        assert str(job.error) == "oops"

    def test_wait__returns_job_fifo(self):
        s1 = RState(name="s1", procedure=_simple, args={"a": 1})
        s2 = RState(name="s2", procedure=_simple, args={"a": 2})
        executor = LocalExecutor()
        executor.submit(s1)
        executor.submit(s2)

        j1 = executor.wait()
        assert j1.output == 1

        j2 = executor.wait()
        assert j2.output == 2

    def test_wait__empty_queue(self):
        executor = LocalExecutor()
        assert executor.wait() is None

    def test_wait__after_draining(self):
        state = RState(name="s", procedure=_simple, args={"a": 1})
        executor = LocalExecutor()
        executor.submit(state)
        executor.wait()
        assert executor.wait() is None
