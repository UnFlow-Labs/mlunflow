from dataclasses import dataclass

from unflow.core.executors.executor import Executor
from unflow.core.unflow_types import RState


@dataclass
class Job:
    state: RState
    output: object = None
    error: Exception | None = None
    completed: bool = False


class LocalExecutor(Executor):
    def __init__(self):
        self.jobs = []

    def submit(self, state: RState):

        job = Job(state)

        try:
            job.output = state.procedure(**state.args)

        except Exception as e:
            job.error = e

        job.completed = True

        self.jobs.append(job)

        return job

    def wait(self):

        if not self.jobs:
            return None

        return self.jobs.pop(0)
