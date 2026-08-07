from time import time

from ...core.executors.executor import Executor
from ...core.unflow_types import Job, RState


class LocalExecutor(Executor):
    def __init__(self):
        self.jobs = []

    def submit(self, state: RState):

        job = Job(state)

        try:
            job.start_time = time()
            job.output = state.procedure(**state.args)

        except Exception as e:
            job.error = e
        finally:
            job.end_time = time()

        job.completed = True

        self.jobs.append(job)

        return job

    def wait(self):

        if not self.jobs:
            return None

        return self.jobs.pop(0)
