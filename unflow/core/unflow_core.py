from functools import wraps
from logging import getLogger

from unflow.core.builder import GraphBuilder
from unflow.core.executors.executor import Executor
from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.scheduler import Scheduler
from unflow.core.unflow_types import RStateStatus

logger = getLogger(__name__)  # Placeholder for a logger instance, if needed


class unflowdecorator:
    def __init__(self, executor: Executor | None = None):
        """
        A decorator that wraps a function to enable unflow's graph-based execution.
        :param executor: An instance of an Executor. If None, LocalExecutor is used.
        :param re_execute_all: If True, all states will be re-executed even if they have been executed before.
        """
        # get the execution path from the caller's frame
        self.graph_builder = GraphBuilder("")
        self.executor = executor or LocalExecutor()

    def get_execution_path(self, func):
        from pathlib import Path

        directory = Path(func.__code__.co_filename).parent
        return str(directory)

    def set_execution_path(self, execution_path: str):
        self.graph_builder.set_execution_path(execution_path)

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # once the function is called, we will build the graph and execute the function
            record = self.run_once(func, args, kwargs)
            if record is None:
                return None
            return record.outcome.outputs if record.status == RStateStatus.COMPLETED else None

        wrapper.run_multiple = lambda combos: self.run_multiple(func, combos)
        wrapper.clear_graph = lambda: self._clear_graph(func)
        wrapper.graph_size = lambda: len(self.graph_builder.compute_graph.graph.nodes)
        wrapper.compute_graph = self.graph_builder.compute_graph
        wrapper.get_outcomes = lambda: self.get_outcomes(func)

        return wrapper

    def _clear_graph(self, func):
        self.graph_builder.compute_graph.clear()
        # get the execution path from the caller's frame
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.db.clear_graph(func.__name__, self.graph_builder.compute_graph.execution_path)

    def build_graph(self, func, *args, **kwargs):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        new_state = self.graph_builder.create_state(func, *args, **kwargs)
        can_transform = self.graph_builder.transform(new_state)
        if not can_transform:
            logger.info("No new state created. The function has already been executed with the same arguments.")
            return None
        self.graph_builder.save_graph(func.__name__)
        return new_state

    def run_once(self, func, args, kwargs):
        new_state = self.build_graph(func, *args, **kwargs)
        if new_state is None:
            return None
        scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
        record, graph = scheduler.run(new_state)
        self.graph_builder.finalize(graph, record, func.__name__, new_state)
        return record

    def get_outcomes(self, func):
        self.graph_builder.load_graph(func.__name__)
        outcomes = {}
        for node in self.graph_builder.compute_graph.graph.nodes:
            state = self.graph_builder.compute_graph.state_map.get(node)
            if state is not None:
                outcome = self.graph_builder.load_outcome(state)
                if outcome is not None:
                    outcomes[node] = outcome
        return outcomes

    def run_multiple(self, func, combos):
        results = []
        new_states = []
        for combo in combos:
            if "args" in combo or "kwargs" in combo:
                args = combo.get("args", [])
                kwargs = combo.get("kwargs", {})
            else:
                args = []
                kwargs = combo
            new_state = self.build_graph(func, *args, **kwargs)
            if new_state is not None:
                new_states.append(new_state)
        for new_state in new_states:
            self.graph_builder.set_status(new_state.name, RStateStatus.READY)
        if len(new_states) > 0:
            scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
            record, graph = scheduler.run(new_states[-1])
            self.graph_builder.finalize(graph, record, func.__name__, new_states[-1])
            results.append(record)

        return results
