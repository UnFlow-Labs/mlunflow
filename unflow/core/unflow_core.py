from functools import wraps

from unflow.core.engine import GraphBuilder
from unflow.core.executors.executor import Executor
from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.scheduler import Scheduler
from unflow.core.unflow_types import RStateStatus
from logging import getLogger

logger = getLogger(__name__)  # Placeholder for a logger instance, if needed


class unflowdecorator:
    def __init__(self, executor: Executor | None = None):
        '''
        A decorator that wraps a function to enable unflow's graph-based execution.
        :param executor: An instance of an Executor. If None, LocalExecutor is used.
        :param re_execute_all: If True, all states will be re-executed even if they have been executed before.
        '''
        self.graph_builder = GraphBuilder(None)
        self.executor = executor or LocalExecutor()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            record = self.run_once(func, args, kwargs)
            if record is None:
                return None
            return record.outputs if record.status == RStateStatus.COMPLETED else None

        wrapper.run_multiple = lambda combos: self.run_multiple(func, combos)
        wrapper.clear_graph = lambda: self._clear_graph(func)
        wrapper.graph_size = lambda: len(self.graph_builder.compute_graph.graph.nodes)
        wrapper.compute_graph = self.graph_builder.compute_graph
        
        return wrapper

    def _clear_graph(self, func):
        self.graph_builder.compute_graph.clear()
        self.graph_builder.db.clear_graph(func.__name__)
    
    def build_graph(self, func, *args, **kwargs):
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
        self.graph_builder.set_graph(graph)
        self.graph_builder.save_graph(func.__name__)
        return record


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
        # when running the functions multiple times the nodes will be treated as independent and will be executed in parallel.
        # to do this every state will be marked as ready for execution and the scheduler will execute them in parallel.
        for new_state in new_states:
            self.graph_builder.set_status(new_state.name, RStateStatus.READY)
        if len(new_states) > 0:
            scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
            record, graph = scheduler.run(new_states[-1])
            self.graph_builder.set_graph(graph)
            self.graph_builder.save_graph(func.__name__)
            results.append(record)

        return results
