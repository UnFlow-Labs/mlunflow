from functools import wraps

from unflow.core.engine import GraphBuilder
from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.scheduler import Scheduler
from unflow.core.unflow_types import RStateStatus


class unflowdecorator:
    def __init__(self):
        self.graph_builder = GraphBuilder(None)
        self.executor = LocalExecutor()

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

    def run_once(self, func, args, kwargs):
        # load the graph from the database before running the function if it exists
        self.graph_builder.load_graph(func.__name__)
        # transform the function into a new state and add it to the graph
        new_state = self.graph_builder.transform(func, *args, **kwargs)
        if new_state is None:
            print("No new state created. The function has already been executed with the same arguments.")
            return None
        self.graph_builder.save_graph(func.__name__)
        scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
        record, graph = scheduler.run(new_state)
        self.graph_builder.set_graph(graph)
        self.graph_builder.save_graph(func.__name__)
        return record

    def run_multiple(self, func, combos):
        """
        combos is a list of dictionaries, where each dictionary contains the arguments for a single run of the function
        """

        results = []
        for combo in combos:
            args = combo.get("args", [])
            kwargs = combo.get("kwargs", {})
            result = self.run_once(func, args, kwargs)
            results.append(result)
        return results
