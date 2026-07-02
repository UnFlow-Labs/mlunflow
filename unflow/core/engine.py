import inspect

from networkx import DiGraph

from unflow.core.simpledb import DB
from unflow.core.unflow_types import RState, Transformation
from unflow.graph.compute_graph import ComputeGraph


class GraphBuilder:
    def __init__(self, graph: ComputeGraph | None = None):
        self.compute_graph = graph if graph is not None else ComputeGraph()
        self.db = DB()

    def normalize(self, func, args, kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def transform(self, func, *args, **kwargs):
        g_name = func.__name__
        params = self.normalize(func, args, kwargs)
        existing_states = [data["state"] for _, data in self.compute_graph.graph.nodes(data=True)]
        new_state = RState(
            name=f"{g_name}_{len(existing_states)}",
            procedure=func,
            args=params,
            kwargs={},
            description="auto",
        )
        # check if the new state is in the graph already
        transformations = []
        for prev_state in existing_states:
            t = Transformation(
                name=f"{prev_state.name}->{new_state.name}",
                state1=prev_state,
                state2=new_state,
                description="auto",
            )
            if t.__has__changed__():
                transformations.append(t)
            else:
                return None

        if len(transformations) > 0 or len(existing_states) == 0:
            self.compute_graph.add_state(new_state)
            for t in transformations:
                self.compute_graph.add_transformation(t.state1, t.state2, t)
            return new_state
        return None

    def save_graph(self, name: str):
        self.db.save_graph(name, self.compute_graph.graph2json())

    def set_graph(self, graph: DiGraph):
        self.compute_graph.graph = graph

    def load_graph(self, name: str):
        graph_json = self.db.load_graph(name)
        if graph_json:
            self.compute_graph.json2graph(graph_json)
