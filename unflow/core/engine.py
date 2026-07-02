import inspect

from networkx import DiGraph
from typer.cli import state

from unflow.core.simpledb import DB
from unflow.core.unflow_types import RState, RStateStatus, Transformation
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
    
    def roll_node_version(self, state: RState):
        if state.name in self.compute_graph.state_map:
           state_version = state.name.split('_')[-1]
           new_version = int(state_version) + 1
           state.name = '_'.join(state.name.split('_')[:-1]) + f'_{new_version}'
           self.compute_graph.state_map[state.name] = state
        return state

    
    def create_state(self, func, *args, **kwargs):
        g_name = func.__name__
        params = self.normalize(func, args, kwargs)
        new_state = RState(
            name=f"{g_name}_{len(self.compute_graph.get_states())}",
            procedure=func,
            args=params,
            kwargs={},
            description="auto",
        )
        return new_state

    def transform(self, state: RState,):
        existing_states = [data["state"] for _, data in self.compute_graph.graph.nodes(data=True)]
        transformations = []
        for prev_state in existing_states:
            t = Transformation(
                name=f"{prev_state.name}->{state.name}",
                state1=prev_state,
                state2=state,
                description="auto",
            )
            if t.__has__changed__():
                transformations.append(t)
            else:
                return False

        if len(transformations) > 0 or len(existing_states) == 0:
            self.compute_graph.add_state(state)
            for t in transformations:
                self.compute_graph.add_transformation(t.state1, t.state2, t)
            return True
        return False

    def save_graph(self, name: str):
        self.db.save_graph(name, self.compute_graph.graph2json())

    def set_graph(self, graph: DiGraph):
        self.compute_graph.graph = graph

    def load_graph(self, name: str):
        graph_json = self.db.load_graph(name)
        if graph_json:
            self.compute_graph.json2graph(graph_json)
    def set_status(self, state_name: str, status: RStateStatus):
        if state_name in self.compute_graph.state_map:
            self.compute_graph.state_map[state_name].status = status
