import inspect
import json

from networkx import DiGraph

from ..core.simpledb import DB
from ..core.unflow_types import ExecutionRecord, Outcome, RState, RStateStatus, Transformation
from ..graph.compute_graph import ComputeGraph


class GraphBuilder:
    def __init__(self, execution_path: str, graph: ComputeGraph | None = None):
        self.compute_graph = graph if graph is not None else ComputeGraph(execution_path)
        self.db = DB()

    def set_execution_path(self, execution_path: str):
        self.compute_graph.execution_path = execution_path

    def normalize(self, func, args, kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def roll_node_version(self, state: RState):
        if state.name in self.compute_graph.state_map:
            state_version = state.name.split("_")[-1]
            new_version = int(state_version) + 1
            state.name = "_".join(state.name.split("_")[:-1]) + f"_{new_version}"
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

    def transform(
        self,
        state: RState,
    ):
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
        self.db.save_graph(name, self.compute_graph.execution_path, self.compute_graph.graph2json())

    def set_graph(self, graph: DiGraph):
        self.compute_graph.graph = graph

    def load_graph(self, name: str):
        graph_json = self.db.load_graph(name, self.compute_graph.execution_path)
        if graph_json:
            self.compute_graph.json2graph(graph_json)

    def set_status(self, state_name: str, status: RStateStatus):
        if state_name in self.compute_graph.state_map:
            self.compute_graph.state_map[state_name].status = status

    def save_outcome(self, state: RState, outcome: Outcome | None):
        if outcome is None:
            return
        outcome_data = outcome.to_json()
        self.db.save_outcome(state.name, self.compute_graph.execution_path, outcome_data)

    def load_outcome(self, state: RState) -> Outcome | None:
        outcome_data = self.db.load_outcome(state.name, self.compute_graph.execution_path)
        if outcome_data:
            outcome_json = json.loads(outcome_data.decode("utf-8"))
            return Outcome.from_dict(outcome_json, state)
        return None

    def save_execution_record(self, state: RState, record):
        if state:
            self.db.save_execution_record(
                graph_path=self.compute_graph.execution_path,
                state_name=state.name,
                status=record.status.value,
                start_time=record.start_time,
                end_time=record.end_time,
                error=str(record.error) if record.error else None,
            )

    def load_execution_record(self, state: RState) -> ExecutionRecord | None:
        pass

    def finalize(self, graph: DiGraph, record: ExecutionRecord, graph_name: str, state: RState):
        self.set_graph(graph)
        self.save_graph(graph_name)
        self.save_execution_record(state, record)
        self.save_outcome(state, record.outcome)
