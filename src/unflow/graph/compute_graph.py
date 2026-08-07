from collections.abc import Callable

import networkx as nx
import orjson
from networkx.readwrite import json_graph

from ..core.json_encoder import dumps
from ..core.unflow_types import RState, RStateStatus, Transformation


class ComputeGraph:
    def __init__(self, execution_path: str):
        self.graph = nx.DiGraph()
        self.state_map: dict[str, RState] = {}
        self.execution_path = execution_path  # Initialize execution_path

    def add_state(self, state: RState):
        # does this state already exist? if so, we don't add it again
        self.graph.add_node(state.name, state=state)
        self.state_map[state.name] = state

    def add_transformation(self, from_state: RState, to_state: RState, transformation: Transformation):
        self.graph.add_edge(from_state.name, to_state.name, transformation=transformation)

    def get_state(self, state_name):
        return self.state_map.get(state_name, None)

    def get_states(self):
        return list(self.state_map.values())

    def get_transformations(self):
        transformations = []
        for _u, _v, data in self.graph.edges(data=True):
            transformations.append(data["transformation"])
        return transformations

    def query_states(
        self,
        status: RStateStatus | str | None = None,
        name_contains: str | None = None,
        args_contains: dict | None = None,
        predicate: Callable[[RState], bool] | None = None,
    ) -> list[RState]:
        matched_states = []
        normalized_status = status.value if isinstance(status, RStateStatus) else status

        for state in self.get_states():
            if normalized_status is not None and state.status.value != normalized_status:
                continue
            if name_contains is not None and name_contains not in state.name:
                continue
            if args_contains is not None:
                has_all_args = all(state.args.get(key) == value for key, value in args_contains.items())
                if not has_all_args:
                    continue
            if predicate is not None and not predicate(state):
                continue
            matched_states.append(state)

        return matched_states

    def query_transformations(
        self,
        from_state: str | None = None,
        to_state: str | None = None,
        has_args_changes: bool | None = None,
        has_procedure_changes: bool | None = None,
        predicate: Callable[[Transformation], bool] | None = None,
    ) -> list[Transformation]:
        matched_transformations = []

        for _u, _v, data in self.graph.edges(data=True):
            transformation = data["transformation"]
            if from_state is not None and transformation.state1.name != from_state:
                continue
            if to_state is not None and transformation.state2.name != to_state:
                continue
            if has_args_changes is not None and bool(transformation.args_changes) != has_args_changes:
                continue
            if has_procedure_changes is not None and bool(transformation.p_changes) != has_procedure_changes:
                continue
            if predicate is not None and not predicate(transformation):
                continue
            matched_transformations.append(transformation)

        return matched_transformations

    def shortest_path(self, from_state: str, to_state: str) -> list[RState]:
        path_node_names = nx.shortest_path(self.graph, source=from_state, target=to_state)
        return [self.state_map[state_name] for state_name in path_node_names]

    def graph2json(
        self,
    ):
        data = json_graph.node_link_data(self.graph)
        for node in data["nodes"]:
            state = self.graph.nodes[node["id"]]["state"]
            node["state"] = state.serialize()
        for edge in data["edges"]:
            transformation = self.graph.edges[edge["source"], edge["target"]]["transformation"]
            edge["transformation"] = transformation.__dict__()
        return dumps(data)

    def json2graph(self, json_object):

        json_object = orjson.loads(json_object)
        self.graph = json_graph.node_link_graph(json_object)
        self.state_map = {}
        for node, data in self.graph.nodes(data=True):
            state = RState.from_dict(data["state"])
            self.graph.nodes[node]["state"] = state
            self.state_map[state.name] = state
        for u, v_, data in self.graph.edges(data=True):
            state1 = self.graph.nodes[u]["state"]
            state2 = self.graph.nodes[v_]["state"]
            transformation = Transformation.from_dict(data["transformation"], state1, state2)
            self.graph.edges[u, v_]["transformation"] = transformation

    def clear(self):
        self.graph.clear()
        self.state_map.clear()
