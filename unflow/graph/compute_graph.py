import orjson

import networkx as nx
from networkx.readwrite import json_graph
from unflow.core.json_encoder import dumps
from unflow.core.unflow_types import RState, Transformation

class ComputeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.state_map = {}

    def add_state(self, state: RState):
        self.graph.add_node(state.name, state=state)
        self.state_map[state.name] = state

    def add_transformation(self, from_state: RState, to_state: RState, transformation: Transformation):
        self.graph.add_edge(from_state.name, to_state.name, transformation=transformation)

    def get_state(self, name):
        return self.state_map.get(name, None)

    def get_states(self):
        return list(self.state_map.values())

    def get_transformations(self):
        transformations = []
        for _u, _v, data in self.graph.edges(data=True):
            transformations.append(data["transformation"])
        return transformations

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
        from unflow.core.unflow_core import RState

        json_object = orjson.loads(json_object)
        self.graph = json_graph.node_link_graph(json_object)
        self.state_map = {}
        for node, data in self.graph.nodes(data=True):
            state = RState.from_dict(data["state"])
            self.graph.nodes[node]["state"] = state
            self.state_map[node] = state
        for u, v_, data in self.graph.edges(data=True):
            state1 = self.graph.nodes[u]["state"]
            state2 = self.graph.nodes[v_]["state"]
            transformation = Transformation.from_dict(data["transformation"], state1, state2)
            self.graph.edges[u, v_]["transformation"] = transformation

    def run(self, target_state: RState):

        # 1. get all ancestors needed for this state
        ancestors = nx.ancestors(self.graph, target_state.name)
        subgraph_nodes = ancestors | {target_state.name}

        # 2. extract induced subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)

        # 3. topological sort based on dependencies
        ordered = list(nx.topological_sort(subgraph))
        print(f"Execution order: {ordered}")
        # 4. execute in correct order
        for state_name in ordered:
            state = self.graph.nodes[state_name]["state"]
            print(f"Running state: {state.name}")
            outputs = state.run()
            print(f"Outputs: {outputs}")