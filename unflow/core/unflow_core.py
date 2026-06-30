import json

import networkx as nx
from networkx.readwrite import json_graph

from unflow.core.simpledb import DB
from unflow.core.unflow_types import Procedure, RState, Transformation


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
        return json.dumps(data, indent=4)

    def json2graph(self, json_object):
        from unflow.core.unflow_core import RState

        json_object = json.loads(json_object)
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


# connect to the database
db = DB()
c_graph = ComputeGraph()  # current command graph


class unflowdecorator:
    def __init__(self, procedure: Procedure):
        self.procedure = procedure

    def __call__(self, *args, **kwargs):
        # every time the decorated function is called, we create a new state and detect the transformation
        # first load the graph from the database if it exists
        g_name = self.procedure.__name__
        graph_data = db.load_graph(g_name)

        if graph_data:
            c_graph.json2graph(graph_data)
        existing_states = [data["state"] for _, data in c_graph.graph.nodes(data=True)]
        # create a new state for this procedure call
        new_state_name = f"{g_name}_{len(c_graph.graph.nodes)}"
        state_args = kwargs if kwargs else args
        new_state = RState(
            name=new_state_name,
            procedure=self.procedure,
            args=state_args,
            description=f"State for {g_name} with args {state_args}",
        )
        # detect transformations from existing states to the new state
        first_time_execution = False
        if len(existing_states) == 0:
            c_graph.add_state(new_state)
            first_time_execution = True
        transformations = []
        for prev_state in existing_states:
            t = Transformation(
                name=f"Transformation_{prev_state.name}_to_{new_state_name}",
                state1=prev_state,
                state2=new_state,
                description=f"Transformation to {new_state_name}",
            )
            if t.__has__changed__():
                transformations.append(t)
            else:
                print("This state has been run before with the same procedure and arguments.")
                transformations = []
                break
        if transformations:
            c_graph.add_state(new_state)
            for t in transformations:
                c_graph.add_transformation(t.state1, t.state2, t)
        if first_time_execution or transformations:
            outs = c_graph.run(new_state)
            # save the updated graph to the database
            data = c_graph.graph2json()
            db.save_graph(g_name, graph_data=data)  # Save the graph to the database
            return outs
