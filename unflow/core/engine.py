
from unflow.core.simpledb import DB
from unflow.graph.compute_graph import ComputeGraph
from unflow.core.unflow_types import RState, Transformation

class ExecutionEngine:
    def __init__(self):
        self.db = DB()
        self.graph = ComputeGraph()

    def run(self, func, *args, **kwargs):
        return self._execute(func, *args, **kwargs)

    def _execute(self, func, *args, **kwargs):
        g_name = func.__name__

        graph_data = self.db.load_graph(g_name)
        if graph_data:
            self.graph.json2graph(graph_data)

        existing_states = [
            data["state"]
            for _, data in self.graph.graph.nodes(data=True)
        ]

        new_state = RState(
            name=f"{g_name}_{len(existing_states)}",
            procedure=func,
            args=args,
            description=kwargs.get("description", ""),
        )

      

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
        
        if len(transformations) > 0 or len(existing_states) == 0:
            self.graph.add_state(new_state)
            for t in transformations:
                self.graph.add_transformation(t.state1, t.state2, t)
            out = self.graph.run(new_state)

            self.db.save_graph(g_name, self.graph.graph2json())
            # self.db.close()
            return out
    def close(self):
        self.db.close()