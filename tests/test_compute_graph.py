from unflow.core.unflow_types import RState, RStateStatus, Transformation
from unflow.graph.compute_graph import ComputeGraph


def _f1():
    return 1


def _f2():
    return 2


class TestComputeGraph:
    def setup_method(self):
        self.graph = ComputeGraph("unflow_test_graph")
        self.s1 = RState(name="s1", procedure=_f1, args={})
        self.s2 = RState(name="s2", procedure=_f2, args={})

    def test_add_state(self):
        self.graph.add_state(self.s1)
        assert self.graph.get_state("s1") == self.s1

    def test_get_states(self):
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        states = self.graph.get_states()
        assert self.s1 in states
        assert self.s2 in states

    def test_get_state__missing(self):
        assert self.graph.get_state("nonexistent") is None

    def test_add_transformation(self):
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        t = Transformation("t", self.s1, self.s2)
        self.graph.add_transformation(self.s1, self.s2, t)

        transformations = self.graph.get_transformations()
        assert len(transformations) == 1
        assert transformations[0].name == "t"

    def test_get_transformations__empty(self):
        assert self.graph.get_transformations() == []

    def test_clear(self):
        self.graph.add_state(self.s1)
        self.graph.clear()
        assert self.graph.get_states() == []
        assert self.graph.get_transformations() == []

    def test_graph2json(self):
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        t = Transformation("t", self.s1, self.s2)
        self.graph.add_transformation(self.s1, self.s2, t)

        data = self.graph.graph2json()
        assert isinstance(data, bytes)

    def test_json2graph_roundtrip(self):
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        t = Transformation("t", self.s1, self.s2)
        self.graph.add_transformation(self.s1, self.s2, t)

        json_data = self.graph.graph2json()
        new_graph = ComputeGraph("unflow_test_graph")
        new_graph.json2graph(json_data)

        assert new_graph.get_state("s1") is not None
        assert new_graph.get_state("s2") is not None
        assert len(new_graph.get_transformations()) == 1

    def test_json2graph_restores_status(self):
        self.s1.run()
        self.graph.add_state(self.s1)
        json_data = self.graph.graph2json()

        new_graph = ComputeGraph("unflow_test_graph")
        new_graph.json2graph(json_data)
        restored = new_graph.get_state("s1")
        assert restored.status == RStateStatus.COMPLETED

    def test_query_states_by_name_and_args(self):
        self.s1.args = {"model": 0, "hesitation": False}
        self.s2.args = {"model": 1, "hesitation": True}
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)

        states = self.graph.query_states(name_contains="s", args_contains={"model": 0})

        assert states == [self.s1]

    def test_query_transformations_by_source_and_args_changes(self):
        self.s1.args = {"x": 1}
        self.s2.args = {"x": 2}
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        t = Transformation("t", self.s1, self.s2)
        self.graph.add_transformation(self.s1, self.s2, t)

        transformations = self.graph.query_transformations(from_state="s1", has_args_changes=True)

        assert transformations == [t]

    def test_shortest_path(self):
        s3 = RState(name="s3", procedure=_f1, args={})
        self.graph.add_state(self.s1)
        self.graph.add_state(self.s2)
        self.graph.add_state(s3)
        t1 = Transformation("t1", self.s1, self.s2)
        t2 = Transformation("t2", self.s2, s3)
        self.graph.add_transformation(self.s1, self.s2, t1)
        self.graph.add_transformation(self.s2, s3, t2)

        path = self.graph.shortest_path("s1", "s3")

        assert [state.name for state in path] == ["s1", "s2", "s3"]
