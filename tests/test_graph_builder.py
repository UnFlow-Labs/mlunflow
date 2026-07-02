from unittest.mock import MagicMock, patch

import pytest

from unflow.core.engine import GraphBuilder
from unflow.graph.compute_graph import ComputeGraph


def _sample(a, b=1):
    return a + b


def _other(x):
    return x * 2


class TestGraphBuilder:
    @pytest.fixture
    def builder(self):
        with patch("unflow.core.engine.DB") as mock_db:
            mock_db.return_value = MagicMock()
            gb = GraphBuilder(ComputeGraph())
        return gb

    def test_init__creates_compute_graph(self, builder):
        assert builder.compute_graph is not None
        assert isinstance(builder.compute_graph, ComputeGraph)

    def test_normalize__binds_positional_args(self, builder):
        result = builder.normalize(_sample, (3,), {})
        assert result == {"a": 3, "b": 1}

    def test_normalize__binds_keyword_args(self, builder):
        result = builder.normalize(_sample, (), {"a": 5, "b": 10})
        assert result == {"a": 5, "b": 10}

    def test_normalize__applies_defaults(self, builder):
        result = builder.normalize(_sample, (7,), {})
        assert result == {"a": 7, "b": 1}

    def test_transform__creates_new_state(self, builder):
        state = builder.transform(_sample, 1, 2)
        assert state is not None
        assert builder.compute_graph.get_state(state.name) == state

    def test_transform__duplicate_args_returns_none(self, builder):
        builder.transform(_sample, 1, 2)
        state = builder.transform(_sample, 1, 2)
        assert state is None

    def test_transform__different_args_creates_new_state(self, builder):
        builder.transform(_sample, 1, 2)
        state2 = builder.transform(_sample, 3, 4)
        assert state2 is not None
        assert len(builder.compute_graph.get_states()) == 2
        assert len(builder.compute_graph.get_transformations()) == 1

    def test_transform__different_procedure_creates_new_state(self, builder):
        builder.transform(_sample, 1)
        state2 = builder.transform(_other, 5)
        assert state2 is not None
        assert len(builder.compute_graph.get_states()) == 2
        assert len(builder.compute_graph.get_transformations()) == 1

    def test_set_graph(self, builder):
        state = builder.transform(_sample, 1, 2)
        assert builder.compute_graph.get_state(state.name) is not None

    def test_save_graph_calls_db(self, builder):
        builder.transform(_sample, 1, 2)
        builder.save_graph("test")
        builder.db.save_graph.assert_called_once()

    def test_load_graph_calls_db(self, builder):
        builder.db.load_graph.return_value = None
        builder.load_graph("test")
        builder.db.load_graph.assert_called_once_with("test")
