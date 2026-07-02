from unittest.mock import MagicMock, patch

import pytest

from unflow.core.unflow_core import unflowdecorator
from unflow.core.unflow_types import RStateStatus


def _train(lr=0.01, epochs=10):
    return {"loss": 0.5, "lr": lr, "epochs": epochs}


class TestUnflowDecorator:
    @pytest.fixture
    def decorator(self):
        with (
            patch("unflow.core.unflow_core.GraphBuilder") as mock_gb_cls,
            patch("unflow.core.unflow_core.LocalExecutor") as mock_exec_cls,
        ):
            mock_gb = MagicMock()
            mock_gb_cls.return_value = mock_gb
            mock_exec = MagicMock()
            mock_exec_cls.return_value = mock_exec

            deco = unflowdecorator()
            deco.graph_builder = mock_gb
            deco.executor = mock_exec

            yield deco, mock_gb

    def test_wraps_function(self, decorator):
        deco, _ = decorator
        wrapped = deco(_train)
        assert wrapped.__name__ == "_train"
        assert hasattr(wrapped, "run_multiple")
        assert hasattr(wrapped, "clear_graph")

    def test_call__returns_output(self, decorator):
        deco, _ = decorator
        wrapped = deco(_train)

        mock_state = MagicMock()
        mock_state.status = RStateStatus.COMPLETED
        mock_state.outputs = {"loss": 0.5}

        deco.run_once = MagicMock(return_value=mock_state)

        result = wrapped(lr=0.01, epochs=10)
        assert result == {"loss": 0.5}

    def test_call__duplicate_returns_none(self, decorator):
        deco, _ = decorator
        wrapped = deco(_train)

        deco.run_once = MagicMock(return_value=None)

        result = wrapped(lr=0.01, epochs=10)
        assert result is None

    def test_call__non_completed_returns_none(self, decorator):
        deco, _ = decorator
        wrapped = deco(_train)

        mock_state = MagicMock()
        mock_state.status = RStateStatus.FAILED
        mock_state.outputs = None

        deco.run_once = MagicMock(return_value=mock_state)

        result = wrapped(lr=0.01, epochs=10)
        assert result is None

    def test_run_multiple(self, decorator):
        deco, _ = decorator
        deco.run_once = MagicMock(return_value="result")
        wrapped = deco(_train)

        combos = [
            {"args": [], "kwargs": {"lr": 0.01}},
            {"args": [], "kwargs": {"lr": 0.1}},
        ]
        results = wrapped.run_multiple(combos)

        assert results == ["result", "result"]
        assert deco.run_once.call_count == 2

    def test_clear_graph(self, decorator):
        deco, mock_gb = decorator
        mock_gb.compute_graph = MagicMock()
        mock_gb.db = MagicMock()
        wrapped = deco(_train)

        wrapped.clear_graph()
        mock_gb.compute_graph.clear.assert_called_once()
        mock_gb.db.clear_graph.assert_called_once()


class TestRunOnce:
    @pytest.fixture
    def deco(self):
        with (
            patch("unflow.core.unflow_core.GraphBuilder") as mock_gb_cls,
            patch("unflow.core.unflow_core.LocalExecutor") as mock_exec_cls,
            patch("unflow.core.unflow_core.Scheduler") as mock_sched_cls,
        ):
            mock_gb = MagicMock()
            mock_gb_cls.return_value = mock_gb
            mock_exec = MagicMock()
            mock_exec_cls.return_value = mock_exec
            mock_sched = MagicMock()
            mock_sched_cls.return_value = mock_sched

            deco = unflowdecorator()
            deco.graph_builder = mock_gb
            deco.executor = mock_exec

            yield deco, mock_gb, mock_sched, mock_sched_cls

    def test_run_once__loads_transforms_saves(self, deco):
        deco, mock_gb, mock_sched, _ = deco
        mock_state = MagicMock()
        mock_state.name = "s0"
        mock_gb.transform.return_value = mock_state
        mock_sched.run.return_value = (mock_state, MagicMock())

        deco.run_once(_train, (), {"lr": 0.01})

        mock_gb.load_graph.assert_called_once_with("_train")
        mock_gb.transform.assert_called_once()
        mock_gb.save_graph.assert_called()
        mock_sched.run.assert_called_once_with(mock_state)

    def test_run_once__transform_returns_none(self, deco):
        deco, mock_gb, _, _ = deco
        mock_gb.transform.return_value = None

        result = deco.run_once(_train, (), {"lr": 0.01})

        assert result is None
        mock_gb.load_graph.assert_called_once()
        mock_gb.save_graph.assert_not_called()
