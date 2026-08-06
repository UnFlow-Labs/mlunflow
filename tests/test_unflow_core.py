from unittest.mock import MagicMock, patch

import pytest

from unflow.core.unflow_core import unflowdecorator
from unflow.core.unflow_types import ExecutionRecord, Outcome, RStateStatus


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
        assert hasattr(wrapped, "query_states")
        assert hasattr(wrapped, "query_transformations")
        assert hasattr(wrapped, "shortest_path")
        assert hasattr(wrapped, "shortest_path_to_lowest_outcome")

    def test_call__returns_output(self, decorator):
        deco, _ = decorator
        wrapped = deco(_train)

        mock_record = MagicMock()
        mock_record.status = RStateStatus.COMPLETED
        mock_record.outcome = Outcome(MagicMock(), {"loss": 0.5})

        deco.run_once = MagicMock(return_value=mock_record)

        result = wrapped(lr=0.01, epochs=10)
        assert result is not None
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

        mock_record = MagicMock()
        mock_record.status = RStateStatus.FAILED
        mock_record.outcome = None

        deco.run_once = MagicMock(return_value=mock_record)

        result = wrapped(lr=0.01, epochs=10)
        assert result is None

    def test_run_multiple(self, decorator):
        deco, mock_gb = decorator
        wrapped = deco(_train)

        s1 = MagicMock(name="s1")
        s1.name = "s1"
        s2 = MagicMock(name="s2")
        s2.name = "s2"
        deco.build_graph = MagicMock(side_effect=[s1, s2])
        mock_gb.compute_graph = MagicMock()

        record = ExecutionRecord(status=RStateStatus.COMPLETED)
        with patch("unflow.core.unflow_core.Scheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched.run.return_value = (record, MagicMock())
            mock_sched_cls.return_value = mock_sched

            combos = [
                {"args": [], "kwargs": {"lr": 0.01}},
                {"args": [], "kwargs": {"lr": 0.1}},
            ]
            results = wrapped.run_multiple(combos)

        assert results == [record]
        assert deco.build_graph.call_count == 2
        assert mock_gb.set_status.call_count == 2

    def test_clear_graph(self, decorator):
        deco, mock_gb = decorator
        mock_gb.compute_graph = MagicMock()
        mock_gb.db = MagicMock()
        wrapped = deco(_train)

        wrapped.clear_graph()
        mock_gb.compute_graph.clear.assert_called_once()
        mock_gb.db.clear_graph.assert_called_once()

    def test_query_states(self, decorator):
        deco, mock_gb = decorator
        wrapped = deco(_train)
        mock_gb.compute_graph.query_states.return_value = ["s1"]

        result = wrapped.query_states(name_contains="s")

        assert result == ["s1"]
        mock_gb.load_graph.assert_called_with("_train")
<<<<<<< HEAD
        mock_gb.compute_graph.query_states.assert_called_once_with(
            status=None,
            name_contains="s",
            args_contains=None,
            predicate=None,
        )
=======
        mock_gb.compute_graph.query_states.assert_called_once_with(name_contains="s")
>>>>>>> origin/main

    def test_query_transformations(self, decorator):
        deco, mock_gb = decorator
        wrapped = deco(_train)
        mock_gb.compute_graph.query_transformations.return_value = ["t1"]

        result = wrapped.query_transformations(from_state="s1")

        assert result == ["t1"]
        mock_gb.load_graph.assert_called_with("_train")
<<<<<<< HEAD
        mock_gb.compute_graph.query_transformations.assert_called_once_with(
            from_state="s1",
            to_state=None,
            has_args_changes=None,
            has_procedure_changes=None,
            predicate=None,
        )
=======
        mock_gb.compute_graph.query_transformations.assert_called_once_with(from_state="s1")
>>>>>>> origin/main

    def test_shortest_path(self, decorator):
        deco, mock_gb = decorator
        wrapped = deco(_train)
        mock_gb.compute_graph.shortest_path.return_value = ["s1", "s2"]

        result = wrapped.shortest_path("s1", "s2")

        assert result == ["s1", "s2"]
        mock_gb.load_graph.assert_called_with("_train")
        mock_gb.compute_graph.shortest_path.assert_called_once_with("s1", "s2")


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
        deco, mock_gb, mock_sched, mock_sched_cls = deco
        mock_state = MagicMock()
        mock_state.name = "s0"
        mock_record = MagicMock()
        graph = MagicMock()

        mock_gb.create_state.return_value = mock_state
        mock_gb.transform.return_value = True
        mock_sched.run.return_value = (mock_record, graph)

        result = deco.run_once(_train, (), {"lr": 0.01})

        mock_gb.load_graph.assert_called_once_with("_train")
        mock_gb.create_state.assert_called_once_with(_train, lr=0.01)
        mock_gb.transform.assert_called_once_with(mock_state)
        mock_gb.save_graph.assert_called()
        mock_sched.run.assert_called_once_with(mock_state)
        mock_sched_cls.assert_called_once_with(mock_gb.compute_graph, deco.executor)
        mock_gb.finalize.assert_called_once_with(graph, mock_record, "_train", mock_state)
        assert result == mock_record

    def test_run_once__transform_returns_none(self, deco):
        deco, mock_gb, _, mock_sched_cls = deco
        mock_state = MagicMock()
        mock_gb.create_state.return_value = mock_state
        mock_gb.transform.return_value = False

        result = deco.run_once(_train, (), {"lr": 0.01})

        assert result is None
        mock_gb.load_graph.assert_called_once()
        mock_gb.save_graph.assert_not_called()
        mock_sched_cls.assert_not_called()
