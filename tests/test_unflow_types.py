import inspect
from pathlib import Path
from unittest.mock import patch

from unflow.core.unflow_types import (
    RState,
    RStateStatus,
    Transformation,
    load_procedure,
    load_procedure_source,
    save_procedure,
    save_procedure_source,
)


class TestRStateStatus:
    def test_enum_values(self):
        assert RStateStatus.PENDING.value == "pending"
        assert RStateStatus.READY.value == "ready"
        assert RStateStatus.RUNNING.value == "running"
        assert RStateStatus.COMPLETED.value == "completed"
        assert RStateStatus.FAILED.value == "failed"


class TestRState:
    def test_init__sets_attributes(self, sample_state):
        assert sample_state.name == "test_state_0"
        assert sample_state.args == {"a": 1, "b": 2}
        assert sample_state.status == RStateStatus.PENDING

    def test_init__extracts_source(self, sample_state):
        expected = inspect.getsource(sample_state.procedure)
        assert sample_state.procedure_source == expected

    def test_init__uses_provided_source(self):
        state = RState(
            name="s",
            procedure=lambda: None,
            args={},
            procedure_source="custom source",
        )
        assert state.procedure_source == "custom source"

    def test_run__returns_output_and_completed(self, sample_state):
        output, status = sample_state.run()
        assert output == 3
        assert status == RStateStatus.COMPLETED
        assert sample_state.status == RStateStatus.COMPLETED

    def test_run__already_completed_returns_none(self, sample_state):
        sample_state.run()
        assert sample_state.run() is None

    def test_serialize__creates_files_and_returns_dict(self, sample_state, pickle_dir):
        with (
            patch("unflow.core.unflow_types.PICKLE_PATH", pickle_dir),
            patch("unflow.core.unflow_types.save_procedure") as mock_save,
            patch("unflow.core.unflow_types.save_procedure_source") as mock_save_src,
        ):
            result = sample_state.serialize()

        assert result["name"] == "test_state_0"
        assert "procedure" in result
        assert "procedure_source" in result
        assert result["args"] == {"a": 1, "b": 2}
        assert result["status"] == "pending"
        mock_save.assert_called_once()
        mock_save_src.assert_called_once()

    def test_from_dict__roundtrip(self, sample_state, pickle_dir):
        with (
            patch("unflow.core.unflow_types.PICKLE_PATH", pickle_dir),
            patch("unflow.core.unflow_types.save_procedure"),
            patch("unflow.core.unflow_types.save_procedure_source"),
        ):
            data = sample_state.serialize()

        with (
            patch("unflow.core.unflow_types.load_procedure", return_value=sample_state.procedure),
            patch("unflow.core.unflow_types.load_procedure_source", return_value=sample_state.procedure_source),
        ):
            restored = RState.from_dict(data)

        assert restored.name == sample_state.name
        assert restored.args == sample_state.args
        assert restored.status == RStateStatus.PENDING

    def test_from_dict__restores_status(self, sample_state, pickle_dir):
        sample_state.run()
        with (
            patch("unflow.core.unflow_types.PICKLE_PATH", pickle_dir),
            patch("unflow.core.unflow_types.save_procedure"),
            patch("unflow.core.unflow_types.save_procedure_source"),
        ):
            data = sample_state.serialize()

        with (
            patch("unflow.core.unflow_types.load_procedure", return_value=sample_state.procedure),
            patch("unflow.core.unflow_types.load_procedure_source", return_value=sample_state.procedure_source),
        ):
            restored = RState.from_dict(data)

        assert restored.status == RStateStatus.COMPLETED


class TestProcedureHelpers:
    def test_save_and_load_procedure(self, tmp_path: Path):
        path = tmp_path / "proc.pkl"

        def my_func():
            return 42

        save_procedure(my_func, path)
        assert path.exists()

        loaded = load_procedure(path)
        assert loaded() == 42

    def test_save_and_load_source(self, tmp_path: Path):
        path = tmp_path / "proc.py"

        def my_func():
            return 42

        save_procedure_source(my_func, path)
        assert path.exists()

        loaded = load_procedure_source(path)
        assert "return 42" in loaded


class TestTransformation:
    def test_init__computes_diffs(self):
        def f1():
            pass

        def f2():
            return 1

        s1 = RState(name="s1", procedure=f1, args={"x": 1})
        s2 = RState(name="s2", procedure=f2, args={"x": 2})

        t = Transformation("t", s1, s2)
        assert t.name == "t"
        assert t.p_changes  # sources differ
        assert t.args_changes  # args differ

    def test_has_changed__true_when_different(self):
        def f1():
            pass

        def f2():
            return 1

        s1 = RState(name="s1", procedure=f1, args={"x": 1})
        s2 = RState(name="s2", procedure=f2, args={"x": 2})
        t = Transformation("t", s1, s2)
        assert t.__has__changed__() is True

    def test_has_changed__false_when_identical(self):
        def f():
            pass

        s1 = RState(name="s1", procedure=f, args={"x": 1})
        s2 = RState(name="s2", procedure=f, args={"x": 1})
        t = Transformation("t", s1, s2)
        assert t.__has__changed__() is False

    def test_dict(self):
        def f():
            pass

        s1 = RState(name="s1", procedure=f, args={"x": 1})
        s2 = RState(name="s2", procedure=f, args={"x": 2})
        t = Transformation("t", s1, s2, description="desc")
        d = t.__dict__()
        assert d["name"] == "t"
        assert d["state1"] == "s1"
        assert d["state2"] == "s2"
        assert d["description"] == "desc"

    def test_from_dict__roundtrip(self):
        def f():
            pass

        s1 = RState(name="s1", procedure=f, args={"x": 1})
        s2 = RState(name="s2", procedure=f, args={"x": 2})
        t = Transformation("t", s1, s2)
        data = t.__dict__()

        restored = Transformation.from_dict(data, s1, s2)
        assert restored.name == "t"
        assert restored.state1 == s1
        assert restored.state2 == s2
