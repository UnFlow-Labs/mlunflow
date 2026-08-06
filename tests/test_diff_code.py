from unflow.core.diff_args import (
    diff_args,
    get_args_changes,
)
from unflow.core.diff_code import (
    diff_procedure,
    get_procedure_changes,
)

SRC_A = """
def foo():
    return 1
"""

SRC_B = """
def foo():
    return 2
"""

SRC_A_STRIPPED = "def foo():\n    return 1"
SRC_B_STRIPPED = "def foo():\n    return 2"


class TestGetProcedureChanges:
    def test_identical_sources(self):
        assert get_procedure_changes(SRC_A, SRC_A) == {}

    def test_different_sources(self):
        changes = get_procedure_changes(SRC_A, SRC_B)
        assert isinstance(changes, dict)
        assert len(changes) > 0
        # should contain diff lines
        lines = "".join(changes.values())
        assert "-    return 1" in lines or "return 1" in lines
        assert "+    return 2" in lines or "return 2" in lines


class TestDiffProcedure:
    def test_identical(self):
        assert diff_procedure(SRC_A, SRC_A) is False

    def test_different(self):
        assert diff_procedure(SRC_A, SRC_B) is True


class TestGetArgsChanges:
    def test_identical_args(self):
        assert get_args_changes({"a": 1}, {"a": 1}) == {}

    def test_values_changed(self):
        changes = get_args_changes({"a": 1}, {"a": 2})
        assert any("a" in str(k) for k in changes)

    def test_key_added(self):
        changes = get_args_changes({"a": 1}, {"a": 1, "b": 2})
        assert len(changes) > 0

    def test_key_removed(self):
        changes = get_args_changes({"a": 1, "b": 2}, {"a": 1})
        assert len(changes) > 0

    def test_empty_dicts(self):
        assert get_args_changes({}, {}) == {}


class TestDiffArgs:
    def test_identical(self):
        assert diff_args({"a": 1}, {"a": 1}) is False

    def test_different(self):
        assert diff_args({"a": 1}, {"a": 2}) is True
