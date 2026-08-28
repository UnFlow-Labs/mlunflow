from unflow.core.diff_args import (
    diff_args,
    get_args_changes,
)
from core.code.diff_code import (
    build_procedure_change_prompt,
    diff_procedure,
    format_procedure_changes_for_llm,
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


class TestProcedureChangesLLMFormatting:
    def test_format_procedure_changes_for_llm(self):
        changes = get_procedure_changes(SRC_A_STRIPPED, SRC_B_STRIPPED)

        payload = format_procedure_changes_for_llm(changes)

        assert payload["change_type"] == "modified"
        assert payload["added_lines"] >= 1
        assert payload["removed_lines"] >= 1
        assert payload["hunk_headers"]
        assert payload["patch"]

    def test_build_procedure_change_prompt(self):
        changes = get_procedure_changes(SRC_A_STRIPPED, SRC_B_STRIPPED)

        prompt = build_procedure_change_prompt(changes)

        assert '"task"' in prompt
        assert '"instructions"' in prompt
        assert '"change"' in prompt


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

    def test_key_added_uses_new_value(self):
        changes = get_args_changes({"a": 1}, {"a": 1, "b": 2})
        assert changes["b"].to_value == 2

    def test_key_removed_uses_old_value(self):
        changes = get_args_changes({"a": 1, "b": 2}, {"a": 1})
        assert changes["b"].from_value == 2

    def test_empty_dicts(self):
        assert get_args_changes({}, {}) == {}


class TestDiffArgs:
    def test_identical(self):
        assert diff_args({"a": 1}, {"a": 1}) is False

    def test_different(self):
        assert diff_args({"a": 1}, {"a": 2}) is True
