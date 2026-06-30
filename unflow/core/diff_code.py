import difflib
from typing import TYPE_CHECKING, Any

import deepdiff
from orderly_set import OrderedSet

if TYPE_CHECKING:
    pass


def get_procedure_changes(source1: str, source2: str) -> dict[int, str]:
    changes = {}
    if source1 != source2:
        diff = difflib.unified_diff(
            source1.splitlines(),
            source2.splitlines(),
            fromfile="previous",
            tofile="current",
            lineterm="",
        )
        # Store the diff as dictionary with line numbers and changes
        changes = dict(enumerate(diff, start=1))
    return changes


def diff_procedure(source1: str, source2: str) -> bool:
    return bool(get_procedure_changes(source1, source2))


def get_args_changes(args1: dict, args2: dict) -> dict[str, tuple[Any, Any]]:
    diff = deepdiff.DeepDiff(args1, args2, ignore_order=True)
    changes = {}
    for change_type in ["values_changed", "type_changes", "dictionary_item_added", "dictionary_item_removed"]:
        if change_type in diff:
            if isinstance(diff[change_type], dict):
                for key, value in diff[change_type].items():
                    if change_type == "values_changed":
                        changes[key] = (value["old_value"], value["new_value"])
                    elif change_type == "type_changes":
                        changes[key] = (value["old_type"], value["new_type"])
                    elif change_type == "dictionary_item_added":
                        changes[key] = (None, value)
                    elif change_type == "dictionary_item_removed":
                        changes[key] = (value, None)
            if isinstance(diff[change_type], OrderedSet):
                for item in diff[change_type]:
                    print(item)
                    changes[item] = (None, None)  # Placeholder for added/removed items
    return changes


def diff_args(args1: dict, args2: dict) -> bool:
    return bool(get_args_changes(args1, args2))
