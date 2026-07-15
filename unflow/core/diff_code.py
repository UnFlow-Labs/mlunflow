import difflib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import deepdiff
from orderly_set import OrderedSet

from unflow.core.change_types import ArgTypeChangeType, ArgValueChangeType

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


def get_arg_name(arg_key: str) -> str:
    """
    Extracts the argument name from a deepdiff key.

    Args:
        arg_key (str): The deepdiff key representing the argument.
    Returns:
        str: The extracted argument name.
    """
    # Split the key by '[' and ']' to isolate the argument name
    parts = arg_key.split("[")
    if len(parts) > 1:
        arg_name = parts[-1].strip("]").replace("'", "")
        return arg_name
    return arg_key  # Return the original key if it doesn't match the expected format


def get_args_changes(args1: dict, args2: dict) -> dict[str, ArgValueChangeType | ArgTypeChangeType | tuple[Any, Any]]:
    diff = deepdiff.DeepDiff(args1, args2, ignore_order=True)
    changes: dict[str, ArgValueChangeType | ArgTypeChangeType | tuple[Any, Any]] = {}
    for change_type in ["values_changed", "type_changes", "dictionary_item_added", "dictionary_item_removed"]:
        if change_type in diff:
            if isinstance(diff[change_type], dict):
                for key, value in diff[change_type].items():
                    arg_name = get_arg_name(key)
                    if change_type == "values_changed":
                        changes[arg_name] = ArgValueChangeType(
                            # get the name of the argument from the key
                            name=f"{arg_name}Change",
                            description=f"Value changed from {value['old_value']} to {value['new_value']}",
                            arg_name=arg_name,
                            from_value=value["old_value"],
                            to_value=value["new_value"],
                        )
                    elif change_type == "type_changes":
                        changes[arg_name] = ArgTypeChangeType(
                            name=f"{arg_name}Change",
                            description=f"Type changed from {value['old_type']} to {value['new_type']}",
                            arg_name=arg_name,
                            from_type=value["old_type"],
                            to_type=value["new_type"],
                        )
                    elif change_type == "dictionary_item_added":
                        arg_name = get_arg_name(key)
                        changes[arg_name] = ArgValueChangeType(
                            name=f"{arg_name}Change",
                            description=f"Argument added with value {value}",
                            arg_name=arg_name,
                            from_value=None,
                            to_value=value,
                        )
                    elif change_type == "dictionary_item_removed":
                        arg_name = get_arg_name(key)
                        changes[arg_name] = ArgValueChangeType(
                            name=f"{arg_name}Change",
                            description=f"Argument removed with value {value}",
                            arg_name=arg_name,
                            from_value=value,
                            to_value=None,
                        )
            elif isinstance(diff[change_type], (OrderedSet, Iterable)):
                for item in diff[change_type]:
                    if change_type == "dictionary_item_added":
                        arg_name = get_arg_name(item)
                        changes[arg_name] = ArgValueChangeType(
                            name=f"{arg_name}Change",
                            description="Argument added with value <added>",
                            arg_name=arg_name,
                            from_value=None,
                            to_value="<added>",
                        )
                    elif change_type == "dictionary_item_removed":
                        arg_name = get_arg_name(item)
                        changes[arg_name] = ArgValueChangeType(
                            name=f"{arg_name}Change",
                            description="Argument removed with value <removed>",
                            arg_name=arg_name,
                            from_value="<removed>",
                            to_value=None,
                        )
    return changes


def diff_args(args1: dict, args2: dict) -> bool:
    return bool(get_args_changes(args1, args2))
