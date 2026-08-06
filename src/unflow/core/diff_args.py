import re
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import deepdiff
from orderly_set import OrderedSet

from unflow.core.change_types import ArgTypeChangeFactory, ArgTypeChangeType, ArgValueChangeFactory, ArgValueChangeType

if TYPE_CHECKING:
    pass

_TOKEN_RE = re.compile(
    r"""
    \[
        (?:
            '((?:\\.|[^'])*)' |      # quoted string key
            "((?:\\.|[^"])*)" |      # double quoted string key
            (-?\d+)                  # integer index
        )
    \]
    |
    \.([A-Za-z_]\w*)                 # attribute
    """,
    re.VERBOSE,
)

# PyTorch internal attributes that should be excluded from comparison
_PYTORCH_INTERNAL_ATTRS = {
    '_forward_hooks',
    '_backward_hooks',
    '_forward_pre_hooks',
    '_backward_pre_hooks',
    '_forward_hooks_with_kwargs',
    '_forward_pre_hooks_with_kwargs',
    '_backward_pre_hooks_with_kwargs',
    '_state_dict_hooks',
    '_state_dict_pre_hooks',
    '_load_state_dict_pre_hooks',
    '_load_state_dict_post_hooks',
    '_forward_hooks_always_called',
    '_is_full_backward_hook',
    '_non_persistent_buffers_set',
}

def _is_pytorch_module(value: Any) -> bool:
    """Check if value is a PyTorch Module or similar."""
    try:
        return hasattr(value, '_parameters') and hasattr(value, '_buffers')
    except Exception:
        return False

def _to_comparable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_comparable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_comparable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_comparable(item) for item in value)
    if isinstance(value, set):
        return {_to_comparable(item) for item in value}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_comparable(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_comparable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _to_comparable(value.dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        obj_dict = vars(value)
        # Filter out PyTorch internal attributes
        if _is_pytorch_module(value):
            obj_dict = {
                k: v for k, v in obj_dict.items() 
                if k not in _PYTORCH_INTERNAL_ATTRS
            }
        return _to_comparable(obj_dict)
    return value


def normalize_args(args: dict) -> dict:
    """
    Normalize the argument dictionary by converting all values to their string representations.
    This is useful for comparing argument dictionaries in a consistent manner.

    Args:
        args (dict): The argument dictionary to normalize.

    Returns:
        dict: A new dictionary with the same keys but with all values converted to strings.
    """
    return _to_comparable(args)

def parse_deepdiff_path(path: str) -> tuple[list[str], str]:
    """
    Returns (parents, key)

    Examples:
        ``root['p1']['p2']['k'] -> (['p1', 'p2'], 'k')``
        ``root['p1'].k          -> (['p1'], 'k')``
        ``root[0].name          -> (['0'], 'name')``
        ``root['a'][2]['b']     -> (['a', '2'], 'b')``
    """
    if path == "root":
        return [], "root"

    tokens = []

    for sq, dq, index, attr in _TOKEN_RE.findall(path):
        if sq:
            tokens.append(sq)
        elif dq:
            tokens.append(dq)
        elif index:
            tokens.append(index)
        else:
            tokens.append(attr)

    if not tokens:
        raise ValueError(f"Invalid DeepDiff path: {path}")

    return [str(token) for token in tokens[:-1]], str(tokens[-1])
  

def get_args_changes(args1: dict, args2: dict) -> dict[str, ArgValueChangeType | ArgTypeChangeType | tuple[Any, Any]]:
    comparable_args1 = normalize_args(args1)
    comparable_args2 = normalize_args(args2)
    diff = deepdiff.DeepDiff(comparable_args1, comparable_args2, ignore_order=True)
    changes: dict[str, ArgValueChangeType | ArgTypeChangeType | tuple[Any, Any]] = {}
    for change_type in ["values_changed", "type_changes", "dictionary_item_added", "dictionary_item_removed"]:
        if change_type in diff:
            if isinstance(diff[change_type], dict):
                for key, value in diff[change_type].items():
                    parents, arg_name = parse_deepdiff_path(key)
                    if change_type == "values_changed":
                        changes[arg_name] = ArgValueChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_value=value["old_value"],
                            to_value=value["new_value"],
                        )
                    elif change_type == "type_changes":
                        changes[arg_name] = ArgTypeChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_type=value["old_type"],
                            to_type=value["new_type"],
                        )
                    elif change_type == "dictionary_item_added":
                        parents, arg_name = parse_deepdiff_path(key)
                        changes[arg_name] = ArgValueChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_value=None,
                            to_value=value,
                        )
                    elif change_type == "dictionary_item_removed":
                        parents, arg_name = parse_deepdiff_path(key)
                        changes[arg_name] = ArgValueChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_value=value,
                            to_value=None,
                        )
            elif isinstance(diff[change_type], (OrderedSet, Iterable)):
                for item in diff[change_type]:
                    if change_type == "dictionary_item_added":
                        parents, arg_name = parse_deepdiff_path(item)
                        arg_name = str(arg_name)
                        parents = [str(parent) for parent in parents]
                        changes[arg_name] =ArgValueChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_value=None,
                            to_value="<added>",
                        )
                    elif change_type == "dictionary_item_removed":
                        parents, arg_name = parse_deepdiff_path(item)
                        arg_name = str(arg_name)
                        parents = [str(parent) for parent in parents]
                        changes[arg_name] = ArgValueChangeFactory(
                            arg_name=arg_name,
                            parents=parents,
                            from_value="<removed>",
                            to_value=None,
                        )
    return changes


def diff_args(args1: dict, args2: dict) -> bool:
    return bool(get_args_changes(args1, args2))