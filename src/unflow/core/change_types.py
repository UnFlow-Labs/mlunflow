from typing import Any

from pydantic import BaseModel


class ChangeType(BaseModel):
    """
    A class representing a change type in the Unflow framework.

    Attributes:
        name (str): The name of the change type.
        description (str): A brief description of the change type.
    """

    name: str  # name to be used in the graph
    description: str


class ArgValueChangeType(ChangeType):
    """
    A class representing an argument value change type in the Unflow framework.

    Attributes:
        name (str): The name of the change type.
        description (str): A brief description of the change type.
    """

    arg_name: str
    description: str
    from_value: Any | None = None
    to_value: Any | None = None


class ArgTypeChangeType(ChangeType):
    """
    A class representing an argument type change type in the Unflow framework.

    Attributes:
        name (str): The name of the change type.
        description (str): A brief description of the change type.
    """

    arg_name: str
    description: str
    from_type: type | None = None
    to_type: type | None = None


class CodeChangeType(ChangeType):
    """
    A class representing a code change type in the Unflow framework.

    Attributes:
        name (str): The name of the change type.
        description (str): A brief description of the change type.
    """

    description: str
    from_code: str | None = None
    to_code: str | None = None

def normalize_arg_name(arg_name: str) -> str:
    """
    Normalize the argument name by removing any leading or trailing whitespace,
    removing underscores, and converting to lowercase. This ensures that argument names are consistent and can be compared accurately.

    Args:
        arg_name (str): The argument name to normalize.
    returns:
        str: The normalized argument name.
    """
    return arg_name.strip().replace("_", "").lower() if isinstance(arg_name, str) else arg_name

def get_full_arg_name(parents: list[str], arg_name: str) -> str:
    """
    Construct the full argument name by joining the parent names and the argument name.

    Args:
        parents (list[str]): A list of parent names.
        arg_name (str): The argument name.

    Returns:
        str: The full argument name.
    """
    return ".".join(parents + [arg_name]) if parents else arg_name

def ArgValueChangeFactory(arg_name: str, parents:list[str], from_value: Any | None, to_value: Any | None) -> ArgValueChangeType:
    """
    Factory function to create an instance of ArgValueChangeType.

    Args:
        arg_name (str): The name of the argument that changed.
        from_value (Any | None): The original value of the argument.
        to_value (Any | None): The new value of the argument.

    Returns:
        ArgValueChangeType: An instance representing the argument value change.
    """
    full_arg_name = get_full_arg_name(parents, arg_name)
    return ArgValueChangeType(
        name=f"{full_arg_name} Changed",
        description=f"Value changed from {from_value} to {to_value}",
        arg_name=normalize_arg_name(arg_name),
        from_value=from_value,
        to_value=to_value,
    )
def ArgTypeChangeFactory(arg_name: str, parents:list[str], from_type: type | None, to_type: type | None) -> ArgTypeChangeType:
    """
    Factory function to create an instance of ArgTypeChangeType.

    Args:
        arg_name (str): The name of the argument that changed.
        from_type (type | None): The original type of the argument.
        to_type (type | None): The new type of the argument.

    Returns:
        ArgTypeChangeType: An instance representing the argument type change.
    """
    full_arg_name = get_full_arg_name(parents, arg_name)
    return ArgTypeChangeType(
        name=f"{full_arg_name} Changed",
        description=f"Type changed from {from_type} to {to_type}",
        arg_name=normalize_arg_name(arg_name),
        from_type=from_type,
        to_type=to_type,
    )
def CodeChangeFactory(from_code: str | None, to_code: str | None) -> CodeChangeType:
    """
    Factory function to create an instance of CodeChangeType.

    Args:
        from_code (str | None): The original code.
        to_code (str | None): The new code.

    Returns:
        CodeChangeType: An instance representing the code change.
    """
    return CodeChangeType(
        name="Code Changed",
        description=f"Code changed from {from_code} to {to_code}",
        from_code=from_code,
        to_code=to_code,
    )