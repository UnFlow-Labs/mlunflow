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
