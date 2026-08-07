import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cloudpickle

from ..core.constants import PICKLE_PATH
from ..core.diff_args import get_args_changes
from ..core.diff_code import get_procedure_changes
from ..core.json_encoder import dumps

Procedure = Callable[..., Any]
DummyProcedure = Callable[..., Any]


class RStateStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def save_procedure(procedure: Procedure, path: Path):
    # Save the procedure to a file or database with the given name
    # This is a placeholder implementation; you can customize it as needed
    with open(path, "wb") as f:
        cloudpickle.dump(procedure, f)


def save_procedure_source(procedure: Procedure, path: Path):
    # Save the source code of the procedure to a file with the given name
    # This is a placeholder implementation; you can customize it as needed

    source_code = inspect.getsource(procedure)
    with open(path, "w") as f:
        f.write(source_code)


def load_procedure_source(path: Path) -> str:
    # Load the source code of the procedure from a file with the given name
    # This is a placeholder implementation; you can customize it as needed
    with open(path) as f:
        return f.read()


def load_procedure(path: Path) -> Procedure:
    # Load the procedure from a file or database with the given name
    # This is a placeholder implementation; you can customize it as needed
    with open(path, "rb") as f:
        return cloudpickle.load(f)


class RState:
    def __init__(
        self,
        name: str,
        procedure: Procedure,
        args: dict | None,
        kwargs: dict | None = None,
        procedure_source: str = "",
        description: str = "",
    ):
        self.name = name
        self.procedure = procedure
        self.description = description
        self.args = args if args is not None else {}
        self.status = RStateStatus.PENDING
        if not procedure_source:
            self.procedure_source = inspect.getsource(procedure)
        else:
            self.procedure_source = procedure_source

    def run(self):
        if self.status == RStateStatus.COMPLETED:
            print(f"State {self.name} has already been executed. Skipping execution.")
            return None
        # Execute the procedure with the provided arguments and keyword arguments
        self.status = RStateStatus.RUNNING
        outputs = self.procedure(**self.args)
        self.status = RStateStatus.COMPLETED
        return outputs, self.status

    def serialize(self):
        name = f".{self.procedure.__name__}_{self.name}"
        os.makedirs(PICKLE_PATH / name, exist_ok=True)
        path_proc = PICKLE_PATH / name / f"{self.procedure.__name__}_{self.name}.pkl"
        save_procedure(self.procedure, path_proc)
        save_procedure_source(self.procedure, path_proc.with_suffix(".py"))
        return {
            "name": self.name,
            "procedure": str(path_proc),
            "procedure_source": str(path_proc.with_suffix(".py")),
            "args": self.args,
            "description": self.description,
            "status": self.status.value,
        }

    @staticmethod
    def from_dict(data: dict):
        state = RState.__new__(RState)
        state.name = data["name"]
        state.procedure = load_procedure(Path(data["procedure"]))
        state.procedure_source = load_procedure_source(Path(data["procedure_source"]))
        state.args = data.get("args", {})
        state.description = data.get("description", "")
        state.status = RStateStatus(data.get("status", "pending"))
        return state


class Transformation:
    """
    A transformation is between two states.
    """

    def __init__(self, name: str, state1: RState, state2: RState, description: str = ""):
        self.name = name
        self.state1 = state1
        self.state2 = state2
        self.description = description
        self.p_changes = get_procedure_changes(state1.procedure_source, state2.procedure_source)
        self.args_changes = get_args_changes(state1.args, state2.args)

    def __has__changed__(self) -> bool:
        return bool(self.p_changes or self.args_changes)

    def __dict__(self):
        return {
            "name": self.name,
            "state1": self.state1.name,
            "state2": self.state2.name,
            "description": self.description,
            "procedure_changes": self.p_changes,
            "args_changes": self.args_changes,
        }

    @staticmethod
    def from_dict(data: dict, state1: RState, state2: RState):
        transformation = Transformation.__new__(Transformation)
        transformation.name = data["name"]
        transformation.state1 = state1
        transformation.state2 = state2
        transformation.description = data.get("description", "")
        transformation.p_changes = data.get("procedure_changes", {})
        transformation.args_changes = data.get("args_changes", {})
        return transformation


class Outcome:
    def __init__(self, state: RState, outputs: dict[str, Any] | None = None):
        """
        An outcome represents the result of executing a state.
        :param state: The state that was executed.
        :param outputs: The outputs produced by the state execution."""
        self.state = state
        self.outputs = outputs

    def __dict__(self):
        return {
            "state": self.state.name,
            "outputs": self.outputs,
        }

    @staticmethod
    def from_dict(data: dict, state: RState):
        outcome = Outcome.__new__(Outcome)
        outcome.state = state
        outcome.outputs = data.get("outputs")
        return outcome

    def to_json(self):
        return dumps(self.__dict__())


@dataclass
class Job:
    state: RState
    output: object = None
    error: Exception | None = None
    completed: bool = False
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class ExecutionRecord:
    status: RStateStatus = RStateStatus.PENDING
    outcome: Outcome | None = None
    error: Exception | None = None
    state_name: str | None = None
    start_time: float | None = None
    end_time: float | None = None
