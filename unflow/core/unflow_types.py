from argparse import Namespace
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cloudpickle

from unflow.core.constants import PICKLE_PATH
from unflow.core.diff_code import get_args_changes, get_procedure_changes

Procedure = Callable[..., Any]
DummyProcedure = Callable[..., Any]


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
        self, name: str, procedure: Procedure, args: dict | None, procedure_source: str = "", description: str = ""
    ):
        self.name = name
        self.procedure = procedure
        self.description = description
        self.args = args if args is not None else {}
        self.completed = False
        if not procedure_source:
            self.procedure_source = inspect.getsource(procedure)

    def run(self):
        if self.completed:
            print(f"State {self.name} has already been executed. Skipping execution.")
            return None
        if isinstance(self.args, dict):
            outputs = self.procedure(**self.args)
        elif isinstance(self.args, (tuple, list)):
            outputs = self.procedure(*self.args)
        elif self.args is None:
            outputs = self.procedure()
        else:
            outputs = self.procedure(self.args)
        self.completed = True
        return outputs, self.completed
    
   

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
            "completed": self.completed,
        }

    @staticmethod
    def from_dict(data: dict):
        state = RState.__new__(RState)
        state.name = data["name"]
        state.procedure = load_procedure(Path(data["procedure"]))
        state.procedure_source = load_procedure_source(Path(data["procedure_source"]))
        state.args = data.get("args", {})
        state.description = data.get("description", "")
        state.completed = data.get("completed", False)
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
