import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from .code.context_files import ContextData
from .code.utils import get_context_changes

from ..core.constants import PICKLE_PATH
from ..core.diff_args import get_args_changes
from .code.diff_code import get_procedure_changes
from ..core.json_encoder import dumps
from ..core.change_types import ArgTypeChangeType, ArgValueChangeType, dict2type 

if TYPE_CHECKING:
    from ..reasoning.detection import Intent  # Verify this path exists in your project

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
        context_data: ContextData|None = None,
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
        self.infered_intent: Intent | None = None  # This will be set by the intent detector if applicable
        self.accpeted_intent: Intent | None = None  # This will be set by the user if applicable
        self.context_data = context_data
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
            "infered_intent": self.infered_intent.__dict__() if self.infered_intent else None,
            "accpeted_intent": self.accpeted_intent.__dict__() if self.accpeted_intent else None,
            "context_data": self.context_data.__dict__() if self.context_data else None,
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
        if data.get("context_data"):
            context_dict = data["context_data"]
            context_files = [Path(path) for path in context_dict.get("context_files", [])]
            base_path_value = context_dict.get("base_path")
            base_path = Path(base_path_value) if base_path_value else None
            state.context_data = ContextData(
                data=context_dict.get("data", {}),
                context_files=context_files,
                hashes=context_dict.get("hashes", {}),
                base_path=base_path,
            )
        else:
            state.context_data = None
        infered_intent_data = data.get("infered_intent")

        if infered_intent_data:
            from ..reasoning.detection import Intent

            state.infered_intent = Intent(**infered_intent_data)
        else:
            state.infered_intent = None
        accpeted_intent_data = data.get("accpeted_intent")
        if accpeted_intent_data:
            from ..reasoning.detection import Intent

            state.accpeted_intent = Intent(**accpeted_intent_data)
        else:
            state.accpeted_intent = None
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
        if state1.context_data is None and state2.context_data is None:
            self.context_changes = {}
        else:
            empty_context = ContextData(data={}, context_files=[], hashes={})
            self.context_changes = get_context_changes(
                state1.context_data or empty_context,
                state2.context_data or empty_context,
            )
        self.args_changes = get_args_changes(state1.args, state2.args)

    def __has__changed__(self) -> bool:
        return bool(self.p_changes or self.args_changes or self.context_changes)

    def __dict__(self):
        return {
            "name": self.name,
            "state1": self.state1.name,
            "state2": self.state2.name,
            "description": self.description,
            "procedure_changes": self.p_changes,
            "context_changes": self.context_changes,
            "args_changes": {k: v for k, v in self.args_changes.items()},
        }
    def args_changes_str(self) -> str:
        changes_str = []
        for _, change in self.args_changes.items():
            changes_str.append(str(change))

        return "\n".join(changes_str)

    @staticmethod
    def from_dict(data: dict, state1: RState, state2: RState):
        transformation = Transformation.__new__(Transformation)
        transformation.name = data["name"]
        transformation.state1 = state1
        transformation.state2 = state2
        transformation.description = data.get("description", "")
        transformation.p_changes = data.get("procedure_changes", {})
        transformation.context_changes = data.get("context_changes", {})
        arg_changes = data.get("args_changes", {})
        transformation.args_changes = {
            k: cast(
                ArgValueChangeType | ArgTypeChangeType | tuple[Any, Any],
                v if isinstance(v, (ArgValueChangeType, ArgTypeChangeType)) else dict2type(v),
            )
            for k, v in arg_changes.items()
        }
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
