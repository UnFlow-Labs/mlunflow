import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
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

# Procedure = Callable[..., Any]
# DummyProcedure = Callable[..., Any]

class Procedure:
    '''
    A Procedure can be a callable function , class with a __call__ method, a python script, or a program in general. 
    It represents a unit of work that can be executed with given arguments and keyword arguments.
    '''
    def __init__(self, func: Callable[..., Any]| None = None, 
                 script_path: Path | None = None, executable: str | None = None,
                 python_path: str | None = None):
        self.func = func
        self.script_path = script_path
        self.executable = executable
        self.python_path = python_path
        self.procedure_source = self._get_procedure_source()
        self.name = self.set_name()
    def set_name(self):
        if self.func:
            return self.func.__name__
        elif self.script_path:
            return self.script_path.name    
        elif self.executable:
            return Path(self.executable).name
        else:
            raise ValueError("Cannot set name for a Procedure without a function, script, or executable.")
    @staticmethod
    def save_callable(func: Callable[..., Any], path: Path):
            with open(path, "wb") as f:
                cloudpickle.dump(func, f)
    @staticmethod
    def load_callable(path: Path) -> Callable[..., Any]:
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    def _get_procedure_source(self) -> str:
        if self.func:
            return inspect.getsource(self.func)
        elif self.script_path:
            return self.script_path.read_text()
        elif self.executable:
            # For an external program, we might not have source code available.
            return f"External executable: {self.executable}"
        else:
            return "No procedure source available."
    
    def __call__(self, *args, **kwargs) -> Any:
        if self.func:
            return self.func(*args, **kwargs)
        elif self.script_path:
            # Execute the script with the provided arguments
            # This is a placeholder implementation; you can customize it as needed
            import subprocess
            cmd = [self.python_path or "python", str(self.script_path)] + list(args)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        elif self.executable:
            # Execute the external program with the provided arguments
            # This is a placeholder implementation; you can customize it as needed
            import subprocess
            cmd = [self.executable] + list(args)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
    
    def _save_procedure_source(self, path: Path):
        source_code = self._get_procedure_source()
        with open(path, "w") as f:
            f.write(source_code)
    @staticmethod
    def get_current_python_path() -> str:
        """Get the path to the current Python interpreter."""
        return os.path.abspath(sys.executable)
    def to_dict(self, tag: str = "") -> dict:
        if self.func:
            func_path = PICKLE_PATH / f"{self.func.__name__}_{tag}.pkl"
            self.save_callable(self.func, func_path)
        self._save_procedure_source(PICKLE_PATH / f"{self.name}_{tag}.py")
        return {
            "func": str(self.func) if self.func else None,
            "script_path": str(self.script_path) if self.script_path else None,
            "executable": self.executable,
            "python_path": self.python_path,
        }
    @staticmethod
    def from_dict(data: dict) -> "Procedure":
        func = Procedure.load_callable(Path(data["func"])) if data.get("func") else None
        return Procedure(
            func=func,
            script_path=Path(data["script_path"]) if data.get("script_path") else None,
            executable=data.get("executable"),
            python_path=data.get("python_path"),
        )
    


class RStateStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"



class RState:
    def __init__(
        self,
        name: str,
        procedure: Procedure,
        args: dict | None,
        kwargs: dict | None = None,
        context_data: ContextData|None = None,
        description: str = "",
    ):
        self.name = name
        self.procedure = procedure
        self.description = description
        self.args = args if args is not None else {}
        self.status = RStateStatus.PENDING
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
        procedure = self.procedure.to_dict(tag=self.name)
        return {
            "name": self.name,
            "procedure": procedure,
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
        state.procedure = Procedure.from_dict(data["procedure"])
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
        self.p_changes = get_procedure_changes(state1.procedure.procedure_source, state2.procedure.procedure_source)
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
