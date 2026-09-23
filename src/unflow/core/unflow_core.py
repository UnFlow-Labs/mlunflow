"""
Unflow Core Module

This module provides the core functionality of the Unflow framework,
including the unflowdecorator class, which enables graph-based execution of functions,
and methods for managing and querying the compute graph.

The unflowdecorator class wraps a function to enable graph-based execution,
allowing for tracking of execution states, outcomes, and dependencies.
It provides methods for running the function with multiple sets of arguments,
clearing the compute graph, querying states and transformations,
and finding the shortest path between states.

Overall, this module provides the essential components for building and
executing graph-based workflows using the Unflow framework,
enabling users to manage and track the execution of functions in a structured and efficient manner.
"""

from collections.abc import Callable
from functools import wraps
import inspect
from logging import getLogger
from numbers import Number
from typing import Any

from ..core.builder import GraphBuilder
from ..core.code.context_files import ContextFileTracker, read_context_files
from ..core.executors.executor import Executor
from ..core.executors.local_exectuor import LocalExecutor
from ..core.scheduler import Scheduler
from ..core.unflow_types import ExecutionRecord, Outcome, RState, RStateStatus, Transformation

logger = getLogger(__name__)  # Placeholder for a logger instance, if needed


class unflowdecorator:
    """
    A decorator that wraps a function to enable unflow's graph-based execution.
    With this decorator, the function can be executed in a graph-based manner,
    allowing for tracking of execution states, outcomes, and dependencies.
    It also provides additional methods for managing and querying the compute graph.

    Example usage:
    ```python
    from ..core.unflow_core import unflowdecorator

    @unflowdecorator()
    def my_function(x, y):
        return x + y

    my_function(1, 2)  # Executes the function and builds the graph
    my_function.run_multiple([{"args": [1, 2]}, {"args": [3, 4]}])
    my_function.clear_graph()  # Clears the compute graph
    my_function.graph_size()  # Returns number of unique executions

    # Query states without lambdas (recommended for common lookups)
    states = my_function.query_states(
        status="completed",
        args_contains={"x": 1, "y": 2},
    )

    # Query transformations without lambdas
    arg_only_transforms = my_function.query_transformations(
        has_args_changes=True,
        has_procedure_changes=False,
    )

    # Optional: add a custom predicate for advanced filtering
    same_y_transforms = my_function.query_transformations(
        filter_func=lambda t: t.state1.args.get("y") == t.state2.args.get("y")
    )
    ```
    """

    def __init__(self, executor: Executor | None = None, context_files=None, intent_detector=None):
        """
        A decorator that wraps a function to enable unflow's graph-based execution.

        Args:
            executor: An instance of an Executor. If None, LocalExecutor is used by default.
            intent_detector: An instance of an IntentDetector. If None, no intent detection is performed.
        """
        # get the execution path from the caller's frame
        self.graph_builder = GraphBuilder("")
        self.executor = executor or LocalExecutor()
        self.context_files = context_files
        self._context_tracker = ContextFileTracker(context_files) if context_files else None
        self._context_watch_started = False
        self.intent_detector = None
        if intent_detector:
            self.intent_detector = intent_detector
            self.intent_detector._set_graph(self.graph_builder.compute_graph)
            self.intent_detector._set_graph_builder(self.graph_builder)
        self.state : RState | None = None

    def _get_context_data(self):
        if not self.context_files:
            return None
        if self._context_tracker is None:
            return read_context_files(self.context_files, strict=False)

        if not self._context_watch_started:
            try:
                self._context_tracker.start()
            except ModuleNotFoundError:
                logger.warning("watchfiles is not installed; context snapshots will be refreshed on-demand")
            self._context_watch_started = True
        self._context_tracker.update()
        return self._context_tracker.snapshot()

    def stop_context_watch(self):
        if self._context_tracker is not None:
            self._context_tracker.stop()

    def __del__(self):
        self.stop_context_watch()

    def get_execution_path(self, func):
        from pathlib import Path

        directory = Path(func.__code__.co_filename).parent
        return str(directory)

    def set_execution_path(self, execution_path: str):
        self.graph_builder.set_execution_path(execution_path)

    def __call__(self, func):
        """
        Wraps the function to enable unflow's graph-based execution.

        The wrapped function will build the compute graph and execute the function when called.

        Also adds additional methods to the wrapped function for graph management and querying:

        - run_multiple: Executes the function with multiple sets of arguments.
        - clear_graph: Clears the compute graph and resets the execution path.
        - graph_size: Returns the number of nodes in the compute graph.
        - compute_graph: Returns the underlying ComputeGraph instance.
        - get_outcomes: Returns the outcomes of all executed states.
        - query_states: Queries states in the compute graph based on provided filters.
        - query_transformations: Queries transformations in the compute graph based on provided filters.
        - query_with_outcomes: Queries states based on both state filters and outcome filters.
        - shortest_path: Finds the shortest path between two states in the compute graph.

        Args:
            func: The function to be wrapped.

        Returns:
            The wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # once the function is called, we will build the graph and execute the function
            record = self._run_once(func, args, kwargs)
            if record is None:
                return None
            return record.outcome.outputs if record.status == RStateStatus.COMPLETED else None

        wrapper.run_multiple = lambda combos: self.run_multiple(func, combos)
        wrapper.clear_graph = lambda: self.clear_graph(func)
        wrapper.graph_size = lambda: len(self.graph_builder.compute_graph.graph.nodes)
        wrapper.compute_graph = self.graph_builder.compute_graph
        wrapper.get_outcomes = lambda: self.get_outcomes(func)
        wrapper.query_states = lambda filter_func=None, **filters: self.query_states(
            func, filter_func=filter_func, **filters
        )
        wrapper.query_transformations = lambda filter_func=None, **filters: self.query_transformations(
            func, filter_func=filter_func, **filters
        )
        wrapper.shortest_path = lambda from_state, to_state: self.shortest_path(func, from_state, to_state)
        wrapper.query_with_outcomes = lambda outcome_filters, **filters: self.query_with_outcomes(
            func, outcome_filters, **filters
        )
        wrapper.print_graph = lambda: self.print_graph(func)
        wrapper.get_intents = lambda save_in_graph=False, mark_as_accept=False: self.get_intents(
            func, save_in_graph=save_in_graph, mark_as_accept=mark_as_accept
        )
        wrapper.stop_context_watch = lambda: self.stop_context_watch()
        return wrapper
    def print_graph(self, func: Callable) -> str:
        """
        Returns a string representation of the compute graph.

        Returns:
            A string representation of the compute graph.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return str(self.graph_builder.compute_graph)
    
    def _set_intent(self, state: RState):
        if self.intent_detector:
            intent = self.intent_detector.detect_intent(state)
            state.infered_intent = intent
            print(f"Inferred Intent: {intent.name} with confidence {intent.confidence}")

    def _predict_outcome(self, state: RState) -> Any:
        if self.intent_detector is None:
            raise ValueError("Intent detector is not set. Please provide an intent detector.")
        # Use the intent detector to predict the outcome based on the inferred intent
        predicted_outcome = self.intent_detector.predict_outcome(state)
        return predicted_outcome    
    
    def get_intents(self, func: Callable, save_in_graph=False, mark_as_accept=False) -> dict[str, Any]:
        """
        Retrieves the inferred intents of all executed states for the given function.
        Please note that this will take a plenty of time if the graph is large and the LLM is slow.

        Args:
            func: The function whose intents are to be retrieved.
            save_in_graph: If True, saves the inferred intents in the graph.
            mark_as_accept: If True, marks the inferred intents as accepted in the graph.
        Returns:
            A dictionary mapping state names to their corresponding inferred intents.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        intents = {}
        for node in self.graph_builder.compute_graph.graph.nodes:
            state = self.graph_builder.compute_graph.state_map.get(node)
            if state is not None and hasattr(state, "infered_intent"):
                if state.infered_intent is None:
                    self._set_intent(state)
                    if save_in_graph:
                        self.graph_builder.save_graph(func.__name__)
                if mark_as_accept and state.infered_intent is not None:
                    state.accpeted_intent = state.infered_intent
                    if save_in_graph:
                        self.graph_builder.save_graph(func.__name__)
                intents[node] = state.infered_intent
        return intents
    

    def clear_graph(self, func: Callable):
        """
        Clears the compute graph and resets the execution path.

        Args:
            func: The function whose graph is to be cleared.
        """
        self.graph_builder.compute_graph.clear()
        # get the execution path from the caller's frame
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.db.clear_graph(func.__name__, self.graph_builder.compute_graph.execution_path)

    def _build_graph(self, func: Callable, *args, **kwargs):
        """
        Builds the compute graph for the given function and its arguments.

        Args:
            func: The function for which the graph is to be built.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        Returns:
            The new state created in the graph, an outcome object of the previously executed state or None if there is no outcome for this."""
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        try:
            procedure_source = inspect.getsource(func)
        except (OSError, TypeError):
            procedure_source = ""
        context_data = self._get_context_data()
        new_state = self.graph_builder.create_state(
            func,
            *args,
            _unflow_procedure_source=procedure_source,
            _unflow_context_data=context_data,
            **kwargs,
        )
        can_transform, prev_state = self.graph_builder.transform(new_state)
        if not can_transform:
            logger.info("No new state created. The function has already been executed with the same arguments.")
            if prev_state is not None:
                return self.graph_builder.load_outcome(prev_state)
            else:
                return None
        # self._set_intent(new_state)
        self.graph_builder.save_graph(func.__name__)
        self._predict_outcome(new_state)
        return new_state

    def _run_once(self, func: Callable, args, kwargs):
        new_state = self._build_graph(func, *args, **kwargs)
        if new_state is None:
            return None
        if isinstance(new_state, Outcome):
            return ExecutionRecord(status=RStateStatus.COMPLETED, start_time=None, end_time=None, error=None, outcome=new_state)
        scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
        record, graph = scheduler.run(new_state)
        self.graph_builder.finalize(graph, record, func.__name__, new_state)
        return record

    def get_outcomes(self, func: Callable) -> dict[str, Outcome]:
        """
        Retrieves the outcomes of all executed states for the given function.

        Args:
            func: The function whose outcomes are to be retrieved.
        Returns:
            A dictionary mapping state names to their corresponding outcomes.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        outcomes = {}
        for node in self.graph_builder.compute_graph.graph.nodes:
            state = self.graph_builder.compute_graph.state_map.get(node)
            if state is not None:
                outcome = self.graph_builder.load_outcome(state)
                if outcome is not None:
                    outcomes[node] = outcome
        return outcomes

    def query_with_outcomes(
        self, func: Callable, outcome_filters: dict[str, Any], states_filter: Callable[[RState], bool] | None = None
    ) -> list[RState]:
        """
        Queries states in the compute graph based on provided filters and outcome filters.

        Args:
            func: The function whose states are to be queried.
            outcome_filters: A dictionary of outcome attributes to filter states by.
            states_filter: A callable that takes an RState
            and returns a boolean indicating whether the state should be included.
        Returns:
            A list of states that match the provided filters and outcome filters.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        states = self.graph_builder.compute_graph.query_states(predicate=states_filter)
        # filter states to only those that apply the outcome filters
        if outcome_filters is not None:
            filtered_states = []
            for state in states:
                outcome = self.graph_builder.load_outcome(state)
                if outcome is not None:
                    match = True
                    for key, value in outcome_filters.items():
                        if getattr(outcome, key, None) != value:
                            match = False
                            break
                    if match:
                        filtered_states.append(state)
            states = filtered_states
        return states

    def query_states(
        self,
        func: Callable,
        filter_func: Callable[[RState], bool] | None = None,
        status: RStateStatus | str | None = None,
        name_contains: str | None = None,
        args_contains: dict | None = None,
    ) -> list[RState]:
        """
        Queries states in the compute graph based on provided filters.

        Args:
            func: The function whose states are to be queried.
            filter_func: A callable that takes an RState and
            returns a boolean indicating whether the state should be included.
            status: Optional state status to filter by.
            name_contains: Optional substring to match within the state name.
            args_contains: Optional partial args dict to match against state args.
        Returns:
            A list of states that match the provided filters.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.query_states(
            status=status,
            name_contains=name_contains,
            args_contains=args_contains,
            predicate=filter_func,
        )

    def query_transformations(
        self,
        func: Callable,
        filter_func: Callable[[Transformation], bool] | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        has_args_changes: bool | None = None,
        has_procedure_changes: bool | None = None,
    ) -> list[Transformation]:
        """
        Queries transformations in the compute graph based on provided filters.

        Args:
            func: The function whose transformations are to be queried.
            filter_func: A callable that takes a
            Transformation and returns a boolean indicating whether the transformation should be included.
            from_state: Optional source state name to filter transformations.
            to_state: Optional target state name to filter transformations.
            has_args_changes: Optional filter for transformations with args changes.
            has_procedure_changes: Optional filter for transformations with procedure/source changes.
        Returns:
            A list of transformations that match the provided filters.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.query_transformations(
            from_state=from_state,
            to_state=to_state,
            has_args_changes=has_args_changes,
            has_procedure_changes=has_procedure_changes,
            predicate=filter_func,
        )

    def shortest_path(self, func: Callable, from_state: str, to_state: str) -> list[RState]:
        """
        Finds the shortest path between two states in the compute graph.
        Args:
            func: The function whose graph is to be queried.
            from_state: The name of the starting state.
            to_state: The name of the target state.
        Returns:
            A list of state names representing the shortest path from from_state to to_state.
        """
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.shortest_path(from_state, to_state)

    def _extract_outcome_score(self, outputs: Any, output_key: str | None = None, scorer: Any = None) -> Number:
        """
        Extracts a numeric score from the outcome outputs based on the provided output_key or scorer function.

        Args:
            outputs: The outcome outputs from which to extract the score.
            output_key: The key to use when extracting the score from a dictionary of outputs.
            scorer: A custom function to compute the score from the outputs.
        Returns:
            The extracted numeric score.
        """
        if scorer is not None:
            score = scorer(outputs)
            if not isinstance(score, Number):
                raise ValueError("scorer must return a numeric value")
            return score

        if output_key is not None:
            if not isinstance(outputs, dict):
                raise ValueError("output_key can only be used when outcome outputs are dictionaries")
            if output_key not in outputs:
                raise ValueError(f"output_key '{output_key}' not found in outcome outputs")
            value = outputs[output_key]
            if not isinstance(value, Number):
                raise ValueError(f"outcome value for key '{output_key}' is not numeric")
            return value

        if not isinstance(outputs, Number):
            raise ValueError("outcome outputs must be numeric when output_key/scorer are not provided")
        return outputs

    def run_multiple(self, func: Callable, combinations: list[dict[str, Any]]) -> list[dict[str, ExecutionRecord]]:
        """
        Executes the function with multiple sets of arguments, building the compute graph for each set.
        After building the graph, it runs the scheduler to execute the states and records the outcomes.

        Args:
            func: The function to be executed.
            combinations: A list of dictionaries, each containing
            a set of arguments (args and/or kwargs) for the function.
        Returns:
            A list of records, each containing the outcome and status of the executed states.
        """
        results = []
        new_states = []
        for combo in combinations:
            if "args" in combo or "kwargs" in combo:
                args = combo.get("args", [])
                kwargs = combo.get("kwargs", {})
            else:
                args = []
                kwargs = combo
            new_state = self._build_graph(func, *args, **kwargs)
            if new_state is not None:
                new_states.append(new_state)
        for new_state in new_states:
            self.graph_builder.set_status(new_state.name, RStateStatus.READY)
        if len(new_states) > 0:
            scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
            record, graph = scheduler.run(new_states[-1])
            self.graph_builder.finalize(graph, record, func.__name__, new_states[-1])
            results.append(record)

        return results
