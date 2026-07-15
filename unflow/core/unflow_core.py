from functools import wraps
from logging import getLogger
from numbers import Number
from statistics import mean

from unflow.core.builder import GraphBuilder
from unflow.core.executors.executor import Executor
from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.scheduler import Scheduler
from unflow.core.unflow_types import RStateStatus

logger = getLogger(__name__)  # Placeholder for a logger instance, if needed


class unflowdecorator:
    def __init__(self, executor: Executor | None = None):
        """
        A decorator that wraps a function to enable unflow's graph-based execution.
        :param executor: An instance of an Executor. If None, LocalExecutor is used.
        :param re_execute_all: If True, all states will be re-executed even if they have been executed before.
        """
        # get the execution path from the caller's frame
        self.graph_builder = GraphBuilder("")
        self.executor = executor or LocalExecutor()

    def get_execution_path(self, func):
        from pathlib import Path

        directory = Path(func.__code__.co_filename).parent
        return str(directory)

    def set_execution_path(self, execution_path: str):
        self.graph_builder.set_execution_path(execution_path)

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # once the function is called, we will build the graph and execute the function
            record = self.run_once(func, args, kwargs)
            if record is None:
                return None
            return record.outcome.outputs if record.status == RStateStatus.COMPLETED else None

        wrapper.run_multiple = lambda combos: self.run_multiple(func, combos)
        wrapper.clear_graph = lambda: self._clear_graph(func)
        wrapper.graph_size = lambda: len(self.graph_builder.compute_graph.graph.nodes)
        wrapper.compute_graph = self.graph_builder.compute_graph
        wrapper.get_outcomes = lambda: self.get_outcomes(func)
        wrapper.query_states = lambda **filters: self.query_states(func, **filters)
        wrapper.query_transformations = lambda **filters: self.query_transformations(func, **filters)
        wrapper.shortest_path = lambda from_state, to_state: self.shortest_path(func, from_state, to_state)
        wrapper.shortest_path_to_lowest_outcome = (
            lambda output_key=None, from_state=None, scorer=None: self.shortest_path_to_lowest_outcome(
                func,
                output_key=output_key,
                from_state=from_state,
                scorer=scorer,
            )
        )
        wrapper.datasets_amplifying_model_differences = (
            lambda dataset_arg="data", model_arg="model", fixed_args=None, output_key=None, scorer=None: self.datasets_amplifying_model_differences(
                func,
                dataset_arg=dataset_arg,
                model_arg=model_arg,
                fixed_args=fixed_args,
                output_key=output_key,
                scorer=scorer,
            )
        )

        return wrapper

    def _clear_graph(self, func):
        self.graph_builder.compute_graph.clear()
        # get the execution path from the caller's frame
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.db.clear_graph(func.__name__, self.graph_builder.compute_graph.execution_path)

    def build_graph(self, func, *args, **kwargs):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        new_state = self.graph_builder.create_state(func, *args, **kwargs)
        can_transform = self.graph_builder.transform(new_state)
        if not can_transform:
            logger.info("No new state created. The function has already been executed with the same arguments.")
            return None
        self.graph_builder.save_graph(func.__name__)
        return new_state

    def run_once(self, func, args, kwargs):
        new_state = self.build_graph(func, *args, **kwargs)
        if new_state is None:
            return None
        scheduler = Scheduler(self.graph_builder.compute_graph, self.executor)
        record, graph = scheduler.run(new_state)
        self.graph_builder.finalize(graph, record, func.__name__, new_state)
        return record

    def get_outcomes(self, func):
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

    def query_states(self, func, **filters):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.query_states(**filters)

    def query_transformations(self, func, **filters):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.query_transformations(**filters)

    def shortest_path(self, func, from_state: str, to_state: str):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)
        return self.graph_builder.compute_graph.shortest_path(from_state, to_state)

    def _extract_outcome_score(self, outputs, output_key=None, scorer=None):
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

    def shortest_path_to_lowest_outcome(self, func, output_key=None, from_state=None, scorer=None):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)

        scored_states = []
        for state in self.graph_builder.compute_graph.get_states():
            outcome = self.graph_builder.load_outcome(state)
            if outcome is None or outcome.outputs is None:
                continue
            score = self._extract_outcome_score(outcome.outputs, output_key=output_key, scorer=scorer)
            scored_states.append((state.name, score))

        if not scored_states:
            raise ValueError("No comparable outcomes found in graph")

        target_state, lowest_score = min(scored_states, key=lambda item: item[1])

        if from_state is not None:
            path_states = self.graph_builder.compute_graph.shortest_path(from_state, target_state)
        else:
            source_nodes = [
                node for node in self.graph_builder.compute_graph.graph.nodes if self.graph_builder.compute_graph.graph.in_degree(node) == 0
            ]
            if not source_nodes:
                raise ValueError("Graph has no source states to compute shortest path")

            best_path = None
            for source in source_nodes:
                try:
                    candidate_path = self.graph_builder.compute_graph.shortest_path(source, target_state)
                except Exception:
                    continue
                if best_path is None or len(candidate_path) < len(best_path):
                    best_path = candidate_path

            if best_path is None:
                raise ValueError("No path found to the lowest outcome state")
            path_states = best_path

        return {
            "target_state": target_state,
            "lowest_outcome": lowest_score,
            "path": [state.name for state in path_states],
        }

    def datasets_amplifying_model_differences(
        self,
        func,
        dataset_arg="data",
        model_arg="model",
        fixed_args=None,
        output_key=None,
        scorer=None,
    ):
        self.set_execution_path(self.get_execution_path(func))
        self.graph_builder.load_graph(func.__name__)

        fixed_args = fixed_args or {}
        grouped_scores = {}

        for state in self.graph_builder.compute_graph.get_states():
            if dataset_arg not in state.args or model_arg not in state.args:
                continue
            if any(state.args.get(key) != value for key, value in fixed_args.items()):
                continue

            outcome = self.graph_builder.load_outcome(state)
            if outcome is None or outcome.outputs is None:
                continue

            score = self._extract_outcome_score(outcome.outputs, output_key=output_key, scorer=scorer)
            dataset_value = state.args[dataset_arg]
            model_value = state.args[model_arg]

            grouped_scores.setdefault(dataset_value, {}).setdefault(model_value, []).append(score)

        results = []
        for dataset_value, model_scores in grouped_scores.items():
            if len(model_scores) < 2:
                continue

            reduced_scores = {model: mean(scores) for model, scores in model_scores.items()}
            min_model = min(reduced_scores, key=reduced_scores.get)
            max_model = max(reduced_scores, key=reduced_scores.get)
            amplification = reduced_scores[max_model] - reduced_scores[min_model]

            results.append(
                {
                    "dataset": dataset_value,
                    "amplification": amplification,
                    "model_scores": reduced_scores,
                    "min_model": min_model,
                    "max_model": max_model,
                }
            )

        return sorted(results, key=lambda item: item["amplification"], reverse=True)

    def run_multiple(self, func, combos):
        results = []
        new_states = []
        for combo in combos:
            if "args" in combo or "kwargs" in combo:
                args = combo.get("args", [])
                kwargs = combo.get("kwargs", {})
            else:
                args = []
                kwargs = combo
            new_state = self.build_graph(func, *args, **kwargs)
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
