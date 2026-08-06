# unflow

`unflow` is a reproducible, graph-based framework for machine learning experimentation.

It wraps your training/evaluation functions and automatically builds a **compute graph** of executions:

- nodes represent normalized function invocation states
- edges represent transformations between states (argument and/or procedure changes)
- duplicate states are de-duplicated to avoid unnecessary re-runs

## Why unflow

- Track exactly what changed between iterations
- Skip duplicate evaluations when inputs and procedure match
- Query states, transformations, shortest paths, and outcomes
- Execute locally or with multiprocessing for parallel workflows
- Persist graph snapshots and outcomes for later inspection

## Quick Example

```python
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train(lr: float = 0.01, epochs: int = 10) -> dict[str, float]:
    return {"loss": 0.5, "lr": lr, "epochs": float(epochs)}


print(train(lr=0.01, epochs=10))
print(train(lr=0.01, epochs=10))  # duplicate state -> not re-executed
print(train(lr=0.02, epochs=10))

print("states:", train.graph_size())
print("outcomes:", train.get_outcomes())
```

Run a full script version:

```bash
uv run python examples/simple_example.py
```

## Core API Helpers

Decorated functions expose graph-aware helpers:

- `run_multiple(combinations)`
- `clear_graph()`
- `graph_size()`
- `get_outcomes()`
- `query_states(...)`
- `query_transformations(...)`
- `query_with_outcomes(...)`
- `shortest_path(from_state, to_state)`

## Read Next

- Start with [Getting Started](getting-started.md)
- Learn the model in [Concepts](concepts.md)
- Pick a runnable path in [Examples](how-to/examples.md)
- See executor behavior in [Local and Parallel Execution](how-to/local-and-parallel.md)
- Browse all modules in [API Reference](api/index.md)

## Which Example First

- Start with `examples/simple_example.py` for state tracking basics
- Use `examples/code_changes.py` for procedure diff behavior
- Use `examples/datadict_changes.py` for nested config argument diffs
- Use `examples/multiprocessing_example.py` for parallel execution
