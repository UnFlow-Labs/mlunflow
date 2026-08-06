# unflow (mlunflow)

`unflow` is a Python framework for reproducible, graph-based machine learning experimentation.

It wraps your training/evaluation functions and builds a **compute graph** of executions where:

- each node is a normalized function invocation state
- each edge is a transformation (arguments and/or procedure source changes)
- repeated identical calls are de-duplicated

This makes it easier to understand *what changed* between runs and to query outcomes across experiment history.

## Why unflow

- Track argument and code changes across iterations
- Avoid re-running identical experiment states
- Query states, transformations, shortest paths, and outcomes
- Run locally by default or in parallel with multiprocessing
- Persist graph snapshots and outcomes for later inspection

## Installation

### Prerequisites

- Python `>=3.11,<3.14.1`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install dependencies (development)

```bash
uv sync
```

### Install as a package

```bash
uv pip install -e .
```

## Quick Start

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

Run it with:

```bash
uv run python examples/simple_example.py
```

## Core API

Decorated functions expose helpers for experiment graph operations:

- `run_multiple(combinations)`
- `clear_graph()`
- `graph_size()`
- `get_outcomes()`
- `query_states(...)`
- `query_transformations(...)`
- `query_with_outcomes(...)`
- `shortest_path(from_state, to_state)`

## Execution Modes

- **LocalExecutor** (default): in-process execution
- **MultiprocessingExecutor**: parallel execution with process pool

Multiprocessing example:

```python
from unflow.core.executors.multiprocessing_executor import MultiprocessingExecutor
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator(executor=MultiprocessingExecutor(max_workers=4))
def train(lr: float, epochs: int) -> dict[str, float]:
   return {"score": lr * epochs}
```

See runnable scripts in [examples/](examples).

## Documentation

Project docs are in [docs/](docs) and include:

- [Getting Started](docs/getting-started.md)
- [Concepts](docs/concepts.md)
- [Examples](docs/how-to/examples.md)
- [Local and Parallel Execution](docs/how-to/local-and-parallel.md)
- [API Reference](docs/api/index.md)

Build and serve locally:

```bash
uv run mkdocs serve
```

Build static docs:

```bash
uv run mkdocs build --strict
```

## Development

Common commands:

```bash
# tests
uv run pytest

# formatting and linting
uv run ruff format
uv run ruff check .

# type checks
uv run mypy unflow/
```

If you use Invoke tasks (`tasks.py`):

```bash
uv run invoke format
uv run invoke static-analysis
uv run invoke test
```

## Repository Layout

- `src/unflow/` — package source
- `tests/` — test suite
- `examples/` — runnable examples from simple to end-to-end ML workflows
- `docs/` — MkDocs content

## License

Apache-2.0
