# UnFlow

[!UnFlow](https://ibb.co/9mDPBLw0)

### Reproducible, graph-based experimentation for machine learning

**UnFlow turns your ML experiments into a computation graph.**

Instead of treating every training run as an isolated experiment, UnFlow records the relationship between executions — what arguments changed, what code changed, which states have already been computed, and what outcomes they produced.

That means you can **reuse identical computation, understand how experiments evolved, and navigate your experiment history as a graph.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

---

## Why UnFlow?

ML experimentation often looks like this:

```text
experiment_001
experiment_002
experiment_003
experiment_004
...
```

But experiments are rarely independent.

You might change:

* learning rate
* batch size
* model architecture
* dataset
* preprocessing
* training code
* evaluation code

Traditional experiment tracking tells you **what happened in each run**.

UnFlow additionally tracks **how runs are related to one another**.

```text
                    lr=0.01
                       │
                       ▼
                 ┌──────────┐
                 │  State A │
                 └────┬─────┘
                      │ lr changed
                      ▼
                 ┌──────────┐
                 │  State B │
                 └────┬─────┘
                      │ epochs changed
                      ▼
                 ┌──────────┐
                 │  State C │
                 └──────────┘
```

This makes experimentation easier to reason about and enables computation reuse.

---

## What UnFlow does

### 🔁 Deduplicates experiments

If you execute the exact same experiment state twice, UnFlow can recognize that the state has already been computed instead of running it again.

```python
train(lr=0.01, epochs=10)
train(lr=0.01, epochs=10)  # already computed
```

### 🧬 Tracks experiment transformations

UnFlow represents changes between experiment states.

For example:

```text
State A
  │
  ├── learning_rate → 0.02
  │
  ▼
State B
```

This lets you reason about *how* an experiment changed, not just its final configuration.

### 🔎 Query experiment history

Explore states, transformations, and outcomes programmatically.

```python
train.query_states(...)
train.query_transformations(...)
train.query_with_outcomes(...)
train.shortest_path(...)
```

### ⚡ Run experiments locally or in parallel

Use the default local executor or execute experiments concurrently with multiprocessing.

```python
@unflowdecorator(
    executor=MultiprocessingExecutor(max_workers=4)
)
def train(lr: float, epochs: int):
    ...
```

### 💾 Persist experiment graphs

Graph snapshots and outcomes can be persisted for later inspection and analysis.

---

# Quick Start

## Installation

UnFlow currently supports Python 3.11+.

With `uv`:

```bash
git clone https://github.com/UnFlow-Labs/mlunflow.git
cd mlunflow

uv sync
```

Or install the package directly:

```bash
uv pip install -e .
```

---

## Your first UnFlow experiment

```python
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train(lr: float = 0.01, epochs: int = 10) -> dict[str, float]:
    # Your training code goes here
    return {
        "loss": 0.5,
        "lr": lr,
        "epochs": float(epochs),
    }


train(lr=0.01, epochs=10)

# Same state → reused instead of recomputed
train(lr=0.01, epochs=10)

# New state → executed
train(lr=0.02, epochs=10)

print("states:", train.graph_size())
print("outcomes:", train.get_outcomes())
```

Run the example:

```bash
uv run python examples/simple_example.py
```

---

# The Core Idea

UnFlow models experimentation as a graph.

Each **node** represents a normalized experiment state.

Each **edge** represents a transformation between states.

```text
                ┌──────────────┐
                │ Experiment A │
                └──────┬───────┘
                       │
                 lr changed
                       │
                       ▼
                ┌──────────────┐
                │ Experiment B │
                └──────┬───────┘
                       │
                epochs changed
                       │
                       ▼
                ┌──────────────┐
                │ Experiment C │
                └──────────────┘
```

Because experiment states are normalized, identical invocations can be detected and reused.

The result is an experiment history that is:

* **reproducible**
* **queryable**
* **deduplicated**
* **relationship-aware**

---

# Core API

Decorated functions expose experiment-graph operations:

| API                       | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| `run_multiple()`          | Execute multiple experiment configurations |
| `clear_graph()`           | Reset the experiment graph                 |
| `graph_size()`            | Inspect the number of tracked states       |
| `get_outcomes()`          | Retrieve recorded outcomes                 |
| `query_states()`          | Search experiment states                   |
| `query_transformations()` | Inspect changes between states             |
| `query_with_outcomes()`   | Query states together with results         |
| `shortest_path()`         | Find a path between experiment states      |

---

# Parallel Execution

UnFlow can execute experiments concurrently.

```python
from unflow.core.executors.multiprocessing_executor import (
    MultiprocessingExecutor,
)
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator(
    executor=MultiprocessingExecutor(max_workers=4)
)
def train(lr: float, epochs: int) -> dict[str, float]:
    return {
        "score": lr * epochs,
    }
```

See the [`examples/`](examples/) directory for runnable examples.

---

# When should you use UnFlow?

UnFlow is particularly useful when you have:

### Hyperparameter exploration

```text
learning rate
    ×
batch size
    ×
epochs
    ×
model configuration
```

and want to understand the resulting experiment space.

### Iterative model development

When experiments evolve incrementally and you want to understand:

> "What changed between this result and that result?"

### Expensive experiments

When repeating the same computation is wasteful and deterministic experiment states can be reused.

### Programmatic experimentation

When you want your experiment history to be queryable from Python rather than locked inside a dashboard.

---

# How is this different from experiment tracking?

Tools such as MLflow and Weights & Biases are excellent at recording experiment runs, metrics, artifacts, and metadata.

UnFlow focuses on a different abstraction:

```text
Traditional experiment tracking

Run A     Run B     Run C     Run D
 │         │         │         │
 ▼         ▼         ▼         ▼
metrics  metrics   metrics   metrics


UnFlow

        State A
           │
     ┌─────┴─────┐
     │           │
   change      change
     │           │
     ▼           ▼
 State B      State C
     │
   change
     │
     ▼
 State D
```

UnFlow is designed to make the **relationships between experiments** first-class.

It can therefore complement existing experiment-tracking systems rather than necessarily replacing them.

---

# Documentation

Full documentation is available in [`docs/`](docs/).

Topics include:

* [Getting Started](docs/)
* Concepts
* Examples
* Local and parallel execution
* API reference

Run the documentation locally:

```bash
uv run mkdocs serve
```

Build the documentation:

```bash
uv run mkdocs build --strict
```

---

# Development

Run the test suite:

```bash
uv run pytest
```

Format:

```bash
uv run ruff format
```

Lint:

```bash
uv run ruff check .
```

Type-check:

```bash
uv run mypy unflow/
```

---

# Project Structure

```text
mlunflow/
├── src/unflow/      # Core library
├── tests/           # Test suite
├── examples/        # Runnable examples
├── docs/            # Documentation
├── pyproject.toml
└── README.md
```

---

# Roadmap

UnFlow is an evolving open-source project.

Areas of interest include:

* richer experiment graph visualization
* persistent graph backends
* distributed execution
* integrations with existing ML experiment trackers
* richer experiment comparison
* dataset and artifact lineage
* experiment caching
* reproducibility tooling
* improved graph querying
* CLI tooling

If this direction is interesting to you, contributions and feedback are welcome.

---

# Contributing

Contributions are welcome.

Before opening a pull request, please:

1. Run the test suite.
2. Run formatting and linting.
3. Run type checks.
4. Add tests for new functionality where appropriate.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information.

---

# License

UnFlow is released under the [Apache-2.0 License](LICENSE).

---

## ⭐ If UnFlow is useful to you

Star the repository, try it on an experiment, and open an issue with what you discover.

**GitHub:** https://github.com/UnFlow-Labs/mlunflow

**Documentation:** https://unflow-labs.github.io/mlunflow/

