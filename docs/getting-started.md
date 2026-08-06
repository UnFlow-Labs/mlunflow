# Getting Started

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Install

```bash
uv sync
```

## First Run

Create a file such as `examples/basic.py`:

```python
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train(lr: float = 0.01, epochs: int = 10) -> dict[str, float]:
    return {"loss": 0.5, "lr": lr, "epochs": float(epochs)}


print(train(lr=0.01, epochs=10))
print(train(lr=0.01, epochs=10))  # duplicate args -> skipped path
```

Run it with:

```bash
uv run python examples/basic.py
```

## Common Development Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mkdocs serve
uv run mkdocs build --strict
```

## Next Steps

- Learn the graph model in [Concepts](concepts.md)
- See local vs parallel executors in [Local and Parallel Execution](how-to/local-and-parallel.md)
- Explore all symbols in [API Reference](api/index.md)
