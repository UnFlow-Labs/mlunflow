# Local and Parallel Execution

## Local Execution (Default)

```python
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train(lr: float) -> dict[str, float]:
    return {"loss": 1.0 / lr}


result = train(lr=0.01)
```

## Parallel Execution (Multiprocessing)

```python
from unflow.core.executors.multiprocessing_executor import MultiprocessingExecutor
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator(executor=MultiprocessingExecutor(max_workers=4))
def train(lr: float, epochs: int) -> dict[str, float]:
    return {"score": lr * epochs}


records = train.run_multiple(
    [
        {"kwargs": {"lr": 0.01, "epochs": 10}},
        {"kwargs": {"lr": 0.02, "epochs": 10}},
        {"kwargs": {"lr": 0.05, "epochs": 10}},
    ]
)
```

## Querying the Graph

After runs, use the attached helpers:

- `train.query_states(...)`
- `train.query_transformations(...)`
- `train.shortest_path(from_state, to_state)`
- `train.get_outcomes()`