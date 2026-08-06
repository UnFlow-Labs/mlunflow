# unflow

`unflow` is a graph-driven execution framework for experimentation workflows.
It tracks function states by argument values, stores outcomes, and lets you query paths and differences across runs.

## Why unflow

- Build a state graph from function executions automatically
- Skip duplicate runs when the same inputs were already evaluated
- Query states and transformations for experiment analysis
- Run locally or with multiprocessing executors

## Quick Example

```python
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train(lr: float, epochs: int) -> dict[str, float]:
	return {"loss": 0.5 / lr, "epochs": float(epochs)}


result = train(lr=0.01, epochs=10)
print(result)

# run multiple combinations
records = train.run_multiple(
	[
		{"kwargs": {"lr": 0.01, "epochs": 10}},
		{"kwargs": {"lr": 0.02, "epochs": 10}},
	]
)
```

## Read Next

- Start with [Getting Started](getting-started.md)
- Learn the model in [Concepts](concepts.md)
- Browse all modules in [API Reference](api/index.md)
