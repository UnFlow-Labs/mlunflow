# Examples

This page maps each runnable example to a specific learning goal so you can pick the right starting point.

## Quick Start Examples

### 1) Basic state tracking and de-dup

- File: `examples/simple_example.py`
- Goal: see function-state tracking and repeated-argument behavior.
- Run:

```bash
uv run python examples/simple_example.py
```

- What to look for:
  - multiple calls with different argument combinations
  - graph growth across runs

### 2) Procedure/code change detection

- File: `examples/code_changes.py`
- Goal: compare two runs where function code changes while arguments stay the same.
- Run:

```bash
uv run python examples/code_changes.py
```

- What to look for:
  - transformation entries that include `p_changes`
  - argument changes vs procedure changes

### 3) Nested config argument diffs (Pydantic)

- File: `examples/datadict_changes.py`
- Goal: inspect diffs for nested structured inputs.
- Run:

```bash
uv run python examples/datadict_changes.py
```

- What to look for:
  - edge transformation output showing nested field differences
  - graph node/edge count after two config variants

### 4) Outcome retrieval

- File: `examples/example_with_outcome.py`
- Goal: read and inspect stored outcomes across states.
- Run:

```bash
uv run python examples/example_with_outcome.py
```

- What to look for:
  - `get_outcomes()` output per state

## Execution Model Example

### Parallel execution with multiprocessing

- File: `examples/multiprocessing_example.py`
- Goal: run multiple configurations with `MultiprocessingExecutor`.
- Run:

```bash
uv run python examples/multiprocessing_example.py
```

- What to look for:
  - multiple process IDs in output
  - successful completion of `run_multiple`

## Advanced End-to-End Experiments

These examples are heavier and intended for deeper experimentation workflows.

### LeNet on MNIST

- Files:
  - `examples/LeNetMNIST/lenet.py`
  - `examples/LeNetMNIST/experiments.py`
  - `examples/LeNetMNIST/queries.py`
- Goal: run experiment hypotheses and query transformations against outcomes.
- Notes:
  - requires `torch`/`torchvision`
  - first run downloads dataset and takes longer

### Hugging Face fine-tuning

- File: `examples/HuggingFace/finetune.py`
- Goal: track argument transformations and outcome deltas in a realistic NLP pipeline.
- Notes:
  - requires `transformers`, `datasets`, `evaluate`
  - network access and model/data downloads are required
  - runtime is significantly longer than quick-start examples

## Which example should I run?

- New to unflow: start with `examples/simple_example.py`
- Want diff semantics: run `examples/code_changes.py` then `examples/datadict_changes.py`
- Need concurrency: run `examples/multiprocessing_example.py`
- Need real ML workflows: use `examples/LeNetMNIST/*` or `examples/HuggingFace/finetune.py`