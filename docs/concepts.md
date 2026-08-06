# Concepts

## Execution State

An execution state (`RState`) represents a function invocation with normalized arguments.
Each state has a lifecycle: `pending` → `ready` → `running` → `completed` or `failed`.

## Transformation

A transformation links two states and records what changed:

- procedure source changes
- argument value/type changes

If no change is detected for a candidate state, that state is not added.

## Compute Graph

The compute graph is a directed acyclic graph (DAG):

- Nodes are states
- Edges are transformations

The graph supports filtering states/transformations and shortest-path queries.

## Scheduling and Execution

`Scheduler` resolves dependencies and executes only required ancestors of a target state.
Executors provide runtime behavior:

- `LocalExecutor`: in-process execution
- `MultiprocessingExecutor`: process pool execution

## Persistence

The framework persists:

- graph snapshots
- outcomes
- execution records

Persistence is handled by `DB` in `unflow.core.simpledb`.