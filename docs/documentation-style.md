# Writing Docs

Use these conventions to keep documentation consistent and maintainable.

## What to Document

- Public classes and functions in `src/unflow`
- Behavior changes in scheduling, state transitions, and persistence
- User-facing setup, examples, and migration notes

## Docstring Style

Prefer short, behavior-first docstrings with clear argument and return descriptions.

Example:

```python
def run_once(self, func, args, kwargs):
    """Build a new state, execute required dependencies, and return one execution record.

    Args:
        func: Callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        ExecutionRecord or None when an equivalent state already exists.
    """
```

## Page Structure

For each new guide page:

1. Start with a one-sentence purpose
2. Add one minimal runnable example
3. Add links to related concept/API pages

## Maintenance Workflow

- Update docs in the same PR as code changes
- Run `uv run mkdocs build --strict` before merging
- Keep examples aligned with tests where possible