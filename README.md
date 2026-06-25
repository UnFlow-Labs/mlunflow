# mlunflow

## 📋 Prerequisites

- uv installed: https://docs.astral.sh/uv/
- invoke: `uv tool install --with PyYAML --python 3.13 invoke`
- Git LFS: https://git-lfs.com
- (optional) GitHub CLI: https://cli.github.com/

## 💻 Development workflow

1. Check out the code: `git clone https://github.com/CUPA-ARC/mlunflow.git`
2. Go to your code and set up dependencies: `cd mlunflow && inv install`
3. Create a new branch: `git checkout -b <Ticket Number>-<Ticket Name>`
4. (Optional) Apply and updates and check if things still work: `inv update`
    - If failed, rollback: `git reset --hard HEAD`
    - If success, commit: `git commit -a -m 'Apply updates'`
5. Make your changes, test them (`inv test`) and commit it (incl. the `uv.lock` file).
    - Make sure to store all test files in `tests/fixtures/` so they are tracked by git lfs.
7. Create a Pull Request (PR) from your feature branch to main: `gh pr new -w`
    - The branch will be automatically deleted after merged.

## 🔧 Development Commands

Some of these commands might be useful while developing the software.

- `inv install`: Installs all development dependencies.
- `inv install --prod`: Installs the software as a python module (without dev dependencies).
- `inv test`: Formats the code and runs tests.
- `inv docker`: Builds the docker image.
- `uv run ...`: If you want to manually run your code, prepred the command with `uv run`, so the right environment will
  be used. e.g., `uv run python my_script.py`. More information at https://docs.astral.sh/uv/
- `uv add <package>`: Add new package to the project and update `pyproject.toml`. More information
  at https://docs.astral.sh/uv/
- You can set up tab completion for `inv` commands. See https://docs.pyinvoke.org/en/stable/invoke.html#shell-tab-completion.

### Other Commands

- `inv update`: Updates the dependencies, applies template changes and runs the tests.
- `inv show-unused`: Shows unused code in your project.
- `inv format`: Formats the code without running tests.

Created using [CUPA-ARC/scratch-python-template](https://github.com/CUPA-ARC/scratch-python-template).
