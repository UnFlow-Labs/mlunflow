import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from invoke import task


def update_python_version(c):
    config_file = Path(".cruft.json")
    cruft = json.loads(config_file.read_text())

    version = get_latest_python_version(c)
    major, minor = reduce_version(*version)
    python_version = f"{major}.{minor}"
    current_python_version = cruft["context"]["cookiecutter"]["python_version_dev"]

    try:
        set_python_version(c, python_version)
        cruft["context"]["cookiecutter"]["python_version_dev"] = python_version
        config_file.write_text(json.dumps(cruft, indent=2))
    except Exception as e:
        print(f"Warning: Could not update to Python {python_version}: {e}")
        set_python_version(c, current_python_version)


def set_python_version(c, python_version):
    c.run(f"echo {python_version} > .python-version")
    install(c)


def get_latest_python_version(c):
    out = c.run("uv python list", hide=True).stdout
    versions = [
        tuple(map(int, m.group(1).split(".")))
        for line in out.splitlines()
        if (m := re.match(r"^cpython-([0-9]+\.[0-9]+\.[0-9]+)", line))
    ]
    return max(versions)[:2]


def reduce_version(major, minor):
    prev_version = 100 * major + minor - 1
    return prev_version // 100, prev_version % 100


CONFIG_DIR = Path(__file__).parent / "unflow" / "conf"


def find_packages_in_config_files():
    import yaml

    packages = set()
    for config_file in CONFIG_DIR.rglob("*.yaml"):
        conf = yaml.safe_load(config_file.read_text())
        packages.update(find_packages(conf))
    return packages


def find_packages(conf):
    packages = set()
    if isinstance(conf, dict):
        for k, v in conf.items():
            if k == "_target_":
                packages.add(v.split(".")[0])
            else:
                packages.update(find_packages(v))
    elif isinstance(conf, list):
        for v in conf:
            packages.update(find_packages(v))
    else:
        return {}
    return packages


@task
def install(c, prod=False):
    if prod:
        c.run("uv tool install .")
    else:
        c.run(f"uv python install {Path('.python-version').read_text().strip()}")
        c.run("uv sync")


@task
def docker(c):
    """Build the Docker image."""
    github_token = c.run("gh auth token", hide=True).stdout.strip()
    with TemporaryDirectory() as tmpdir:
        token_file = Path(tmpdir) / "github_token"
        with open(token_file, "w") as f:
            f.write(github_token)
        c.run(f"docker build --secret id=github_token,src={token_file} -t mlunflow .")


@task
def format(c):
    """Format code and fix lint issues using ruff."""
    c.run("uv run ruff format")
    c.run("uv run ruff check --fix")


@task
def static_analysis(c):
    """Run static analysis."""
    c.run("uv run mypy unflow/")
    with Path("pyproject.toml").open("rb") as f:
        try:
            toml_ignores = tomllib.load(f)["tool"]["deptry"]["per_rule_ignores"]
        except KeyError:
            toml_ignores = {}
    toml_ignores = defaultdict(list, toml_ignores)
    toml_ignores["DEP002"] += find_packages_in_config_files()

    ignores = ",".join([f"{code}={'|'.join(packages)}" for code, packages in toml_ignores.items()])
    if ignores:
        c.run(f"uv run deptry . --per-rule-ignores '{ignores}'")
    else:
        c.run("uv run deptry .")


@task()
def test(c):
    """Run all tests and checks."""
    format(c)
    static_analysis(c)
    c.run("uv run pytest --cov")


@task
def update(c):
    """Update dependencies and run tests."""
    c.run("uvx cruft update -y")
    if has_merge_conflicts(c):
        input("Merge conflicts detected! Resolve conflicts and press enter to continue!")
    update_python_version(c)
    c.run("uv sync -U")
    test(c)
    c.run("echo 'All updates tests have been executed'")


@task
def show_unused(c):
    """Show unused code using vulture."""
    c.run("uv run vulture unflow/")


def has_merge_conflicts(c):
    """Check if there are any merge conflicts using git status."""
    return bool(c.run("git diff --name-only --diff-filter=U", hide=True, warn=True).stdout.strip())
