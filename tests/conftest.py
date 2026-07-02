from pathlib import Path

import pytest

from unflow.core.simpledb import DB
from unflow.core.unflow_types import RState


def sample_func(a, b=1):
    return a + b


def sample_func_v2(a, b=2):
    return a * b


def failing_func(_x):
    msg = "intentional failure"
    raise ValueError(msg)


@pytest.fixture
def sample_state():
    return RState(
        name="test_state_0",
        procedure=sample_func,
        args={"a": 1, "b": 2},
        kwargs={},
        description="test",
    )


@pytest.fixture
def pickle_dir(tmp_path: Path) -> Path:
    p = tmp_path / "pickles"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def tmp_db(tmp_path: Path) -> DB:
    db = DB(tmp_path / "test.db")
    yield db
    db.close()
