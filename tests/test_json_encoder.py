import datetime
import decimal
from dataclasses import dataclass

import orjson

from unflow.core.json_encoder import dumps


class TestDumps:
    def test_none(self):
        assert dumps(None) == b"null"

    def test_str(self):
        assert dumps("hello") == b'"hello"'

    def test_int(self):
        assert dumps(42) == b"42"

    def test_float(self):
        result = orjson.loads(dumps(3.14))
        assert result == 3.14

    def test_bool(self):
        assert dumps(True) == b"true"
        assert dumps(False) == b"false"

    def test_list(self):
        assert dumps([1, 2, 3]) == b"[1,2,3]"

    def test_dict(self):
        result = orjson.loads(dumps({"a": 1, "b": 2}))
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        data = {"x": {"y": [1, 2, None]}}
        result = orjson.loads(dumps(data))
        assert result == data

    def test_datetime(self):
        dt = datetime.datetime(2025, 1, 15, 10, 30, 0)
        result = orjson.loads(dumps(dt))
        assert result == "2025-01-15T10:30:00"

    def test_date(self):
        d = datetime.date(2025, 1, 15)
        result = orjson.loads(dumps(d))
        assert result == "2025-01-15"

    def test_decimal(self):
        dec = decimal.Decimal("3.14")
        result = orjson.loads(dumps(dec))
        assert result == 3.14

    def test_tuple(self):
        result = orjson.loads(dumps((1, 2)))
        assert result == [1, 2]

    def test_set(self):
        result = orjson.loads(dumps({1, 2, 3}))
        assert sorted(result) == [1, 2, 3]

    def test_generic_object(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = orjson.loads(dumps(Obj()))
        assert result == {"x": 1, "y": "hello"}

    def test_cycle_detection(self):
        d = {}
        d["self"] = d
        result = orjson.loads(dumps(d))
        assert result["self"] == "<cycle>"

    def test_dataclass(self):
        @dataclass
        class Point:
            x: int
            y: int

        result = orjson.loads(dumps(Point(1, 2)))
        assert result == {"x": 1, "y": 2}

    def test_pydantic_model(self):
        from pydantic import BaseModel

        class Model(BaseModel):
            name: str
            value: int

        result = orjson.loads(dumps(Model(name="test", value=42)))
        assert result == {"name": "test", "value": 42}

    def test_numpy_array(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0])
        result = orjson.loads(dumps(arr))
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_number(self):
        import numpy as np

        result = orjson.loads(dumps(np.float64(3.14)))
        assert result == 3.14

    def test_non_str_dict_keys_are_stringified(self):
        result = orjson.loads(dumps({1: "one", 2: "two"}))
        assert result == {"1": "one", "2": "two"}

    def test_serialization_error_not_raised_for_valid_input(self):
        dumps([1, 2, 3])  # should not raise
