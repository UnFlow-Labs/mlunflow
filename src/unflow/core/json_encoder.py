import datetime
import decimal
from dataclasses import asdict, is_dataclass

import numpy as np
import orjson
from pydantic import BaseModel


class SerializationError(Exception):
    pass


def dumps(obj, *, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY):
    active_stack = set()

    def _convert(o):
        oid = id(o)
        track_identity = isinstance(o, (dict, list, set, tuple)) or hasattr(o, "__dict__")

        # ---- cycle detection ----
        if track_identity:
            if oid in active_stack:
                return "<cycle>"
            active_stack.add(oid)

        try:
            # ---- fast primitives ----
            if o is None or isinstance(o, (str, int, float, bool)):
                return o

            # ---- datetime ----
            if isinstance(o, (datetime.datetime, datetime.date)):
                return o.isoformat()

            # ---- decimal ----
            if isinstance(o, decimal.Decimal):
                return float(o)

            # ---- Pydantic v2 (model_dump) / v1 (dict) ----
            if BaseModel is not None and isinstance(o, BaseModel):
                # v2
                if hasattr(o, "model_dump"):
                    return _convert(o.model_dump())
                # v1 fallback
                return _convert(o.dict())

            # ---- dataclass ----
            if is_dataclass(o):
                return _convert(asdict(o))

            # ---- numpy ----
            if np is not None:
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, np.number):
                    return o.item()

            # ---- dict ----
            if isinstance(o, dict):
                return {str(k): _convert(v) for k, v in o.items()}

            # ---- iterables ----
            if isinstance(o, (list, tuple, set)):
                return [_convert(x) for x in o]

            # ---- generic objects ----
            if hasattr(o, "__dict__"):
                return _convert(vars(o))

            # ---- fallback ----
            return str(o)
        finally:
            if track_identity:
                active_stack.remove(oid)

    return orjson.dumps(_convert(obj), option=option)
