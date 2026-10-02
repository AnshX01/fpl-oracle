"""
Safe JSON serialization utilities for FPL Oracle.
Guarantees clean JSON conversion for NumPy, pandas, datetime, NaNs/infinities, and Pydantic models.
"""

import math
import json
from datetime import datetime, date
from pathlib import PurePath
from uuid import UUID
from typing import Any, Dict, List, Set, Union
import numpy as np
import pandas as pd
from starlette.responses import JSONResponse
import fastapi.encoders as fastapi_enc

def safe_json_serialize(obj: Any, seen: Set[int] = None) -> Any:
    """
    Recursively serialize any object into JSON-compliant standard Python types:
    - NumPy integers -> int
    - NumPy / Python floats -> float, or None if NaN / Inf
    - NumPy bools -> bool
    - NumPy arrays -> list
    - pandas DataFrame -> list of dicts with sanitized values
    - pandas Series -> dict with sanitized values
    - pandas Timestamp / datetime / date -> ISO format string
    - UUID / PurePath -> string
    - Pydantic models -> dict
    - NaN / Inf -> None (renders as null in JSON)
    - Dict keys -> str
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    # Detect and break cyclic references for complex container types
    if isinstance(obj, (dict, list, tuple, set)):
        if obj_id in seen:
            return None
        seen.add(obj_id)

    try:
        # None and basic types
        if obj is None:
            return None

        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)

        if isinstance(obj, (int, np.integer)):
            return int(obj)

        if isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj) or np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        if isinstance(obj, str):
            return obj

        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()

        if isinstance(obj, (PurePath, UUID)):
            return str(obj)

        if isinstance(obj, pd.DataFrame):
            # Replace NaNs in DataFrame before converting to records
            clean_df = obj.replace({np.nan: None, np.inf: None, -np.inf: None})
            records = clean_df.to_dict(orient="records")
            return [safe_json_serialize(row, seen) for row in records]

        if isinstance(obj, pd.Series):
            clean_s = obj.replace({np.nan: None, np.inf: None, -np.inf: None})
            return {str(k): safe_json_serialize(v, seen) for k, v in clean_s.to_dict().items()}

        if isinstance(obj, np.ndarray):
            return [safe_json_serialize(x, seen) for x in obj.tolist()]

        # Pydantic v2 or v1 model
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return safe_json_serialize(obj.model_dump(), seen)
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return safe_json_serialize(obj.dict(), seen)

        if isinstance(obj, dict):
            return {str(k): safe_json_serialize(v, seen) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [safe_json_serialize(x, seen) for x in obj]

        # Objects with custom to_dict
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return safe_json_serialize(obj.to_dict(), seen)

        # Dataclass or objects with __dict__
        if hasattr(obj, "__dict__"):
            return safe_json_serialize(vars(obj), seen)

        # Fallback to string representation
        return str(obj)
    finally:
        if obj_id in seen:
            seen.remove(obj_id)


class SafeJSONResponse(JSONResponse):
    """
    FastAPI / Starlette Response class that serializes response data safely using safe_json_serialize.
    Guarantees no NaN, Inf, NumPy, or pandas types break JSON rendering.
    """
    def render(self, content: Any) -> bytes:
        sanitized = safe_json_serialize(content)
        return json.dumps(
            sanitized,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")


def register_fastapi_safe_encoders():
    """
    Register safe encoders in FastAPI's ENCODERS_BY_TYPE mapping and regenerate class tuples.
    This protects any internal jsonable_encoder calls made by FastAPI routers.
    """
    fastapi_enc.ENCODERS_BY_TYPE[np.integer] = int
    fastapi_enc.ENCODERS_BY_TYPE[np.floating] = lambda v: None if (np.isnan(v) or np.isinf(v)) else float(v)
    fastapi_enc.ENCODERS_BY_TYPE[np.bool_] = bool
    fastapi_enc.ENCODERS_BY_TYPE[np.ndarray] = lambda v: [safe_json_serialize(x) for x in v.tolist()]
    fastapi_enc.ENCODERS_BY_TYPE[pd.DataFrame] = lambda df: safe_json_serialize(df)
    fastapi_enc.ENCODERS_BY_TYPE[pd.Series] = lambda s: safe_json_serialize(s)
    fastapi_enc.ENCODERS_BY_TYPE[pd.Timestamp] = lambda ts: ts.isoformat()

    try:
        fastapi_enc.encoders_by_class_tuples = fastapi_enc.generate_encoders_by_class_tuples(
            fastapi_enc.ENCODERS_BY_TYPE
        )
    except Exception:
        pass
