"""
Unit tests for safe JSON serialization engine (safe_json.py).
Ensures NaN, Inf, NumPy types, Pandas types, and complex nested data structures
are flawlessly sanitized without runtime serialization errors.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fpl_oracle.server.safe_json import SafeJSONResponse, safe_json_serialize


class SampleModel(BaseModel):
    name: str
    score: float


def test_safe_json_primitive_sanitization():
    """Verify NaN, Inf, -Inf are sanitized to None."""
    data = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf_val": float("-inf"),
        "valid_float": 42.5,
        "valid_str": "FPL Oracle",
        "valid_int": 10,
        "valid_bool": True,
        "none_val": None,
    }
    sanitized = safe_json_serialize(data)

    assert sanitized["nan_val"] is None
    assert sanitized["inf_val"] is None
    assert sanitized["neg_inf_val"] is None
    assert sanitized["valid_float"] == 42.5
    assert sanitized["valid_str"] == "FPL Oracle"
    assert sanitized["valid_int"] == 10
    assert sanitized["valid_bool"] is True


def test_safe_json_numpy_and_pandas_conversions():
    """Verify NumPy scalar/array types and Pandas Series/DataFrame conversions."""
    df = pd.DataFrame(
        {
            "element": [1, 2],
            "xp": [6.5, np.nan],
            "flag": [True, False],
        }
    )
    series = pd.Series([10, 20, 30], name="points")

    data = {
        "np_int": np.int64(42),
        "np_float": np.float64(3.1415),
        "np_nan": np.float64(np.nan),
        "np_arr": np.array([1, 2, 3]),
        "pd_df": df,
        "pd_series": series,
        "pd_timestamp": pd.Timestamp("2026-10-03 14:00:00"),
    }
    sanitized = safe_json_serialize(data)

    assert isinstance(sanitized["np_int"], int)
    assert sanitized["np_int"] == 42
    assert isinstance(sanitized["np_float"], float)
    assert round(sanitized["np_float"], 4) == 3.1415
    assert sanitized["np_nan"] is None
    assert sanitized["np_arr"] == [1, 2, 3]
    assert isinstance(sanitized["pd_df"], list)
    assert len(sanitized["pd_df"]) == 2
    assert sanitized["pd_df"][1]["xp"] is None
    assert sanitized["pd_series"] in ({0: 10, 1: 20, 2: 30}, {"0": 10, "1": 20, "2": 30})
    assert "2026-10-03" in sanitized["pd_timestamp"]


def test_safe_json_pydantic_and_datetime():
    """Verify Pydantic models, sets, tuples, and datetime objects."""
    dt = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
    model = SampleModel(name="Palmer", score=8.2)

    data = {
        "model": model,
        "dt": dt,
        "tuple_items": (1, 2, "three"),
        "set_items": {100, 200},
    }
    sanitized = safe_json_serialize(data)

    assert sanitized["model"]["name"] == "Palmer"
    assert sanitized["model"]["score"] == 8.2
    assert "2026-10-03" in sanitized["dt"]
    assert sanitized["tuple_items"] == [1, 2, "three"]
    assert sorted(sanitized["set_items"]) == [100, 200]


def test_safe_json_response_class():
    """Verify SafeJSONResponse renders valid JSON bytes without encoding crash."""
    resp = SafeJSONResponse(content={"xp": np.float64(7.5), "nan": float("nan")})
    assert resp.status_code == 200
    body = resp.body.decode("utf-8")
    assert '"xp": 7.5' in body or '"xp":7.5' in body
    assert '"nan": null' in body or '"nan":null' in body
