import json
from datetime import UTC, datetime, timedelta

import pytest

from fpl_oracle.league.validation import freeze_forecast, score_forecast, validation_summary


def test_forward_evidence_rejects_hindsight_and_fake_calibration(tmp_path):
    now = datetime.now(UTC)
    payload = dict(
        deadline_utc=(now + timedelta(hours=1)).isoformat(),
        source_kind="official_forward",
        snapshot_id="fixture",
        observed_manager_ids=[1, 2],
        owner_manager_id=1,
        probability=0.6,
        target_gameweek=6,
        scope="actual_observed_manager_outcome",
    )
    saved = freeze_forecast(payload, tmp_path, now)
    record = json.loads(open(saved["path"]).read())
    with pytest.raises(ValueError):
        freeze_forecast(payload, tmp_path, now + timedelta(hours=2))
    with pytest.raises(ValueError):
        freeze_forecast(payload | {"source_kind": "synthetic"}, tmp_path, now)
    scored = score_forecast(
        record, dict(source_kind="official_finalized_manager_history", gameweek=6, scores={1: 50, 2: 45})
    )
    assert scored["brier"] == pytest.approx(0.16)
    summary = validation_summary([scored])
    assert summary["count"] == 1
    assert summary["promotion_allowed"] is False
    assert summary["calibrated"] is False
    assert (
        score_forecast(
            record | {"scope": "counterfactual_plan"},
            dict(source_kind="official_finalized_manager_history", gameweek=6, scores={1: 50, 2: 45}),
        )["status"]
        == "not_scoreable"
    )
    with pytest.raises(ValueError):
        score_forecast(record, dict(source_kind="manual", gameweek=6, scores={1: 50, 2: 45}))
