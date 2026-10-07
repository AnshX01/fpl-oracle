import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.briefing.decision_card import decision_card_generator
from fpl_oracle.domain.manager_state import EffectiveManagerState
from fpl_oracle.ml.holdout import score_frozen_predictions
from fpl_oracle.ml.temporal import temporal_folds
from fpl_oracle.server.analysis import AnalysisService
from scripts.verify_ledger import validate_evidence_file


def test_temporal_subset_group_boundaries():
    X = pd.DataFrame({"x": range(32)})
    meta = pd.DataFrame({"season": ["2025-26"] * 32, "round": np.repeat(range(1, 9), 4)})
    X.attrs["meta"] = meta
    subset = X.iloc[4:]
    for train, test in temporal_folds(subset):
        m = meta.loc[subset.index]
        assert m.iloc[train]["round"].max() < m.iloc[test]["round"].min()
        assert set(m.iloc[train]["round"]).isdisjoint(m.iloc[test]["round"])


def test_analysis_coalesces_worker_and_returns_copies():
    async def run():
        service = AnalysisService()
        boot = BootstrapStatic.model_validate({"elements": [], "events": [], "teams": [], "element_types": []})
        calls = []

        def project(*args, **kwargs):
            calls.append(kwargs)
            return {6: pd.DataFrame({"element": [1], "expected_points": [4.0]})}

        with (
            patch("fpl_oracle.server.analysis.news_analyzer.get_player_news_signals", new=AsyncMock()) as news,
            patch("fpl_oracle.server.analysis.news_analyzer.get_reconciled_inputs", return_value={}),
            patch("fpl_oracle.server.analysis.projection_engine.predict_multi_gameweeks", side_effect=project),
        ):
            first, second = await asyncio.gather(
                service.projections(6, 1, boot, []), service.projections(6, 1, boot, [])
            )
            assert len(calls) == 1 and news.await_count == 1
            first[6].loc[0, "expected_points"] = 99
            assert second[6].loc[0, "expected_points"] == 4
            service._news_at = 0
            await service.projections(6, 1, boot, [])
            assert news.await_count == 2

    asyncio.run(run())


def test_missing_squad_card_never_fabricates_advice():
    async def run():
        with (
            patch(
                "fpl_oracle.domain.manager_state.manager_state_service.get_current_state",
                new=AsyncMock(return_value=EffectiveManagerState()),
            ),
            patch("fpl_oracle.server.analysis.analysis_service.projections", new=AsyncMock(return_value={})),
        ):
            card = await decision_card_generator.generate_decision_card()
            assert card["status"] == "unavailable" and card["transfers"] is None and card["captain"] is None

    asyncio.run(run())


def test_plain_actuals_cannot_enter_official_holdout_log(tmp_path):
    past = datetime.now(UTC) - timedelta(days=2)
    payload = {
        "gameweek": 6,
        "schema_version": 2,
        "source_kind": "official_forward",
        "frozen_at": past.isoformat(),
        "deadline_utc": (past + timedelta(hours=1)).isoformat(),
        "predictions": [{"element": i, "expected_points": 3, "p10": 1, "p90": 5} for i in range(10)],
    }
    (tmp_path / "gw6_predictions.json").write_text(json.dumps(payload))
    log = tmp_path / "log.json"
    with pytest.raises(ValueError, match="finalized official"):
        score_frozen_predictions(6, {i: 3 for i in range(10)}, custom_holdout_dir=tmp_path, custom_log_path=log)
    assert not log.exists()


def test_evidence_full_final_pytest_summary_and_nan(tmp_path):
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {
        "schema_version": 1,
        "command": ["pytest"],
        "exit_code": 0,
        "stdout": "===== test session starts =====\n===== 1 failed, 30 passed in 1s =====",
        "stderr": "",
        "code_head": head,
        "result": "PASS",
        "artifacts": [],
        "gates": [],
    }
    p = tmp_path / "run.json"
    p.write_text(json.dumps(record))
    assert not validate_evidence_file(p)[0]
    record["stdout"] = "===== 30 passed in 1s ====="
    record["gates"] = [{"name": "coverage", "status": "PASS", "value": float("nan"), "min": 75, "max": 85}]
    p.write_text(json.dumps(record))
    assert not validate_evidence_file(p)[0]
    record["gates"] = [{"name": "coverage", "status": "PASS", "value": 80, "min": 75, "max": 85}]
    p.write_text(json.dumps(record))
    assert validate_evidence_file(p)[0]
