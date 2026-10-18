import asyncio
import json
from types import SimpleNamespace

from fpl_oracle.server import retraining


def run_check(monkeypatch, tmp_path, finalized, held):
    import importlib

    module = importlib.reload(retraining)  # the autouse stub replaced the function; reload restores it
    events = [SimpleNamespace(id=g, finished=True, data_checked=g in finalized) for g in range(1, 9)]

    async def bootstrap(force_refresh=False):
        return SimpleNamespace(events=events), False

    calls = []

    async def refresh():
        calls.append("refresh")
        return {"finalized_gameweeks": sorted(finalized)}

    async def retrain():
        calls.append("retrain")

    out = tmp_path / "master_history.csv"
    out.with_suffix(".meta.json").write_text(json.dumps({"finalized_gameweeks": held}))
    monkeypatch.setattr(module.fpl_client, "get_bootstrap_static", bootstrap)
    monkeypatch.setattr(module.historical_manager, "output_file", out)
    monkeypatch.setattr(module.historical_manager, "refresh_current_season", refresh)
    monkeypatch.setattr(module, "retrain_finalized_gameweeks", retrain)
    return asyncio.run(module.ensure_latest_data()), calls


def test_new_finalized_gameweek_is_pulled_and_trained_before_advice(monkeypatch, tmp_path):
    basis, calls = run_check(monkeypatch, tmp_path, finalized={1, 2, 3, 4, 5, 6, 7}, held=[1, 2, 3, 4, 5, 6])
    assert calls == ["refresh", "retrain"]
    assert basis["form_through_gameweek"] == 7


def test_up_to_date_history_is_not_refetched(monkeypatch, tmp_path):
    basis, calls = run_check(monkeypatch, tmp_path, finalized={1, 2, 3, 4, 5, 6}, held=[1, 2, 3, 4, 5, 6])
    assert calls == ["retrain"]
    assert basis["form_through_gameweek"] == 6
