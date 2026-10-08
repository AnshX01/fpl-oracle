import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd

from fpl_oracle.server.analysis import AnalysisService, publication_projections


def test_news_ttl_does_not_rebuild_snapshot_in_slow_atomic_job(monkeypatch):
    from fpl_oracle.ml.model_registry import model_registry
    from fpl_oracle.ml.predict import projection_engine
    from fpl_oracle.news.analyse import news_analyzer

    async def run():
        service = AnalysisService()
        boot = SimpleNamespace(model_dump=lambda **kw: {"elements": []})
        news = AsyncMock(return_value=[])
        monkeypatch.setattr(news_analyzer, "get_player_news_signals", news)
        monkeypatch.setattr(news_analyzer, "get_reconciled_inputs", lambda *a: {})
        monkeypatch.setattr(model_registry, "get_active_version", lambda: {"version": "fixed"})
        calls = []

        def predict(*args, **kw):
            calls.append(1)
            return {6: pd.DataFrame([dict(element=1, expected_points=5)])}

        monkeypatch.setattr(projection_engine, "predict_multi_gameweeks", predict)
        token = publication_projections.set({})
        try:
            first = await service.projections(6, 5, boot, [])
            service._news_at = -1000  # Simulated >5-minute calculation.
            first[6].loc[0, "expected_points"] = 999  # Caller must not mutate pin.
            second = await service.projections(6, 8, boot, [])
            assert second[6].expected_points.iloc[0] == 5
            assert len(calls) == 1
            news.assert_awaited_once()
        finally:
            publication_projections.reset(token)

    asyncio.run(run())
