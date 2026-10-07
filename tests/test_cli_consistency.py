"""CLI commands consume the same configured API operations, not template squads."""

from unittest.mock import AsyncMock

from fpl_oracle import cli
from fpl_oracle.server.routes import api


def test_cli_commands_render_shared_operations(monkeypatch):
    rows = [
        {"element": i, "position": "MID", "web_name": f"Player{i}", "value": 50, "expected_points": 3}
        for i in range(1, 16)
    ]
    squad = {
        "starters": rows[:11],
        "bench": rows[11:],
        "formation": "3-5-2",
        "captain": rows[0],
        "vice_captain": rows[1],
        "starters_expected_points": 33,
        "target_gameweek": 6,
    }
    get_squad = AsyncMock(return_value=squad)
    optimize = AsyncMock(
        return_value={
            "recommended_plan": {"recommendation_summary": "Roll"},
            "candidate_plans": [],
            "hit_verdict": "No hit",
        }
    )
    chips = AsyncMock(return_value={"chip_plan_table": []})
    monkeypatch.setattr(api, "get_squad", get_squad)
    monkeypatch.setattr(api, "run_optimizer", optimize)
    monkeypatch.setattr(api, "get_chip_strategy", chips)
    cli.analyze()
    cli.optimize()
    cli.chips()
    assert get_squad.await_count == optimize.await_count == chips.await_count == 1


def test_cli_missing_squad_cannot_generate_template(monkeypatch):
    get_squad = AsyncMock(return_value={"status": "unavailable", "starters": []})
    monkeypatch.setattr(api, "get_squad", get_squad)
    cli.analyze()
    assert get_squad.await_count == 1


def test_cli_actual_configured_squad_renders_without_independent_template(configured_advisor, monkeypatch):
    from fpl_oracle.news.analyse import news_analyzer

    monkeypatch.setattr(news_analyzer, "get_player_news_signals", AsyncMock(return_value=[]))
    cli.analyze()
