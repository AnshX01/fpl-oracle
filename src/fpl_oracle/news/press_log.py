"""Forward validation of applied press signals: log them, then score against who actually played."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fpl_oracle import config

logger = logging.getLogger("fpl_oracle.news.press_log")


def _path() -> Path:
    return config.DATA_DIR / "news_applied.jsonl"


def _read() -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in _path().read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def log_applied(signals: list[dict[str, Any]], gameweek: int) -> int:
    """Append each applied press signal once per (player, gameweek, quote). Returns rows added."""
    rows = _read()
    seen = {(r["player_id"], r["gameweek"], r["quote"]) for r in rows}
    new = [
        {
            "player_id": s["element_id"],
            "player": s["web_name"],
            "team": s["team_name"],
            "gameweek": gameweek,
            "applied_chance": s["chance_of_playing"],
            "baseline_chance": s["baseline_chance"],
            "category": s["signal_type"].removeprefix("PRESS_").lower(),
            "minutes_limit": s.get("expected_minutes_limit"),
            "quote": s["quote"],
            "url": s["source_url"],
            "logged_at": datetime.now(UTC).isoformat(),
            "played": None,
        }
        for s in signals
        if s.get("applied_to_production")
        and str(s.get("signal_type", "")).startswith("PRESS_")
        and s.get("quote")
        and (s["element_id"], gameweek, s["quote"]) not in seen
    ]
    if new:
        with _path().open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in new)
    return len(new)


def score_gameweek(gameweek: int, minutes: dict[int, int]) -> dict[str, Any]:
    """Mark logged signals of a finished gameweek with whether the player played (minutes > 0); report precision."""
    rows = _read()
    for r in rows:
        if r["gameweek"] == gameweek and r["played"] is None and r["player_id"] in minutes:
            r["played"] = minutes[r["player_id"]] > 0
    if rows:
        _path().write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return report(rows)


def report(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """precision = share of 0% (ruled out) signals whose player did not play; doubts report the played rate."""
    done = [r for r in (_read() if rows is None else rows) if r["played"] is not None]
    out = [r for r in done if r["applied_chance"] == 0]
    doubt = [r for r in done if 0 < r["applied_chance"] < 100]
    return {
        "scored": len(done),
        "pending": len(_read() if rows is None else rows) - len(done),
        "ruled_out_n": len(out),
        "ruled_out_precision": round(sum(not r["played"] for r in out) / len(out), 3) if out else None,
        "doubt_n": len(doubt),
        "doubt_played_rate": round(sum(bool(r["played"]) for r in doubt) / len(doubt), 3) if doubt else None,
    }


async def score_finished() -> dict[str, Any] | None:
    """Score every logged gameweek that FPL has finished and checked (called from the forward job)."""
    from fpl_oracle.api.fpl_client import fpl_client

    pending = {r["gameweek"] for r in _read() if r["played"] is None}
    if not pending:
        return None
    boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
    result = None
    for ev in boot.events:
        if ev.id in pending and ev.finished and ev.data_checked and not stale:
            live, live_stale = await fpl_client.get_live_gameweek(ev.id, force_refresh=True)
            if not live_stale and isinstance(live.get("elements"), list):
                result = score_gameweek(ev.id, {int(e["id"]): int(e["stats"]["minutes"]) for e in live["elements"]})
    return result
