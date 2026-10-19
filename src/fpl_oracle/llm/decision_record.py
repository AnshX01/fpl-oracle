"""
Compact decision record for the chat advisor.

Built from the advice the app has already published (advice_publisher.current["result"]:
squadData, decisionCard, contingencyPlans), so chat quotes exactly what the manager sees.
It is injected into the system prompt every turn and returned by the get_current_decision tool.
"""

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("fpl_oracle.llm.decision_record")

_ARMBAND_KEYS = ("element", "web_name", "expected_points", "p10", "p90")
SECTIONS = ("team", "transfers", "chip", "lineup", "alternatives", "league", "meta", "raw")


def _clean(v: Any, depth: int = 3, items: int = 6, chars: int = 240) -> Any:
    """JSON-safe, size-bounded copy of an arbitrary planner blob."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, dict):
        if depth == 0:
            return f"<{len(v)} fields>"
        return {str(k): _clean(x, depth - 1, items, chars) for k, x in list(v.items())[:30]}
    if isinstance(v, (list, tuple, set)):
        seq = list(v)
        out = [_clean(x, depth - 1, items, chars) for x in seq[:items]]
        if len(seq) > items:
            out.append(f"... {len(seq) - items} more")
        return out
    if v is None or isinstance(v, (bool, int, str)):
        return v if not isinstance(v, str) or len(v) <= chars else v[:chars] + "..."
    if isinstance(v, float):
        return None if v != v else round(v, 3)
    s = str(v)
    return s if len(s) <= chars else s[:chars] + "..."


def _fixtures(rows: list[dict[str, Any]] | None) -> list[str]:
    return [
        f"GW{f.get('event')} {f.get('opp_short')}({'H' if f.get('is_home') else 'A'},fdr{f.get('difficulty')})"
        for f in (rows or [])[:5]
    ]


def _player(p: dict[str, Any], form: dict[int, dict[str, Any]]) -> dict[str, Any]:
    eid = p.get("element")
    out = {
        "id": eid,
        "name": p.get("web_name"),
        "pos": p.get("position"),
        "team": p.get("team_short"),
        "price": p.get("now_cost"),
        "sell": p.get("selling_price"),
        "xp": p.get("expected_points"),
        "p10": p.get("p10"),
        "p90": p.get("p90"),
        "defcon_xp": p.get("exp_defcon_pts"),
        "play_chance": p.get("chance_of_playing"),
        "status": p.get("status"),
        "news": p.get("news") or None,
        "minutes_limit": p.get("expected_minutes_limit"),
        "bench_order": p.get("bench_order"),
        "armband": "C" if p.get("is_captain") else ("VC" if p.get("is_vice_captain") else None),
        "fixtures": _fixtures(p.get("next_fixtures")),
        **form.get(int(eid or 0), {}),
    }
    return {k: _clean(v) for k, v in out.items() if v not in (None, "", [])}


def _plan_option(plan: dict[str, Any]) -> dict[str, Any]:
    names = lambda key: [p.get("web_name") for p in plan.get(key) or [] if isinstance(p, dict)]  # noqa: E731
    return _clean(
        {
            "title": plan.get("title"),
            "type": plan.get("plan_type"),
            "in": names("transfers_in"),
            "out": names("transfers_out"),
            "hits": plan.get("hits"),
            "net_xp": plan.get("net_expected_points"),
            "delta_vs_plan_a": plan.get("delta_vs_plan_a"),
            "trigger": plan.get("trigger_condition"),
            "summary": plan.get("action_summary"),
        }
    )


def build_decision_record(result: dict[str, Any], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """Faithful summary of one published advice result. `extras`: optional live context (form, chips)."""
    extras = extras or {}
    squad = result.get("squadData") or {}
    card = result.get("decisionCard") or {}
    plans = result.get("contingencyPlans") or {}
    form = extras.get("form") or {}
    starters = sorted(squad.get("starters") or [], key=lambda p: -(p.get("expected_points") or 0))
    bench = sorted(squad.get("bench") or [], key=lambda p: p.get("bench_order") or 99)
    captain, vice = squad.get("captain") or card.get("captain") or {}, squad.get("vice_captain") or {}
    transfers = card.get("transfers") or {}
    chip = card.get("chip") or {}

    def moves(key: str) -> list[dict[str, Any]]:
        return [
            _clean(
                {
                    "id": p.get("element"),
                    "name": p.get("web_name"),
                    "price": p.get("cost"),
                    "xp_gw": p.get("expected_points"),
                }
            )
            for p in transfers.get(key) or []
        ]

    top = [(p.get("web_name"), p.get("expected_points")) for p in starters[:3]]
    return {
        "status": "ready",
        "gameweek": card.get("gameweek") or squad.get("target_gameweek"),
        "snapshot_id": result.get("snapshot_id"),
        "deadline": _clean(card.get("deadline")),
        "meta": _clean(
            {
                "model_version": card.get("model_version"),
                "projection_snapshot_id": card.get("projection_snapshot_id"),
                "data_freshness": card.get("data_freshness"),
                "data_as_of": result.get("data_as_of"),
                "is_stale": bool(card.get("is_stale") or squad.get("is_stale")),
                "decision_scope": card.get("decision_scope"),
                "decision_readiness": card.get("decision_readiness"),
                "advice_only": card.get("advice_disclaimer"),
            },
            depth=3,
        ),
        "team": {
            "manager": squad.get("manager_name"),
            "team_name": squad.get("team_name"),
            "overall_points": squad.get("overall_points"),
            "overall_rank": squad.get("overall_rank"),
            "bank_millions": squad.get("bank_millions"),
            "free_transfers": squad.get("free_transfers"),
            "squad_value": squad.get("total_squad_value"),
            "selling_value": squad.get("total_selling_value"),
            "chips_used": squad.get("chips_used"),
            "chips_remaining": extras.get("chips"),
            "note": "Players are the squad AFTER the recommended moves (what the manager fields).",
        },
        "transfers": _clean(
            {
                "action": transfers.get("action"),
                "in": moves("in"),
                "out": moves("out"),
                "hits": transfers.get("hits_count"),
                "hit_cost": transfers.get("hit_cost"),
                "free_transfers_used": transfers.get("ft_used"),
                "free_transfers_left": transfers.get("ft_remaining"),
                "free_transfers_next_gw": transfers.get("ft_next_gw"),
                "bank_after_millions": transfers.get("bank_after"),
                "gain_this_gw_xp": transfers.get("expected_gain_gw"),
                "gain_vs_roll_over_horizon_xp": transfers.get("net_gain_vs_roll"),
                "horizon_xp": transfers.get("horizon_pts"),
                "no_regret": transfers.get("no_regret_flag"),
                "summary": transfers.get("action_summary"),
                "hit_policy": card.get("hit_policy"),
                "why": card.get("two_line_reasoning"),
                "versus_saving": transfers.get("comparison"),
            },
            depth=3,
        ),
        "chip": _clean(
            {
                "play_now": chip.get("recommend"),
                "chip": chip.get("chip_display_name"),
                "gain_vs_hold_xp": chip.get("gain_vs_hold"),
                "comparison": chip.get("comparison"),
                "check": chip.get("check"),
                "later_provisional": chip.get("later"),
                "reason": chip.get("reason"),
                "set_1_warning": chip.get("set_1_deadline_warning"),
                "now_vs_later": card.get("ranked_options"),
            },
            items=10,
        ),
        "lineup": {
            "formation": squad.get("formation") or card.get("formation"),
            "captain": _clean({k: captain.get(k) for k in _ARMBAND_KEYS}),
            "vice_captain": _clean({k: vice.get(k) for k in _ARMBAND_KEYS}),
            "captain_reason": (card.get("captain") or {}).get("reason")
            or f"Highest-xP starters: {top}. Captain and vice come from the same optimiser as the XI.",
            "xi_xp": squad.get("starters_expected_points"),
            "total_xp_with_captain": squad.get("total_expected_points"),
            "xi": [_player(p, form) for p in starters],
            "bench": [_player(p, form) for p in bench],
            "bench_reason": "Bench order is the optimiser's choice; first bench player is first auto-sub.",
            "bench_note": card.get("bench_note"),
        },
        "alternatives": {
            "chip_options": _clean(card.get("ranked_options"), items=10),
            "contingency": {
                k: _plan_option(plans[k]) for k in ("plan_a", "plan_b", "plan_c") if isinstance(plans.get(k), dict)
            },
            "resource_frontier": _clean(card.get("resource_frontier"), depth=3, items=8),
        },
        "league": _clean(
            {
                "rivals": card.get("rivals"),
                "win_probability": card.get("win_prob"),
                "scenarios": card.get("rival_scenarios"),
            },
            depth=3,
        ),
        "caveats": _clean(card.get("caveats"), items=6),
        "what_changed": card.get("what_changed"),
        "close_calls": _clean(card.get("close_calls"), items=4),
        "press_signals": _clean([n.get("text") for n in card.get("press_signals") or []], items=8),
    }


def published_result() -> tuple[dict[str, Any] | None, str]:
    """(result, reason) for the app's current publication; result is None unless status is ready."""
    from fpl_oracle.server.advice_job import advice_publisher

    cur = advice_publisher.current
    if cur and cur.get("status") == "ready" and isinstance(cur.get("result"), dict):
        return cur["result"], ""
    # While a new run calculates (or after a restart), the last published advice for this team still stands.
    previous = (cur or {}).get("previous_result")
    if isinstance(previous, dict) and previous.get("decisionCard"):
        return (
            previous,
            f"Advice from {(cur or {}).get('previous_published_at') or 'the last run'}; an update is running.",
        )
    from fpl_oracle.data.store import data_store
    from fpl_oracle.server.advice_job import load_last_advice, profile_key

    saved = load_last_advice(profile_key(data_store.get_profile()))
    if saved and isinstance(saved.get("result"), dict):
        return saved["result"], f"Advice from {saved.get('published_at')}; refresh the app for newer data."
    if not cur:
        return None, "No advice has been published yet. Open the app's Advice screen to calculate it."
    return None, f"Advice is {cur.get('status')} ({cur.get('stage') or 'no stage'}); it is not published yet."


async def _extras(result: dict[str, Any]) -> dict[str, Any]:
    """Best-effort live context the publication lacks: player form and chips left per set."""
    extras: dict[str, Any] = {}
    try:
        from fpl_oracle.api.fpl_client import fpl_client

        boot, _ = await fpl_client.get_bootstrap_static()
        ids = {p.get("element") for k in ("starters", "bench") for p in (result.get("squadData") or {}).get(k, [])}
        extras["form"] = {
            e.id: {
                "form": e.form,
                "minutes": e.minutes,
                "season_points": e.total_points,
                "owned_pct": e.selected_by_percent,
            }
            for e in boot.elements
            if e.id in ids
        }
    except Exception as e:
        logger.warning("Decision record form context unavailable: %s", type(e).__name__)
    try:
        from fpl_oracle.domain.manager_state import manager_state_service

        state = await manager_state_service.get_current_state()
        extras["chips"] = {"set_1_gw1_19": state.chips_remaining_set_1, "set_2_gw20_38": state.chips_remaining_set_2}
    except Exception as e:
        logger.warning("Decision record chip context unavailable: %s", type(e).__name__)
    return extras


async def current_decision_record() -> dict[str, Any]:
    result, reason = published_result()
    if result is None:
        return {"status": "unavailable", "reason": reason}
    record = build_decision_record(result, await _extras(result))
    if reason:
        record["as_of_note"] = reason
    return record


def section_of(record: dict[str, Any], section: str | None) -> dict[str, Any]:
    """Slice of a record for the get_current_decision tool; 'raw' is the lightly trimmed published card."""
    if record.get("status") != "ready" or not section or section == "all":
        return record
    if section == "raw":
        result, _ = published_result()
        card = (result or {}).get("decisionCard") or {}
        plans = (result or {}).get("contingencyPlans") or {}
        return {
            "status": "ready",
            "decision_card": _clean(card, depth=5, items=15, chars=600),
            "contingency_plans": _clean(plans, depth=5, items=15, chars=600),
        }
    if section not in record:
        return {"status": "error", "error": f"Unknown section {section!r}; use one of {SECTIONS}"}
    return {
        "status": "ready",
        "gameweek": record["gameweek"],
        "snapshot_id": record["snapshot_id"],
        section: record[section],
    }


def prompt_section(record: dict[str, Any]) -> str:
    if record.get("status") != "ready":
        return (
            "## PUBLISHED DECISION: NOT READY\n"
            f"{record.get('reason')}\n"
            "Tell the manager the app has no published advice yet, then use the tools "
            "(optimise_transfers, captain_options, plan_chips, get_my_team) for any number you quote."
        )
    return (
        f"## PUBLISHED DECISION RECORD (GW{record.get('gameweek')}, snapshot {str(record.get('snapshot_id'))[:12]})\n"
        "This is exactly the advice shown to the manager in the app. It is authoritative: explain it, defend it "
        "with its numbers, and compare the manager's ideas against it. Prices are millions, xP is projected points "
        "for the gameweek unless named horizon, fdr is fixture difficulty. Call get_current_decision for any "
        "section or the raw card.\n" + json.dumps(record, default=str, separators=(",", ":"))
    )
