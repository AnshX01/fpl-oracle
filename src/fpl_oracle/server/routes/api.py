"""
FastAPI REST API routes for FPL Oracle.
"""

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.api.rules_checker import rules_checker
from fpl_oracle.briefing.decision_card import decision_card_generator, format_decision_card_markdown
from fpl_oracle.briefing.review import post_gameweek_reviewer
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.data.fuzzy_match import fuzzy_matcher
from fpl_oracle.data.store import data_store
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.llm.agent import expert_agent
from fpl_oracle.llm.provider import get_llm_status
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.news.ingest import news_ingestion
from fpl_oracle.optimise.contingency import contingency_engine
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.optimise.transfers import transfer_optimizer
from fpl_oracle.server.jobs import get_jobs_status, run_job_on_demand
from fpl_oracle.server.pipeline import sync_pipeline
from fpl_oracle.server.safe_json import SafeJSONResponse, safe_json_serialize

router = APIRouter(prefix="/api", default_response_class=SafeJSONResponse)


class RollbackRequest(BaseModel):
    target_version: str


class ProfileUpdateRequest(BaseModel):
    manager_id: int | None = None
    target_league_id: int | None = None
    risk_preference: str | None = None
    llm_provider: str | None = None
    bank: float | None = None
    free_transfers: int | None = None
    manual_squad: list[int] | None = None


class MatchSquadRequest(BaseModel):
    raw_text: str


class ManualSquadRequest(BaseModel):
    player_ids: list[int]
    bank: float | None = 0.0
    free_transfers: int | None = 1


class OptimizeRequest(BaseModel):
    locked_in: list[int] | None = None
    locked_out: list[int] | None = None
    excluded_teams: list[int] | None = None
    custom_budget: float | None = None


class PanicRequest(BaseModel):
    query: str | None = None
    ruled_out_ids: list[int] | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = "default"


@router.get("/health")
async def get_health():
    game_state = await game_state_manager.get_game_state()
    cache_ages = cache_manager.get_all_cache_ages()
    model_status = projection_engine.get_model_status()
    news_status = news_ingestion.get_news_status()
    llm_status = get_llm_status()

    rules_ver = rules_checker.get_last_result()
    if rules_ver is None:
        try:
            boot, _ = await fpl_client.get_bootstrap_static()
            rules_ver = rules_checker.verify(boot)
        except Exception:
            pass

    return {
        "status": "healthy" if not fpl_client.is_stale_mode else "degraded",
        "api_reachability": not fpl_client.is_stale_mode,
        "rules_verification": rules_ver.to_dict()
        if rules_ver
        else {"verified": True, "rules_source": "cached", "mismatches": []},
        "season": game_state.season,
        "current_gameweek": game_state.current_gw,
        "next_gameweek": game_state.next_gw,
        "game_state": {
            "phase": game_state.phase.value,
            "deadline_time": game_state.deadline_time,
            "seconds_to_deadline": game_state.seconds_to_deadline,
            "is_live": game_state.is_live,
            "bonus_added": game_state.bonus_added,
            "leagues_updated": game_state.leagues_updated,
            "blank_gws": game_state.blank_gws,
            "double_gws": game_state.double_gws,
            "postponed_fixtures_count": game_state.postponed_fixtures_count,
        },
        "cache_age_seconds": cache_ages,
        "model": model_status,
        "news": news_status,
        "llm": llm_status,
        "is_stale": fpl_client.is_stale_mode,
        "stale": fpl_client.is_stale_mode,
        "last_sync": fpl_client.last_sync_time.isoformat() if fpl_client.last_sync_time else None,
        "data_as_of": game_state.data_as_of,
    }


@router.get("/game-state")
async def get_game_state_endpoint():
    state = await game_state_manager.get_game_state()
    return safe_json_serialize(state.model_dump())


@router.get("/profile")
def get_profile():
    p = data_store.get_profile()
    from fpl_oracle.config import app_config

    diag = app_config.get_redacted_status()
    return {
        "manager_id": p.manager_id,
        "manager_id_configured": diag["manager_id_configured"],
        "manager_id_redacted": diag["manager_id_redacted"],
        "target_league_id": p.target_league_id,
        "league_id_configured": diag["league_id_configured"],
        "league_id_redacted": diag["league_id_redacted"],
        "risk_preference": p.risk_preference,
        "llm_provider": p.llm_provider,
        "bank": p.bank,
        "free_transfers": p.free_transfers,
        "manual_squad": json.loads(p.manual_squad) if p.manual_squad else None,
        "status_diagnostic": diag,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@router.post("/profile")
def update_profile(req: ProfileUpdateRequest):
    kwargs: dict[str, Any] = {}
    if req.risk_preference is not None:
        kwargs["risk_preference"] = req.risk_preference
    if req.bank is not None:
        kwargs["bank"] = req.bank
    if req.free_transfers is not None:
        kwargs["free_transfers"] = req.free_transfers
    if req.manual_squad is not None:
        kwargs["manual_squad"] = req.manual_squad

    data_store.update_profile(**kwargs)
    return {"status": "success", "profile": get_profile()}


@router.get("/config/status")
def get_config_status():
    """Diagnostic status endpoint returning configured variables without exposing secrets."""
    from fpl_oracle.config import app_config

    return app_config.get_redacted_status()


@router.post("/sync/trigger")
async def trigger_sync_endpoint():
    """Trigger the background synchronization and analysis pipeline. Returns run ID."""
    return sync_pipeline.trigger_sync()


@router.get("/sync/stream")
async def get_sync_stream():
    """
    Server-Sent Events (SSE) stream broadcasting background analysis pipeline progress.
    Client disconnection does not terminate the underlying background analysis.
    """
    return StreamingResponse(
        sync_pipeline.subscribe(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sync/status")
def get_sync_status():
    """Return status and latest summary of the background synchronization pipeline."""
    return sync_pipeline.get_status()


@router.get("/squad")
async def get_squad(manager_id: int | None = None):
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)

    fixtures_list, _ = await fpl_client.get_fixtures()
    team_map = {t.id: t for t in boot.teams}

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures=fixtures_list)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
        user_squad_df = squad_res["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    # Attach current GW projections to user squad
    if not target_df.empty and "expected_points" in target_df.columns:
        xp_map = {int(r["element"]): float(r["expected_points"]) for _, r in target_df.iterrows()}
        p10_map = {int(r["element"]): float(r.get("p10", 0.0)) for _, r in target_df.iterrows()}
        p90_map = {int(r["element"]): float(r.get("p90", 0.0)) for _, r in target_df.iterrows()}
        defcon_map = {int(r["element"]): float(r.get("exp_defcon_pts", 0.0)) for _, r in target_df.iterrows()}

        user_squad_df["expected_points"] = user_squad_df["element"].map(xp_map).fillna(user_squad_df["expected_points"])
        user_squad_df["p10"] = user_squad_df["element"].map(p10_map).fillna(0.0)
        user_squad_df["p90"] = user_squad_df["element"].map(p90_map).fillna(0.0)
        user_squad_df["exp_defcon_pts"] = user_squad_df["element"].map(defcon_map).fillna(0.0)

    lineup_res = lineup_optimizer.select_lineup_and_captain(user_squad_df)

    def get_next_fixtures(team_id: int):
        f_list = []
        for f in fixtures_list:
            if f.event and f.event >= target_gw and f.event < target_gw + 5:
                if f.team_h == team_id:
                    opp = team_map.get(f.team_a)
                    f_list.append(
                        {
                            "event": f.event,
                            "opp_short": opp.short_name if opp else "OPP",
                            "is_home": True,
                            "difficulty": f.team_h_difficulty or 3,
                        }
                    )
                elif f.team_a == team_id:
                    opp = team_map.get(f.team_h)
                    f_list.append(
                        {
                            "event": f.event,
                            "opp_short": opp.short_name if opp else "OPP",
                            "is_home": False,
                            "difficulty": f.team_a_difficulty or 3,
                        }
                    )
        return f_list[:5]

    news_map = {}
    try:
        raw_signals = await news_analyzer.get_player_news_signals(boot, target_gw=target_gw)
        for sig in raw_signals:
            news_map[sig["element_id"]] = sig
    except Exception:
        pass

    starters_out = []
    for _, s in lineup_res["starters"].iterrows():
        elem_id = int(s["element"])
        sig = news_map.get(elem_id, {})
        starters_out.append(
            {
                "element": elem_id,
                "web_name": s["web_name"],
                "team": int(s["team"]),
                "team_short": team_map[s["team"]].short_name if s["team"] in team_map else "PL",
                "position": s["position"],
                "now_cost": round(float(s["value"]) / 10.0, 1),
                "purchase_price": round(float(s.get("purchase_price", s["value"])) / 10.0, 1),
                "selling_price": round(float(s.get("selling_price", s["value"])) / 10.0, 1),
                "expected_points": round(float(s["expected_points"]), 2),
                "p10": round(float(s.get("p10", 0.0)), 2),
                "p90": round(float(s.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(s.get("exp_defcon_pts", 0.0)), 2),
                "chance_of_playing": s.get("chance_of_playing", 100),
                "status": s.get("status", "a"),
                "news": s.get("news", ""),
                "news_quote": sig.get("quote", ""),
                "source_url": sig.get("source_url", ""),
                "news_mode": "gated_active"
                if sig.get("applied_to_production")
                else ("shadow" if sig.get("quote") else "official"),
                "expected_minutes_limit": sig.get("expected_minutes_limit"),
                "reconciliation_reason": sig.get("reconciliation_reason", ""),
                "is_captain": elem_id == lineup_res["captain"]["element"],
                "is_vice_captain": elem_id == lineup_res["vice_captain"]["element"],
                "next_fixtures": get_next_fixtures(int(s["team"])),
            }
        )

    bench_out = []
    for idx, (_, b) in enumerate(lineup_res["bench"].iterrows(), start=1):
        b_elem_id = int(b["element"])
        b_sig = news_map.get(b_elem_id, {})
        bench_out.append(
            {
                "element": b_elem_id,
                "web_name": b["web_name"],
                "team": int(b["team"]),
                "team_short": team_map[b["team"]].short_name if b["team"] in team_map else "PL",
                "position": b["position"],
                "now_cost": round(float(b["value"]) / 10.0, 1),
                "purchase_price": round(float(b.get("purchase_price", b["value"])) / 10.0, 1),
                "selling_price": round(float(b.get("selling_price", b["value"])) / 10.0, 1),
                "expected_points": round(float(b["expected_points"]), 2),
                "p10": round(float(b.get("p10", 0.0)), 2),
                "p90": round(float(b.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(b.get("exp_defcon_pts", 0.0)), 2),
                "chance_of_playing": b.get("chance_of_playing", 100),
                "status": b.get("status", "a"),
                "news": b.get("news", ""),
                "news_quote": b_sig.get("quote", ""),
                "source_url": b_sig.get("source_url", ""),
                "news_mode": "gated_active"
                if b_sig.get("applied_to_production")
                else ("shadow" if b_sig.get("quote") else "official"),
                "expected_minutes_limit": b_sig.get("expected_minutes_limit"),
                "reconciliation_reason": b_sig.get("reconciliation_reason", ""),
                "bench_order": idx,
                "next_fixtures": get_next_fixtures(int(b["team"])),
            }
        )

    squad_val = round(sum(s["now_cost"] for s in starters_out + bench_out), 1)
    sell_val = round(sum(s["selling_price"] for s in starters_out + bench_out), 1)
    team_val = round(sell_val + effective_state.bank_millions, 1)

    return safe_json_serialize(
        {
            "manager_id": effective_state.manager_id,
            "manager_name": effective_state.manager_name,
            "team_name": effective_state.team_name,
            "overall_points": effective_state.overall_points,
            "overall_rank": effective_state.overall_rank,
            "mode": effective_state.mode.value,
            "confidence": effective_state.confidence,
            "target_gameweek": target_gw,
            "bank_millions": effective_state.bank_millions,
            "bank": effective_state.bank_millions,
            "bank_source": effective_state.bank_source,
            "has_bank_override": effective_state.has_bank_override,
            "free_transfers": effective_state.free_transfers,
            "available_transfers": effective_state.free_transfers,
            "ft_source": effective_state.ft_source,
            "has_ft_override": effective_state.has_ft_override,
            "total_squad_value": squad_val,
            "total_selling_value": sell_val,
            "total_team_value": team_val,
            "chips_used": [c.get("name") for c in effective_state.chips_used],
            "formation": lineup_res["formation"],
            "captain": lineup_res["captain"],
            "vice_captain": lineup_res["vice_captain"],
            "starters": starters_out,
            "bench": bench_out,
            "starters_expected_points": lineup_res.get("starters_expected_points", 0.0),
            "captain_bonus_expected_points": lineup_res.get(
                "captain_bonus_expected_points", lineup_res["captain"]["expected_points"]
            ),
            "total_expected_points": lineup_res["total_gameweek_expected_points"],
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.post("/squad/match")
async def match_squad_names(req: MatchSquadRequest):
    boot, _ = await fpl_client.get_bootstrap_static()
    team_map = {t.id: t.short_name for t in boot.teams}
    res = fuzzy_matcher.parse_and_match_squad(req.raw_text, boot.elements, team_map)
    return safe_json_serialize(res)


@router.post("/squad/manual")
async def save_manual_squad(req: ManualSquadRequest):
    if len(req.player_ids) != 15:
        raise HTTPException(status_code=400, detail="A valid squad must contain exactly 15 players.")
    boot, _ = await fpl_client.get_bootstrap_static()
    elem_dict = {e.id: e for e in boot.elements}

    # Verify positions and 3-per-club rule
    pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    team_counts: dict[int, int] = {}
    for pid in req.player_ids:
        if pid not in elem_dict:
            raise HTTPException(status_code=400, detail=f"Player ID {pid} is invalid.")
        elem = elem_dict[pid]
        pos_counts[elem.element_type] += 1
        team_counts[elem.team] = team_counts.get(elem.team, 0) + 1
        if team_counts[elem.team] > 3:
            raise HTTPException(status_code=400, detail=f"Exceeded max 3 players for club {elem.team}.")

    if pos_counts[1] != 2 or pos_counts[2] != 5 or pos_counts[3] != 5 or pos_counts[4] != 3:
        raise HTTPException(status_code=400, detail="Invalid formation! Required: 2 GKP, 5 DEF, 5 MID, 3 FWD.")

    data_store.update_profile(manual_squad=req.player_ids, bank=req.bank, free_transfers=req.free_transfers)
    return safe_json_serialize(
        {"status": "success", "message": "Custom squad saved successfully.", "player_ids": req.player_ids}
    )


@router.get("/projections")
async def get_projections(position: str | None = None, team_id: int | None = None, horizon: int = Query(5, ge=1, le=8)):
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    _, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, horizon, boot, fixtures)
    df = horizon_proj.get(target_gw, pd.DataFrame())

    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if team_id:
        df = df[df["team"] == team_id]

    results = []
    for _, r in df.sort_values(by="expected_points", ascending=False).head(50).iterrows():
        results.append(
            {
                "element": int(r["element"]),
                "web_name": r["web_name"],
                "team": int(r["team"]),
                "position": r["position"],
                "cost": round(r["value"] / 10.0, 1),
                "expected_points": round(float(r["expected_points"]), 2),
                "p10": round(float(r.get("p10", 0.0)), 2),
                "p90": round(float(r.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(r.get("exp_defcon_pts", 0.0)), 2),
            }
        )

    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "horizon": horizon,
            "players": results,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.post("/optimize")
async def run_optimizer(req: OptimizeRequest | None = None):
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    # Get user squad from single source of truth
    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        user_squad_df = squad_optimizer.solve_best_squad(target_df, budget=1000.0)["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    locked_in = req.locked_in if req else None
    locked_out = req.locked_out if req else None
    excl_teams = req.excluded_teams if req else None

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=float(effective_state.bank_tenths),
        free_transfers=int(effective_state.free_transfers),
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        locked_in_ids=locked_in,
        locked_out_ids=locked_out,
        excluded_team_ids=excl_teams,
    )
    res["stale"] = is_stale
    res["is_stale"] = is_stale
    res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    res["manager_state"] = {
        "mode": effective_state.mode.value,
        "bank_millions": effective_state.bank_millions,
        "bank_source": effective_state.bank_source,
        "free_transfers": effective_state.free_transfers,
        "ft_source": effective_state.ft_source,
    }
    return safe_json_serialize(res)


@router.get("/chips")
async def get_chip_strategy():
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)
    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 8, boot, fixtures)
    pool_df = horizon_proj.get(target_gw, pd.DataFrame())

    hist = None
    if effective_state.manager_id:
        try:
            hist, _ = await fpl_client.get_manager_history(effective_state.manager_id)
        except Exception:
            pass

    squad_df, _, _ = await _get_effective_user_squad(pool_df, boot)

    res = chip_planner.generate_chip_strategy(
        current_gw=curr_gw or 5,
        current_squad_df=squad_df,
        horizon_projections=horizon_proj,
        fixtures=fixtures,
        bootstrap=boot,
        manager_history=hist,
    )
    res["stale"] = is_stale
    res["is_stale"] = is_stale
    res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(res)


@router.get("/league")
async def get_league_intel(league_id: int | None = None):
    effective_state = await manager_state_service.get_current_state()
    profile = data_store.get_profile()
    l_id = league_id or profile.target_league_id
    if not l_id:
        return safe_json_serialize(
            {
                "status": "unconfigured",
                "message": "No target mini-league ID configured. Enter a mini-league ID to see rival analysis.",
                "league_name": "None",
                "total_teams": 0,
                "standings": [],
                "template_players": [],
                "differential_players": [],
                "simulation": {"user_win_probability_pct": 0.0, "expected_final_rank": 1.0},
                "strategy": {
                    "mode_title": "Setup Required",
                    "rationale": "Set your mini-league ID in settings to activate rival analysis.",
                    "tactical_recommendations": [],
                },
                "stale": False,
                "is_stale": False,
                "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
            }
        )

    standings_data = await league_standings_manager.get_league_standings(l_id)
    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, _ = await fpl_client.get_current_and_next_gw()

    if not standings_data.get("standings"):
        return safe_json_serialize(
            {
                "status": "empty",
                "message": f"No standings found for mini-league {l_id}.",
                "league_name": standings_data.get("league_name", f"League #{l_id}"),
                "total_teams": 0,
                "standings": [],
                "template_players": [],
                "differential_players": [],
                "simulation": {"user_win_probability_pct": 0.0, "expected_final_rank": 1.0},
                "strategy": {
                    "mode_title": "No Standings",
                    "rationale": "No standings data returned from FPL API for this league.",
                    "tactical_recommendations": [],
                },
                "stale": is_stale,
                "is_stale": is_stale,
                "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
            }
        )

    rivals_res = await rival_analyzer.analyze_rivals(
        standings=standings_data["standings"],
        user_manager_id=effective_state.manager_id,
        current_gw=curr_gw or 5,
        bootstrap=boot,
    )

    user_pts = effective_state.overall_points
    # Find user's rank in this mini-league if present, otherwise fallback to overall_rank or 1
    user_mini_rank = None
    if effective_state.manager_id and standings_data.get("standings"):
        for row in standings_data["standings"]:
            if row.get("entry") == effective_state.manager_id:
                user_mini_rank = row.get("rank")
                break
    user_rank: int = (
        int(user_mini_rank)
        if user_mini_rank is not None
        else (int(effective_state.overall_rank) if effective_state.overall_rank else 1)
    )

    # Monte Carlo simulation
    fixtures, _ = await fpl_client.get_fixtures()
    sim_target_gw = (curr_gw + 1) if curr_gw else 2
    projections_horizon = projection_engine.predict_multi_gameweeks(sim_target_gw, 5, boot, fixtures)
    proj_df = projections_horizon.get(sim_target_gw)

    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) < 15:
        user_squad_df = (
            proj_df.sort_values(by="expected_points", ascending=False).head(15)
            if proj_df is not None and not proj_df.empty
            else pd.DataFrame()
        )

    mc_res = monte_carlo_simulator.simulate_league(
        user_points=user_pts,
        user_squad_df=user_squad_df,
        rival_squads=rivals_res["rival_squads"],
        projections_df=proj_df,
        horizon_gws=5,
        projections_by_gw=projections_horizon,
    )

    strategy = league_strategy_advisor.evaluate_strategy(
        user_rank=user_rank, user_total_points=user_pts, rivals_analysis=rivals_res, user_squad_df=user_squad_df
    )

    return safe_json_serialize(
        {
            "status": "success",
            "league_name": standings_data["league_name"],
            "total_teams": standings_data["total_teams"],
            "standings": standings_data["standings"],
            "template_players": rivals_res["template_players"],
            "differential_players": rivals_res["differential_players"],
            "simulation": mc_res,
            "strategy": strategy,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.get("/news")
async def get_news_signals(
    gw: int | None = None,
):
    """Return all reconciled news signals with verbatim quotes, source links, and mode badges (N7)."""
    boot, _ = await fpl_client.get_bootstrap_static()
    target_gw = gw or 6
    signals = await news_analyzer.get_player_news_signals(boot, target_gw=target_gw)
    return safe_json_serialize(
        {
            "target_gw": target_gw,
            "count": len(signals),
            "signals": signals,
        }
    )


@router.get("/decision-card")
async def get_decision_card_endpoint():
    """
    Returns the comprehensive per-gameweek unified decision card (D1).
    Reads from shared game state, projection snapshots, beam search transfers,
    chip calendar, and joint rival simulation.
    """
    card = await decision_card_generator.generate_decision_card()
    return safe_json_serialize(card)


@router.get("/decision-card/export", response_class=PlainTextResponse)
async def export_decision_card_endpoint():
    """
    Returns a printable plain text / markdown gameweek decision card (D2).
    """
    card = await decision_card_generator.generate_decision_card()
    markdown_text = format_decision_card_markdown(card)
    return PlainTextResponse(content=markdown_text, media_type="text/plain; charset=utf-8")


@router.get("/briefing")
async def get_briefing():
    return await weekly_briefing_generator.generate_briefing()


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    ans = await expert_agent.answer(user_message=req.message, session_id=req.session_id or "default")
    return safe_json_serialize({"response": ans, "data_as_of": fpl_client.get_data_as_of("bootstrap-static")})


@router.get("/price-changes")
async def get_price_changes():
    boot, is_stale = await fpl_client.get_bootstrap_static()
    preds = price_change_predictor.analyze_price_changes(boot)
    return safe_json_serialize(
        {
            "rises": [p for p in preds if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:10],
            "falls": [p for p in preds if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:10],
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


async def _get_effective_user_squad(target_df: pd.DataFrame, boot: Any) -> tuple[pd.DataFrame, float, int]:
    effective_state = await manager_state_service.get_current_state()
    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
        user_squad_df = squad_res["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    if not target_df.empty and "expected_points" in target_df.columns:
        xp_map = {int(r["element"]): float(r["expected_points"]) for _, r in target_df.iterrows()}
        p10_map = {int(r["element"]): float(r.get("p10", 0.0)) for _, r in target_df.iterrows()}
        p90_map = {int(r["element"]): float(r.get("p90", 0.0)) for _, r in target_df.iterrows()}
        user_squad_df["expected_points"] = user_squad_df["element"].map(xp_map).fillna(user_squad_df["expected_points"])
        user_squad_df["p10"] = user_squad_df["element"].map(p10_map).fillna(0.0)
        user_squad_df["p90"] = user_squad_df["element"].map(p90_map).fillna(0.0)

    return user_squad_df, float(effective_state.bank_tenths), int(effective_state.free_transfers)


@router.get("/contingency/plans")
async def get_contingency_plans():
    """Returns precomputed Plan A, Plan B (injury pivot), and Plan C (differential/price pivot)."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)
    profile = data_store.get_profile()

    plans = contingency_engine.generate_contingency_plans(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=free_transfers,
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        risk_preference=profile.risk_preference or "balanced",
    )
    plans["stale"] = is_stale
    plans["is_stale"] = is_stale
    plans["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(plans)


@router.get("/contingency/matrix")
async def get_contingency_matrix():
    """Returns 'What if Player X is ruled out' matrix comparing auto-sub vs emergency transfer."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)

    matrix = contingency_engine.compute_injury_matrix(
        squad_df=user_squad_df, player_pool_df=target_df, bank=bank, free_transfers=free_transfers, bootstrap=boot
    )
    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "contingency_matrix": matrix,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.post("/contingency/panic")
async def post_contingency_panic(req: PanicRequest):
    """Emergency 1-click crisis solver for breaking team news (e.g. 'Saka ruled out 6 weeks')."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)

    crisis_res = contingency_engine.panic_button_reoptimize(
        query=req.query or "",
        squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=free_transfers,
        ruled_out_ids=req.ruled_out_ids,
    )
    crisis_res["stale"] = is_stale
    crisis_res["is_stale"] = is_stale
    crisis_res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(crisis_res)


@router.get("/contingency/checklist")
async def get_pre_deadline_checklist():
    """Generates 5-point operational pre-deadline audit."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)
    game_state = await game_state_manager.get_game_state()

    profile = data_store.get_profile()
    history = None
    if profile.manager_id:
        try:
            history, _ = await fpl_client.get_manager_history(profile.manager_id)
        except Exception:
            pass

    chips_status = chip_planner.get_remaining_chips(history)
    checklist = contingency_engine.generate_pre_deadline_checklist(
        squad_df=user_squad_df, bootstrap=boot, game_state_data=game_state.model_dump(), chips_status=chips_status
    )
    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "seconds_to_deadline": game_state.seconds_to_deadline,
            "checklist": checklist,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.get("/review")
async def get_gameweek_review(gameweek: int | None = None):
    """Produces post-gameweek review and performance diagnostics."""
    profile = data_store.get_profile()
    return await post_gameweek_reviewer.generate_gameweek_review(profile.manager_id, gameweek)


@router.get("/system/jobs")
async def get_system_jobs():
    """Returns execution status and telemetry for all background scheduler jobs."""
    status = get_jobs_status()
    return safe_json_serialize(status)


@router.post("/system/jobs/{job_id}/run")
async def trigger_job_run(job_id: str):
    """Manually triggers immediate execution of a scheduled background job."""
    res = await run_job_on_demand(job_id)
    if not res.get("success", False) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return safe_json_serialize(res)


@router.get("/system/models")
async def get_model_versions_endpoint():
    """Returns active model version, metric history, and rollback checkpoints."""
    active = model_registry.get_active_version()
    history = model_registry.get_version_history()
    return safe_json_serialize({"active_version": active, "history": history})


@router.post("/system/models/rollback")
async def rollback_model_endpoint(req: RollbackRequest):
    """Restores an archived model version checkpoint to active production."""
    res = model_registry.rollback_to_version(req.target_version)
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("error", "Rollback failed"))
    return safe_json_serialize(res)


@router.get("/system/status")
async def get_system_status():
    """Comprehensive system diagnostic telemetry."""
    game_state = await game_state_manager.get_game_state()
    active_model = model_registry.get_active_version()
    news_stat = news_ingestion.get_news_status()
    jobs_stat = get_jobs_status()
    db_runs = data_store.get_job_runs(limit=5)
    prices = data_store.get_latest_price_snapshots(limit=10)

    return safe_json_serialize(
        {
            "game_state": {
                "phase": game_state.phase.value,
                "seconds_to_deadline": round(game_state.seconds_to_deadline, 1)
                if game_state.seconds_to_deadline is not None
                else None,
                "current_gw": game_state.current_gw,
                "next_gw": game_state.next_gw,
            },
            "model": {
                "active_version": active_model.get("version"),
                "mae": active_model.get("ml_mae"),
                "spearman": active_model.get("ml_spearman"),
                "status": active_model.get("status"),
            },
            "scheduler": {"running": jobs_stat["scheduler_running"], "total_jobs": jobs_stat["total_jobs"]},
            "news": news_stat,
            "price_snapshots_recorded": len(prices),
            "recent_job_runs": db_runs,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )
