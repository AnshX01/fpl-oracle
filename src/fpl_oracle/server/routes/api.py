"""
FastAPI REST API routes for FPL Oracle.
"""

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.api.rules_checker import rules_checker
from fpl_oracle.briefing.review import post_gameweek_reviewer
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.data.fuzzy_match import fuzzy_matcher
from fpl_oracle.data.store import data_store
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.llm.agent import expert_agent
from fpl_oracle.llm.provider import get_llm_status
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine
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
    return {
        "manager_id": p.manager_id,
        "target_league_id": p.target_league_id,
        "risk_preference": p.risk_preference,
        "llm_provider": p.llm_provider,
        "bank": p.bank,
        "free_transfers": p.free_transfers,
        "manual_squad": json.loads(p.manual_squad) if p.manual_squad else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@router.post("/profile")
def update_profile(req: ProfileUpdateRequest):
    kwargs: dict[str, Any] = {}
    if req.manager_id is not None:
        kwargs["manager_id"] = req.manager_id
    if req.target_league_id is not None:
        kwargs["target_league_id"] = req.target_league_id
    if req.risk_preference is not None:
        kwargs["risk_preference"] = req.risk_preference
    if req.llm_provider is not None:
        kwargs["llm_provider"] = req.llm_provider
    if req.bank is not None:
        kwargs["bank"] = req.bank
    if req.free_transfers is not None:
        kwargs["free_transfers"] = req.free_transfers
    if req.manual_squad is not None:
        kwargs["manual_squad"] = req.manual_squad

    data_store.update_profile(**kwargs)
    return {"status": "success", "profile": get_profile()}


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
    profile = data_store.get_profile()
    m_id = manager_id or profile.manager_id

    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(
        target_gw,
        5,
        boot,
        fixtures=await fpl_client.get_fixtures()[0] if False else (await fpl_client.get_fixtures())[0],
    )
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df = None
    bank = float(profile.bank or 0.5) * 10.0
    free_transfers = int(profile.free_transfers or 1)
    chips_used = []
    manager_name = "Manager"
    team_name = "My FPL Squad"
    overall_points = 0
    overall_rank = None

    if m_id:
        try:
            entry, _ = await fpl_client.get_manager_entry(m_id)
            picks, _ = await fpl_client.get_manager_picks(m_id, curr_gw or 5)
            history, _ = await fpl_client.get_manager_history(m_id)
            transfers, _ = await fpl_client.get_manager_transfers(m_id)
            manager_name = f"{entry.player_first_name} {entry.player_last_name}"
            team_name = entry.name
            overall_points = entry.summary_overall_points or 0
            overall_rank = entry.summary_overall_rank
            chips_used = [c.name for c in history.chips]
            if picks.entry_history:
                bank = float(picks.entry_history.bank)
            if profile.bank is not None and profile.bank > 0:
                bank = float(profile.bank * 10.0)

            auto_ft = transfer_optimizer.calculate_banked_free_transfers(history)
            if profile.free_transfers and profile.free_transfers > 1:
                free_transfers = profile.free_transfers
            else:
                free_transfers = auto_ft

            picks_ids = [p.element for p in picks.picks]
            user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
            user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, transfers, boot)
        except Exception:
            pass
    if (user_squad_df is None or len(user_squad_df) < 15) and profile.manual_squad:
        try:
            manual_ids = (
                json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
            )
            if manual_ids and len(manual_ids) == 15:
                user_squad_df = target_df[target_df["element"].isin(manual_ids)].copy()
                team_name = "Custom / Pasted Squad"
                bank = (profile.bank or 0.0) * 10.0
                user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)
        except Exception:
            pass

    if user_squad_df is None or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
        user_squad_df = squad_res["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    # Formations and Lineup
    lineup_res = lineup_optimizer.select_lineup_and_captain(user_squad_df)

    # Next 5 fixtures per player
    fixtures_list, _ = await fpl_client.get_fixtures()
    team_map = {t.id: t for t in boot.teams}

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

    starters_out = []
    for _, s in lineup_res["starters"].iterrows():
        starters_out.append(
            {
                "element": int(s["element"]),
                "web_name": s["web_name"],
                "team": int(s["team"]),
                "team_short": team_map[s["team"]].short_name if s["team"] in team_map else "PL",
                "position": s["position"],
                "now_cost": round(s["value"] / 10.0, 1),
                "purchase_price": round(float(s.get("purchase_price", s["value"])) / 10.0, 1),
                "selling_price": round(float(s.get("selling_price", s["value"])) / 10.0, 1),
                "expected_points": round(float(s["expected_points"]), 2),
                "p10": round(float(s.get("p10", 0.0)), 2),
                "p90": round(float(s.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(s.get("exp_defcon_pts", 0.0)), 2),
                "is_captain": s["element"] == lineup_res["captain"]["element"],
                "is_vice_captain": s["element"] == lineup_res["vice_captain"]["element"],
                "next_fixtures": get_next_fixtures(int(s["team"])),
            }
        )

    bench_out = []
    for _, b in lineup_res["bench"].iterrows():
        bench_out.append(
            {
                "element": int(b["element"]),
                "web_name": b["web_name"],
                "team": int(b["team"]),
                "team_short": team_map[b["team"]].short_name if b["team"] in team_map else "PL",
                "position": b["position"],
                "now_cost": round(b["value"] / 10.0, 1),
                "purchase_price": round(float(b.get("purchase_price", b["value"])) / 10.0, 1),
                "selling_price": round(float(b.get("selling_price", b["value"])) / 10.0, 1),
                "expected_points": round(float(b["expected_points"]), 2),
                "p10": round(float(b.get("p10", 0.0)), 2),
                "p90": round(float(b.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(b.get("exp_defcon_pts", 0.0)), 2),
                "next_fixtures": get_next_fixtures(int(b["team"])),
            }
        )
    squad_val = round(float(user_squad_df["value"].sum()) / 10.0, 1)
    sell_val = round(float(user_squad_df.get("selling_price", user_squad_df["value"]).sum()) / 10.0, 1)
    bank_m = round(bank / 10.0, 2)
    team_val = round(sell_val + bank_m, 1)

    return safe_json_serialize(
        {
            "manager_id": m_id,
            "manager_name": manager_name,
            "team_name": team_name,
            "overall_points": overall_points,
            "overall_rank": overall_rank,
            "bank_millions": bank_m,
            "free_transfers": free_transfers,
            "available_transfers": free_transfers,
            "total_squad_value": squad_val,
            "total_selling_value": sell_val,
            "total_team_value": team_val,
            "chips_used": chips_used,
            "formation": lineup_res["formation"],
            "captain": lineup_res["captain"],
            "vice_captain": lineup_res["vice_captain"],
            "starters": starters_out,
            "bench": bench_out,
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
    profile = data_store.get_profile()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    # Get user squad
    user_squad_df = None
    bank = 5.0
    if profile.manager_id:
        try:
            picks, _ = await fpl_client.get_manager_picks(profile.manager_id, curr_gw or 5)
            picks_ids = [p.element for p in picks.picks]
            user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
            if picks.entry_history:
                bank = picks.entry_history.bank
        except Exception:
            pass

    if (user_squad_df is None or len(user_squad_df) < 15) and profile.manual_squad:
        try:
            manual_ids = (
                json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
            )
            if manual_ids and len(manual_ids) == 15:
                user_squad_df = target_df[target_df["element"].isin(manual_ids)].copy()
                bank = (profile.bank or 0.0) * 10.0
                user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)
        except Exception:
            pass

    if user_squad_df is None or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        user_squad_df = squad_optimizer.solve_best_squad(target_df, budget=1000.0)["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    locked_in = req.locked_in if req else None
    locked_out = req.locked_out if req else None
    excl_teams = req.excluded_teams if req else None

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=profile.free_transfers or 1,
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
    return safe_json_serialize(res)


@router.get("/chips")
async def get_chip_strategy():
    profile = data_store.get_profile()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

    horizon_proj = projection_engine.predict_multi_gameweeks(next_gw or 6, 8, boot, fixtures)
    pool_df = horizon_proj.get(next_gw or 6, pd.DataFrame())

    hist = None
    squad_df = None
    if profile.manager_id:
        try:
            hist, _ = await fpl_client.get_manager_history(profile.manager_id)
            picks, _ = await fpl_client.get_manager_picks(profile.manager_id, curr_gw or 5)
            picks_ids = [p.element for p in picks.picks]
            squad_df = pool_df[pool_df["element"].isin(picks_ids)].copy()
        except Exception:
            pass

    if (squad_df is None or len(squad_df) < 15) and profile.manual_squad:
        try:
            manual_ids = (
                json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
            )
            if manual_ids and len(manual_ids) == 15:
                squad_df = pool_df[pool_df["element"].isin(manual_ids)].copy()
        except Exception:
            pass

    if squad_df is None or len(squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        squad_df = squad_optimizer.solve_best_squad(pool_df, budget=1000.0)["squad"]

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

    standings_data = await league_standings_manager.get_league_standings(l_id, max_pages=2)
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
        user_manager_id=profile.manager_id,
        current_gw=curr_gw or 5,
        bootstrap=boot,
        max_rivals_to_inspect=8,
    )

    user_pts = 0
    if profile.manager_id:
        try:
            entry, _ = await fpl_client.get_manager_entry(profile.manager_id)
            user_pts = entry.summary_overall_points or 0
        except Exception:
            pass

    # Monte Carlo simulation
    fixtures, _ = await fpl_client.get_fixtures()
    proj_df = projection_engine.predict_gameweek((curr_gw or 5) + 1, boot, fixtures)

    user_squad_mock = proj_df.sort_values(by="expected_points", ascending=False).head(15)

    mc_res = monte_carlo_simulator.simulate_league(
        user_points=user_pts,
        user_squad_df=user_squad_mock,
        rival_squads=rivals_res["rival_squads"],
        projections_df=proj_df,
        horizon_gws=5,
    )

    strategy = league_strategy_advisor.evaluate_strategy(
        user_rank=1, user_total_points=user_pts, rivals_analysis=rivals_res, user_squad_df=user_squad_mock
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
    profile = data_store.get_profile()
    curr_gw, _ = await fpl_client.get_current_and_next_gw()
    bank = float(profile.bank or 0.5) * 10.0
    free_transfers = int(profile.free_transfers or 1)
    user_squad_df = None

    if profile.manager_id:
        try:
            picks, _ = await fpl_client.get_manager_picks(profile.manager_id, curr_gw or 5)
            history, _ = await fpl_client.get_manager_history(profile.manager_id)
            transfers, _ = await fpl_client.get_manager_transfers(profile.manager_id)
            if picks.entry_history:
                bank = float(picks.entry_history.bank)
            if profile.bank is not None and profile.bank > 0:
                bank = float(profile.bank * 10.0)
            auto_ft = transfer_optimizer.calculate_banked_free_transfers(history)
            if profile.free_transfers and profile.free_transfers > 1:
                free_transfers = profile.free_transfers
            else:
                free_transfers = auto_ft
            picks_ids = [p.element for p in picks.picks]
            user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
            user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, transfers, boot)
        except Exception:
            pass

    if (user_squad_df is None or len(user_squad_df) < 15) and profile.manual_squad:
        try:
            manual_ids = (
                json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
            )
            if manual_ids and len(manual_ids) == 15:
                user_squad_df = target_df[target_df["element"].isin(manual_ids)].copy()
                bank = float(profile.bank or 0.0) * 10.0
                user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)
        except Exception:
            pass

    if user_squad_df is None or len(user_squad_df) < 15:
        from fpl_oracle.optimise.squad import squad_optimizer

        squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
        user_squad_df = squad_res["squad"].copy()
        user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, None, boot)

    return user_squad_df, bank, free_transfers


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
