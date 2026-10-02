"""
FastAPI REST API routes for FPL Oracle.
"""

import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.transfers import transfer_optimizer
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.llm.agent import expert_agent
from fpl_oracle.data.store import data_store
from fpl_oracle.data.fuzzy_match import fuzzy_matcher
from fpl_oracle.server.safe_json import SafeJSONResponse, safe_json_serialize

router = APIRouter(prefix="/api", default_response_class=SafeJSONResponse)

class ProfileUpdateRequest(BaseModel):
    manager_id: Optional[int] = None
    target_league_id: Optional[int] = None
    risk_preference: Optional[str] = None
    llm_provider: Optional[str] = None
    bank: Optional[float] = None
    free_transfers: Optional[int] = None

class MatchSquadRequest(BaseModel):
    raw_text: str

class ManualSquadRequest(BaseModel):
    player_ids: List[int]
    bank: Optional[float] = 0.0
    free_transfers: Optional[int] = 1

class OptimizeRequest(BaseModel):
    locked_in: Optional[List[int]] = None
    locked_out: Optional[List[int]] = None
    excluded_teams: Optional[List[int]] = None
    custom_budget: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

@router.get("/health")
async def get_health():
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    return {
        "status": "healthy",
        "season": "2026/27",
        "current_gameweek": curr_gw,
        "next_gameweek": next_gw,
        "is_stale": fpl_client.is_stale_mode,
        "last_sync": fpl_client.last_sync_time.isoformat() if fpl_client.last_sync_time else None
    }

@router.get("/profile")
def get_profile():
    p = data_store.get_profile()
    return {
        "manager_id": p.manager_id,
        "target_league_id": p.target_league_id,
        "risk_preference": p.risk_preference,
        "llm_provider": p.llm_provider,
        "bank": p.bank,
        "free_transfers": p.free_transfers
    }

@router.post("/profile")
def update_profile(req: ProfileUpdateRequest):
    data_store.update_profile(
        manager_id=req.manager_id,
        target_league_id=req.target_league_id,
        risk_preference=req.risk_preference,
        llm_provider=req.llm_provider,
        bank=req.bank,
        free_transfers=req.free_transfers
    )
    return {"status": "success", "profile": get_profile()}

@router.get("/squad")
async def get_squad(manager_id: Optional[int] = None):
    profile = data_store.get_profile()
    m_id = manager_id or profile.manager_id

    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures=await fpl_client.get_fixtures()[0] if False else (await fpl_client.get_fixtures())[0])
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df = None
    bank = 5.0
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
                bank = picks.entry_history.bank

            picks_ids = [p.element for p in picks.picks]
            user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
            user_squad_df = transfer_optimizer.compute_squad_selling_prices(user_squad_df, transfers, boot)
        except Exception:
            pass
    if (user_squad_df is None or len(user_squad_df) < 15) and profile.manual_squad:
        try:
            manual_ids = json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
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
                    f_list.append({
                        "event": f.event,
                        "opp_short": opp.short_name if opp else "OPP",
                        "is_home": True,
                        "difficulty": f.team_h_difficulty or 3
                    })
                elif f.team_a == team_id:
                    opp = team_map.get(f.team_h)
                    f_list.append({
                        "event": f.event,
                        "opp_short": opp.short_name if opp else "OPP",
                        "is_home": False,
                        "difficulty": f.team_a_difficulty or 3
                    })
        return f_list[:5]

    starters_out = []
    for _, s in lineup_res["starters"].iterrows():
        starters_out.append({
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
            "next_fixtures": get_next_fixtures(int(s["team"]))
        })

    bench_out = []
    for _, b in lineup_res["bench"].iterrows():
        bench_out.append({
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
            "next_fixtures": get_next_fixtures(int(b["team"]))
        })

    return safe_json_serialize({
        "manager_id": m_id,
        "manager_name": manager_name,
        "team_name": team_name,
        "overall_points": overall_points,
        "overall_rank": overall_rank,
        "bank_millions": round(bank / 10.0, 2),
        "chips_used": chips_used,
        "formation": lineup_res["formation"],
        "captain": lineup_res["captain"],
        "vice_captain": lineup_res["vice_captain"],
        "starters": starters_out,
        "bench": bench_out,
        "total_expected_points": lineup_res["total_gameweek_expected_points"]
    })

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
    team_counts = {}
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

    data_store.update_profile(
        manual_squad=req.player_ids,
        bank=req.bank,
        free_transfers=req.free_transfers
    )
    return safe_json_serialize({
        "status": "success",
        "message": "Custom squad saved successfully.",
        "player_ids": req.player_ids
    })

@router.get("/projections")
async def get_projections(
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    horizon: int = Query(5, ge=1, le=8)
):
    boot, _ = await fpl_client.get_bootstrap_static()
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
        results.append({
            "element": int(r["element"]),
            "web_name": r["web_name"],
            "team": int(r["team"]),
            "position": r["position"],
            "cost": round(r["value"] / 10.0, 1),
            "expected_points": round(float(r["expected_points"]), 2),
            "p10": round(float(r.get("p10", 0.0)), 2),
            "p90": round(float(r.get("p90", 0.0)), 2),
            "exp_defcon_pts": round(float(r.get("exp_defcon_pts", 0.0)), 2)
        })

    return safe_json_serialize({"gameweek": target_gw, "horizon": horizon, "players": results})

@router.post("/optimize")
async def run_optimizer(req: Optional[OptimizeRequest] = None):
    profile = data_store.get_profile()
    boot, _ = await fpl_client.get_bootstrap_static()
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
            manual_ids = json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
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
        excluded_team_ids=excl_teams
    )
    return safe_json_serialize(res)

@router.get("/chips")
async def get_chip_strategy():
    profile = data_store.get_profile()
    boot, _ = await fpl_client.get_bootstrap_static()
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
            manual_ids = json.loads(profile.manual_squad) if isinstance(profile.manual_squad, str) else profile.manual_squad
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
        manager_history=hist
    )
    return safe_json_serialize(res)

@router.get("/league")
async def get_league_intel(league_id: Optional[int] = None):
    profile = data_store.get_profile()
    l_id = league_id or profile.target_league_id
    if not l_id:
        return safe_json_serialize({
            "status": "unconfigured",
            "message": "No target mini-league ID configured. Enter a mini-league ID to see rival analysis.",
            "league_name": "None",
            "total_teams": 0,
            "standings": [],
            "template_players": [],
            "differential_players": [],
            "simulation": {"user_win_probability_pct": 0.0, "expected_final_rank": 1.0},
            "strategy": {"mode_title": "Setup Required", "rationale": "Set your mini-league ID in settings to activate rival analysis.", "tactical_recommendations": []}
        })

    standings_data = await league_standings_manager.get_league_standings(l_id, max_pages=2)
    boot, _ = await fpl_client.get_bootstrap_static()
    curr_gw, _ = await fpl_client.get_current_and_next_gw()

    if not standings_data.get("standings"):
        return safe_json_serialize({
            "status": "empty",
            "message": f"No standings found for mini-league {l_id}.",
            "league_name": standings_data.get("league_name", f"League #{l_id}"),
            "total_teams": 0,
            "standings": [],
            "template_players": [],
            "differential_players": [],
            "simulation": {"user_win_probability_pct": 0.0, "expected_final_rank": 1.0},
            "strategy": {"mode_title": "No Standings", "rationale": "No standings data returned from FPL API for this league.", "tactical_recommendations": []}
        })

    rivals_res = await rival_analyzer.analyze_rivals(
        standings=standings_data["standings"],
        user_manager_id=profile.manager_id,
        current_gw=curr_gw or 5,
        bootstrap=boot,
        max_rivals_to_inspect=8
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
        horizon_gws=5
    )

    strategy = league_strategy_advisor.evaluate_strategy(
        user_rank=1,
        user_total_points=user_pts,
        rivals_analysis=rivals_res,
        user_squad_df=user_squad_mock
    )

    return safe_json_serialize({
        "status": "success",
        "league_name": standings_data["league_name"],
        "total_teams": standings_data["total_teams"],
        "standings": standings_data["standings"],
        "template_players": rivals_res["template_players"],
        "differential_players": rivals_res["differential_players"],
        "simulation": mc_res,
        "strategy": strategy
    })

@router.get("/briefing")
async def get_briefing():
    return await weekly_briefing_generator.generate_briefing()

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    ans = await expert_agent.answer(user_message=req.message, session_id=req.session_id or "default")
    return safe_json_serialize({"response": ans})

@router.get("/price-changes")
async def get_price_changes():
    boot, _ = await fpl_client.get_bootstrap_static()
    preds = price_change_predictor.analyze_price_changes(boot)
    return safe_json_serialize({
        "rises": [p for p in preds if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:10],
        "falls": [p for p in preds if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:10]
    })
