"""
Weekly Gameweek Briefing Automation.
Generates an executive, data-backed pre-deadline briefing card and markdown export.
"""

import asyncio
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.data.store import data_store
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.server.analysis import analysis_service
from fpl_oracle.server.safe_json import safe_json_serialize


class WeeklyBriefingGenerator:
    def __init__(self):
        pass

    async def generate_briefing(self, manager_id: int | None = None) -> dict[str, Any]:
        profile = data_store.get_profile()
        state = await manager_state_service.get_current_state()
        m_id = state.manager_id
        if manager_id is not None and manager_id != m_id:
            return {
                "status": "unavailable",
                "markdown": None,
                "reason": "Requested manager does not match configured squad",
            }
        if len(state.squad) != 15:
            return {
                "status": "unavailable",
                "markdown": None,
                "reason": state.error_message or "Configured squad unavailable",
                "is_stale": state.is_stale,
            }

        boot, is_stale = await fpl_client.get_bootstrap_static()
        fixtures, fixtures_stale = await fpl_client.get_fixtures()
        is_stale = is_stale or fixtures_stale
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 1)
        effective_curr_gw = curr_gw or (target_gw - 1 if target_gw > 1 else 1)

        # Deadline time for next gameweek
        next_event = next((e for e in boot.events if e.id == target_gw), None)
        deadline_str = next_event.deadline_time if next_event else "Unknown"

        # Predictions for next 5 gameweeks
        horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures)
        target_df = horizon_proj.get(target_gw, pd.DataFrame())

        user_squad_df = state.to_squad_dataframe()
        if target_df.empty or not set(user_squad_df["element"]).issubset(set(target_df["element"])):
            return {
                "status": "unavailable",
                "markdown": None,
                "reason": "Projection snapshot missing configured players",
            }
        for column in target_df.columns:
            if column not in ("element", "selling_price", "purchase_price", "price_provenance"):
                user_squad_df[column] = user_squad_df["element"].map(target_df.set_index("element")[column])
        user_history = None
        if m_id:
            user_history, history_stale = await fpl_client.get_manager_history(m_id)
            is_stale = is_stale or history_stale
        is_stale = is_stale or state.is_stale
        available_chips = state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2
        chips_used = [c["name"] for c in state.chips_used if (c["event"] <= 19) == (target_gw <= 19)]
        transfers_res = await analysis_service.joint_plan(
            current_squad_df=user_squad_df,
            player_pool_df=target_df,
            bank=state.bank_tenths,
            free_transfers=state.free_transfers,
            horizon_projections=horizon_proj,
            current_gw=effective_curr_gw,
            target_gw=target_gw,
            available_chips=available_chips,
            chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
            chips_already_used=chips_used,
            risk_preference="points",
        )
        lineup_res = transfers_res["recommended_plan"].get("lineup")
        if not lineup_res:
            lineup_res = await asyncio.to_thread(
                lineup_optimizer.select_lineup_and_captain,
                user_squad_df,
                risk_preference="points",
            )
        chip_res = await analysis_service.chip_strategy(
            current_gw=effective_curr_gw,
            current_squad_df=user_squad_df,
            horizon_projections=horizon_proj,
            fixtures=fixtures,
            bootstrap=boot,
            manager_history=user_history,
        )
        chip_res = analysis_service.bind_chip_schedule(chip_res, transfers_res)

        from fpl_oracle.chips.measured_windows import measured_windows

        measured = await asyncio.to_thread(measured_windows, user_squad_df, horizon_proj)
        measured = [row for row in measured if row["chip"] in available_chips]
        chip_res["measured_windows"] = measured

        # 4. Price Changes Tonight
        price_preds = price_change_predictor.analyze_price_changes(boot)
        imminent_rises = [p for p in price_preds if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:3]
        imminent_falls = [p for p in price_preds if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:3]

        # 5. News & Injuries
        news_signals = await analysis_service.news_signals(boot, target_gw)
        squad_elem_ids = set(user_squad_df["element"].tolist())
        squad_news = [s for s in news_signals if s["element_id"] in squad_elem_ids]

        # 6. Mini-league Intelligence
        league_id = profile.target_league_id
        league_summary = None
        if league_id:
            try:
                standings = await league_standings_manager.get_league_standings(league_id)
                rivals_res = await rival_analyzer.analyze_rivals(
                    standings=standings["standings"],
                    user_manager_id=m_id,
                    current_gw=effective_curr_gw,
                    bootstrap=boot,
                )
                user_total = 0
                if m_id:
                    user_entry, _ = await fpl_client.get_manager_entry(m_id)
                    user_total = user_entry.summary_overall_points or 0

                user_rank = None
                for row in standings.get("standings", []):
                    if row.get("entry") == m_id:
                        user_rank = row.get("rank")
                        break

                if user_rank is not None:
                    strategy = league_strategy_advisor.evaluate_strategy(
                        user_rank=user_rank,
                        user_total_points=user_total,
                        rivals_analysis=rivals_res,
                        user_squad_df=user_squad_df,
                    )
                    league_summary = {
                        "league_name": standings["league_name"],
                        "strategy_mode": strategy["mode_title"],
                        "tactics": strategy["tactical_recommendations"],
                    }
                else:
                    league_summary = {
                        "league_name": standings.get("league_name", f"League #{league_id}"),
                        "strategy_mode": "Unavailable",
                        "tactics": ["Manager not found in mini-league standings."],
                    }
            except Exception:
                pass

        # Build Markdown Export
        cap = lineup_res["captain"]
        vc = lineup_res["vice_captain"]
        rec_plan = transfers_res["recommended_plan"]

        markdown = f"""# Gameweek {target_gw}

{deadline_str}

## Your plan

| Captain | Vice-captain | Formation | Predicted points |
|---|---|---|---|
| {cap["web_name"]} | {vc["web_name"]} | {lineup_res["formation"]} | {lineup_res["total_gameweek_expected_points"]} |

**Transfers:** {rec_plan.get("transfers_count", 0)} · **Hit cost:** {rec_plan.get("hit_cost", 0)} points

"""
        for out, incoming in zip(rec_plan.get("transfers_out", []), rec_plan.get("transfers_in", []), strict=False):
            markdown += f"- {out['web_name']} → {incoming['web_name']}\n"
        if not rec_plan.get("transfers_in"):
            markdown += "Save your free transfer.\n"
        hit_policy = transfers_res.get("hit_policy", {})
        markdown += "\n## Hits\n"
        if hit_policy.get("warning"):
            markdown += hit_policy["warning"] + "\n"
        if hit_policy.get("cooldown_until", 0) > target_gw:
            markdown += f"No new hit before GW{hit_policy['cooldown_until']}.\n"
        taken_hits = [r for r in hit_policy.get("records", []) if r.get("kind") == "taken"]
        if taken_hits:
            markdown += "\n| Week | Cost | Actual player gain | Result |\n|---|---|---|---|\n"
            for row in sorted(taken_hits, key=lambda r: r["gameweek"], reverse=True)[:6]:
                markdown += f"| GW{row['gameweek']} | {row['hit_cost']} | {row.get('realised_gain', 'Not settled')} | {row['status']} |\n"
        if any(r.get("hit_requires_prior_success") for r in rec_plan.get("trajectory", [])):
            markdown += "\nFuture hits need a new check.\n"
        markdown += "\n## Starting XI\n\n| Player | Position | Predicted points |\n|---|---|---|\n"
        for _, player in lineup_res["starters"].iterrows():
            role = (
                " (C)" if player["element"] == cap["element"] else " (VC)" if player["element"] == vc["element"] else ""
            )
            markdown += f"| {player['web_name']}{role} | {player['position']} | {player['expected_points']:.1f} |\n"
        markdown += "\n### Bench\n\n"
        for _, player in lineup_res["bench"].iterrows():
            markdown += (
                f"- {player['web_name']} ({player['position']}) · {player['expected_points']:.1f} predicted pts\n"
            )
        markdown += "\n## Chips\n\n"
        if transfers_res.get("chip_measurement_status") == "measured_through_expiry" and chip_res["chip_plan_table"]:
            markdown += "| Chip | Plan week | Alternative week | Est. extra points |\n|---|---|---|---|\n"
            for row in chip_res["chip_plan_table"]:
                week = f"GW{row['recommended_gw']}" if row["recommended_gw"] is not None else "No clear benefit"
                gain = f"{row['expected_gain']:+.1f}" if row["status"] == "measured" else "Search inconclusive"
                equal_week = (
                    f"GW{row['equal_weight_gameweek']}" if row.get("equal_weight_gameweek") is not None else "Not used"
                )
                markdown += f"| {row['chip']} | {week} | {equal_week} | {gain} |\n"

        else:
            markdown += (
                "No chips remain in this set.\n"
                if not available_chips
                else "Chip timing unavailable. Update data and try again.\n"
            )
        if measured:
            markdown += "\n### Weekly chip comparisons\n\n| Week | Triple Captain extra pts | Bench Boost extra pts |\n|---|---|---|\n"
            for gw in sorted(horizon_proj):
                gains = {row["chip"]: row["expected_gain"] for row in measured if row["gameweek"] == gw}
                markdown += f"| GW{gw} | {gains.get('3xc', 'Unavailable')} | {gains.get('bboost', 'Unavailable')} |\n"
        markdown += f"\n## Price watch\n\n- Possible rises: {', '.join(p['web_name'] for p in imminent_rises) or 'No signal'}\n- Possible falls: {', '.join(p['web_name'] for p in imminent_falls) or 'No signal'}\n"
        strategy = transfers_res.get("league_objective", {}).get("strategy", {})
        markdown += f"\n## Your league\n\n{strategy.get('mode_title', 'Focus on points')}\n"
        if strategy.get("status") == "automatic":
            markdown += f"\nRank #{strategy['rank']} · {strategy['gap_to_leader']} points behind first.\n"

        briefing_dict = {
            "target_gameweek": target_gw,
            "deadline_time": deadline_str,
            "is_stale": is_stale,
            "stale": is_stale,
            "data_as_of": state.source_timestamp,
            "snapshot_id": target_df.attrs.get("snapshot_id"),
            "bank_tenths": state.bank_tenths,
            "free_transfers": state.free_transfers,
            "captain": cap,
            "vice_captain": vc,
            "lineup": lineup_res,
            "transfers": transfers_res,
            "chips": chip_res,
            "price_changes": {"rises": imminent_rises, "falls": imminent_falls},
            "squad_news": squad_news,
            "league_summary": league_summary,
            "markdown": markdown,
        }

        return safe_json_serialize(briefing_dict)


weekly_briefing_generator = WeeklyBriefingGenerator()
