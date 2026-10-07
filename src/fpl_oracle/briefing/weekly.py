"""
Weekly Gameweek Briefing Automation.
Generates an executive, data-backed pre-deadline briefing card and markdown export.
"""

import asyncio
from datetime import UTC, datetime
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
            risk_preference=profile.risk_preference or "balanced",
        )
        lineup_res = transfers_res["recommended_plan"].get("lineup")
        if not lineup_res:
            lineup_res = await asyncio.to_thread(
                lineup_optimizer.select_lineup_and_captain,
                user_squad_df,
                risk_preference=profile.risk_preference or "balanced",
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

        markdown = f"""# FPL Oracle - Gameweek {target_gw} Executive Briefing

**Generated:** {datetime.now(UTC).strftime("%A, %d %B %Y %H:%M UTC")}
**Deadline:** {deadline_str}
**Status:** {"[STALE DATA] (API Unavailable)" if is_stale else "[LIVE & SYNCHRONIZED]"}

---

## 1. Executive Summary & Core Decisions
- **Captain:** **{cap["web_name"]}** ({cap["expected_points"]} projected points, {cap["multiplier"]}x multiplier).
- **Vice-Captain:** **{vc["web_name"]}** ({vc["expected_points"]} projected points).
- **Transfers Decision:** {rec_plan.get("recommendation_summary", rec_plan.get("plan_type", "Unavailable"))}
- **Hit Verdict:** {transfers_res["hit_verdict"]}
- **Starting Formation:** {lineup_res["formation"]} (Starting XI before captain bonus: **{lineup_res["starters_expected_points"]}** xP; total including captain/chip: **{lineup_res["total_gameweek_expected_points"]}** xP).

---

## 2. Recommended Starting XI & Bench

### Starting XI:
"""
        for _, s in lineup_res["starters"].iterrows():
            is_c = " (C)" if s["element"] == cap["element"] else (" (VC)" if s["element"] == vc["element"] else "")
            markdown += f"- **{s['web_name']}** ({s['position']}) - {s['expected_points']} xP [Floor: {s.get('p10', 0.0)}, Ceiling: {s.get('p90', 0.0)}, DefCon: +{s.get('exp_defcon_pts', 0.0)}]{is_c}\n"

        markdown += "\n### Bench Substitutes (in priority order):\n"
        for idx, (_, b) in enumerate(lineup_res["bench"].iterrows(), 1):
            markdown += f"{idx}. **{b['web_name']}** ({b['position']}) - {b['expected_points']} xP\n"

        markdown += f"""
---

## 3. Conditional Chip Roadmap (reassess every deadline)
{chip_res.get("set_1_deadline_warning") or "Set 1 chips are active through GW19."}

| Chip | Recommended GW | Expected Gain | Tactical Notes |
|---|---|---|---|
"""
        for c in chip_res.get("chip_plan_table", [])[:4]:
            markdown += (
                f"| **{c['chip']}** | GW {c['recommended_gw']} | Not separately estimated | {c['reasoning']} |\n"
            )

        markdown += "\nFuture resource value is not forecast. Legal preservation alternatives:\n"
        for row in transfers_res.get("resource_frontier", []):
            markdown += f"- Preserve {row['chip']} through GW{row['retained_through_gameweek']}: {row['discounted_horizon_cost_to_preserve']} discounted xP window cost; later value above {row['tail_value_break_even_at_horizon_end']} xP could reverse the choice.\n"

        markdown += f"""
---

## 4. Heuristic Transfer Momentum (price timing unverified)
- **Positive momentum:** {", ".join([r["web_name"] for r in imminent_rises]) if imminent_rises else "No signal"}
- **Negative momentum:** {", ".join([f["web_name"] for f in imminent_falls]) if imminent_falls else "No signal"}

---

## 5. Mini-League & Tactical Situation
- **Strategy Mode:** {league_summary["strategy_mode"] if league_summary else "Global Points Optimization"}
- **Key Tactical Directives:**
"""
        if league_summary and league_summary.get("tactics"):
            for t in league_summary["tactics"]:
                markdown += f"  - {t}\n"
        else:
            markdown += (
                "  - Focus on maximizing overall expected points and banking free transfers for winter swings.\n"
            )

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
