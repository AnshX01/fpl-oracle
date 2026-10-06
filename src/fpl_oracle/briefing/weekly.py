"""
Weekly Gameweek Briefing Automation.
Generates an executive, data-backed pre-deadline briefing card and markdown export.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.data.store import data_store
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.optimise.transfers import transfer_optimizer
from fpl_oracle.server.safe_json import safe_json_serialize


class WeeklyBriefingGenerator:
    def __init__(self):
        pass

    async def generate_briefing(self, manager_id: int | None = None) -> dict[str, Any]:
        profile = data_store.get_profile()
        m_id = manager_id or profile.manager_id

        boot, is_stale = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6

        # Deadline time for next gameweek
        next_event = next((e for e in boot.events if e.id == target_gw), None)
        deadline_str = next_event.deadline_time if next_event else "2026-10-10T10:00:00Z"

        # Predictions for next 5 gameweeks
        horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
        target_df = horizon_proj.get(target_gw, pd.DataFrame())

        # Load user squad
        bank = 5.0
        ft = profile.free_transfers or 1
        user_squad_df = None
        user_history = None

        if m_id:
            try:
                picks, _ = await fpl_client.get_manager_picks(m_id, curr_gw or 5)
                picks_ids = [p.element for p in picks.picks]
                user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
                if picks.entry_history:
                    bank = picks.entry_history.bank
                user_history, _ = await fpl_client.get_manager_history(m_id)
            except Exception:
                pass

        if user_squad_df is None or len(user_squad_df) < 15:
            from fpl_oracle.optimise.squad import squad_optimizer

            squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
            user_squad_df = squad_res["squad"].copy()

        # 1. Lineup & Captaincy
        lineup_res = lineup_optimizer.select_lineup_and_captain(user_squad_df)

        # 2. Transfer Optimization
        transfers_res = transfer_optimizer.evaluate_transfer_options(
            current_squad_df=user_squad_df,
            player_pool_df=target_df,
            bank=bank,
            free_transfers=ft,
            horizon_projections=horizon_proj,
            current_gw=curr_gw or 5,
            target_gw=target_gw,
        )

        # 3. Chip Strategy
        chip_res = chip_planner.generate_chip_strategy(
            current_gw=curr_gw or 5,
            current_squad_df=user_squad_df,
            horizon_projections=horizon_proj,
            fixtures=fixtures,
            bootstrap=boot,
            manager_history=user_history,
        )

        # 4. Price Changes Tonight
        price_preds = price_change_predictor.analyze_price_changes(boot)
        imminent_rises = [p for p in price_preds if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:3]
        imminent_falls = [p for p in price_preds if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:3]

        # 5. News & Injuries
        news_signals = await news_analyzer.get_player_news_signals(boot)
        squad_elem_ids = set(user_squad_df["element"].tolist())
        squad_news = [s for s in news_signals if s["element_id"] in squad_elem_ids]

        # 6. Mini-league Intelligence
        league_id = profile.target_league_id
        league_summary = None
        if league_id:
            try:
                standings = await league_standings_manager.get_league_standings(league_id, max_pages=1)
                rivals_res = await rival_analyzer.analyze_rivals(
                    standings=standings["standings"],
                    user_manager_id=m_id,
                    current_gw=curr_gw or 5,
                    bootstrap=boot,
                    max_rivals_to_inspect=5,
                )
                user_total = 0
                if m_id:
                    user_entry, _ = await fpl_client.get_manager_entry(m_id)
                    user_total = user_entry.summary_overall_points or 0
                strategy = league_strategy_advisor.evaluate_strategy(
                    user_rank=1, user_total_points=user_total, rivals_analysis=rivals_res, user_squad_df=user_squad_df
                )
                league_summary = {
                    "league_name": standings["league_name"],
                    "strategy_mode": strategy["mode_title"],
                    "tactics": strategy["tactical_recommendations"],
                }
            except Exception:
                pass

        # Build Markdown Export
        cap = lineup_res["captain"]
        vc = lineup_res["vice_captain"]
        rec_plan = transfers_res["recommended_plan"]

        markdown = f"""# ⚽ FPL Oracle — Gameweek {target_gw} Executive Briefing

**Generated:** {datetime.now(UTC).strftime("%A, %d %B %Y %H:%M UTC")}
**Deadline:** {deadline_str}
**Status:** {"⚠️ STALE DATA (API Unavailable)" if is_stale else "🟢 LIVE & SYNCHRONIZED"}

---

## 1. Executive Summary & Core Decisions
- **Captain:** **{cap["web_name"]}** ({cap["expected_points"]} projected points, {cap["multiplier"]}x multiplier).
- **Vice-Captain:** **{vc["web_name"]}** ({vc["expected_points"]} projected points).
- **Transfers Decision:** {rec_plan["recommendation_summary"]}
- **Hit Verdict:** {transfers_res["hit_verdict"]}
- **Starting Formation:** {lineup_res["formation"]} (Projected starting points: **{lineup_res["starters_expected_points"]}** xP).

---

## 2. Recommended Starting XI & Bench

### Starting XI:
"""
        for _, s in lineup_res["starters"].iterrows():
            is_c = " (C)" if s["element"] == cap["element"] else (" (VC)" if s["element"] == vc["element"] else "")
            markdown += f"- **{s['web_name']}** ({s['position']}) — {s['expected_points']} xP [Floor: {s.get('p10', 0.0)}, Ceiling: {s.get('p90', 0.0)}, DefCon: +{s.get('exp_defcon_pts', 0.0)}]{is_c}\n"

        markdown += "\n### Bench Substitutes (in priority order):\n"
        for idx, (_, b) in enumerate(lineup_res["bench"].iterrows(), 1):
            markdown += f"{idx}. **{b['web_name']}** ({b['position']}) — {b['expected_points']} xP\n"

        markdown += f"""
---

## 3. 2026/27 Chip Strategy & Set 1 Deadlines
{chip_res.get("set_1_deadline_warning") or "Set 1 chips are active through GW19."}

| Chip | Recommended GW | Expected Gain | Tactical Notes |
|---|---|---|---|
"""
        for c in chip_res.get("chip_plan_table", [])[:4]:
            markdown += (
                f"| **{c['chip']}** | GW {c['recommended_gw']} | +{c['expected_gain']} pts | {c['reasoning']} |\n"
            )

        markdown += f"""
---

## 4. Market & Price Change Alert Tonight
- **Imminent Rises (+£0.1m):** {", ".join([r["web_name"] for r in imminent_rises]) if imminent_rises else "None at immediate trigger"}
- **Imminent Falls (-£0.1m):** {", ".join([f["web_name"] for f in imminent_falls]) if imminent_falls else "None at immediate trigger"}

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
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
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
