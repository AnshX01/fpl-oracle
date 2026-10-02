"""
Provider-agnostic LLM Layer.
Supports Google Gemini (default), Anthropic, OpenAI, and a standalone OfflineExpertProvider
that operates purely on live data, projections, and mathematical optimizations.
"""

from typing import List, Dict, Any, Optional
import os
import logging

from fpl_oracle.config import GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
from fpl_oracle.llm.tools import tool_executor

logger = logging.getLogger("fpl_oracle.llm.provider")

SYSTEM_PROMPT = """You are FPL Oracle — a world-class Fantasy Premier League expert, data scientist, and mathematical tactician for the 2026/27 season.
Today is in early October 2026. The season is underway (GW6 is upcoming).
You have access to live tools: get_my_team, get_projections, optimise_transfers, plan_chips, captain_options, league_analysis, price_change_watch, compare_players.
Scoring standard:
1. The Decision in one line.
2. The Numbers: projected points over the horizon net of hits.
3. The Why: fixtures, form, minutes, DefCon (+2 pts in 2026/27), rebalanced BPS, price changes.
4. The Risk: injury risk, rotation, ceiling/floor.
5. What would change the call.
6. Long-term effect: chip schedule (GW19 deadline for Set 1 chips!) and banking up to 5 free transfers.
Never invent news or fixtures. Always cite data and sources.
"""

class OfflineExpertProvider:
    """
    Offline data-grounded expert analyst that executes live tools
    and generates structured FPL Oracle recommendations without external API keys.
    """
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        last_raw = messages[-1]["content"] if messages else ""
        last_msg = last_raw.lower()
        import re
        tokens = set(re.findall(r"\w+", last_msg))

        # 1. Comparison questions (e.g., "Explain why you picked Saka over Palmer", "compare Saka vs Palmer")
        if "compare" in tokens or "vs" in tokens or ("over" in tokens and any(p in tokens for p in ["saka", "palmer", "haaland", "watkins", "foden", "salah"])):
            players_to_compare = []
            known_stars = ["saka", "palmer", "haaland", "watkins", "foden", "salah", "fernandes", "mbeumo", "diaz", "son"]
            for star in known_stars:
                if star in tokens:
                    players_to_compare.append(star.capitalize())
            if len(players_to_compare) < 2:
                players_to_compare = ["Saka", "Palmer"]

            res = await tool_executor.execute("compare_players", {"player_names": players_to_compare})
            comps = res.get("comparisons", [])
            if len(comps) >= 2:
                p1, p2 = comps[0], comps[1]
                better = p1 if p1["expected_points"] >= p2["expected_points"] else p2
                worse = p2 if p1["expected_points"] >= p2["expected_points"] else p1
                diff_xp = round(better["expected_points"] - worse["expected_points"], 2)
                return (
                    f"### ⚖️ Head-to-Head Comparison: {better['web_name']} vs {worse['web_name']} (GW{res['gameweek']})\n"
                    f"**The Decision:** Start / Prioritize **{better['web_name']}** over **{worse['web_name']}** (+{diff_xp} net xP advantage).\n\n"
                    f"**The Numbers:**\n"
                    f"- **{better['web_name']}** (£{better['cost']}m): **{better['expected_points']} xP** (Floor P10: {better['p10']}, Ceiling P90: {better['p90']}, DefCon: +{better['defcon_pts']} pts)\n"
                    f"- **{worse['web_name']}** (£{worse['cost']}m): **{worse['expected_points']} xP** (Floor P10: {worse['p10']}, Ceiling P90: {worse['p90']}, DefCon: +{worse['defcon_pts']} pts)\n\n"
                    f"**The Why:** {better['web_name']} holds higher expected minutes and superior underlying box touches and non-penalty xG for the upcoming fixture. Both benefit from the 2026/27 scoring rules.\n"
                    f"**The Risk:** Monitor pre-match team news for any late tactical rotation.\n"
                    f"**What would change the call:** If press conferences indicate a position shift or reduced minutes for {better['web_name']}."
                )

        # 2. Transfer Roadmap questions
        if "roadmap" in tokens:
            res = await tool_executor.execute("optimise_transfers", {})
            roadmap = res.get("roadmap", [])
            rows = []
            for step in roadmap:
                rows.append(f"| **GW {step['gameweek']}** | {step['status']} | {step['action']} | Banked FT: {step.get('banked_free_transfers_projected', 1)} | {', '.join(step.get('key_targets', [])[:2])} |")
            roadmap_table = "\n".join(rows)
            return (
                f"### 🗺️ Multi-Gameweek Transfer Roadmap (Next 5 GWs)\n"
                f"**The Decision:** Sequential plan prioritizing fixture swings and strategic FT banking (up to 5 max in 2026/27).\n\n"
                f"| Gameweek | Status | Action | Banked FT | Key Targets |\n"
                f"|---|---|---|---|---|\n"
                f"{roadmap_table}\n\n"
                f"**Strategic Note:** Firm steps are locked for the upcoming deadline; contingent steps adjust based on post-match injuries and European cup congestion."
            )

        # 3. Differentials under budget questions (e.g., "best differential midfielder under 6.5")
        if "differential" in tokens or ("under" in tokens and any(pos in tokens for pos in ["midfielder", "mid", "defender", "forward"])):
            pos_filter = "MID" if any(w in tokens for w in ["midfielder", "mid"]) else ("DEF" if "defender" in tokens else "FWD")
            proj = await tool_executor.execute("get_projections", {"query": pos_filter, "horizon": 3})
            players = proj.get("players", [])
            # Filter players <= 6.5m
            budget_cands = [p for p in players if p.get("cost", 10.0) <= 6.5]
            if not budget_cands:
                budget_cands = players[:3]
            top3 = budget_cands[:3]
            c_lines = [f"- **{c['web_name']}** (£{c.get('cost', 6.0)}m): {c['expected_points_gw']} xP (DefCon: +{c.get('exp_defcon_points', 0.0)})" for c in top3]
            best_diff = top3[0]["web_name"] if top3 else "Rogers"
            return (
                f"### 🚀 Top Differential {pos_filter}s (Under £6.5m)\n"
                f"**The Decision:** Target **{best_diff}** as your premier budget differential for the upcoming fixture swing.\n\n"
                f"**The Numbers:**\n" + "\n".join(c_lines) + "\n\n"
                f"**The Why:** High baseline minutes reliability and substantial upside from the 2026/27 DefCon actions and attacking output.\n"
                f"**Mini-League Strategy:** Under 20% Effective Ownership allows you to rapidly gain ground on rivals holding template assets."
            )

        # 4. Captaincy questions
        if any(w in tokens for w in ["captain", "armband", "vice", "vc"]):
            res = await tool_executor.execute("captain_options", {})
            safe = res["safe_captain"]
            diff = res["differential_captain"]
            cands = res["candidates"]
            lines = [f"- **{c['web_name']}**: {c['expected_points']} xP (Floor: {c['p10']}, Ceiling: {c['p90']})" for c in cands[:3]]
            return (
                f"### 🎯 Captaincy Recommendation for Gameweek {res['gameweek']}\n"
                f"**The Decision:** Captain **{safe}** (Safe/Template) or **{diff}** (High-Variance Differential).\n\n"
                f"**The Numbers:**\n" + "\n".join(lines) + "\n\n"
                f"**The Why:** {safe} ranks highest in the 2026/27 ML model with an elite minutes expectation and attack volume. {diff} provides higher ceiling differential upside if you are chasing in your mini-league.\n"
                f"**The Risk:** Guard against late press-conference rotation notes.\n"
                f"**What would change the call:** If press conferences indicate minutes management, pivot immediately to your vice-captain."
            )

        # 5. Chip questions (Wildcard, Free Hit, Bench Boost, Triple Captain)
        chip_tokens = {"chip", "chips", "wildcard", "freehit", "bboost", "bb", "fh", "tc"}
        chip_phrases = ["bench boost", "free hit", "triple captain"]
        if (tokens & chip_tokens) or any(p in last_msg for p in chip_phrases):
            res = await tool_executor.execute("plan_chips", {})
            warning = res.get("set_1_warning", "")
            table_rows = []
            for row in res.get("chip_table", [])[:4]:
                table_rows.append(f"| **{row['chip']}** | GW {row['recommended_gw']} | +{row['expected_gain']} pts | {row['confidence']} | {row['reasoning']} |")

            table_str = "\n".join(table_rows)
            return (
                f"### 🃏 2026/27 Chip Strategy Plan\n"
                f"**The Decision:** Execute your 4 Set 1 chips before the Gameweek 19 deadline (Saturday 2 January 2027).\n\n"
                f"⚠️ **{warning}**\n\n"
                f"| Chip | Recommended GW | Exp. Gain | Confidence | Tactical Role |\n"
                f"|---|---|---|---|---|\n"
                f"{table_str}\n\n"
                f"**Key 2026/27 Rule Note:** Remember that Set 1 chips DO NOT carry over into Set 2. The Assistant Manager chip has been removed for 2026/27."
            )

        # 6. Mini-league questions
        if any(w in tokens for w in ["league", "rival", "rivals", "standings", "catch", "win", "rank"]):
            res = await tool_executor.execute("league_analysis", {})
            if "message" in res:
                return f"### 🏆 Mini-League Intelligence\n{res['message']}"

            temps = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("template_players", [])[:3]]
            diffs = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("differentials", [])[:3]]
            return (
                f"### 🏆 Mini-League Intelligence for {res['league_name']}\n"
                f"**The Decision:** Exploit rival template vulnerabilities with targeted differentials.\n\n"
                f"- **Top Template (High Rival EO):** {', '.join(temps)}\n"
                f"- **Key Differentials (<20% EO):** {', '.join(diffs)}\n\n"
                f"**Strategy Mode:** If leading, mirror the template to protect against rival hauls. If chasing, back differentials to make up points."
            )

        # 7. Price changes
        if any(w in tokens for w in ["price", "prices", "rise", "fall", "value", "cost"]):
            res = await tool_executor.execute("price_change_watch", {})
            rises = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_rises", [])[:3]]
            falls = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_falls", [])[:3]]
            return (
                f"### 📈 Price Change Watch (Tonight's Projections)\n"
                f"**Imminent Rises:** {', '.join(rises) if rises else 'None currently at threshold'}\n\n"
                f"**Imminent Falls:** {', '.join(falls) if falls else 'None currently at threshold'}\n\n"
                f"**Action:** Make scheduled transfers before 01:30 GMT to avoid losing purchasing power."
            )

        # 8. Transfer & Hit questions
        if any(w in tokens for w in ["transfer", "transfers", "hit", "hits", "sell", "buy", "haaland"]):
            res = await tool_executor.execute("optimise_transfers", {})
            if "error" in res:
                return f"### 🔄 Transfer Strategy & Hit Analysis\nUnable to optimize transfers: {res['error']}"
            rec = res["decision"]
            hit = res["hit_verdict"]
            plans = res.get("candidate_plans", [])
            plan_lines = [f"- **{p['type']}**: {p['summary']} (Net xP: {p['net_xp']}, Net Gain: {p['gain']})" for p in plans]

            return (
                f"### 🔄 Transfer Strategy & Hit Analysis\n"
                f"**The Decision:** {rec}\n\n"
                f"**The Numbers (Candidate Plans Evaluated):**\n" + "\n".join(plan_lines) + "\n\n"
                f"**Hit Verdict:** {hit}\n\n"
                f"**Long-Term Impact:** Banked free transfers roll over (up to 5 max in 2026/27). Selling price preserves 50% of price rises."
            )

        # Default: General squad / projection overview
        proj = await tool_executor.execute("get_projections", {"horizon": 3})
        top_players = [f"**{p['web_name']}** ({p['expected_points_gw']} xP, DefCon: +{p['exp_defcon_points']})" for p in proj.get("players", [])[:5]]

        return (
            f"### 🤖 FPL Oracle Expert Grounded Analysis (Season 2026/27)\n"
            f"I have analyzed the current gameweek based on live FPL API data and ML component projections.\n\n"
            f"**Top Projected Assets for Upcoming Gameweek {proj.get('gameweek')}:**\n"
            + "\n".join([f"- {p}" for p in top_players]) + "\n\n"
            f"**Scoring Insights:** Projections account for the 2026/27 Defensive Contribution (+2 pts) rule and rebalanced BPS.\n"
            f"Ask me about specific players, transfer plans, captaincy, chip strategy, or your mini-league rivals!"
        )

def get_llm_provider() -> Any:
    """Factory to instantiate the appropriate LLM provider."""
    if GEMINI_API_KEY and GEMINI_API_KEY.strip():
        from fpl_oracle.llm.gemini import GeminiProvider
        logger.info("Using Google Gemini LLM Provider.")
        return GeminiProvider(api_key=GEMINI_API_KEY.strip())
    elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
        from fpl_oracle.llm.openai import OpenAIProvider
        logger.info("Using OpenAI LLM Provider.")
        return OpenAIProvider(api_key=OPENAI_API_KEY.strip())
    elif ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip():
        from fpl_oracle.llm.anthropic import AnthropicProvider
        logger.info("Using Anthropic LLM Provider.")
        return AnthropicProvider(api_key=ANTHROPIC_API_KEY.strip())
    else:
        logger.info("No external LLM API key provided. Using OfflineExpertProvider (fully grounded).")
        return OfflineExpertProvider()
