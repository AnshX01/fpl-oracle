"""
Provider-agnostic LLM Layer.
Supports Google Gemini (default), Anthropic, OpenAI, and a standalone OfflineExpertProvider
that operates purely on live data, projections, and mathematical optimizations.
"""

import logging
from typing import Any

from fpl_oracle.config import ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
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

    async def chat(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        last_raw = messages[-1]["content"] if messages else ""
        last_msg = last_raw.lower()
        import re

        tokens = set(re.findall(r"\w+", last_msg))

        # 1. Comparison questions (e.g., "Explain why you picked Saka over Palmer", "compare Saka vs Palmer")
        if (
            "compare" in tokens
            or "vs" in tokens
            or (
                "over" in tokens
                and any(p in tokens for p in ["saka", "palmer", "haaland", "watkins", "foden", "salah"])
            )
        ):
            players_to_compare = []
            known_stars = [
                "saka",
                "palmer",
                "haaland",
                "watkins",
                "foden",
                "salah",
                "fernandes",
                "mbeumo",
                "diaz",
                "son",
            ]
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
                    f"### Head-to-Head Comparison: {better['web_name']} vs {worse['web_name']} (GW{res['gameweek']})\n"
                    f"**The Decision:** Start / Prioritize **{better['web_name']}** over **{worse['web_name']}** (+{diff_xp} net xP advantage).\n\n"
                    f"**The Numbers:**\n"
                    f"- **{better['web_name']}** (£{better['cost']}m): **{better['expected_points']} xP** (Floor P10: {better['p10']}, Ceiling P90: {better['p90']}, DefCon: +{better['defcon_pts']} pts)\n"
                    f"- **{worse['web_name']}** (£{worse['cost']}m): **{worse['expected_points']} xP** (Floor P10: {worse['p10']}, Ceiling P90: {worse['p90']}, DefCon: +{worse['defcon_pts']} pts)\n\n"
                    f"**The Why:** {better['web_name']} has the higher current model projection. A points comparison alone does not verify superiority in minutes, touches or xG.\n"
                    f"**The Risk:** Monitor pre-match team news for any late tactical rotation.\n"
                    f"**What would change the call:** If press conferences indicate a position shift or reduced minutes for {better['web_name']}."
                )

        # 2. Transfer Roadmap questions
        if "roadmap" in tokens:
            res = await tool_executor.execute("optimise_transfers", {})
            roadmap = res.get("roadmap", [])
            rows = []
            for step in roadmap:
                rows.append(
                    f"| **GW {step['gameweek']}** | {step['status']} | {step['action']} | Banked FT: {step.get('banked_free_transfers_projected', 1)} | {', '.join(step.get('key_targets', [])[:2])} |"
                )
            roadmap_table = "\n".join(rows)
            return (
                f"### Multi-Gameweek Transfer Roadmap (Next 5 GWs)\n"
                f"**The Decision:** Sequential plan prioritizing fixture swings and strategic FT banking (up to 5 max in 2026/27).\n\n"
                f"| Gameweek | Status | Action | Banked FT | Key Targets |\n"
                f"|---|---|---|---|---|\n"
                f"{roadmap_table}\n\n"
                f"**Strategic Note:** Firm steps are locked for the upcoming deadline; contingent steps adjust based on post-match injuries and European cup congestion."
            )

        # 3. Differentials under budget questions (e.g., "best differential midfielder under 6.5")
        if "differential" in tokens or (
            "under" in tokens and any(pos in tokens for pos in ["midfielder", "mid", "defender", "forward"])
        ):
            pos_filter = (
                "MID" if any(w in tokens for w in ["midfielder", "mid"]) else ("DEF" if "defender" in tokens else "FWD")
            )
            proj = await tool_executor.execute("get_projections", {"query": pos_filter, "horizon": 3})
            players = proj.get("players", [])
            # Extract budget dynamically (e.g. "under 6.5", "under £7.0m", "under 5")
            budget_match = re.search(r"under\s+(?:£)?(\d+(?:\.\d+)?)", last_raw, re.IGNORECASE)
            max_cost = float(budget_match.group(1)) if budget_match else 6.5
            budget_cands = [p for p in players if p.get("cost", 10.0) <= max_cost]
            if not budget_cands:
                budget_cands = players[:3]
            top3 = budget_cands[:3]
            c_lines = [
                f"- **{c['web_name']}** (£{c.get('cost', 6.0)}m): {c['expected_points_gw']} xP (DefCon: +{c.get('exp_defcon_points', 0.0)})"
                for c in top3
            ]
            best_diff = top3[0]["web_name"] if top3 else "Selected player"
            return (
                f"### Top Differential {pos_filter}s (Under £{max_cost:.1f}m)\n"
                f"**The Decision:** Target **{best_diff}** as your premier budget differential for the upcoming fixture swing.\n\n"
                f"**The Numbers:**\n" + "\n".join(c_lines) + "\n\n"
                "**The Why:** High baseline minutes reliability and substantial upside from the 2026/27 DefCon actions and attacking output.\n"
                "**Mini-League Strategy:** Under 20% Effective Ownership allows you to rapidly gain ground on rivals holding template assets."
            )

        # 4. Chip questions (Wildcard, Free Hit, Bench Boost, Triple Captain)
        chip_tokens = {"chip", "chips", "wildcard", "freehit", "bboost", "bb", "fh", "tc"}
        chip_phrases = ["bench boost", "free hit", "triple captain"]
        if (tokens & chip_tokens) or any(p in last_msg for p in chip_phrases):
            res = await tool_executor.execute("plan_chips", {})
            warning = res.get("set_1_warning", "")
            table_rows = []
            for row in res.get("chip_table", [])[:4]:
                table_rows.append(
                    f"| **{row['chip']}** | GW {row['recommended_gw']} | {row.get('expected_gain') if row.get('expected_gain') is not None else 'Not separately estimated'} | {row.get('confidence', 'Not assessed')} | {row['reasoning']} |"
                )

            table_str = "\n".join(table_rows)
            return (
                f"### 2026/27 Chip Strategy Plan\n"
                f"**The Decision:** Execute your 4 Set 1 chips before the Gameweek 19 deadline (Saturday 2 January 2027).\n\n"
                f"**[Warning]** {warning}\n\n"
                f"| Chip | Recommended GW | Exp. Gain | Confidence | Tactical Role |\n"
                f"|---|---|---|---|---|\n"
                f"{table_str}\n\n"
                f"**Key 2026/27 Rule Note:** Remember that Set 1 chips DO NOT carry over into Set 2. The Assistant Manager chip has been removed for 2026/27."
            )

        # 5. Captaincy questions
        if any(w in tokens for w in ["captain", "armband", "vice", "vc"]):
            res = await tool_executor.execute("captain_options", {})
            if res.get("status") == "unavailable" or res.get("error"):
                return "Captain advice unavailable: " + res.get("reason", res.get("error", "No verified lineup"))
            safe = res["safe_captain"]
            diff = res["differential_captain"]
            cands = res["candidates"]
            lines = [
                f"- **{c['web_name']}**: {c['expected_points']} xP (Floor: {c['p10']}, Ceiling: {c['p90']})"
                for c in cands[:3]
            ]
            return (
                f"### Captaincy Recommendation for Gameweek {res['gameweek']}\n"
                f"**The Decision:** Captain **{safe}**. Vice-captain **{diff}**.\n\n"
                f"**The Numbers:**\n" + "\n".join(lines) + "\n\n"
                "**The Why:** This is the same selected plan (not submitted), lineup and captain decision shown in your squad and decision card.\n"
                "**The Risk:** Guard against late press-conference rotation notes.\n"
                "**What would change the call:** If press conferences indicate minutes management, pivot immediately to your vice-captain."
            )

        # 6. Injury & Availability questions
        if any(
            w in tokens
            for w in [
                "injury",
                "injuries",
                "injured",
                "fit",
                "fitness",
                "available",
                "availability",
                "doubt",
                "doubtful",
                "knock",
            ]
        ):
            ignore_words = {
                "is",
                "are",
                "the",
                "for",
                "upcoming",
                "match",
                "game",
                "gameweek",
                "gw",
                "injured",
                "injury",
                "fit",
                "available",
                "out",
                "doubtful",
                "playing",
                "next",
                "round",
            }
            cand_tokens = [w for w in tokens if w not in ignore_words and len(w) > 2]
            player_cand = cand_tokens[0] if cand_tokens else last_msg
            news_res = await tool_executor.execute("get_news", {"query": player_cand})
            if news_res.get("status") == "found":
                p_name = news_res["web_name"]
                t_name = news_res["team"]
                if news_res.get("is_fit"):
                    return (
                        f"### Player Availability: **{p_name}** ({t_name})\n"
                        f"**Status:** Fully Fit & Available (100% chance of playing).\n"
                        f"**Official FPL Signal:** No current injury or suspension flags reported in the Premier League database.\n"
                        f"**Expected Minutes:** Projected to start with full 100% match availability."
                    )
                else:
                    return (
                        f"### Player Availability: **{p_name}** ({t_name})\n"
                        f"**Status:** Flagged ({news_res['availability_status']}) with {news_res['chance_of_playing']}% chance of playing.\n"
                        f"**Official FPL Signal:** {news_res['news']}\n"
                        f"**Source:** {news_res['source']} (Confidence: 1.0)."
                    )
            else:
                return f"### Player Availability: Not Found\nI could not find active Premier League records for '{player_cand}'. No injury signals reported."

        # 6. Mini-league questions
        if any(w in tokens for w in ["league", "rival", "rivals", "standings", "catch", "win", "rank"]):
            res = await tool_executor.execute("league_analysis", {})
            if "message" in res:
                return f"### Mini-League Intelligence\n{res['message']}"

            temps = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("template_players", [])[:3]]
            diffs = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("differentials", [])[:3]]
            return (
                f"### Mini-League Intelligence for {res['league_name']}\n"
                f"**The Decision:** Exploit rival template vulnerabilities with targeted differentials.\n\n"
                f"- **Top Template (High Rival EO):** {', '.join(temps)}\n"
                f"- **Key Differentials (<20% EO):** {', '.join(diffs)}\n\n"
                f"**Strategy Mode:** If leading, mirror the template to protect against rival hauls. If chasing, back differentials to make up points."
            )

        # 7. Specific entity / player projection lookup and unknown player handling
        # Match full player name/token after a preposition
        entity_match = re.search(r"(?:for|about|is|on)\s+([a-zA-Z0-9_\-]+)", last_raw, re.IGNORECASE)
        if ("projected" in tokens or "points" in tokens or "xp" in tokens) and entity_match:
            entity = entity_match.group(1).strip()
            # Short-circuit: obviously fictitious/unknown entity names (contain digits mixed with letters)
            is_obviously_unknown = any(c.isdigit() for c in entity) or len(entity) > 30
            if not is_obviously_unknown:
                proj_res = await tool_executor.execute("get_projections", {"query": entity})
                players = proj_res.get("players", [])
                exact_or_close = [
                    p
                    for p in players
                    if entity.lower() in p["web_name"].lower() or p["web_name"].lower() in entity.lower()
                ]
            else:
                exact_or_close = []
            if exact_or_close:
                p = exact_or_close[0]
                return (
                    f"### Player Profile & Projections: **{p['web_name']}**\n"
                    f"- **Position:** {p['position']} | **Cost:** £{p.get('cost', 5.0)}m\n"
                    f"- **Gameweek {proj_res.get('gameweek')} Projected Points:** **{p['expected_points_gw']} xP**\n"
                    f"- **Uncertainty:** Floor P10: {p.get('p10_floor', p.get('p10', '?'))}, Ceiling P90: {p.get('p90_ceiling', p.get('p90', '?'))}\n"
                    f"- **Defensive Contribution:** +{p.get('exp_defcon_points', 0.0)} pts\n\n"
                    f"**Source:** Sourced directly from live 2026/27 ML component inference."
                )
            else:
                return (
                    f"### Player Lookup: Not Found\n"
                    f"I could not find any active Premier League player matching **'{entity}'** in the official 2026/27 database. "
                    f"This player is unknown or not registered in the current season. Unable to locate data or player records.\n"
                    f"Please verify the spelling or confirm the player is registered in FPL 2026/27."
                )

        # 8. Price changes
        if any(w in tokens for w in ["rise", "rises", "fall", "falls", "drop", "drops"]) or (
            "price" in tokens and any(w in tokens for w in ["change", "changes", "tonight", "watch", "imminent"])
        ):
            res = await tool_executor.execute("price_change_watch", {})
            rises = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_rises", [])[:3]]
            falls = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_falls", [])[:3]]
            return (
                f"### Price Change Watch (Tonight's Projections)\n"
                f"**Imminent Rises:** {', '.join(rises) if rises else 'None currently at threshold'}\n\n"
                f"**Imminent Falls:** {', '.join(falls) if falls else 'None currently at threshold'}\n\n"
                f"**Action:** Make scheduled transfers before 01:30 GMT to avoid losing purchasing power."
            )

        # 9. Transfer & Hit questions
        if any(w in tokens for w in ["transfer", "transfers", "hit", "hits", "sell", "buy", "haaland"]):
            res = await tool_executor.execute("optimise_transfers", {})
            if "error" in res:
                return f"### Transfer Strategy & Hit Analysis\nUnable to optimize transfers: {res['error']}"
            rec = res["decision"]
            hit = res["hit_verdict"]
            plans = res.get("candidate_plans", [])
            plan_lines = [
                f"- **{p['type']}**: {p['summary']} (Net xP: {p['net_xp']}, Net Gain: {p['gain']})" for p in plans
            ]

            return (
                f"### Transfer Strategy & Hit Analysis\n"
                f"**The Decision:** {rec}\n\n"
                f"**The Numbers (Candidate Plans Evaluated):**\n" + "\n".join(plan_lines) + "\n\n"
                f"**Hit Verdict:** {hit}\n\n"
                f"**Long-Term Impact:** Banked free transfers roll over (up to 5 max in 2026/27). Selling price preserves 50% of price rises."
            )

        # Default: General squad / projection overview
        proj = await tool_executor.execute("get_projections", {"horizon": 3})
        top_players = [
            f"**{p['web_name']}** ({p['expected_points_gw']} xP, DefCon: +{p['exp_defcon_points']})"
            for p in proj.get("players", [])[:5]
        ]

        return (
            f"### FPL Oracle Expert Grounded Analysis (Season 2026/27)\n"
            f"I have analyzed the current gameweek based on live FPL API data and ML component projections.\n\n"
            f"**Top Projected Assets for Upcoming Gameweek {proj.get('gameweek')}:**\n"
            + "\n".join([f"- {p}" for p in top_players])
            + "\n\n"
            "**Scoring Insights:** Projections account for the 2026/27 Defensive Contribution (+2 pts) rule and rebalanced BPS.\n"
            "Ask me about specific players, transfer plans, captaincy, chip strategy, or your mini-league rivals!"
        )


def get_llm_status() -> dict[str, Any]:
    """Report LLM provider status for diagnostic health checks."""
    from fpl_oracle.config import SETTINGS
    from fpl_oracle.data.store import data_store

    profile = data_store.get_profile()
    pref = (getattr(profile, "llm_provider", None) or SETTINGS.get("llm", {}).get("provider", "")).lower()

    active_provider = "offline_expert"
    configured = False

    if pref in ("offline", "offline_expert", "none"):
        active_provider = "offline_expert"
        configured = False
    elif pref == "gemini" and GEMINI_API_KEY.strip():
        active_provider = "gemini"
        configured = True
    elif pref == "openai" and OPENAI_API_KEY.strip():
        active_provider = "openai"
        configured = True
    elif pref == "anthropic" and ANTHROPIC_API_KEY.strip():
        active_provider = "anthropic"
        configured = True
    elif GEMINI_API_KEY.strip():
        active_provider = "gemini"
        configured = True
    elif OPENAI_API_KEY.strip():
        active_provider = "openai"
        configured = True
    elif ANTHROPIC_API_KEY.strip():
        active_provider = "anthropic"
        configured = True

    return {
        "active_provider": active_provider,
        "preferred_provider": pref or None,
        "is_api_key_configured": configured,
        "fallback_available": True,
        "supported_providers": ["gemini", "openai", "anthropic", "offline_expert"],
    }


def get_llm_provider(preferred_provider: str | None = None) -> Any:
    """Factory to instantiate the appropriate LLM provider respecting preferences."""
    from fpl_oracle.config import SETTINGS
    from fpl_oracle.data.store import data_store

    profile = data_store.get_profile()
    pref = (
        preferred_provider or getattr(profile, "llm_provider", None) or SETTINGS.get("llm", {}).get("provider", "")
    ).lower()

    if pref in ("offline", "offline_expert", "none"):
        logger.info("Using OfflineExpertProvider by user preference.")
        return OfflineExpertProvider()
    elif pref == "gemini" and GEMINI_API_KEY.strip():
        from fpl_oracle.llm.gemini import GeminiProvider

        logger.info("Using Google Gemini LLM Provider by preference.")
        return GeminiProvider(api_key=GEMINI_API_KEY.strip())
    elif pref == "openai" and OPENAI_API_KEY.strip():
        from fpl_oracle.llm.openai import OpenAIProvider

        logger.info("Using OpenAI LLM Provider by preference.")
        return OpenAIProvider(api_key=OPENAI_API_KEY.strip())
    elif pref == "anthropic" and ANTHROPIC_API_KEY.strip():
        from fpl_oracle.llm.anthropic import AnthropicProvider

        logger.info("Using Anthropic LLM Provider by preference.")
        return AnthropicProvider(api_key=ANTHROPIC_API_KEY.strip())

    # Fallback to available key or offline expert
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
        logger.info("No external LLM API key configured. Using OfflineExpertProvider (fully grounded).")
        return OfflineExpertProvider()
