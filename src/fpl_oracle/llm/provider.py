"""
Provider-agnostic LLM Layer.
Supports Google Gemini (default), Anthropic, OpenAI, and a standalone OfflineExpertProvider
that operates purely on live data, projections, and mathematical optimizations.
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fpl_oracle.config import ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
from fpl_oracle.llm.tools import tool_executor

logger = logging.getLogger("fpl_oracle.llm.provider")

VOICE_CONTRACT = Path(__file__).parent / "prompts" / "advisor_system.md"

SYSTEM_PROMPT = """You are the FPL Oracle: the manager's hired Fantasy Premier League expert and decision-support advisor.
You know the whole tool: squad, bank, free transfers, chips per set, form, fixtures, availability, projections, the published decision and how it was made.

Operating rules:
1. THE PUBLISHED DECISION IS THE ANSWER. The decision record below is what the app shows the manager. Explain it, justify it with its numbers, and debate it. If the manager proposes something else, compare it with the record using evaluate_plan, compare_options, explain_why_not or compare_players, then concede or push back with the computed numbers. Do not silently replace the published advice with a new plan.
2. STRICT GROUNDING. Quote only names, prices, xP, probabilities and fixtures from the record or tool results. Never invent news, injuries or numbers. If data is missing or stale, say so. Label forecasts as estimates.
3. TOOLS. Use get_current_decision for any section of the decision (or raw card). Use the other tools for anything the record lacks (other players, other transfers, price changes, news, fixtures).
4. Earlier chat turns may refer to older advice; the record in this prompt wins.
5. Action first. Answer the question asked; use the structured elements of the voice contract for full recommendations, not for every follow-up.
"""


OFFLINE_MODE_PREFIX = "[OFFLINE EXPERT MODE: Running purely on live local data, projections, and mathematical optimizations without external LLM keys.]\n\n"


async def build_system_prompt() -> str:
    """Voice contract + rules + today's date + the published decision record, rebuilt every turn."""
    from fpl_oracle.llm.decision_record import current_decision_record, prompt_section

    try:
        voice = VOICE_CONTRACT.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Advisor voice contract missing: %s", VOICE_CONTRACT)
        voice = ""
    try:
        record = await current_decision_record()
    except Exception as e:
        logger.warning("Decision record unavailable for chat prompt: %s", type(e).__name__)
        record = {"status": "unavailable", "reason": "The published decision could not be read."}
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return "\n\n".join(
        p for p in (SYSTEM_PROMPT.strip(), voice, f"Today is {today} (UTC).", prompt_section(record)) if p
    )


_WHY_NOT_WORDS = {"instead", "rather"}
_EXPLAIN_WORDS = {"why", "justify", "explain", "defend", "reason", "reasoning", "decision", "recommend", "recommended"}
_POSITION_WORDS = {
    "goalkeeper": "GKP",
    "keeper": "GKP",
    "defender": "DEF",
    "midfielder": "MID",
    "mid": "MID",
    "forward": "FWD",
}


def _fmt_move(m: dict[str, Any] | str) -> str:
    if isinstance(m, str):  # trimmed lists end with a "+N more" note
        return m
    return f"{m.get('name')} (£{m.get('price')}m, {m.get('xp_gw')} xP)"


def _find_players(text: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool players named in `text`, in the order they are mentioned (best-projected player per name)."""
    low = text.lower()
    hits: dict[str, tuple[int, dict[str, Any]]] = {}
    for p in pool:  # pool is ranked best first
        name = p["web_name"].lower()
        if name in hits:
            continue
        m = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", low)
        if m:
            hits[name] = (m.start(), p)
    return [p for _, p in sorted(hits.values(), key=lambda t: t[0])]


class OfflineExpertProvider:
    """
    Offline data-grounded expert analyst. With a published decision it answers from the decision
    record (what to do, why, alternatives, why-not comparisons); otherwise it runs the live tools.
    No external LLM keys needed, and the same question always gets the same answer.
    """

    is_offline: bool = True

    async def chat(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        raw_res = await self._generate_response(messages)
        if not raw_res.startswith(OFFLINE_MODE_PREFIX):
            return OFFLINE_MODE_PREFIX + raw_res
        return raw_res

    # ---- published-decision answers -------------------------------------------------------

    @staticmethod
    def _render_decision(rec: dict[str, Any]) -> str:
        t, chip, lineup = rec.get("transfers") or {}, rec.get("chip") or {}, rec.get("lineup") or {}
        team = rec.get("team") or {}
        cap, vice = lineup.get("captain") or {}, lineup.get("vice_captain") or {}
        outs, ins = t.get("out") or [], t.get("in") or []
        if ins and chip.get("play_now") and str(chip.get("chip") or "").lower() in ("wildcard", "free hit"):
            # A rebuild is not a list of like-for-like swaps
            move = (
                f"play your {chip.get('chip')}: bring in {', '.join(_fmt_move(i) for i in ins)}; "
                f"sell {', '.join(_fmt_move(o) for o in outs)}"
            )
        elif ins:
            move = "; ".join(f"sell {_fmt_move(o)} and buy {_fmt_move(i)}" for o, i in zip(outs, ins, strict=False))
        else:
            move = "roll the free transfer"
        chip_line = f"play {chip.get('chip')}" if chip.get("play_now") else "save your chips"
        lines = [
            f"### GW{rec.get('gameweek')} decision",
            f"**Decision:** {move}; captain {cap.get('web_name')} (vice {vice.get('web_name')}); {chip_line}.",
            "",
            "**Why:**",
        ]
        if t.get("why"):
            lines.append(f"- {t['why']}")
        if t.get("gain_vs_roll_over_horizon_xp") is not None:
            lines.append(
                f"- Transfers: +{t['gain_vs_roll_over_horizon_xp']} xP over the planning horizon vs rolling "
                f"({t.get('gain_this_gw_xp')} xP this GW); horizon total {t.get('horizon_xp')} xP."
            )
        if cap:
            lines.append(
                f"- Captain {cap.get('web_name')}: {cap.get('expected_points')} xP "
                f"(floor {cap.get('p10')}, ceiling {cap.get('p90')}); {lineup.get('captain_reason')}"
            )
        if chip:
            lines.append(
                f"- Chip: {'play ' + str(chip.get('chip')) if chip.get('play_now') else 'hold'}"
                f" (gain vs hold {chip.get('gain_vs_hold_xp')} xP). {chip.get('reason') or ''}"
            )
        alt = rec.get("alternatives") or {}
        lines += ["", "**Alternatives considered:**"]
        for o in (chip.get("now_vs_later") or [])[:6]:
            if isinstance(o, dict):
                lines.append(
                    f"- {o.get('chip')}: {o.get('net_gain_vs_hold')} xP vs holding"
                    f"{' (recommended)' if o.get('recommended') else ''}"
                )
        for plan in (alt.get("contingency") or {}).values():
            lines.append(f"- {plan.get('title')}: {plan.get('net_xp')} xP ({plan.get('delta_vs_plan_a')} vs Plan A)")
        lines += [
            "",
            f"**Cost:** {t.get('hits') or 0} hit(s) (-{t.get('hit_cost') or 0} pts), "
            f"{t.get('free_transfers_used')} free transfer(s) used, {t.get('free_transfers_next_gw')} banked next GW, "
            f"bank after £{t.get('bank_after_millions')}m (now £{team.get('bank_millions')}m).",
        ]
        for key, plan in (alt.get("contingency") or {}).items():
            if plan.get("trigger") and key != "plan_a":
                lines.append(f"**What would change this:** {plan['trigger']}")
        meta = rec.get("meta") or {}
        lines.append(
            f"**Data:** model {meta.get('model_version')}, as of {meta.get('data_as_of')}"
            f"{'; DATA IS STALE' if meta.get('is_stale') else ''}."
        )
        dl = rec.get("deadline") or {}
        lines.append(f"**Deadline:** {dl.get('label') or dl.get('deadline_time')}")
        return "\n".join(lines)

    @staticmethod
    async def _why_not(rec: dict[str, Any], text: str, tokens: set[str]) -> str | None:
        """'Why not X?': hits/chips use the planner's scored plans; players are compared by projection."""
        if {"free", "hit"} <= tokens or "freehit" in tokens:
            action = "freehit"
        elif tokens & {"hit", "hits"}:
            action = "hit"
        elif "wildcard" in tokens:
            action = "wildcard"
        elif "bench" in tokens:
            action = "bboost"
        elif "triple" in tokens:
            action = "3xc"
        elif "chip" in tokens:
            action = "chip"
        else:
            action = None
        chip = rec.get("chip") or {}
        names = {"wildcard": "wildcard", "freehit": "free hit", "bboost": "bench boost", "3xc": "triple captain"}
        if action in names and names[action] in str(chip.get("chip") or "").lower():
            # The published chip decision already measured now against later: answer from it, no re-solve.
            check = chip.get("check") or {}
            lines = [f"### {chip.get('chip')}: {'play it now' if chip.get('play_now') else 'keep it'}"]
            if chip.get("comparison"):
                lines.append(str(chip["comparison"]))
            if check:
                lines.append(
                    f"Measured: {check.get('gain')} predicted points versus {check.get('versus')}, against a forecast "
                    f"uncertainty of {check.get('uncertainty')} (needs {check.get('needed')} to act at "
                    f"{int(float(check.get('confidence') or 0.75) * 100)}% confidence)."
                )
            for later in chip.get("later_provisional") or []:
                if isinstance(later, dict):
                    lines.append(
                        f"Pencilled in for later (re-decided every week): {later.get('chip_display_name')} in "
                        f"gameweek {later.get('gameweek')}, about +{later.get('gain')} points."
                    )
            return "\n\n".join(lines)
        if action:
            res = await tool_executor.execute("explain_why_not", {"action_type": action})
            if "error" in res:
                return None
            nums = ", ".join(f"{k}={v}" for k, v in (res.get("numbers") or {}).items())
            return f"### {res['query_action']}\n**{res['verdict']}.** {res['explanation']}" + (
                f"\nNumbers: {nums}" if nums else ""
            )

        pool_res = await tool_executor.execute("get_projections", {"horizon": 5, "limit": 800})
        pool = pool_res.get("players", [])
        named = _find_players(text, pool)
        t, lineup = rec.get("transfers") or {}, rec.get("lineup") or {}
        rec_in_ids = {m.get("id") for m in t.get("in") or []}
        named = [p for p in named if p["element"] not in rec_in_ids] or named
        if not named:
            return None
        x = named[0]
        by_id = {p["element"]: p for p in pool}
        horizon = pool_res.get("gameweeks") or []
        if "captain" in tokens:
            ref = by_id.get((lineup.get("captain") or {}).get("element"))
            role = "captain"
        else:
            ins = [by_id[m["id"]] for m in t.get("in") or [] if m.get("id") in by_id]
            ref = next((p for p in ins if p["position"] == x["position"]), ins[0] if ins else None)
            role = "recommended signing"

        def line(p: dict[str, Any]) -> str:
            return (
                f"**{p['web_name']}** ({p['position']}, £{p['cost']}m, {p['ownership_pct']}% owned): "
                f"{p['expected_points_horizon']} xP over GW{horizon[0]}-{horizon[-1]} "
                f"(this GW {p['expected_points_gw']}, floor {p['p10_floor']}, ceiling {p['p90_ceiling']})"
            )

        if ref is None and role == "captain":
            return None
        if ref is None:
            return (
                f"### Why not {x['web_name']}?\n- {line(x)}\n"
                f"**Verdict:** the optimiser's best plan this week makes no signing ({t.get('summary')})."
            )

        diff = round(ref["expected_points_horizon"] - x["expected_points_horizon"], 2)
        verdict = (
            f"{ref['web_name']} leads by {diff:+.2f} xP over the horizon."
            if diff >= 0
            else f"{x['web_name']} projects {-diff:.2f} xP higher over the horizon, so the gap has to come from "
            "budget, the 3-per-club limit or the other moves in the plan (see the cost check below)."
        )
        out = [f"### Why not {x['web_name']}?", f"- {line(x)}", f"- {role}: {line(ref)}", f"**Verdict:** {verdict}"]
        outs = t.get("out") or []
        if role == "recommended signing" and len(outs) == 1 and len(t.get("in") or []) == 1:
            ev = await tool_executor.execute(
                "evaluate_plan", {"proposed_transfers": [{"out": outs[0]["id"], "in": x["element"]}]}
            )
            if "error" not in ev:
                out.append(
                    f"**Planner check, {outs[0]['name']} -> {x['web_name']}:** {ev['verdict']}. {ev['numeric_reason']} "
                    f"Regret vs best plan {ev['regret_xp']} xP."
                )
        return "\n".join(out)

    # ---- main dispatch ---------------------------------------------------------------------

    async def _generate_response(self, messages: list[dict[str, str]]) -> str:
        last_raw = messages[-1]["content"] if messages else ""
        last_msg = last_raw.lower()
        token_list = re.findall(r"\w+", last_msg)
        tokens = set(token_list)

        rec = await tool_executor.execute("get_current_decision", {})
        ready = rec.get("status") == "ready"

        # 1. "Why not X / X instead of Y": the published pick against X, using projections.
        if ready and (("why" in tokens and "not" in tokens) or tokens & _WHY_NOT_WORDS):
            answer = await self._why_not(rec, last_raw, tokens)
            if answer:
                return answer

        # 2. Comparison questions (e.g., "compare A vs B", "A over B")
        pool: list[dict[str, Any]] | None = None
        if "compare" in tokens or "vs" in tokens or "over" in tokens:
            pool = (await tool_executor.execute("get_projections", {"horizon": 1, "limit": 800})).get("players", [])
            players_to_compare = [p["web_name"] for p in _find_players(last_raw, pool)][:2]
            if len(players_to_compare) < 2 and tokens & {"compare", "vs"}:
                return "Name both players you want to compare."
            res = (
                await tool_executor.execute("compare_players", {"player_names": players_to_compare})
                if len(players_to_compare) == 2
                else {}
            )
            comps = res.get("comparisons", [])
            if len(comps) >= 2:
                p1, p2 = comps[0], comps[1]
                better = p1 if p1["expected_points"] >= p2["expected_points"] else p2
                worse = p2 if p1["expected_points"] >= p2["expected_points"] else p1
                diff_xp = round(better["expected_points"] - worse["expected_points"], 2)
                return (
                    f"### Player Comparison: {p1['web_name']} vs {p2['web_name']} (GW{res['gameweek']})\n"
                    f"- **{better['web_name']}** ({better.get('position', '')}, £{better.get('cost', '')}m): **{better['expected_points']} est. points** "
                    f"(P10 Floor: {better.get('p10_floor', better.get('p10', 0))} | P90 Ceiling: {better.get('p90_ceiling', better.get('p90', 0))} | DefCon: +{better.get('defcon_pts', 0.0)})\n"
                    f"- **{worse['web_name']}** ({worse.get('position', '')}, £{worse.get('cost', '')}m): **{worse['expected_points']} est. points** "
                    f"(P10 Floor: {worse.get('p10_floor', worse.get('p10', 0))} | P90 Ceiling: {worse.get('p90_ceiling', worse.get('p90', 0))} | DefCon: +{worse.get('defcon_pts', 0.0)})\n\n"
                    f"**Verdict:** {better['web_name']} leads by +{diff_xp} est. points for GW{res['gameweek']}."
                )

        # 3. Justify / explain the published decision
        if ready and tokens & _EXPLAIN_WORDS:
            return self._render_decision(rec)

        # 4. Transfer Roadmap questions
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
                "### Transfers\n| Week | Status | Move | Free transfers | Players |\n|---|---|---|---|---|\n"
                + roadmap_table
            )

        # 5. Differentials under budget (e.g., "best differential midfielder under 6.5")
        pos_filter = next((code for word, code in _POSITION_WORDS.items() if word in tokens), "")
        if "differential" in tokens or ("under" in tokens and pos_filter):
            proj = await tool_executor.execute("get_projections", {"query": pos_filter, "horizon": 3, "limit": 800})
            budget_match = re.search(r"under\s+(?:£)?(\d+(?:\.\d+)?)", last_raw, re.IGNORECASE)
            max_cost = float(budget_match.group(1)) if budget_match else 6.5
            budget_cands = [
                p
                for p in proj.get("players", [])
                if p.get("cost", 10.0) <= max_cost
                and (p.get("ownership_pct", 0.0) < 10.0 or "differential" not in tokens)
            ]
            if not budget_cands:
                return f"No matching players found within £{max_cost:.1f}m. Try a different budget."
            c_lines = [
                f"- **{c['web_name']}** (£{c['cost']}m, {c['ownership_pct']}% owned): {c['expected_points_horizon']} xP over the next {len(proj['gameweeks'])} GWs (this GW {c['expected_points_gw']}, DefCon: +{c['exp_defcon_points']})"
                for c in budget_cands[:3]
            ]
            label = "differential " if "differential" in tokens else ""
            return (
                f"### {pos_filter or 'All'} {label}options under £{max_cost:.1f}m (best projected first)\n"
                + "\n".join(c_lines)
            )

        # 6. Chip questions (Wildcard, Free Hit, Bench Boost, Triple Captain)
        chip_tokens = {"chip", "chips", "wildcard", "freehit", "bboost", "bb", "fh", "tc"}
        chip_phrases = ["bench boost", "free hit", "triple captain"]
        if (tokens & chip_tokens) or any(p in last_msg for p in chip_phrases):
            res = await tool_executor.execute("plan_chips", {})
            warning = res.get("set_1_warning", "")
            table_rows = [
                f"| {row['chip']} | {('GW' + str(row['recommended_gw'])) if row['recommended_gw'] is not None else 'Save'} | {row.get('expected_gain', 'Unknown')} |"
                for row in res.get("chip_table", [])[:4]
            ]
            published = ""
            if ready:
                c = rec["chip"]
                published = (
                    f"Published decision: {'play ' + str(c.get('chip')) if c.get('play_now') else 'save your chips this week'}"
                    f" (gain vs hold {c.get('gain_vs_hold_xp')} xP). Chips left: {(rec['team'] or {}).get('chips_remaining')}.\n\n"
                )
            return (
                published
                + "### Chips\n| Chip | Week | Est. extra points |\n|---|---|---|\n"
                + "\n".join(table_rows)
                + ("\n" + warning if warning else "")
            )

        # 7. Captaincy questions
        if any(w in tokens for w in ["captain", "armband", "vice", "vc"]):
            res = await tool_executor.execute("captain_options", {})
            if res.get("status") == "unavailable" or res.get("error"):
                return "Captain advice unavailable: " + res.get("reason", res.get("error", "No verified lineup"))
            cap, vice = res["captain_name"], res["vice_captain_name"]
            lines = [
                f"- **{c['web_name']}** ({c.get('position', '')}, {c.get('ownership_pct', 0)}% owned): **{c['expected_points']} est. points** "
                f"(Floor: {c.get('p10_floor', c.get('p10', 0))} | Ceiling: {c.get('p90_ceiling', c.get('p90', 0))} | DefCon: +{c.get('defcon_pts', 0.0)})"
                for c in res["candidates"][:3]
            ]
            diff = res.get("differential_option")
            return (
                f"GW{res['gameweek']}: captain {cap}, vice captain {vice}.\n"
                f"- **Captain:** **{cap}** (highest projected expected points in the selected XI).\n"
                f"- **Vice-captain:** **{vice}** (takes the armband if {cap} does not play).\n"
                + (f"- **Low-ownership option:** **{diff}** (under 10% owned).\n" if diff else "")
                + "\n**Top Candidates Breakdown:**\n"
                + "\n".join(lines)
            )

        # 8. Injury & Availability questions
        if tokens & {
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
        }:
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
            if pool is None:
                pool = (await tool_executor.execute("get_projections", {"horizon": 1, "limit": 800})).get("players", [])
            named = _find_players(last_raw, pool)
            cand_tokens = [w for w in token_list if w not in ignore_words and len(w) > 2]
            player_cand = named[0]["web_name"] if named else (cand_tokens[0] if cand_tokens else last_msg)
            news_res = await tool_executor.execute("get_news", {"player_name": player_cand})
            if news_res.get("status") == "found":
                p_name = news_res["web_name"]
                if news_res.get("is_fit"):
                    return f"{p_name}: available. No injury reported."
                chance = news_res.get("chance_of_playing")
                return f"{p_name}: {news_res['news']}" + (
                    f" {chance}% chance of playing." if chance is not None else " Availability unknown."
                )
            if news_res.get("status") == "stale":
                return "Team news is old. Refresh before deciding."
            return f"Player not found: {player_cand}."

        # 9. Mini-league questions
        if any(w in tokens for w in ["league", "rival", "rivals", "standings", "catch", "win", "rank"]):
            res = await tool_executor.execute("league_analysis", {})
            if "message" in res:
                return f"### Mini-League Intelligence\n{res['message']}"

            temps = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("template_players", [])[:3]]
            diffs = [f"{p['web_name']} ({p['effective_ownership']}%)" for p in res.get("differentials", [])[:3]]
            return f"### {res['league_name']}\nCommon players: {', '.join(temps) or 'Unknown'}\nDifferentials: {', '.join(diffs) or 'Unknown'}"

        # 10. Specific entity / player projection lookup and unknown player handling
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
                    f"- **Defensive Contribution:** +{p.get('exp_defcon_points', 0.0)} pts"
                )
            else:
                return (
                    f"### Player Lookup: Not Found\n"
                    f"I could not find any active Premier League player matching **'{entity}'** in the official 2026/27 database. "
                    f"This player is unknown or not registered in the current season. Unable to locate data or player records.\n"
                    f"Please verify the spelling or confirm the player is registered in FPL 2026/27."
                )

        # 11. Price changes
        if any(w in tokens for w in ["rise", "rises", "fall", "falls", "drop", "drops"]) or (
            "price" in tokens and any(w in tokens for w in ["change", "changes", "tonight", "watch", "imminent"])
        ):
            res = await tool_executor.execute("price_change_watch", {})
            rises = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_rises", [])[:3]]
            falls = [f"**{p['web_name']}** ({p['urgency_message']})" for p in res.get("imminent_falls", [])[:3]]
            return (
                "Possible rises: " + (", ".join(rises) or "None") + "\nPossible falls: " + (", ".join(falls) or "None")
            )

        # 12. Transfer & Hit questions
        if any(w in tokens for w in ["transfer", "transfers", "hit", "hits", "sell", "buy", "swap", "move", "replace"]):
            if ready:
                return self._render_decision(rec)
            res = await tool_executor.execute("optimise_transfers", {})
            if "error" in res:
                return f"### Transfer Strategy & Hit Analysis\nUnable to optimize transfers: {res['error']}"
            rec_plan = res["decision"]
            hit = res["hit_verdict"]
            rec_transfers = res.get("recommended_transfers", [])
            transfers_text = ""
            if rec_transfers:
                t_lines = [
                    f"- **OUT:** {t['transfer_out']} (£{t.get('cost_out', 'N/A')}m) -> **IN:** {t['transfer_in']} (£{t.get('cost_in', 'N/A')}m)"
                    for t in rec_transfers
                ]
                transfers_text = "\n**Recommended Moves:**\n" + "\n".join(t_lines) + "\n"

            plans = res.get("candidate_plans", [])
            plan_lines = [
                f"- **{p.get('type', p.get('plan_type'))}**: {p.get('summary', '')} (Net xP: {p.get('net_xp')}, Gain vs Roll: {p.get('gain', p.get('gain_vs_roll'))})"
                for p in plans
            ]

            return (
                f"### Transfer Strategy & Mathematical Working\n"
                f"**Verdict:** {rec_plan}\n"
                f"{transfers_text}\n"
                f"**Hit Analysis:** {hit}\n\n"
                f"**Evaluated Candidate Plans:**\n" + "\n".join(plan_lines)
            )

        # Default: the published decision, else a general projection overview
        if ready:
            return self._render_decision(rec)
        proj = await tool_executor.execute("get_projections", {"horizon": 3})
        top_players = [
            f"**{p['web_name']}** ({p['expected_points_gw']} xP, DefCon: +{p['exp_defcon_points']})"
            for p in proj.get("players", [])[:5]
        ]
        note = "No published decision is ready yet; showing the top projected players instead.\n"
        return f"### GW{proj.get('gameweek')} player options\n{note}" + "\n".join(
            f"- {player}" for player in top_players
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
