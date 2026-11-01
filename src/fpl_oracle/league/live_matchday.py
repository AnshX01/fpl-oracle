"""
Live Matchday Operations & Real-Time League Standings Engine (Pillar 5).
Tracks live gameweek event scores, calculates real-time provisional BPS bonus points (3-2-1),
simulates live formation-legal bench autosubs and captaincy rollover, and re-ranks
mini-league standings in real time against rivals.
Zero hardcoded player names in production code.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from fpl_oracle.api.fpl_client import fpl_client

logger = logging.getLogger("fpl_oracle.league.live_matchday")


@dataclass
class LivePlayerScore:
    element_id: int
    web_name: str
    position: str
    team_id: int
    minutes: int
    goals: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    bps: int
    official_bonus: int
    provisional_bonus: int
    live_points: int
    status: str  # "YET_TO_PLAY", "PLAYING", "FINISHED"
    fixture_id: int | None = None
    multiplier: int = 1


@dataclass
class LiveAutosub:
    out_element_id: int
    out_web_name: str
    in_element_id: int
    in_web_name: str
    sub_type: str  # "GOALKEEPER", "OUTFIELD", "CAPTAIN_PASS"
    reason: str


@dataclass
class LiveManagerSquad:
    manager_id: int
    manager_name: str
    team_name: str
    active_chip: str | None
    transfer_cost: int
    live_gw_gross: int
    live_gw_net: int
    live_total_points: int
    starters: list[LivePlayerScore]
    bench: list[LivePlayerScore]
    autosubs: list[LiveAutosub]
    effective_captain_id: int
    players_played: int
    players_playing: int
    players_remaining: int


@dataclass
class LiveLeagueRow:
    live_rank: int
    previous_rank: int
    rank_movement: int
    manager_id: int
    manager_name: str
    team_name: str
    live_gw_points: int
    live_total_points: int
    gap_to_leader: int
    players_remaining: int
    captain_name: str
    active_chip: str | None


@dataclass
class LiveSafetyScore:
    safety_score: float  # The parity threshold to avoid rank drop
    user_live_net_points: int
    safety_gap: float  # user_live_net_points - safety_score
    projected_arrow: str  # "GREEN_ARROW", "GREY_ARROW", "RED_ARROW"
    field_effective_ownership: dict[str, float]  # Top threats by EO
    benchmark_source: str  # "Mini-League Pool (N=X)"
    summary_text: str


@dataclass
class LiveMatchdaySnapshot:
    gameweek: int
    league_id: int
    league_name: str
    user_manager_id: int | None
    user_live_rank: int | None
    user_live_total: int | None
    standings: list[LiveLeagueRow]
    user_squad: LiveManagerSquad | None
    fixtures_summary: dict[str, int] = field(default_factory=dict)
    is_live: bool = True
    is_stale: bool = False
    safety_score: LiveSafetyScore | None = None


def compute_fixture_provisional_bonus(player_bps: dict[int, int]) -> dict[int, int]:
    """
    Computes official FPL provisional bonus points (3, 2, 1) from match BPS values,
    strictly handling ties according to Premier League rules.
    """
    eligible = {eid: bps for eid, bps in player_bps.items() if bps > 0}
    if not eligible:
        return {}

    unique_scores = sorted(set(eligible.values()), reverse=True)
    groups = [[eid for eid, bps in eligible.items() if bps == sc] for sc in unique_scores]

    bonus_map: dict[int, int] = {}

    # Rank 1
    if len(groups[0]) >= 3:
        for eid in groups[0]:
            bonus_map[eid] = 3
        return bonus_map
    elif len(groups[0]) == 2:
        for eid in groups[0]:
            bonus_map[eid] = 3
        if len(groups) > 1:
            for eid in groups[1]:
                bonus_map[eid] = 1
        return bonus_map
    else:
        bonus_map[groups[0][0]] = 3

    # Rank 2
    if len(groups) > 1:
        if len(groups[1]) >= 2:
            for eid in groups[1]:
                bonus_map[eid] = 2
            return bonus_map
        else:
            bonus_map[groups[1][0]] = 2

    # Rank 3
    if len(groups) > 2:
        for eid in groups[2]:
            bonus_map[eid] = 1

    return bonus_map


class LiveMatchdayEngine:
    """
    Real-Time Matchday Engine:
    - Calculates BPS provisional bonus points with exact FPL tie rules.
    - Resolves provisional autosubs and captaincy handoff.
    - Computes real-time mini-league standings with transfer hits deduction.
    """

    def compute_fixture_bonuses(
        self,
        fixtures: list[Any],
        live_elements: list[dict[str, Any]],
    ) -> dict[int, int]:
        """Maps element_id -> provisional bonus points across all active fixtures."""
        bonus_by_element: dict[int, int] = {}
        fixture_players: dict[int, dict[int, int]] = {}

        for elem in live_elements:
            eid = int(elem.get("id", 0))
            stats = elem.get("stats", {})
            bps = int(stats.get("bps", 0) or 0)
            explains = elem.get("explain", [])
            for ex in explains:
                fix_id = ex.get("fixture")
                if fix_id:
                    if fix_id not in fixture_players:
                        fixture_players[fix_id] = {}
                    fix_bps = bps
                    for s in ex.get("stats", []):
                        if s.get("identifier") == "bps":
                            fix_bps = int(s.get("value", 0))
                            break
                    fixture_players[fix_id][eid] = fix_bps

        for fix in fixtures:
            fix_id = getattr(fix, "id", None)
            if fix_id is None or not getattr(fix, "started", False):
                continue

            has_official_bonus = any(
                getattr(s, "identifier", "") == "bonus" and (getattr(s, "a", None) or getattr(s, "h", None))
                for s in getattr(fix, "stats", [])
            )
            if has_official_bonus and getattr(fix, "finished", False):
                continue

            players_bps = fixture_players.get(fix_id, {})
            provisional = compute_fixture_provisional_bonus(players_bps)
            for peid, pbonus in provisional.items():
                bonus_by_element[peid] = bonus_by_element.get(peid, 0) + pbonus

        return bonus_by_element

    def resolve_autosubs(
        self,
        picks: list[Any],
        scores_by_element: dict[int, LivePlayerScore],
        active_chip: str | None = None,
    ) -> tuple[list[LivePlayerScore], list[LivePlayerScore], list[LiveAutosub], int]:
        """
        Executes official FPL provisional autosubstitution and captain handoff rules.
        """
        starters_picks = [p for p in picks if getattr(p, "position", 1) <= 11]
        bench_picks = [p for p in picks if getattr(p, "position", 1) > 11]

        captain = next((p for p in picks if getattr(p, "is_captain", False)), None)
        vice = next((p for p in picks if getattr(p, "is_vice_captain", False)), None)

        autosubs: list[LiveAutosub] = []
        effective_cap_id = getattr(captain, "element", 0) if captain else 0

        # Bench Boost: all 15 are active starters
        if active_chip == "bboost":
            all_scores = [
                scores_by_element[getattr(p, "element", 0)]
                for p in picks
                if getattr(p, "element", 0) in scores_by_element
            ]
            return all_scores, [], [], effective_cap_id

        # 1. Captaincy Rollover conforming to official FPL rules:
        # If captain finished match with 0 minutes, armband passes to vice-captain.
        # Dual-Zero Rule: If BOTH captain and vice-captain finished match with 0 minutes,
        # NO player gets double points (captain multiplier is lost). Autosubs replace unplayed
        # players at 1x points; an autosub coming on for the captain NEVER inherits the 2x multiplier.
        if captain and vice:
            cap_eid = getattr(captain, "element", 0)
            vice_eid = getattr(vice, "element", 0)
            cap_score = scores_by_element.get(cap_eid)
            vice_score = scores_by_element.get(vice_eid)
            if cap_score and cap_score.status == "FINISHED" and cap_score.minutes == 0:
                if vice_score and vice_score.status == "FINISHED" and vice_score.minutes == 0:
                    effective_cap_id = 0
                    autosubs.append(
                        LiveAutosub(
                            out_element_id=cap_eid,
                            out_web_name=cap_score.web_name,
                            in_element_id=0,
                            in_web_name="None (Multiplier Lost)",
                            sub_type="CAPTAIN_LOST",
                            reason="Both Captain and Vice-Captain finished match with 0 minutes. Captain multiplier lost.",
                        )
                    )
                else:
                    effective_cap_id = vice_eid
                    vice_name = vice_score.web_name if vice_score else getattr(cap_score, "web_name", "Vice-Captain")
                    autosubs.append(
                        LiveAutosub(
                            out_element_id=cap_eid,
                            out_web_name=cap_score.web_name,
                            in_element_id=vice_eid,
                            in_web_name=vice_name,
                            sub_type="CAPTAIN_PASS",
                            reason="Captain finished match with 0 minutes. Armband passed to Vice-Captain.",
                        )
                    )

        current_xi = [
            scores_by_element[getattr(s, "element", 0)]
            for s in starters_picks
            if getattr(s, "element", 0) in scores_by_element
        ]
        bench_players = [
            scores_by_element[getattr(b, "element", 0)]
            for b in bench_picks
            if getattr(b, "element", 0) in scores_by_element
        ]

        bench_gk = [b for b in bench_players if b.position == "GKP"]
        bench_outfield = [b for b in bench_players if b.position != "GKP"]

        subbed_in_ids: set[int] = set()
        subbed_out_players: list[LivePlayerScore] = []
        final_xi = list(current_xi)

        for starter in current_xi:
            if starter.status == "FINISHED" and starter.minutes == 0:
                if starter.position == "GKP":
                    if bench_gk and bench_gk[0].element_id not in subbed_in_ids:
                        bgk = bench_gk[0]
                        if bgk.minutes > 0 or bgk.status != "FINISHED":
                            subbed_in_ids.add(bgk.element_id)
                            subbed_out_players.append(starter)
                            final_xi.remove(starter)
                            final_xi.append(bgk)
                            autosubs.append(
                                LiveAutosub(
                                    out_element_id=starter.element_id,
                                    out_web_name=starter.web_name,
                                    in_element_id=bgk.element_id,
                                    in_web_name=bgk.web_name,
                                    sub_type="GOALKEEPER",
                                    reason="Starting Goalkeeper did not play.",
                                )
                            )
                else:
                    for b_out in bench_outfield:
                        if b_out.element_id in subbed_in_ids:
                            continue
                        if b_out.minutes == 0 and b_out.status == "FINISHED":
                            continue

                        temp_xi = [p for p in final_xi if p.element_id != starter.element_id] + [b_out]
                        counts = {"DEF": 0, "MID": 0, "FWD": 0}
                        for p in temp_xi:
                            if p.position in counts:
                                counts[p.position] += 1

                        if 3 <= counts["DEF"] <= 5 and 2 <= counts["MID"] <= 5 and 1 <= counts["FWD"] <= 3:
                            subbed_in_ids.add(b_out.element_id)
                            subbed_out_players.append(starter)
                            final_xi.remove(starter)
                            final_xi.append(b_out)
                            autosubs.append(
                                LiveAutosub(
                                    out_element_id=starter.element_id,
                                    out_web_name=starter.web_name,
                                    in_element_id=b_out.element_id,
                                    in_web_name=b_out.web_name,
                                    sub_type="OUTFIELD",
                                    reason=f"Starter did not play; formation valid ({counts['DEF']}-{counts['MID']}-{counts['FWD']}).",
                                )
                            )
                            break

        final_bench = [p for p in bench_players if p.element_id not in subbed_in_ids] + subbed_out_players
        return final_xi, final_bench, autosubs, effective_cap_id

    def calculate_safety_score(
        self,
        manager_squads: dict[int, LiveManagerSquad],
        user_manager_id: int | None = None,
    ) -> LiveSafetyScore | None:
        """
        Calculates Ragabolly/LiveFPL-style matchday Safety Score and projected arrow movement.
        """
        if not manager_squads:
            return None

        element_mults: dict[int, float] = {}
        element_names: dict[int, str] = {}
        total_squads = len(manager_squads)
        total_net_points = 0.0

        for squad in manager_squads.values():
            total_net_points += squad.live_gw_net
            for p in squad.starters:
                mult = (
                    3.0
                    if squad.active_chip == "3xc"
                    and squad.effective_captain_id > 0
                    and p.element_id == squad.effective_captain_id
                    else (2.0 if squad.effective_captain_id > 0 and p.element_id == squad.effective_captain_id else 1.0)
                )
                element_mults[p.element_id] = element_mults.get(p.element_id, 0.0) + mult
                element_names[p.element_id] = p.web_name

        avg_score = round(total_net_points / max(1, total_squads), 1)

        eo_table: dict[str, float] = {}
        sorted_eos = sorted(element_mults.items(), key=lambda x: x[1], reverse=True)
        for eid, total_m in sorted_eos[:5]:
            name = element_names.get(eid, f"Element_{eid}")
            eo_pct = round((total_m / total_squads) * 100.0, 1)
            eo_table[name] = eo_pct

        user_squad = manager_squads.get(user_manager_id) if user_manager_id else None
        user_net = user_squad.live_gw_net if user_squad else int(avg_score)
        gap = round(user_net - avg_score, 1)

        if gap >= 2.0:
            arrow = "GREEN_ARROW"
            summary = f"Safety buffer +{gap:.1f} pts above safety threshold ({avg_score:.1f}): Projected green arrow."
        elif gap <= -2.0:
            arrow = "RED_ARROW"
            summary = f"Deficit of {abs(gap):.1f} pts below safety threshold ({avg_score:.1f}): At risk of red arrow."
        else:
            arrow = "GREY_ARROW"
            summary = f"Parity boundary ({gap:+.1f} pts vs safety threshold {avg_score:.1f}): Rank holds steady."

        return LiveSafetyScore(
            safety_score=avg_score,
            user_live_net_points=user_net,
            safety_gap=gap,
            projected_arrow=arrow,
            field_effective_ownership=eo_table,
            benchmark_source=f"League Pool (N={total_squads})",
            summary_text=summary,
        )

    async def get_live_matchday_standings(
        self,
        league_id: int,
        gameweek: int,
        user_manager_id: int | None = None,
        max_entries: int = 50,
    ) -> LiveMatchdaySnapshot:
        """
        Orchestrates full matchday live pipeline:
        1. Ingests live gameweek stats and fixtures.
        2. Calculates provisional BPS bonus.
        3. Simulates autosubs for user and all mini-league rivals.
        4. Re-ranks classic league table in real time.
        """
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures(event=gameweek)
        live_data, _ = await fpl_client.get_live_gameweek(gameweek)

        elem_map = {e.id: e for e in boot.elements}
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

        fixtures_summary = {
            "total": len(fixtures),
            "finished": sum(1 for f in fixtures if getattr(f, "finished", False)),
            "in_progress": sum(
                1 for f in fixtures if getattr(f, "started", False) and not getattr(f, "finished", False)
            ),
            "unplayed": sum(1 for f in fixtures if not getattr(f, "started", False)),
        }

        # Match status by team
        team_fixture_status: dict[int, str] = {}
        for f in fixtures:
            t_h = getattr(f, "team_h", 0)
            t_a = getattr(f, "team_a", 0)
            st = (
                "FINISHED"
                if getattr(f, "finished", False)
                else ("PLAYING" if getattr(f, "started", False) else "YET_TO_PLAY")
            )
            team_fixture_status[t_h] = st
            team_fixture_status[t_a] = st

        live_elements = live_data.get("elements", [])
        provisional_bonuses = self.compute_fixture_bonuses(fixtures, live_elements)

        scores_by_element: dict[int, LivePlayerScore] = {}
        for item in live_elements:
            eid = int(item.get("id", 0))
            stats = item.get("stats", {})
            elem = elem_map.get(eid)
            t_id = getattr(elem, "team", 0) if elem else 0
            pos_id = getattr(elem, "element_type", 3) if elem else 3
            name = getattr(elem, "web_name", f"Player {eid}") if elem else f"Player {eid}"

            mins = int(stats.get("minutes", 0) or 0)
            bps = int(stats.get("bps", 0) or 0)
            off_bonus = int(stats.get("bonus", 0) or 0)
            prov_bonus = provisional_bonuses.get(eid, 0)
            base_pts = int(stats.get("total_points", 0) or 0)

            # Add provisional bonus if official bonus not yet applied
            added_bonus = prov_bonus if off_bonus == 0 else 0
            total_live = base_pts + added_bonus

            match_st = team_fixture_status.get(t_id, "FINISHED" if mins > 0 else "YET_TO_PLAY")
            if mins > 0 and match_st == "YET_TO_PLAY":
                match_st = "PLAYING"

            scores_by_element[eid] = LivePlayerScore(
                element_id=eid,
                web_name=name,
                position=pos_map.get(pos_id, "MID"),
                team_id=t_id,
                minutes=mins,
                goals=int(stats.get("goals_scored", 0) or 0),
                assists=int(stats.get("assists", 0) or 0),
                clean_sheets=int(stats.get("clean_sheets", 0) or 0),
                goals_conceded=int(stats.get("goals_conceded", 0) or 0),
                bps=bps,
                official_bonus=off_bonus,
                provisional_bonus=prov_bonus,
                live_points=total_live,
                status=match_st,
            )

        # Standings ingestion
        standings_data, _ = await fpl_client.get_classic_league_standings(league_id)
        if hasattr(standings_data, "model_dump"):
            standings_dict = standings_data.model_dump(mode="json")
        elif isinstance(standings_data, dict):
            standings_dict = standings_data
        else:
            standings_dict = {}

        league_name = standings_dict.get("league", {}).get("name", f"League #{league_id}")
        raw_rows = standings_dict.get("standings", {}).get("results", [])[:max_entries]

        manager_squads: dict[int, LiveManagerSquad] = {}
        league_rows: list[LiveLeagueRow] = []

        for row in raw_rows:
            m_id = int(row.get("entry", 0))
            m_name = str(row.get("player_name", f"Manager {m_id}"))
            t_name = str(row.get("entry_name", f"Team {m_id}"))
            official_total = int(row.get("total", 0) or 0)
            official_event = int(row.get("event_total", 0) or 0)
            prior_rank = int(row.get("last_rank", row.get("rank", 1)) or 1)

            # Fetch picks
            picks_data: dict[str, Any] = {}
            try:
                picks_data, _ = await fpl_client.get_manager_picks(m_id, gameweek)
            except Exception as exc:
                logger.debug("Could not fetch picks for manager %s: %s", m_id, exc)

            if hasattr(picks_data, "model_dump"):
                picks_dict = picks_data.model_dump(mode="json")
            elif isinstance(picks_data, dict):
                picks_dict = picks_data
            else:
                picks_dict = {}

            picks = picks_dict.get("picks", [])
            active_chip = picks_dict.get("active_chip")
            hits_cost = int(picks_dict.get("entry_history", {}).get("event_transfers_cost", 0) or 0)

            final_xi, final_bench, autosubs, eff_cap_id = self.resolve_autosubs(
                picks=picks,
                scores_by_element=scores_by_element,
                active_chip=active_chip,
            )

            # Calculate Live Gameweek Points
            gross_pts = 0
            players_played = 0
            players_playing = 0
            players_rem = 0

            cap_multiplier = 3 if active_chip == "3xc" else 2

            for p in final_xi:
                mult = cap_multiplier if (eff_cap_id > 0 and p.element_id == eff_cap_id) else 1
                gross_pts += p.live_points * mult
                if p.status == "FINISHED":
                    players_played += 1
                elif p.status == "PLAYING":
                    players_playing += 1
                else:
                    players_rem += 1

            net_pts = gross_pts - hits_cost
            prior_total = official_total if official_event == 0 else (official_total - official_event + hits_cost)
            live_total = prior_total + net_pts

            cap_player = scores_by_element.get(eff_cap_id)
            cap_name = (
                cap_player.web_name
                if (eff_cap_id > 0 and cap_player)
                else ("None (Multiplier Lost)" if eff_cap_id == 0 else "Captain")
            )

            squad_obj = LiveManagerSquad(
                manager_id=m_id,
                manager_name=m_name,
                team_name=t_name,
                active_chip=active_chip,
                transfer_cost=hits_cost,
                live_gw_gross=gross_pts,
                live_gw_net=net_pts,
                live_total_points=live_total,
                starters=final_xi,
                bench=final_bench,
                autosubs=autosubs,
                effective_captain_id=eff_cap_id,
                players_played=players_played,
                players_playing=players_playing,
                players_remaining=players_rem,
            )
            manager_squads[m_id] = squad_obj

            league_rows.append(
                LiveLeagueRow(
                    live_rank=1,
                    previous_rank=prior_rank,
                    rank_movement=0,
                    manager_id=m_id,
                    manager_name=m_name,
                    team_name=t_name,
                    live_gw_points=net_pts,
                    live_total_points=live_total,
                    gap_to_leader=0,
                    players_remaining=players_rem,
                    captain_name=cap_name,
                    active_chip=active_chip,
                )
            )

        # Sort by live total points descending
        league_rows.sort(key=lambda r: (-r.live_total_points, -r.live_gw_points, r.manager_name))
        leader_total = league_rows[0].live_total_points if league_rows else 0

        user_live_rank = None
        user_live_total = None

        for idx, row in enumerate(league_rows, start=1):
            row.live_rank = idx
            row.rank_movement = row.previous_rank - row.live_rank
            row.gap_to_leader = max(0, leader_total - row.live_total_points)
            if user_manager_id and row.manager_id == user_manager_id:
                user_live_rank = idx
                user_live_total = row.live_total_points

        user_squad = manager_squads.get(user_manager_id) if user_manager_id else None
        safety_obj = self.calculate_safety_score(manager_squads, user_manager_id=user_manager_id)

        return LiveMatchdaySnapshot(
            gameweek=gameweek,
            league_id=league_id,
            league_name=league_name,
            user_manager_id=user_manager_id,
            user_live_rank=user_live_rank,
            user_live_total=user_live_total,
            standings=league_rows,
            user_squad=user_squad,
            fixtures_summary=fixtures_summary,
            is_live=fixtures_summary.get("in_progress", 0) > 0 or fixtures_summary.get("finished", 0) > 0,
            safety_score=safety_obj,
        )


live_matchday_engine = LiveMatchdayEngine()
