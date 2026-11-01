"""
Unit tests for FPL Rule Polyhedra, Complex Autosubs, Formation Legality & Captaincy Cascades.

Verifies:
1. Formation-Locked Autosub Skips (The 3-DEF Rule and formation polyhedron bounds):
   - In 3-DEF formations (3-5-2, 3-4-3), a non-playing starting defender forces skipping
     Sub 1 (MID/FWD) to promote the first eligible bench DEF.
   - Upper bounds (max 5 DEF, max 5 MID, max 3 FWD) strictly enforced on all autosub substitutions.
   - Unreplacable starters stay at 0 points without breaking formation legality.
2. Dual-Zero Captaincy & Autosub Cascades:
   - When Captain plays 0 minutes and Vice-Captain plays > 0 minutes: Vice-Captain receives 2x points.
   - When BOTH Captain and Vice-Captain play 0 minutes:
     * Multiplier is completely lost (0x extra captain points).
     * Autosubs replace both Captain and Vice-Captain at 1x points.
     * Autosubs NEVER inherit the captain multiplier.
     * Triple Captain chip burned with no multiplier applied (1x).
3. Bench Ordering Advice in lineup.py:
   - 3-DEF formation with rotation risk prioritizes bench DEF at Sub 1 over higher-xP attackers.
   - Structured bench ordering advice returned with locked minimum positions and reasoning.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fpl_oracle.league.live_matchday import (
    LiveMatchdayEngine,
    LivePlayerScore,
)
from fpl_oracle.optimise.lineup import LineupOptimizer, _order_bench
from fpl_oracle.sim.match_sim import SimulationResult

# ==============================================================================
# 1. Formation-Locked Autosub Skips (The 3-DEF Rule & Polyhedron Invariants)
# ==============================================================================


def test_3_def_rule_skips_mid_and_promotes_def_in_352():
    """
    In a 3-5-2 starting formation:
    If 1 starting DEF plays 0 minutes:
    Sub 1 (MID with 12 pts) CANNOT come on because the team would drop to 2 DEF!
    Sub 1 MUST be skipped and Sub 2 (DEF with 2 pts) brought on.
    """
    engine = LiveMatchdayEngine()

    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="GKP",
            position="GKP",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        2: LivePlayerScore(
            element_id=2,
            web_name="DEF 1",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        3: LivePlayerScore(
            element_id=3,
            web_name="DEF 2",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        4: LivePlayerScore(
            element_id=4,
            web_name="DEF Absent",
            position="DEF",
            team_id=1,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
    }
    # 5 MIDs (elements 5-9)
    for i in range(5, 10):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"MID {i}",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=3,
            status="FINISHED",
        )
    # 2 FWDs (elements 10-11)
    for i in range(10, 12):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"FWD {i}",
            position="FWD",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=2,
            status="FINISHED",
        )

    # Bench:
    # 12: GKP
    # 13: Sub 1 - MID (played, 12 points haul!)
    # 14: Sub 2 - DEF (played, 2 points)
    # 15: Sub 3 - FWD (played, 5 points)
    scores[12] = LivePlayerScore(
        element_id=12,
        web_name="Bench GKP",
        position="GKP",
        team_id=3,
        minutes=0,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=0,
        official_bonus=0,
        provisional_bonus=0,
        live_points=0,
        status="FINISHED",
    )
    scores[13] = LivePlayerScore(
        element_id=13,
        web_name="Sub 1 MID",
        position="MID",
        team_id=3,
        minutes=90,
        goals=1,
        assists=1,
        clean_sheets=0,
        goals_conceded=0,
        bps=32,
        official_bonus=3,
        provisional_bonus=0,
        live_points=12,
        status="FINISHED",
    )
    scores[14] = LivePlayerScore(
        element_id=14,
        web_name="Sub 2 DEF",
        position="DEF",
        team_id=3,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=12,
        official_bonus=0,
        provisional_bonus=0,
        live_points=2,
        status="FINISHED",
    )
    scores[15] = LivePlayerScore(
        element_id=15,
        web_name="Sub 3 FWD",
        position="FWD",
        team_id=3,
        minutes=90,
        goals=1,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=20,
        official_bonus=0,
        provisional_bonus=0,
        live_points=5,
        status="FINISHED",
    )

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 5), is_vice_captain=(i == 6)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, cap_id = engine.resolve_autosubs(picks, scores)
    xi_ids = {p.element_id for p in final_xi}

    assert 14 in xi_ids, "Sub 2 DEF must be brought in to preserve minimum 3 defenders"
    assert 13 not in xi_ids, "Sub 1 MID must be skipped to avoid illegal 2-DEF formation"
    assert 4 not in xi_ids, "Unplayed starter DEF 4 must be subbed off"
    assert len(autosubs) == 1
    assert autosubs[0].in_element_id == 14
    assert autosubs[0].sub_type == "OUTFIELD"


def test_3_def_rule_skips_fwd_and_promotes_def_in_343():
    """
    In a 3-4-3 starting formation:
    If 1 starting DEF plays 0 minutes:
    Sub 1 is a FWD (with 10 pts).
    Bringing in FWD would drop DEF to 2 and raise FWD to 4 (both illegal!).
    Sub 1 MUST be skipped, and Sub 2 (DEF) brought on.
    """
    engine = LiveMatchdayEngine()

    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="GKP",
            position="GKP",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        2: LivePlayerScore(
            element_id=2,
            web_name="DEF 1",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        3: LivePlayerScore(
            element_id=3,
            web_name="DEF 2",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        4: LivePlayerScore(
            element_id=4,
            web_name="DEF Absent",
            position="DEF",
            team_id=1,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
    }
    # 4 MIDs (elements 5-8)
    for i in range(5, 9):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"MID {i}",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=3,
            status="FINISHED",
        )
    # 3 FWDs (elements 9-11)
    for i in range(9, 12):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"FWD {i}",
            position="FWD",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=2,
            status="FINISHED",
        )

    # Bench:
    # 12: GKP
    # 13: Sub 1 - FWD (played, 10 pts)
    # 14: Sub 2 - DEF (played, 2 pts)
    # 15: Sub 3 - MID (played, 3 pts)
    scores[12] = LivePlayerScore(
        element_id=12,
        web_name="Bench GKP",
        position="GKP",
        team_id=3,
        minutes=0,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=0,
        official_bonus=0,
        provisional_bonus=0,
        live_points=0,
        status="FINISHED",
    )
    scores[13] = LivePlayerScore(
        element_id=13,
        web_name="Sub 1 FWD",
        position="FWD",
        team_id=3,
        minutes=90,
        goals=1,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=25,
        official_bonus=0,
        provisional_bonus=0,
        live_points=10,
        status="FINISHED",
    )
    scores[14] = LivePlayerScore(
        element_id=14,
        web_name="Sub 2 DEF",
        position="DEF",
        team_id=3,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=10,
        official_bonus=0,
        provisional_bonus=0,
        live_points=2,
        status="FINISHED",
    )
    scores[15] = LivePlayerScore(
        element_id=15,
        web_name="Sub 3 MID",
        position="MID",
        team_id=3,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=10,
        official_bonus=0,
        provisional_bonus=0,
        live_points=3,
        status="FINISHED",
    )

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 5), is_vice_captain=(i == 6)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, cap_id = engine.resolve_autosubs(picks, scores)
    xi_ids = {p.element_id for p in final_xi}

    assert 14 in xi_ids, "Sub 2 DEF must be brought in"
    assert 13 not in xi_ids, "Sub 1 FWD must be skipped (would create 2 DEF and 4 FWD)"
    def_count = sum(1 for p in final_xi if p.position == "DEF")
    fwd_count = sum(1 for p in final_xi if p.position == "FWD")
    assert def_count >= 3
    assert fwd_count <= 3


def test_formation_max_bounds_enforced_in_live_matchday():
    """
    Starting formation: 4-5-1 (4 DEF, 5 MID, 1 FWD).
    DEF 4 misses out (0 mins).
    Bench:
    Sub 1: MID (played, 8 pts)
    Sub 2: DEF (played, 2 pts)
    Bringing on Sub 1 (MID) would create 6 MID (illegal in FPL: max is 5).
    Sub 1 MUST be skipped, and Sub 2 (DEF) brought on.
    """
    engine = LiveMatchdayEngine()

    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="GKP",
            position="GKP",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        2: LivePlayerScore(
            element_id=2,
            web_name="DEF 1",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        3: LivePlayerScore(
            element_id=3,
            web_name="DEF 2",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        4: LivePlayerScore(
            element_id=4,
            web_name="DEF 3",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        5: LivePlayerScore(
            element_id=5,
            web_name="DEF Absent",
            position="DEF",
            team_id=1,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
    }
    # 5 MIDs (elements 6-10)
    for i in range(6, 11):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"MID {i}",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=3,
            status="FINISHED",
        )
    # 1 FWD (element 11)
    scores[11] = LivePlayerScore(
        element_id=11,
        web_name="FWD 11",
        position="FWD",
        team_id=2,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=10,
        official_bonus=0,
        provisional_bonus=0,
        live_points=2,
        status="FINISHED",
    )

    # Bench:
    # 12: GKP
    # 13: Sub 1 - MID (played, 8 pts)
    # 14: Sub 2 - DEF (played, 2 pts)
    scores[12] = LivePlayerScore(
        element_id=12,
        web_name="Bench GKP",
        position="GKP",
        team_id=3,
        minutes=0,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=0,
        official_bonus=0,
        provisional_bonus=0,
        live_points=0,
        status="FINISHED",
    )
    scores[13] = LivePlayerScore(
        element_id=13,
        web_name="Sub 1 MID",
        position="MID",
        team_id=3,
        minutes=90,
        goals=1,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=25,
        official_bonus=0,
        provisional_bonus=0,
        live_points=8,
        status="FINISHED",
    )
    scores[14] = LivePlayerScore(
        element_id=14,
        web_name="Sub 2 DEF",
        position="DEF",
        team_id=3,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=10,
        official_bonus=0,
        provisional_bonus=0,
        live_points=2,
        status="FINISHED",
    )
    scores[15] = LivePlayerScore(
        element_id=15,
        web_name="Sub 3 FWD",
        position="FWD",
        team_id=3,
        minutes=0,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=0,
        official_bonus=0,
        provisional_bonus=0,
        live_points=0,
        status="FINISHED",
    )

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 6), is_vice_captain=(i == 7)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, cap_id = engine.resolve_autosubs(picks, scores)
    xi_ids = {p.element_id for p in final_xi}

    assert 14 in xi_ids, "Sub 2 DEF must be brought in"
    assert 13 not in xi_ids, "Sub 1 MID must be skipped to avoid illegal 6-MID formation"
    mid_count = sum(1 for p in final_xi if p.position == "MID")
    assert mid_count <= 5


def test_match_sim_3_def_rule_skips_mid_in_vectorized_draws():
    """
    SimulationResult.get_squad_score_distribution strictly skips Sub 1 (MID)
    and promotes Sub 2 (DEF) when a 3-DEF formation has an unplayed defender.
    """
    n_sims = 10
    n_players = 15
    player_ids = np.arange(1, 16)
    positions = np.array(
        ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "GKP", "MID", "DEF", "FWD"]
    )

    # Starters: 1-11 (1 GKP, 3 DEF [2, 3, 4], 5 MID [5-9], 2 FWD [10-11])
    # Bench: 12 (GKP), 13 (Sub 1 MID), 14 (Sub 2 DEF), 15 (Sub 3 FWD)
    minutes = np.full((n_players, n_sims), 90)
    # Player 4 (DEF 3) plays 0 minutes across all sims
    minutes[3, :] = 0

    scores = np.zeros((n_players, n_sims))
    # Give Sub 1 (MID, id 13) 10 points per sim
    scores[12, :] = 10.0
    # Give Sub 2 (DEF, id 14) 2 points per sim
    scores[13, :] = 2.0
    # Give other starters 3 points each
    for i in range(n_players):
        if i not in (3, 12, 13):
            scores[i, :] = 3.0

    res = SimulationResult(
        scores=scores,
        player_ids=player_ids,
        player_names=[f"P_{i}" for i in player_ids],
        positions=positions,
        teams=np.ones(n_players, dtype=int),
        n_sims=n_sims,
        analytic_xp=np.zeros(n_players),
        minutes=minutes,
        goals=np.zeros((n_players, n_sims)),
        assists=np.zeros((n_players, n_sims)),
        clean_sheets=np.zeros((n_players, n_sims)),
        goals_conceded=np.zeros((n_players, n_sims)),
        saves=np.zeros((n_players, n_sims)),
        defcon=np.zeros((n_players, n_sims)),
        bonus=np.zeros((n_players, n_sims)),
        cards=np.zeros((n_players, n_sims)),
        own_goals=np.zeros((n_players, n_sims)),
        penalties_missed=np.zeros((n_players, n_sims)),
        penalties_saved=np.zeros((n_players, n_sims)),
    )

    starter_ids = list(range(1, 12))
    bench_ids = [12, 13, 14, 15]

    squad_scores = res.get_squad_score_distribution(
        starter_ids=starter_ids,
        bench_ids=bench_ids,
        captain_id=5,
        vice_id=6,
    )

    # Calculation:
    # 10 playing starters with 3 pts = 30 pts
    # Captain (id 5) plays 90 mins: +3 pts extra = 33 pts
    # Unplayed DEF (id 4) = 0 pts
    # Sub 1 (id 13, MID) CANNOT come on (team would have 2 DEF).
    # Sub 2 (id 14, DEF) COMES ON with 2 pts.
    # Total squad score per sim = 33 + 2 = 35.0.
    # If Sub 1 (MID) had come on illegally, score would be 33 + 10 = 43.0.
    assert np.allclose(squad_scores, 35.0), f"Expected 35.0 (DEF subbed in), got {squad_scores}"


# ==============================================================================
# 2. Dual-Zero Captaincy & Autosub Cascades
# ==============================================================================


def test_dual_zero_captaincy_in_match_sim_multiplier_lost():
    """
    When BOTH Captain AND Vice-Captain play 0 minutes:
    - FPL rule: No player receives double points (multiplier is completely lost).
    - Autosubs replace unplayed players at 1x points.
    - An autosub coming on NEVER inherits the 2x multiplier!
    """
    n_sims = 5
    n_players = 15
    player_ids = np.arange(1, 16)
    positions = np.array(
        ["GKP", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD", "FWD", "FWD", "GKP", "MID", "MID", "DEF"]
    )

    minutes = np.full((n_players, n_sims), 90)
    # Captain (id 6) and Vice-Captain (id 7) BOTH play 0 minutes
    minutes[5, :] = 0  # id 6 (index 5)
    minutes[6, :] = 0  # id 7 (index 6)

    scores = np.full((n_players, n_sims), 4.0)
    scores[5, :] = 0.0  # id 6 (C)
    scores[6, :] = 0.0  # id 7 (VC)
    # Bench: id 13 (Sub 1 MID, 6 pts), id 14 (Sub 2 MID, 5 pts)
    scores[12, :] = 6.0
    scores[13, :] = 5.0

    res = SimulationResult(
        scores=scores,
        player_ids=player_ids,
        player_names=[f"P_{i}" for i in player_ids],
        positions=positions,
        teams=np.ones(n_players, dtype=int),
        n_sims=n_sims,
        analytic_xp=np.zeros(n_players),
        minutes=minutes,
        goals=np.zeros((n_players, n_sims)),
        assists=np.zeros((n_players, n_sims)),
        clean_sheets=np.zeros((n_players, n_sims)),
        goals_conceded=np.zeros((n_players, n_sims)),
        saves=np.zeros((n_players, n_sims)),
        defcon=np.zeros((n_players, n_sims)),
        bonus=np.zeros((n_players, n_sims)),
        cards=np.zeros((n_players, n_sims)),
        own_goals=np.zeros((n_players, n_sims)),
        penalties_missed=np.zeros((n_players, n_sims)),
        penalties_saved=np.zeros((n_players, n_sims)),
    )

    starter_ids = list(range(1, 12))
    bench_ids = [12, 13, 14, 15]

    total_scores = res.get_squad_score_distribution(
        starter_ids=starter_ids,
        bench_ids=bench_ids,
        captain_id=6,
        vice_id=7,
    )

    # 9 starters who played 90 mins with 4 pts each = 36 pts
    # C (id 6) = 0 pts
    # VC (id 7) = 0 pts
    # Captain multiplier = 0 (lost because both played 0 mins!)
    # Autosubs: Sub 1 (id 13, MID) comes on for C with 6 pts (1x)
    #           Sub 2 (id 14, MID) comes on for VC with 5 pts (1x)
    # Expected total = 36 + 6 + 5 = 47.0.
    # If Sub 1 had inherited 2x, total would be 36 + 12 + 5 = 53.0.
    assert np.allclose(total_scores, 47.0), f"Expected 47.0 (no 2x multiplier), got {total_scores}"


def test_dual_zero_captaincy_in_live_matchday():
    """
    When Captain and Vice-Captain both finish with 0 minutes in Live Matchday:
    - effective_cap_id becomes 0.
    - CAPTAIN_LOST autosub event is recorded.
    - All players in final XI score at 1x multiplier.
    - Autosubs replace both C and VC at 1x points.
    """
    engine = LiveMatchdayEngine()

    # Standard valid 4-4-2 starting lineup
    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="GKP",
            position="GKP",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        2: LivePlayerScore(
            element_id=2,
            web_name="DEF 1",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        3: LivePlayerScore(
            element_id=3,
            web_name="DEF 2",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        4: LivePlayerScore(
            element_id=4,
            web_name="DEF 3",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        5: LivePlayerScore(
            element_id=5,
            web_name="DEF 4",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        6: LivePlayerScore(
            element_id=6,
            web_name="MID 1",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=3,
            status="FINISHED",
        ),
        7: LivePlayerScore(
            element_id=7,
            web_name="MID 2",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=3,
            status="FINISHED",
        ),
        # Captain (8) and Vice-Captain (9) both finish with 0 minutes
        8: LivePlayerScore(
            element_id=8,
            web_name="Captain Benched",
            position="MID",
            team_id=2,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        9: LivePlayerScore(
            element_id=9,
            web_name="Vice Benched",
            position="MID",
            team_id=2,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        10: LivePlayerScore(
            element_id=10,
            web_name="FWD 1",
            position="FWD",
            team_id=3,
            minutes=90,
            goals=1,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=24,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
        11: LivePlayerScore(
            element_id=11,
            web_name="FWD 2",
            position="FWD",
            team_id=3,
            minutes=90,
            goals=0,
            assists=1,
            clean_sheets=0,
            goals_conceded=0,
            bps=18,
            official_bonus=0,
            provisional_bonus=0,
            live_points=5,
            status="FINISHED",
        ),
        # Bench
        12: LivePlayerScore(
            element_id=12,
            web_name="Bench GKP",
            position="GKP",
            team_id=1,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        13: LivePlayerScore(
            element_id=13,
            web_name="Sub 1 MID",
            position="MID",
            team_id=2,
            minutes=90,
            goals=1,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=25,
            official_bonus=0,
            provisional_bonus=0,
            live_points=7,
            status="FINISHED",
        ),
        14: LivePlayerScore(
            element_id=14,
            web_name="Sub 2 MID",
            position="MID",
            team_id=2,
            minutes=90,
            goals=1,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=25,
            official_bonus=0,
            provisional_bonus=0,
            live_points=7,
            status="FINISHED",
        ),
        15: LivePlayerScore(
            element_id=15,
            web_name="Sub 3 DEF",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=6,
            status="FINISHED",
        ),
    }

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 8), is_vice_captain=(i == 9)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, eff_cap_id = engine.resolve_autosubs(picks, scores)

    assert eff_cap_id == 0, "effective_cap_id must be 0 when both C and VC play 0 minutes"
    lost_event = next((a for a in autosubs if a.sub_type == "CAPTAIN_LOST"), None)
    assert lost_event is not None, "A CAPTAIN_LOST autosub event must be recorded"
    assert lost_event.in_element_id == 0

    # Both bench players (13 and 14) are in final XI replacing 8 and 9
    xi_ids = {p.element_id for p in final_xi}
    assert 13 in xi_ids
    assert 14 in xi_ids
    assert 8 not in xi_ids
    assert 9 not in xi_ids


def test_triple_captain_dual_zero_chip_burned():
    """
    If Triple Captain chip is active and both C and VC play 0 minutes:
    No player gets 3x points; chip is burned, and all active players score at 1x.
    """
    engine = LiveMatchdayEngine()

    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="GKP",
            position="GKP",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=20,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        2: LivePlayerScore(
            element_id=2,
            web_name="DEF 1",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        3: LivePlayerScore(
            element_id=3,
            web_name="DEF 2",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        4: LivePlayerScore(
            element_id=4,
            web_name="DEF 3",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        5: LivePlayerScore(
            element_id=5,
            web_name="DEF 4",
            position="DEF",
            team_id=1,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        6: LivePlayerScore(
            element_id=6,
            web_name="MID 1",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        7: LivePlayerScore(
            element_id=7,
            web_name="MID 2",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        # Captain (8) and Vice (9) both finish with 0 mins
        8: LivePlayerScore(
            element_id=8,
            web_name="Captain Benched",
            position="MID",
            team_id=2,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        9: LivePlayerScore(
            element_id=9,
            web_name="Vice Benched",
            position="MID",
            team_id=2,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        10: LivePlayerScore(
            element_id=10,
            web_name="FWD 1",
            position="FWD",
            team_id=3,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        11: LivePlayerScore(
            element_id=11,
            web_name="FWD 2",
            position="FWD",
            team_id=3,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        # Bench
        12: LivePlayerScore(
            element_id=12,
            web_name="Bench GKP",
            position="GKP",
            team_id=1,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="FINISHED",
        ),
        13: LivePlayerScore(
            element_id=13,
            web_name="Sub 1 MID",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        14: LivePlayerScore(
            element_id=14,
            web_name="Sub 2 MID",
            position="MID",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
        15: LivePlayerScore(
            element_id=15,
            web_name="Sub 3 FWD",
            position="FWD",
            team_id=3,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=10,
            official_bonus=0,
            provisional_bonus=0,
            live_points=4,
            status="FINISHED",
        ),
    }

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 8), is_vice_captain=(i == 9)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, eff_cap_id = engine.resolve_autosubs(picks, scores, active_chip="3xc")
    assert eff_cap_id == 0

    # Verify score calculation: no 3x multiplier applied to anyone
    cap_multiplier = 3
    gross_pts = 0
    for p in final_xi:
        mult = cap_multiplier if (eff_cap_id > 0 and p.element_id == eff_cap_id) else 1
        gross_pts += p.live_points * mult

    # 11 players in final_xi, each with 4 pts = 44 pts (no 3x bonus)
    assert gross_pts == 44


# ==============================================================================
# 3. Bench Ordering Advice in lineup.py
# ==============================================================================


def test_bench_ordering_prioritizes_def_eligibility_under_3_def_rotation_risk():
    """
    When starting in 3-5-2:
    Starting DEF has rotation risk (chance_of_playing = 50%).
    Bench contains:
    - Bench MID with high xP (6.0 xP)
    - Bench DEF with modest xP (3.0 xP)
    Bench DEF MUST be ordered ahead of Bench MID (Sub 1) because only Bench DEF
    can legally preserve formation eligibility if the flagged DEF does not feature.
    """
    starters_data = [
        {"element": 1, "position": "GKP", "expected_points": 4.0, "chance_of_playing": 100},
        {"element": 2, "position": "DEF", "expected_points": 4.0, "chance_of_playing": 100},
        {"element": 3, "position": "DEF", "expected_points": 4.0, "chance_of_playing": 100},
        # Flagged starter DEF: 50% chance of playing
        {"element": 4, "position": "DEF", "expected_points": 2.0, "chance_of_playing": 50},
        {"element": 5, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        {"element": 6, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        {"element": 7, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        {"element": 8, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        {"element": 9, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        {"element": 10, "position": "FWD", "expected_points": 5.0, "chance_of_playing": 100},
        {"element": 11, "position": "FWD", "expected_points": 5.0, "chance_of_playing": 100},
    ]

    bench_data = [
        {"element": 12, "position": "GKP", "expected_points": 0.0, "chance_of_playing": 100},
        # Bench MID has higher xP (6.0), but CANNOT legally sub in for flagged DEF in 3-5-2!
        {"element": 13, "position": "MID", "expected_points": 6.0, "chance_of_playing": 100},
        # Bench DEF has lower xP (3.0), but is the ONLY legal coverage!
        {"element": 14, "position": "DEF", "expected_points": 3.0, "chance_of_playing": 100},
        {"element": 15, "position": "FWD", "expected_points": 2.0, "chance_of_playing": 100},
    ]

    starters_df = pd.DataFrame(starters_data)
    bench_df = pd.DataFrame(bench_data)

    ordered = _order_bench(bench_df, starters_df)
    assert ordered.iloc[0]["position"] == "GKP", "GKP must be slot 0 on bench"
    # Sub 1 (slot 1) MUST be the DEF (element 14) to cover the formation minimum
    assert ordered.iloc[1]["element"] == 14, (
        f"Expected Sub 1 to be DEF 14, got {ordered.iloc[1]['element']} ({ordered.iloc[1]['position']})"
    )
    assert ordered.iloc[1]["position"] == "DEF"


def test_lineup_optimizer_returns_bench_ordering_advice():
    """
    LineupOptimizer.select_lineup_and_captain returns structured bench_ordering_advice
    identifying locked minimum requirements and explaining why position eligibility is prioritized.
    """
    squad_data = [
        {
            "element": 1,
            "web_name": "GKP 1",
            "position": "GKP",
            "expected_points": 4.5,
            "chance_of_playing": 100,
            "team": 1,
        },
        {
            "element": 2,
            "web_name": "DEF 1",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100,
            "team": 1,
        },
        {
            "element": 3,
            "web_name": "DEF 2",
            "position": "DEF",
            "expected_points": 4.8,
            "chance_of_playing": 100,
            "team": 2,
        },
        # Flagged starting defender
        {
            "element": 4,
            "web_name": "DEF 3",
            "position": "DEF",
            "expected_points": 4.0,
            "chance_of_playing": 50,
            "team": 2,
        },
        {
            "element": 5,
            "web_name": "MID 1",
            "position": "MID",
            "expected_points": 7.5,
            "chance_of_playing": 100,
            "team": 3,
        },
        {
            "element": 6,
            "web_name": "MID 2",
            "position": "MID",
            "expected_points": 7.0,
            "chance_of_playing": 100,
            "team": 3,
        },
        {
            "element": 7,
            "web_name": "MID 3",
            "position": "MID",
            "expected_points": 6.5,
            "chance_of_playing": 100,
            "team": 4,
        },
        {
            "element": 8,
            "web_name": "MID 4",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100,
            "team": 4,
        },
        {
            "element": 9,
            "web_name": "MID 5",
            "position": "MID",
            "expected_points": 5.5,
            "chance_of_playing": 100,
            "team": 5,
        },
        {
            "element": 10,
            "web_name": "FWD 1",
            "position": "FWD",
            "expected_points": 8.0,
            "chance_of_playing": 100,
            "team": 5,
        },
        {
            "element": 11,
            "web_name": "FWD 2",
            "position": "FWD",
            "expected_points": 6.0,
            "chance_of_playing": 100,
            "team": 6,
        },
        # Bench
        {
            "element": 12,
            "web_name": "GKP 2",
            "position": "GKP",
            "expected_points": 0.5,
            "chance_of_playing": 100,
            "team": 6,
        },
        {
            "element": 13,
            "web_name": "DEF 4",
            "position": "DEF",
            "expected_points": 3.0,
            "chance_of_playing": 100,
            "team": 7,
        },
        {
            "element": 14,
            "web_name": "DEF 5",
            "position": "DEF",
            "expected_points": 2.5,
            "chance_of_playing": 100,
            "team": 7,
        },
        {
            "element": 15,
            "web_name": "FWD 3",
            "position": "FWD",
            "expected_points": 2.0,
            "chance_of_playing": 100,
            "team": 8,
        },
    ]

    df = pd.DataFrame(squad_data)
    opt = LineupOptimizer()
    res = opt.select_lineup_and_captain(df, simulate_captaincy=False)

    assert "bench_ordering_advice" in res
    advice = res["bench_ordering_advice"]
    assert "locked_min_positions" in advice
    assert "DEF" in advice["locked_min_positions"]
    assert "at_risk_locked_positions" in advice
    assert "DEF" in advice["at_risk_locked_positions"]
    assert len(advice["notes"]) > 0
    assert "Formation-locked DEF requirement" in advice["notes"][0]
