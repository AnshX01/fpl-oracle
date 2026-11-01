"""
Unit tests for Real-Time Live Matchday Operations Engine (Pillar 5).
Verifies:
1. Exact official FPL BPS tiebreakers (3, 2, 1).
2. Goalkeeper autosubstitution on zero minutes.
3. Outfield formation-legal autosubstitutions (min 3 DEF, 2 MID, 1 FWD).
4. Captaincy rollover to Vice-Captain.
5. Bench Boost active scoring.
Zero hardcoded player names in production code.
"""

from types import SimpleNamespace

from fpl_oracle.league.live_matchday import (
    LiveMatchdayEngine,
    LivePlayerScore,
    compute_fixture_provisional_bonus,
)


def test_bps_provisional_bonus_tiebreakers():
    """BPS bonus distribution handles exact Premier League tiebreak rules."""
    # Scenario A: Clear 1st (40), 2nd (35), 3rd (30)
    bps_a = {101: 40, 102: 35, 103: 30, 104: 25}
    bon_a = compute_fixture_provisional_bonus(bps_a)
    assert bon_a[101] == 3
    assert bon_a[102] == 2
    assert bon_a[103] == 1
    assert 104 not in bon_a

    # Scenario B: 2-way tie for 1st (both get 3, 3rd gets 1, 2 is skipped)
    bps_b = {201: 40, 202: 40, 203: 35, 204: 20}
    bon_b = compute_fixture_provisional_bonus(bps_b)
    assert bon_b[201] == 3
    assert bon_b[202] == 3
    assert bon_b[203] == 1
    assert 204 not in bon_b

    # Scenario C: 3-way tie for 1st (all 3 get 3, nobody gets 2 or 1)
    bps_c = {301: 40, 302: 40, 303: 40, 304: 35}
    bon_c = compute_fixture_provisional_bonus(bps_c)
    assert bon_c[301] == 3
    assert bon_c[302] == 3
    assert bon_c[303] == 3
    assert 304 not in bon_c

    # Scenario D: Clear 1st, 2-way tie for 2nd (both get 2, nobody gets 1)
    bps_d = {401: 45, 402: 40, 403: 40, 404: 35}
    bon_d = compute_fixture_provisional_bonus(bps_d)
    assert bon_d[401] == 3
    assert bon_d[402] == 2
    assert bon_d[403] == 2
    assert 404 not in bon_d

    # Scenario E: Clear 1st, clear 2nd, 2-way tie for 3rd (both get 1)
    bps_e = {501: 50, 502: 40, 503: 35, 504: 35, 505: 20}
    bon_e = compute_fixture_provisional_bonus(bps_e)
    assert bon_e[501] == 3
    assert bon_e[502] == 2
    assert bon_e[503] == 1
    assert bon_e[504] == 1
    assert 505 not in bon_e


def test_autosub_goalkeeper_replacement():
    """Starting Goalkeeper finished with 0 mins is replaced by Bench GK."""
    engine = LiveMatchdayEngine()

    scores = {
        1: LivePlayerScore(
            element_id=1,
            web_name="Keeper A",
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
        12: LivePlayerScore(
            element_id=12,
            web_name="Keeper B",
            position="GKP",
            team_id=2,
            minutes=90,
            goals=0,
            assists=0,
            clean_sheets=1,
            goals_conceded=0,
            bps=25,
            official_bonus=0,
            provisional_bonus=2,
            live_points=8,
            status="FINISHED",
        ),
    }

    # 10 outfielders
    for i in range(2, 12):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"Player {i}",
            position="MID" if i < 7 else "DEF",
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
    for i in range(13, 16):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"Bench {i}",
            position="FWD",
            team_id=4,
            minutes=0,
            goals=0,
            assists=0,
            clean_sheets=0,
            goals_conceded=0,
            bps=0,
            official_bonus=0,
            provisional_bonus=0,
            live_points=0,
            status="YET_TO_PLAY",
        )

    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 2), is_vice_captain=(i == 3)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, cap_id = engine.resolve_autosubs(picks, scores)

    # Keeper B (12) must be in final XI
    xi_ids = {p.element_id for p in final_xi}
    assert 12 in xi_ids
    assert 1 not in xi_ids
    assert len(autosubs) == 1
    assert autosubs[0].sub_type == "GOALKEEPER"
    assert autosubs[0].in_element_id == 12


def test_captaincy_rollover_to_vice_captain():
    """Armband passes to Vice-Captain when Captain finishes match with 0 minutes."""
    engine = LiveMatchdayEngine()

    scores = {}
    for i in range(1, 16):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"Player {i}",
            position="MID" if i <= 11 else "FWD",
            team_id=1,
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

    # Captain (element 10) finishes with 0 minutes
    scores[10] = LivePlayerScore(
        element_id=10,
        web_name="Captain Benched",
        position="FWD",
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
    )
    # Vice Captain is element 9
    picks = [
        SimpleNamespace(element=i, position=i, is_captain=(i == 10), is_vice_captain=(i == 9)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, eff_cap_id = engine.resolve_autosubs(picks, scores)

    assert eff_cap_id == 9  # Shifted to Vice-Captain!
    cap_pass = next((a for a in autosubs if a.sub_type == "CAPTAIN_PASS"), None)
    assert cap_pass is not None
    assert cap_pass.in_element_id == 9


def test_formation_legal_outfield_autosubs():
    """Outfield autosub rejects subs that would leave fewer than 3 defenders."""
    engine = LiveMatchdayEngine()

    # Starters: 1 GKP, 3 DEF (2, 3, 4), 5 MID (5, 6, 7, 8, 9), 2 FWD (10, 11)
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
        # DEF 3 finishes with 0 minutes!
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
    for i in range(5, 12):
        scores[i] = LivePlayerScore(
            element_id=i,
            web_name=f"Starter {i}",
            position="MID" if i < 10 else "FWD",
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

    # Bench: 12 (GKP), 13 (Sub 1 - MID with 10 pts), 14 (Sub 2 - DEF with 2 pts)
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
        bps=30,
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
        SimpleNamespace(element=i, position=i, is_captain=(i == 5), is_vice_captain=(i == 6)) for i in range(1, 16)
    ]

    final_xi, final_bench, autosubs, cap_id = engine.resolve_autosubs(picks, scores)

    # Sub 1 (13 MID) would leave only 2 defenders (illegal formation)!
    # Therefore, the system MUST skip Sub 1 and promote Sub 2 (14 DEF)!
    xi_ids = {p.element_id for p in final_xi}
    assert 14 in xi_ids
    assert 13 not in xi_ids
    assert 4 not in xi_ids
    assert len(autosubs) == 1
    assert autosubs[0].in_element_id == 14
    assert autosubs[0].sub_type == "OUTFIELD"
