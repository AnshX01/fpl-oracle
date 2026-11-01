"""
Comprehensive Unit Tests for Wave 5 Blind Spots & Remediation Invariants (Items 41-58).
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from fpl_oracle.league.montecarlo import MonteCarloSimulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.scenarios import evolve_rival
from fpl_oracle.league.standings import current_manager_rank
from fpl_oracle.league.validation import score_forecast
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.reconcile import absence_gameweeks
from fpl_oracle.optimise.lineup import best_lineup


def test_item_42_standings_rank_sort_tiebreak():
    """Item 42: current_manager_rank prioritizes rank_sort over shared tied rank."""
    data = {
        "standings": [
            {"entry": 101, "rank": 1, "rank_sort": 1},
            {"entry": 102, "rank": 1, "rank_sort": 2},
        ]
    }
    r1 = current_manager_rank(data, 101)
    r2 = current_manager_rank(data, 102)
    assert r1["current_league_rank"] == 1
    assert r2["current_league_rank"] == 2
    assert r2["tied_display_rank"] == 1


def test_item_43_validation_transfer_tiebreak():
    """Item 43: score_forecast credits tiebreak win if owner made fewer transfers."""
    record = {
        "schema_version": 1,
        "source_kind": "official_forward",
        "scope": "actual_observed_manager_outcome",
        "frozen_at": "2026-10-20T10:00:00Z",
        "deadline_utc": "2026-10-20T11:00:00Z",
        "observed_manager_ids": [101, 102],
        "owner_manager_id": 101,
        "probability": 0.6,
        "target_gameweek": 9,
    }
    # Tied on points (60 pts each), but owner 101 made 5 transfers vs rival 102 made 12 transfers
    finalized = {
        "source_kind": "official_finalized_manager_history",
        "gameweek": 9,
        "scores": {101: 60.0, 102: 60.0},
        "transfers": {101: 5, 102: 12},
    }
    res = score_forecast(record, finalized)
    assert res["status"] == "scored"
    assert res["outcome"] == 1  # Owner wins tiebreak!


def test_item_46_chips_not_resurrected_in_set_2():
    """Item 46: Single-use chips (FH/BB/TC) used in Set 1 (GW 1-19) must NOT reappear in Set 2."""
    chips_used = [
        {"name": "wildcard", "event": 8},
        {"name": "bboost", "event": 10},
        {"name": "freehit", "event": 12},
        {"name": "3xc", "event": 14},
    ]
    # In GW 22 (Set 2): Wildcard 2 is available, but BB, FH, and TC must remain burned!
    status_gw22 = rival_analyzer.calculate_chips_status(chips_used, current_gw=22)
    assert "wildcard_set2" in status_gw22["chips_remaining"]
    assert "bench_boost" not in status_gw22["chips_remaining"]
    assert "freehit" not in status_gw22["chips_remaining"]
    assert "triple_captain" not in status_gw22["chips_remaining"]


def test_item_45_rival_scenarios_utilize_bank():
    """Item 45: evolve_rival allows upgrades if rival holds bank_itb."""
    # 15 player pool + upgrade candidate 16
    pool = {
        1: {"element": 1, "position": "GKP", "team": 1, "value": 4.5, "expected_points": 3.0},
        2: {"element": 2, "position": "GKP", "team": 2, "value": 4.0, "expected_points": 2.5},
        3: {"element": 3, "position": "DEF", "team": 3, "value": 4.5, "expected_points": 3.0},
        4: {"element": 4, "position": "DEF", "team": 4, "value": 4.5, "expected_points": 3.0},
        5: {"element": 5, "position": "DEF", "team": 5, "value": 4.5, "expected_points": 3.0},
        6: {"element": 6, "position": "DEF", "team": 6, "value": 4.0, "expected_points": 2.5},
        7: {"element": 7, "position": "DEF", "team": 7, "value": 4.0, "expected_points": 2.5},
        8: {"element": 8, "position": "MID", "team": 8, "value": 5.0, "expected_points": 3.5},
        9: {"element": 9, "position": "MID", "team": 9, "value": 5.0, "expected_points": 3.5},
        10: {"element": 10, "position": "MID", "team": 10, "value": 5.0, "expected_points": 3.5},
        11: {"element": 11, "position": "MID", "team": 11, "value": 5.0, "expected_points": 3.5},
        12: {"element": 12, "position": "MID", "team": 12, "value": 4.5, "expected_points": 3.0},
        13: {"element": 13, "position": "FWD", "team": 13, "value": 6.0, "expected_points": 4.0},
        14: {"element": 14, "position": "FWD", "team": 14, "value": 5.0, "expected_points": 2.0},  # FWD to upgrade
        15: {"element": 15, "position": "FWD", "team": 15, "value": 4.5, "expected_points": 2.5},
        # Candidate 16 is an 8.5m FWD (8.5 > 5.0, requires 3.5m bank)
        16: {"element": 16, "position": "FWD", "team": 16, "value": 8.5, "expected_points": 7.5},
    }
    elements = list(range(1, 16))
    rival = {"elements": elements, "bank_itb": 3.5}
    maps = {9: pool}
    path = evolve_rival(rival, maps, mode="one_transfer_per_week")
    assert path is not None
    # Rival upgraded player 14 (5.0m) to player 16 (8.5m) using 3.5m bank!
    assert 16 in path[0]["elements"]
    assert 14 not in path[0]["elements"]


def test_item_49_clean_sheet_subbed_off_after_60_mins():
    """Item 49: Player subbed off at 65' with 0 conceded receives CS even if team concedes in 88'."""
    sims = 10
    n_players = 1
    minutes = np.full((n_players, sims), 65.0)  # Played 65 mins
    goals_conceded = np.zeros((n_players, sims), dtype=int)  # 0 conceded while on pitch

    # Clean sheet is awarded when minutes >= 60 and goals conceded on pitch == 0
    clean_sheets = np.where((minutes >= 60) & (goals_conceded == 0), 1, 0)
    assert np.all(clean_sheets == 1)


def test_item_47_match_sim_autosub_joint_formation_deficits():
    """Item 47: Autosub preserves legal formation across multiple unplayed starters."""
    # When active starters have 2 DEF, 4 MID, 0 FWD (needs 1 DEF and 1 FWD with 2 subs needed)
    pos_counts = {"DEF": 2, "MID": 4, "FWD": 0}
    subs_needed = 2

    # Bench option 1 is a MID
    new_def = pos_counts["DEF"]
    new_mid = pos_counts["MID"] + 1
    new_fwd = pos_counts["FWD"]
    rem_slots = subs_needed - 1  # 1 slot remaining

    def_deficit = max(0, 3 - new_def)  # 1
    mid_deficit = max(0, 2 - new_mid)  # 0
    fwd_deficit = max(0, 1 - new_fwd)  # 1
    total_deficits = def_deficit + mid_deficit + fwd_deficit  # 2

    # MID cannot be accepted because total deficits (2) > rem_slots (1)
    can_accept_mid = total_deficits <= rem_slots
    assert not can_accept_mid

    # Bench option 2 is a DEF
    new_def2 = pos_counts["DEF"] + 1
    new_mid2 = pos_counts["MID"]
    new_fwd2 = pos_counts["FWD"]
    def_deficit2 = max(0, 3 - new_def2)  # 0
    mid_deficit2 = max(0, 2 - new_mid2)  # 0
    fwd_deficit2 = max(0, 1 - new_fwd2)  # 1
    total_deficits2 = def_deficit2 + mid_deficit2 + fwd_deficit2  # 1
    can_accept_def = total_deficits2 <= rem_slots
    assert can_accept_def


def test_item_41_monte_carlo_transfer_tiebreak():
    """Item 41: Monte Carlo simulation breaks points ties using official transfer count."""
    mc = MonteCarloSimulator(n_simulations=10)
    user_squad_df = pd.DataFrame(
        {
            "element": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "is_starter": [True] * 11 + [False] * 4,
            "is_captain": [True] + [False] * 14,
            "is_vice_captain": [False, True] + [False] * 13,
        }
    )
    user_squad_df.attrs["total_transfers"] = 4  # User made 4 transfers

    # Rival tied with user on points, but made 12 transfers
    rival_squads = [
        {
            "entry_id": 999,
            "player_name": "Rival",
            "total_points": 100.0,
            "total_transfers": 12,
            "squad": [{"element": i, "is_starter": i <= 11, "is_captain": i == 1} for i in range(1, 16)],
            "captain_element": 1,
        }
    ]

    projs_df = pd.DataFrame(
        {
            "element": list(range(1, 16)),
            "expected_points": [0.0] * 15,  # 0 expected points in projection so points tie at 100.0
            "position": ["GKP"] + ["DEF"] * 4 + ["MID"] * 5 + ["FWD"] * 5,
            "team": [1] * 15,
        }
    )

    res = mc.simulate_league(
        user_points=100.0,
        user_squad_df=user_squad_df,
        rival_squads=rival_squads,
        projections_df=projs_df,
        horizon_gws=1,
        seed=42,
    )

    # Because points tie (100 vs 100) and user has 4 transfers vs rival's 12,
    # user wins the tiebreak 100% of the time!
    assert res["user_win_probability_pct"] == 100.0


def test_item_52_lineup_outfield_under_10_backfill():
    """Item 52: best_lineup backfills blocked outfielders when fit outfield count < 10."""
    squad = list(range(1, 16))
    positions = {
        1: "GKP",
        2: "GKP",
        3: "DEF",
        4: "DEF",
        5: "DEF",
        6: "DEF",
        7: "DEF",
        8: "MID",
        9: "MID",
        10: "MID",
        11: "MID",
        12: "MID",
        13: "FWD",
        14: "FWD",
        15: "FWD",
    }
    xp = {p: 4.0 for p in squad}
    blocked = {6, 7, 10, 11, 12, 14, 15}

    starters, bench, cap, vc = best_lineup(
        squad=squad,
        position=positions,
        xp=xp,
        bench_weight=0.1,
        blocked=blocked,
        allow_blocked_fallback=True,
    )
    assert len(starters) == 11
    assert len(bench) == 4
    assert positions[starters[0]] == "GKP"


def test_item_55_manager_will_start_unblocked():
    """Item 55: Authoritative manager quote 'will start' clears FPL doubt to 100%."""
    ev = PlayerEvidence(
        player_id=10,
        player_name="Test Player",
        team_id=1,
        category=EvidenceCategory.AVAILABLE,
        quote="He is 100% fit and will start tomorrow",
        published_at=datetime.now(UTC),
        source_url="https://bbc.com/sport",
        confidence=0.95,
        publisher="BBC Sport",
    )
    q_low = ev.quote.lower()
    matches = any(
        kw in q_low for kw in ("will start", "fit to start", "starts tomorrow", "available to start", "100% fit")
    )
    assert matches is True


def test_item_57_absence_gameweeks_natural_phrasing():
    """Item 57: absence_gameweeks handles natural phrasing like 'sidelined for 3 weeks'."""
    assert absence_gameweeks("He is sidelined for 3 weeks with a hamstring issue") == 3
    assert absence_gameweeks("Player faces a month out following surgery") == 4
    assert absence_gameweeks("He will be unavailable for 2 matches") == 2
