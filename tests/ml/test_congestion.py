"""
Tests for Midweek European & Domestic Cup Congestion & Turnaround Fatigue Engine.
Accurately models turnaround hours and rotation risk for European competitions,
midweek Premier League rounds, and domestic cup ties.
"""

import pandas as pd

from fpl_oracle.ml.components.congestion import TurnaroundCongestionEngine, turnaround_congestion_engine
from fpl_oracle.optimise.lineup import LineupOptimizer


def test_club_congestion_assessment():
    # MCI in GW9 (Carabao Cup / UCL week)
    status_mci = turnaround_congestion_engine.assess_club_congestion("MCI", 9, "2026-10-31T11:30:00Z")
    assert status_mci.has_midweek is True
    assert status_mci.rotation_risk_tier in ("MODERATE", "ACUTE")
    assert status_mci.fatigue_discount < 1.0

    # Non-European club with no cup in GW5
    status_bre = turnaround_congestion_engine.assess_club_congestion("BRE", 5)
    assert status_bre.has_midweek is False
    assert status_bre.rotation_risk_tier == "LOW"
    assert status_bre.fatigue_discount == 1.0


def test_player_fatigue_multiplier_by_position():
    cong = turnaround_congestion_engine.assess_club_congestion("ARS", 9, "2026-10-31T11:30:00Z")

    # Goalkeepers are immune to turnaround fatigue
    gkp_mult = turnaround_congestion_engine.compute_player_fatigue_multiplier("GKP", cong)
    assert gkp_mult == 1.0

    # Defenders and Midfielders have calculated fatigue multipliers
    def_mult = turnaround_congestion_engine.compute_player_fatigue_multiplier("DEF", cong, age=26)
    fwd_mult = turnaround_congestion_engine.compute_player_fatigue_multiplier("FWD", cong, age=24)
    vet_def_mult = turnaround_congestion_engine.compute_player_fatigue_multiplier("DEF", cong, age=33)

    assert def_mult < 1.0
    assert fwd_mult < 1.0
    # Veterans have a higher fatigue risk penalty
    assert vet_def_mult <= def_mult


def test_apply_congestion_to_projections():
    df = pd.DataFrame(
        [
            {
                "element": 1,
                "web_name": "GKP1",
                "position": "GKP",
                "team_short": "MCI",
                "expected_points": 5.0,
                "expected_minutes": 90.0,
                "p_starts": 0.95,
            },
            {
                "element": 2,
                "web_name": "DEF1",
                "position": "DEF",
                "team_short": "MCI",
                "expected_points": 5.0,
                "expected_minutes": 90.0,
                "p_starts": 0.95,
            },
            {
                "element": 3,
                "web_name": "MID1",
                "position": "MID",
                "team_short": "BRE",
                "expected_points": 5.0,
                "expected_minutes": 90.0,
                "p_starts": 0.95,
            },
        ]
    )

    adjusted = turnaround_congestion_engine.apply_congestion_to_projections(df, 9)
    # GKP should not be reduced
    assert adjusted.loc[0, "expected_points"] == 5.0
    # DEF for MCI should receive fatigue reduction
    assert adjusted.loc[1, "expected_points"] < 5.0
    assert adjusted.loc[1, "congestion_fatigue_factor"] < 1.0
    # Non-European club should be untouched
    assert adjusted.loc[2, "expected_points"] == 5.0
    assert adjusted.loc[2, "congestion_fatigue_factor"] == 1.0
    # Wire turnaround_hours column
    assert "turnaround_hours" in adjusted.columns
    assert adjusted.loc[1, "turnaround_hours"] < 72.0
    assert adjusted.loc[2, "turnaround_hours"] >= 72.0


def test_newcastle_excluded_from_europe():
    """Newcastle finished outside European spots in 2025/26 and must have zero European fatigue."""
    engine = TurnaroundCongestionEngine()
    assert "NEW" not in engine.ucl_clubs
    assert "NEW" not in engine.uel_uecl_clubs

    status_new = engine.assess_club_congestion("NEW", 5)
    assert status_new.has_midweek is False
    assert status_new.fatigue_discount == 1.0
    assert status_new.estimated_turnaround_hours >= 72.0


def test_brighton_carabao_cup_round_4_tracked():
    """Non-European clubs like Brighton playing Man City in Carabao Cup R4 must be tracked."""
    engine = TurnaroundCongestionEngine()
    status_bha = engine.assess_club_congestion("BHA", 9, "2026-10-31T15:00:00Z")
    assert status_bha.has_midweek is True
    assert status_bha.estimated_turnaround_hours < 72.0
    assert status_bha.fatigue_discount < 1.0
    assert status_bha.rotation_risk_tier in ("MODERATE", "ACUTE")


def test_dynamic_active_clubs_tracking():
    """Active clubs can be dynamically registered or overridden per competition/round."""
    engine = TurnaroundCongestionEngine()
    # Initially BRE is not registered for GW15 cup quarter-finals
    status_before = engine.assess_club_congestion("BRE", 15)
    assert status_before.has_midweek is False

    # Dynamically register BRE as advancing to cup quarter-finals
    engine.register_cup_active_clubs(15, ["BRE"], nominal_kickoff="2026-12-09T20:00:00Z")
    status_after = engine.assess_club_congestion("BRE", 15)
    assert status_after.has_midweek is True
    assert status_after.estimated_turnaround_hours < 72.0


def test_midweek_premier_league_exact_turnaround():
    """GW13 Midweek PL round followed by GW14 weekend creates < 72h turnarounds for clubs like CHE/LIV."""
    engine = TurnaroundCongestionEngine()
    # CHE played Dec 2 at 19:30 and plays Dec 5 at 12:30 (65.0h turnaround)
    status_che = engine.assess_club_congestion("CHE", 14)
    assert status_che.has_midweek is True
    assert status_che.estimated_turnaround_hours <= 72.0
    assert status_che.rotation_risk_tier in ("MODERATE", "ACUTE")


def test_festive_crunch_exact_turnaround():
    """The Festive Crunch (GW17 to GW20) features compressed 65h-72h turnarounds for all clubs."""
    engine = TurnaroundCongestionEngine()
    # Arsenal plays GW18 on Dec 30 at 19:30 and GW19 on Jan 2 at 15:00 (67.5h turnaround)
    status_ars_19 = engine.assess_club_congestion("ARS", 19)
    assert status_ars_19.has_midweek is True
    assert status_ars_19.estimated_turnaround_hours <= 72.0


def test_turnaround_hours_activates_production_lineup_drag():
    """
    turnaround_hours wired into apply_congestion_to_projections is passed to LineupOptimizer,
    triggering TurnaroundDragService to bench fatigued assets for fresh starters.
    """
    engine = TurnaroundCongestionEngine()
    # 15-player squad where tired MCI midfielder (5.0 xP, 67h turnaround) contests with fresh BRE midfielder (4.8 xP)
    rows = [
        {
            "element": 1,
            "web_name": "GKP_1",
            "position": "GKP",
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "MCI",
            "kickoff_time": "2026-10-31T15:00:00Z",
        },
        {
            "element": 2,
            "web_name": "GKP_2",
            "position": "GKP",
            "expected_points": 1.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 0.0,
            "team_short": "BRE",
        },
        {
            "element": 3,
            "web_name": "DEF_1",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 4,
            "web_name": "DEF_2",
            "position": "DEF",
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 5,
            "web_name": "DEF_3",
            "position": "DEF",
            "expected_points": 4.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 6,
            "web_name": "DEF_4",
            "position": "DEF",
            "expected_points": 2.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 10.0,
            "team_short": "BRE",
        },
        {
            "element": 7,
            "web_name": "DEF_5",
            "position": "DEF",
            "expected_points": 1.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 0.0,
            "team_short": "BRE",
        },
        {
            "element": 8,
            "web_name": "MID_1",
            "position": "MID",
            "expected_points": 7.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 9,
            "web_name": "MID_2",
            "position": "MID",
            "expected_points": 6.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 10,
            "web_name": "MID_3",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 11,
            "web_name": "MID_Tired",
            "position": "MID",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 85.0,
            "team_short": "MCI",
            "kickoff_time": "2026-10-31T15:00:00Z",
        },
        {
            "element": 12,
            "web_name": "MID_Fresh",
            "position": "MID",
            "expected_points": 4.8,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 13,
            "web_name": "FWD_1",
            "position": "FWD",
            "expected_points": 7.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 14,
            "web_name": "FWD_2",
            "position": "FWD",
            "expected_points": 7.0,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
        {
            "element": 15,
            "web_name": "FWD_3",
            "position": "FWD",
            "expected_points": 6.5,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team_short": "BRE",
        },
    ]
    squad_df = pd.DataFrame(rows)
    # Apply congestion for GW9 (MCI has short turnaround, BRE does not)
    projections = engine.apply_congestion_to_projections(squad_df, 9)
    assert "turnaround_hours" in projections.columns
    assert projections.loc[10, "turnaround_hours"] < 72.0
    assert projections.loc[11, "turnaround_hours"] >= 72.0

    # LineupOptimizer evaluates the squad without needing explicit midweek_data
    opt = LineupOptimizer()
    lineup_res = opt.select_lineup_and_captain(projections, simulate_captaincy=False)
    starters = [int(p["element"]) for p in lineup_res["starters"].to_dict(orient="records")]

    # Fresh element 12 starts; tired element 11 is rotated to bench
    assert 12 in starters
    assert 11 not in starters
    # Goalkeeper (element 1) starts despite MCI quick turnaround
    assert 1 in starters
