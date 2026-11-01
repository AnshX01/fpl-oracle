"""
Tests for Tactical Archetype Matchup Engine (Peak Frontier Pillar 1).
"""

import pandas as pd

from fpl_oracle.ml.components.tactical_matchups import tactical_matchup_engine


def test_classify_team_archetypes():
    arch_mci = tactical_matchup_engine.classify_team_archetype("MCI")
    assert arch_mci.name == "HIGH_PRESS_HIGH_LINE"

    arch_eve = tactical_matchup_engine.classify_team_archetype("EVE")
    assert arch_eve.name == "COMPACT_LOW_BLOCK"

    arch_bou = tactical_matchup_engine.classify_team_archetype("BOU", opponent_strength=3.0)
    assert arch_bou.name == "BALANCED_TRANSITION"


def test_compute_matchup_modifiers():
    # Forward facing High Line (space behind)
    mods_fwd_high = tactical_matchup_engine.compute_matchup_modifiers("TOT", "FWD")
    assert mods_fwd_high["xg_mult"] > 1.0  # Transition boost

    # Forward facing Low Block (congested box)
    mods_fwd_low = tactical_matchup_engine.compute_matchup_modifiers("EVE", "FWD")
    assert mods_fwd_low["xg_mult"] < 1.0

    # Creative Midfielder with high xT facing Low Block (high box possession and cross volume)
    mods_mid_creative = tactical_matchup_engine.compute_matchup_modifiers("EVE", "MID", xt_creativity=0.35)
    assert mods_mid_creative["xa_mult"] > 1.0


def test_enhance_projections():
    df = pd.DataFrame(
        [
            {
                "element": 1,
                "position": "FWD",
                "opponent_short": "TOT",
                "expected_points": 6.0,
                "expected_threat_xt": 0.2,
                "set_piece_threat_share": 0.1,
            },
            {
                "element": 2,
                "position": "FWD",
                "opponent_short": "EVE",
                "expected_points": 6.0,
                "expected_threat_xt": 0.2,
                "set_piece_threat_share": 0.1,
            },
        ]
    )
    enhanced = tactical_matchup_engine.enhance_projections(df)
    assert enhanced.loc[0, "tactical_archetype"] == "HIGH_PRESS_HIGH_LINE"
    assert enhanced.loc[0, "expected_points"] > 6.0
    assert enhanced.loc[1, "tactical_archetype"] == "COMPACT_LOW_BLOCK"
    assert enhanced.loc[1, "expected_points"] < 6.0


def test_enhance_projections_preserves_congestion_fatigue():
    df = pd.DataFrame(
        [
            {
                "element": 1,
                "position": "FWD",
                "opponent_short": "TOT",
                "exp_appearance": 2.0,
                "exp_goals_pts": 4.0,
                "exp_assists_pts": 1.0,
                "exp_cs_pts": 0.0,
                "expected_threat_xt": 0.2,
                "set_piece_threat_share": 0.1,
                "congestion_fatigue_factor": 0.85,
            }
        ]
    )
    enhanced = tactical_matchup_engine.enhance_projections(df)
    assert enhanced.loc[0, "expected_points"] < 7.0
