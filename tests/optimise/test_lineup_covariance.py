"""
Tests for Markowitz Portfolio Backline Covariance Regularization in Lineup Optimization.
Verifies that defensive stack wipeout correlations are penalized under risk-averse postures,
diversifying across independent matches, while preserving elite high-xP stacks and aggressive ceiling chases.
"""

import pandas as pd

from fpl_oracle.optimise.lineup import lineup_optimizer


def _make_sample_squad():
    rows = [
        {
            "element": 10,
            "web_name": "Raya",
            "position": "GKP",
            "team": 1,
            "expected_points": 4.5,
            "p10": 2.0,
            "p90": 8.0,
        },
        {
            "element": 11,
            "web_name": "Fabianski",
            "position": "GKP",
            "team": 19,
            "expected_points": 3.0,
            "p10": 1.0,
            "p90": 6.0,
        },
        # Arsenal DEF pair (double-up risk)
        {
            "element": 1,
            "web_name": "Gabriel",
            "position": "DEF",
            "team": 1,
            "expected_points": 4.2,
            "p10": 1.0,
            "p90": 9.0,
        },
        {
            "element": 2,
            "web_name": "Calafiori",
            "position": "DEF",
            "team": 1,
            "expected_points": 4.1,
            "p10": 1.0,
            "p90": 8.0,
        },
        # Diversified DEF alternatives from other teams
        {
            "element": 3,
            "web_name": "Alexander-Arnold",
            "position": "DEF",
            "team": 14,
            "expected_points": 4.0,
            "p10": 1.0,
            "p90": 9.0,
        },
        {
            "element": 4,
            "web_name": "Konsa",
            "position": "DEF",
            "team": 2,
            "expected_points": 3.95,
            "p10": 1.0,
            "p90": 8.0,
        },
        {
            "element": 5,
            "web_name": "Mykolenko",
            "position": "DEF",
            "team": 9,
            "expected_points": 3.0,
            "p10": 1.0,
            "p90": 6.0,
        },
        # Midfielders
        {
            "element": 20,
            "web_name": "Palmer",
            "position": "MID",
            "team": 6,
            "expected_points": 6.5,
            "p10": 2.0,
            "p90": 12.0,
        },
        {
            "element": 21,
            "web_name": "Saka",
            "position": "MID",
            "team": 1,
            "expected_points": 6.0,
            "p10": 2.0,
            "p90": 11.0,
        },
        {
            "element": 22,
            "web_name": "Rogers",
            "position": "MID",
            "team": 2,
            "expected_points": 5.2,
            "p10": 2.0,
            "p90": 9.0,
        },
        {
            "element": 23,
            "web_name": "Mbeumo",
            "position": "MID",
            "team": 4,
            "expected_points": 5.0,
            "p10": 1.0,
            "p90": 10.0,
        },
        {
            "element": 24,
            "web_name": "Semenyo",
            "position": "MID",
            "team": 15,
            "expected_points": 4.5,
            "p10": 1.0,
            "p90": 8.0,
        },
        # Forwards
        {
            "element": 30,
            "web_name": "Haaland",
            "position": "FWD",
            "team": 15,
            "expected_points": 7.5,
            "p10": 2.0,
            "p90": 15.0,
        },
        {
            "element": 31,
            "web_name": "Isak",
            "position": "FWD",
            "team": 14,
            "expected_points": 6.2,
            "p10": 2.0,
            "p90": 12.0,
        },
        {
            "element": 32,
            "web_name": "Wood",
            "position": "FWD",
            "team": 18,
            "expected_points": 5.0,
            "p10": 1.0,
            "p90": 9.0,
        },
    ]
    return pd.DataFrame(rows)


def test_conservative_lineup_diversifies_defensive_double_up():
    """Conservative posture penalizes Arsenal defensive double-up in favor of diversified alternatives."""
    df = _make_sample_squad()

    # Conservative posture (cov_weight ~ 0.35)
    res_cons = lineup_optimizer.select_lineup_and_captain(df, risk_preference="conservative")
    starters_cons = set(res_cons["starters"]["element"])

    # Element 1 (Gabriel, 4.2) starts
    assert 1 in starters_cons
    # Element 3 (Alexander-Arnold, 4.0, Liverpool) starts
    assert 3 in starters_cons
    # Element 4 (Konsa, 3.95, Villa) starts to diversify
    assert 4 in starters_cons
    # Correlated double-up element 2 (Calafiori, Arsenal) is moved to bench
    assert 2 not in starters_cons


def test_aggressive_lineup_preserves_raw_xp_double_up():
    """Aggressive posture ignores covariance penalty to maximize raw ceiling points."""
    df = _make_sample_squad()

    # Aggressive posture (cov_weight = 0.0)
    res_agg = lineup_optimizer.select_lineup_and_captain(df, risk_preference="aggressive")
    starters_agg = set(res_agg["starters"]["element"])

    # Both Arsenal defenders start because raw xP 4.2 and 4.1 > 3.95
    assert 1 in starters_agg
    assert 2 in starters_agg
    assert 3 in starters_agg


def test_large_xp_gap_preserves_elite_double_up_even_when_conservative():
    """When the next diversified defender is significantly weaker, the double-up is preserved despite the penalty."""
    df = _make_sample_squad()
    # Give GKP a different team so Arsenal is strictly a double-up, not a triple stack with Raya
    df.loc[df["element"] == 10, "team"] = 4  # Brentford GKP
    # Boost Calafiori to elite xP (6.0) and lower alternative defenders (Konsa 2.5, Mykolenko 2.0)
    df.loc[df["element"] == 2, "expected_points"] = 6.0
    df.loc[df["element"] == 4, "expected_points"] = 2.5
    df.loc[df["element"] == 5, "expected_points"] = 2.0

    res = lineup_optimizer.select_lineup_and_captain(df, risk_preference="conservative")
    starters = set(res["starters"]["element"])

    # Both Arsenal defenders start because the quality gap over Konsa/Mykolenko
    # outweighs the 2-player defensive covariance penalty
    assert 1 in starters
    assert 2 in starters
