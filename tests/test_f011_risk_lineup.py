import pandas as pd
import pytest

from fpl_oracle.optimise.lineup import LineupOptimizer


@pytest.fixture
def risk_test_squad():
    """
    Construct a 15-player squad where:
    - 1 GKP starter, 1 GKP bench
    - 3 DEF starters (locked)
    - 2 MID starters (locked)
    - 1 FWD starter (locked)
    - Total locked starters: 1 + 3 + 2 + 1 = 7 players.
    - 3 flex spots filled by high xP players (8.0 xP).
    - Total locked + flex = 10 starters.
    - Exactly 1 slot remains contested between:
      1) Player 11 (MID, Doubtful high-variance star):
         expected_points = 5.0, chance_of_playing = 75.0, p10 = 2.0, p90 = 10.0
      2) Player 12 (DEF, Secure starter):
         expected_points = 4.8, chance_of_playing = 100.0, p10 = 3.8, p90 = 5.8
    - Players 13, 14, 15 have very low xP (1.0 xP) and are clearly on the bench.
    """
    rows = [
        # GKPs (2)
        {
            "element": 1,
            "web_name": "GKP_1",
            "position": "GKP",
            "expected_points": 5.0,
            "p10": 3.0,
            "p90": 7.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 2,
            "web_name": "GKP_2",
            "position": "GKP",
            "expected_points": 1.0,
            "p10": 0.0,
            "p90": 2.0,
            "chance_of_playing": 100.0,
        },
        # DEFs (5)
        {
            "element": 3,
            "web_name": "DEF_1",
            "position": "DEF",
            "expected_points": 6.0,
            "p10": 4.0,
            "p90": 8.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 4,
            "web_name": "DEF_2",
            "position": "DEF",
            "expected_points": 6.0,
            "p10": 4.0,
            "p90": 8.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 5,
            "web_name": "DEF_3",
            "position": "DEF",
            "expected_points": 6.0,
            "p10": 4.0,
            "p90": 8.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 6,
            "web_name": "DEF_SECURE",
            "position": "DEF",
            "expected_points": 4.8,
            "p10": 3.8,
            "p90": 5.8,
            "chance_of_playing": 100.0,
        },
        {
            "element": 7,
            "web_name": "DEF_BENCH",
            "position": "DEF",
            "expected_points": 1.0,
            "p10": 0.0,
            "p90": 2.0,
            "chance_of_playing": 100.0,
        },
        # MIDs (5)
        {
            "element": 8,
            "web_name": "MID_1",
            "position": "MID",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 9,
            "web_name": "MID_2",
            "position": "MID",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 10,
            "web_name": "MID_3",
            "position": "MID",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 11,
            "web_name": "MID_DOUBTFUL",
            "position": "MID",
            "expected_points": 5.0,
            "p10": 2.0,
            "p90": 10.0,
            "chance_of_playing": 75.0,
        },
        {
            "element": 12,
            "web_name": "MID_BENCH",
            "position": "MID",
            "expected_points": 1.0,
            "p10": 0.0,
            "p90": 2.0,
            "chance_of_playing": 100.0,
        },
        # FWDs (3)
        {
            "element": 13,
            "web_name": "FWD_1",
            "position": "FWD",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 14,
            "web_name": "FWD_2",
            "position": "FWD",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
        {
            "element": 15,
            "web_name": "FWD_3",
            "position": "FWD",
            "expected_points": 7.0,
            "p10": 5.0,
            "p90": 9.0,
            "chance_of_playing": 100.0,
        },
    ]
    return pd.DataFrame(rows)


def test_xi_is_pure_expected_points_in_every_mode(risk_test_squad):
    """The XI maximises expected points; risk modes only tilt the captain."""
    optimizer = LineupOptimizer()

    for mode in ["points", "balanced", "conservative", "aggressive"]:
        res = optimizer.select_lineup_and_captain(risk_test_squad, risk_preference=mode)
        starters = set(res["starters"]["element"])
        assert 11 in starters and 6 not in starters, f"xP-best XI expected in {mode} mode"


def test_balanced_captain_is_top_expected_points_and_rankings_agree():
    """Captain must be the xP leader (no hidden p90 tilt); captain_rankings and vice text must match it."""
    optimizer = LineupOptimizer()
    squad = pd.DataFrame(
        [
            ("GKP", 4.0, 3.0, 5.0, 1),
            ("GKP", 1.0, 0.0, 2.0, 2),
            ("DEF", 4.0, 2.0, 6.0, 3),
            ("DEF", 4.0, 2.0, 6.0, 4),
            ("DEF", 4.0, 2.0, 6.0, 5),
            ("DEF", 4.0, 2.0, 6.0, 6),
            ("DEF", 1.0, 0.0, 2.0, 7),
            ("MID", 7.0, 5.0, 10.0, 8),  # xP leader, modest ceiling
            ("MID", 5.5, 1.0, 16.0, 9),  # huge ceiling, lower xP
            ("MID", 4.0, 2.0, 6.0, 10),
            ("MID", 4.0, 2.0, 6.0, 11),
            ("MID", 1.0, 0.0, 2.0, 12),
            ("FWD", 4.0, 2.0, 6.0, 13),
            ("FWD", 4.0, 2.0, 6.0, 14),
            ("FWD", 1.0, 0.0, 2.0, 15),
        ],
        columns=["position", "expected_points", "p10", "p90", "element"],
    )
    squad["web_name"] = squad["element"].map(lambda e: f"P{e}")
    res = optimizer.select_lineup_and_captain(squad, risk_preference="balanced", simulate_captaincy=False)
    assert res["captain"]["element"] == 8
    assert res["captain_rankings"][0]["element"] == 8
    assert res["vice_captain"]["element"] == 9
    assert "5.5 xP" in res["vice_captain"]["reasoning"]


def _squad(rows):
    df = pd.DataFrame(rows, columns=["element", "position", "expected_points", "chance_of_playing", "team"])
    df["web_name"] = df["element"].map(lambda e: f"P{e}")
    return df


def test_vice_is_availability_adjusted():
    rows = [(1, "GKP", 4.0, 100.0, 1), (2, "GKP", 1.0, 100.0, 1)]
    rows += [(e, "DEF", 5.0, 100.0, 2) for e in (3, 4, 5, 6)]
    rows += [(e, "MID", 6.0, 100.0, 3) for e in (7, 8, 9)]
    rows += [(10, "MID", 7.5, 50.0, 4), (11, "MID", 1.0, 100.0, 5)]  # doubtful 7.5 xP -> 3.75 effective
    rows += [
        (12, "FWD", 9.0, 100.0, 6),
        (13, "FWD", 6.0, 100.0, 7),
        (14, "FWD", 1.0, 100.0, 7),
        (15, "DEF", 1.0, 100.0, 2),
    ]
    res = LineupOptimizer().select_lineup_and_captain(_squad(rows), simulate_captaincy=False)
    assert res["captain"]["element"] == 12
    # raw-xP vice would be the 50% doubt (7.5); availability-adjusted vice is a fit 6.0 player
    assert res["vice_captain"]["element"] != 10
    assert 10 in {r["element"] for r in res["bench_risk_starters"]}


def test_bench_order_prefers_legal_swaps_over_raw_xp():
    # Starters 3-5-2: a bench DEF can replace any outfielder, a bench FWD only a MID/FWD.
    rows = [(1, "GKP", 4.0, 100.0, 1), (2, "GKP", 1.0, 100.0, 1)]
    rows += [(e, "DEF", 5.0, 100.0, 2) for e in (3, 4, 5)] + [(6, "DEF", 2.0, 100.0, 2), (7, "DEF", 0.5, 100.0, 2)]
    rows += [(e, "MID", 6.0, 100.0, 3) for e in (8, 9, 10, 11, 12)]
    rows += [(13, "FWD", 7.0, 100.0, 4), (14, "FWD", 7.0, 100.0, 4), (15, "FWD", 2.05, 100.0, 4)]
    res = LineupOptimizer().select_lineup_and_captain(_squad(rows), simulate_captaincy=False)
    assert res["formation"] == "3-5-2"
    assert list(res["bench"]["element"])[:3] == [2, 6, 15]


def test_missing_chance_uses_status():
    from fpl_oracle.optimise.lineup import play_probability

    df = pd.DataFrame({"chance_of_playing": [None, None, None, 75.0], "status": ["i", "d", "a", "a"]})
    assert list(play_probability(df)) == [0.0, 0.5, 1.0, 0.75]
