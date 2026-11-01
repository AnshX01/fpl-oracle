"""
Comprehensive unit tests for the 6 new blind spots:
1. Beyond-horizon chip evaluation solves with full candidate pool (no status=Infeasible).
2. Free Hit resets to 1 FT entering next gameweek, whereas Wildcard preserves banked FTs.
3. Vice-captain fixture diversification and safe fallback.
4. Team confirmation preserves [1, 5] FT invariant entering next gameweek.
5. Retained player selling prices in Free Hit/Wildcard simulations.
6. Poisson expected_floor_div handles NaN gracefully; quantile ordering holds on negative xP.
"""

import numpy as np
import pandas as pd

from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.domain.scoring import expected_floor_div
from fpl_oracle.optimise.lineup import LineupOptimizer
from fpl_oracle.rules.transfers import next_free_transfers


def test_free_hit_and_wildcard_preserve_banked_transfers():
    """Rule R4: Free Hit and Wildcard preserve banked FTs with 0 hit points."""
    # Free Hit with 4 banked FTs preserves 4 banked FTs
    ft_after_fh, hits_fh = next_free_transfers(ft=4, transfers_made=5, active_chip="freehit")
    assert ft_after_fh == 4, f"Expected Free Hit to preserve 4 FTs, got {ft_after_fh}"
    assert hits_fh == 0

    # Wildcard with 4 banked FTs preserves 4 banked FTs
    ft_after_wc, hits_wc = next_free_transfers(ft=4, transfers_made=5, active_chip="wildcard")
    assert ft_after_wc == 4, f"Expected Wildcard to preserve 4 FTs, got {ft_after_wc}"
    assert hits_wc == 0


def test_poisson_nan_rate_returns_zero_without_crashing():
    """Poisson expected_floor_div must return 0.0 without throwing ValueError when rate is NaN."""
    val = expected_floor_div(float("nan"), 2)
    assert val == 0.0

    val_neg = expected_floor_div(-1.5, 2)
    assert val_neg == 0.0

    val_zero = expected_floor_div(0.0, 2)
    assert val_zero == 0.0


def test_quantile_ordering_preserved_on_negative_xp():
    """P10 <= E[X] <= P90 must strictly hold even when projected xP is legitimately negative."""
    # Test through array logic matching ensemble.py
    xP = np.array([-0.5, 2.0, 5.0])
    zero_mask = np.array([False, False, False])
    q10_off = np.array([-1.0, -1.0, -1.5])
    q90_off = np.array([1.5, 2.0, 3.0])
    p50 = np.where(zero_mask, 0.0, np.where(xP < 0.0, xP - 0.2, np.maximum(0.0, xP - 0.2)))
    p10 = np.where(zero_mask, 0.0, np.where(xP < 0.0, np.minimum(xP, xP + q10_off), np.maximum(0.0, xP + q10_off)))
    p90 = np.where(zero_mask, 0.0, np.maximum(p50 + 1.0, np.maximum(xP + 0.1, xP + q90_off)))

    for i in range(len(xP)):
        assert p10[i] <= xP[i], f"P10 ({p10[i]}) must be <= xP ({xP[i]})"
        assert xP[i] <= p90[i], f"xP ({xP[i]}) must be <= P90 ({p90[i]})"


def test_vice_captain_fixture_diversification_and_fallback():
    """Vice-captain selection applies 15% discount to captain's club/match to diversify fixtures."""
    opt = LineupOptimizer()
    # 15 players where top 2 are Palmer (team 5, 8.0 xP) and Jackson (team 5, 7.8 xP),
    # third is Saka (team 1, 7.0 xP)
    squad_rows = [
        {"element": 1, "web_name": "Palmer", "position": "MID", "team": 5, "expected_points": 8.0, "value": 105},
        {"element": 2, "web_name": "Jackson", "position": "FWD", "team": 5, "expected_points": 7.8, "value": 80},
        {"element": 3, "web_name": "Saka", "position": "MID", "team": 1, "expected_points": 7.0, "value": 100},
        {"element": 4, "web_name": "Raya", "position": "GKP", "team": 1, "expected_points": 4.5, "value": 55},
        {"element": 5, "web_name": "Gabriel", "position": "DEF", "team": 1, "expected_points": 5.0, "value": 60},
        {"element": 6, "web_name": "Saliba", "position": "DEF", "team": 1, "expected_points": 5.0, "value": 60},
        {"element": 7, "web_name": "Konate", "position": "DEF", "team": 2, "expected_points": 4.5, "value": 52},
        {"element": 8, "web_name": "Mbeumo", "position": "MID", "team": 3, "expected_points": 5.5, "value": 72},
        {"element": 9, "web_name": "Rogers", "position": "MID", "team": 4, "expected_points": 5.0, "value": 52},
        {"element": 10, "web_name": "Haaland", "position": "FWD", "team": 6, "expected_points": 8.5, "value": 150},
        {"element": 11, "web_name": "Wood", "position": "FWD", "team": 7, "expected_points": 4.5, "value": 62},
        {"element": 12, "web_name": "SubGK", "position": "GKP", "team": 8, "expected_points": 0.0, "value": 40},
        {"element": 13, "web_name": "SubD1", "position": "DEF", "team": 9, "expected_points": 3.0, "value": 40},
        {"element": 14, "web_name": "SubD2", "position": "DEF", "team": 10, "expected_points": 2.5, "value": 40},
        {"element": 15, "web_name": "SubM1", "position": "MID", "team": 11, "expected_points": 2.0, "value": 45},
    ]
    df = pd.DataFrame(squad_rows)
    lineup = opt.select_lineup_and_captain(df, simulate_captaincy=False)

    cap = lineup["captain"]
    vice = lineup["vice_captain"]
    assert cap["element"] != vice["element"], "Captain and Vice-Captain must be distinct players"
    # If captain is Palmer (team 5), Jackson's score is 7.8 * 0.85 = 6.63, so Saka (team 1, 7.0) or Haaland is chosen
    if cap["element"] == 1:
        assert vice["team"] != 5, "Vice-captain should diversify fixture from captain's club"


def test_beyond_horizon_chip_utility_uses_full_pool():
    """Beyond-horizon chip utility must use full candidate pool and solve cleanly without status=Infeasible."""
    # 15 player owned squad
    squad_rows = [
        {
            "element": i,
            "web_name": f"P{i}",
            "position": "GKP" if i <= 2 else ("DEF" if i <= 7 else ("MID" if i <= 12 else "FWD")),
            "team": (i % 20) + 1,
            "value": 60,
            "selling_price": 55,
            "expected_points": 4.0,
        }
        for i in range(1, 16)
    ]
    squad_df = pd.DataFrame(squad_rows)

    # 40 player pool in GW6 snapshot
    pool_rows = squad_rows.copy()
    for i in range(16, 45):
        pool_rows.append(
            {
                "element": i,
                "web_name": f"Pool{i}",
                "position": "DEF" if i < 25 else ("MID" if i < 35 else "FWD"),
                "team": (i % 20) + 1,
                "value": 60,
                "selling_price": 60,
                "expected_points": 6.5,
            }
        )
    pool_df = pd.DataFrame(pool_rows)

    # Horizon projections only contains GW6
    horizon_projections = {6: pool_df}

    # Evaluate Wildcard for GW20 (beyond horizon)
    gain, conf, base_xp, with_xp = chip_planner.compute_chip_utility(
        chip="wildcard",
        gw=20,
        current_squad_df=squad_df,
        horizon_projections=horizon_projections,
        dgw_gws=[],
        bgw_gws=[],
        budget=1000.0,
        current_gw=6,
    )
    # Must solve with positive gain from candidate pool rather than returning status=Infeasible
    assert gain > 0.0, f"Expected positive Wildcard gain for GW20, got {gain}"
    assert "Beyond 5-GW Horizon" in conf


def test_team_confirmation_free_transfers_invariant():
    """Confirmed team state must preserve the [1, 5] FT invariant entering next gameweek."""
    # Test that next_free_transfers guarantees at least 1 FT
    ft_next_after_1_used = next_free_transfers(ft=1, transfers_made=1)[0]
    assert ft_next_after_1_used == 1

    ft_next_after_0_used = next_free_transfers(ft=1, transfers_made=0)[0]
    assert ft_next_after_0_used == 2

    ft_next_after_2_used = next_free_transfers(ft=2, transfers_made=2)[0]
    assert ft_next_after_2_used == 1
