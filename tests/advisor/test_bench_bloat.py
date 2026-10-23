"""
Unit tests for Bench Bloat Elimination & Capital Harvesting (Elite Squad Structure).
"""

import pandas as pd
import pytest

from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.health import SquadHealthDiagnosis


def make_bloated_squad_and_pool():
    # 15 players:
    # 11 strong starters (GKP 1, DEF 3-5, MID 8-10, FWD 13-14)
    # Bench:
    # GKP 2: Backup GKP costing £5.0m (50) -> £1.0m bloated vs £4.0m floor!
    # DEF 6: Sub 1 costing £5.0m (50) -> within allowed £5.0m Sub 1 safety buffer
    # DEF 7: Sub 2 costing £5.0m (50) -> £1.0m bloated vs £4.0m floor!
    # MID 12: Sub 3 costing £6.0m (60) -> £1.5m bloated vs £4.5m floor!
    # Total dead bench capital: 10 + 0 + 10 + 15 = 35 tenths (£3.5m)!
    squad_rows = [
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "value": 55, "now_cost": 55, "purchase_price": 55, "expected_points": 5.0, "chance_of_playing": 100.0, "team": 1},
        {"element": 2, "web_name": "GKP_BLOAT", "position": "GKP", "value": 50, "now_cost": 50, "purchase_price": 50, "expected_points": 2.0, "chance_of_playing": 100.0, "team": 2},
        {"element": 3, "web_name": "DEF_1", "position": "DEF", "value": 60, "now_cost": 60, "purchase_price": 60, "expected_points": 5.5, "chance_of_playing": 100.0, "team": 3},
        {"element": 4, "web_name": "DEF_2", "position": "DEF", "value": 55, "now_cost": 55, "purchase_price": 55, "expected_points": 5.0, "chance_of_playing": 100.0, "team": 4},
        {"element": 5, "web_name": "DEF_3", "position": "DEF", "value": 50, "now_cost": 50, "purchase_price": 50, "expected_points": 4.5, "chance_of_playing": 100.0, "team": 5},
        {"element": 6, "web_name": "DEF_SUB1", "position": "DEF", "value": 50, "now_cost": 50, "purchase_price": 50, "expected_points": 3.8, "chance_of_playing": 100.0, "team": 6},
        {"element": 7, "web_name": "DEF_SUB2_BLOAT", "position": "DEF", "value": 50, "now_cost": 50, "purchase_price": 50, "expected_points": 3.5, "chance_of_playing": 100.0, "team": 7},
        {"element": 8, "web_name": "MID_1", "position": "MID", "value": 80, "now_cost": 80, "purchase_price": 80, "expected_points": 6.5, "chance_of_playing": 100.0, "team": 8},
        {"element": 9, "web_name": "MID_2", "position": "MID", "value": 75, "now_cost": 75, "purchase_price": 75, "expected_points": 6.0, "chance_of_playing": 100.0, "team": 9},
        {"element": 10, "web_name": "MID_3", "position": "MID", "value": 70, "now_cost": 70, "purchase_price": 70, "expected_points": 5.8, "chance_of_playing": 100.0, "team": 10},
        {"element": 11, "web_name": "MID_4", "position": "MID", "value": 60, "now_cost": 60, "purchase_price": 60, "expected_points": 4.8, "chance_of_playing": 100.0, "team": 11},
        {"element": 12, "web_name": "MID_SUB3_BLOAT", "position": "MID", "value": 60, "now_cost": 60, "purchase_price": 60, "expected_points": 3.2, "chance_of_playing": 100.0, "team": 12},
        {"element": 13, "web_name": "FWD_1", "position": "FWD", "value": 90, "now_cost": 90, "purchase_price": 90, "expected_points": 7.5, "chance_of_playing": 100.0, "team": 13},
        {"element": 14, "web_name": "FWD_2", "position": "FWD", "value": 75, "now_cost": 75, "purchase_price": 75, "expected_points": 6.0, "chance_of_playing": 100.0, "team": 14},
        {"element": 15, "web_name": "FWD_3", "position": "FWD", "value": 70, "now_cost": 70, "purchase_price": 70, "expected_points": 5.5, "chance_of_playing": 100.0, "team": 15},
    ]
    squad_df = pd.DataFrame(squad_rows)

    # Pool targets
    pool_rows = squad_rows + [
        # Cheap enablers
        {"element": 20, "web_name": "DEF_ENABLER", "position": "DEF", "value": 40, "now_cost": 40, "expected_points": 2.0, "chance_of_playing": 100.0, "team": 16},
        {"element": 21, "web_name": "MID_ENABLER", "position": "MID", "value": 45, "now_cost": 45, "expected_points": 2.2, "chance_of_playing": 100.0, "team": 17},
        # Premium upgrade targets
        {"element": 22, "web_name": "PREMIUM_MID", "position": "MID", "value": 75, "now_cost": 75, "expected_points": 7.2, "chance_of_playing": 100.0, "team": 18},
    ]
    pool_df = pd.DataFrame(pool_rows)
    return squad_df, pool_df


def test_diagnose_capital_efficiency_detects_bench_bloat():
    """Verify that excessive funds in Sub 2/Sub 3 and backup GKP are flagged as Bench Bloat."""
    squad_df, _ = make_bloated_squad_and_pool()
    diag = SquadHealthDiagnosis()
    report = diag.diagnose_capital_efficiency(squad_df)

    assert report.bench_bloat_detected is True
    # £3.5m dead capital (35 tenths)
    assert report.dead_bench_capital_tenths == 35
    assert report.dead_bench_capital_m == 3.5
    assert len(report.bench_harvest_candidates) == 3
    # Check roles identified
    roles = [c["role"] for c in report.bench_harvest_candidates]
    assert "Backup GKP" in roles
    assert "Sub 2" in roles
    assert "Sub 3" in roles
    assert "Bench Bloat Detected" in report.recommendation


def test_diagnose_capital_efficiency_clean_on_lean_bench():
    """Verify that a lean squad (4.0 GKP, 4.0 Sub 2, 4.5 Sub 3) has no bloat detected."""
    squad_df, _ = make_bloated_squad_and_pool()
    # Lean bench:
    squad_df.loc[squad_df["element"] == 2, ["value", "now_cost", "purchase_price"]] = 40
    squad_df.loc[squad_df["element"] == 7, ["value", "now_cost", "purchase_price"]] = 40
    squad_df.loc[squad_df["element"] == 12, ["value", "now_cost", "purchase_price"]] = 45

    diag = SquadHealthDiagnosis()
    report = diag.diagnose_capital_efficiency(squad_df)

    assert report.bench_bloat_detected is False
    assert report.dead_bench_capital_tenths == 0
    assert len(report.bench_harvest_candidates) == 0


def test_find_bench_harvest_upgrades():
    """Verify that AffordabilityPathFinder generates 2-step enabler downgrade -> starter upgrade moves."""
    squad_df, pool_df = make_bloated_squad_and_pool()
    finder = AffordabilityPathFinder()

    # Harvest from MID_SUB3_BLOAT (releasing £1.5m) to upgrade starting MID_4 (60) to PREMIUM_MID (75)
    plans = finder.find_bench_harvest_upgrades(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank_tenths=0,
        min_gain_xp=1.5,
    )

    assert len(plans) > 0
    top = plans[0]
    # Harvest out element 12 (MID_SUB3_BLOAT) -> enabler 21 (MID_ENABLER)
    assert top["harvest_out_element"] == 12
    assert top["harvest_in_element"] == 21
    # Upgrade starting MID_4 (11) -> PREMIUM_MID (22)
    assert top["upgrade_out_element"] == 11
    assert top["upgrade_in_element"] == 22
    assert top["net_xp_gain"] >= 2.0
    assert "Harvest £1.5m" in top["summary"]
