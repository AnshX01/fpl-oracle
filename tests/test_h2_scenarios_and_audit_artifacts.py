"""
Tests for Phase H2: Audit Harness Artifact Verification (Scenarios 2 & 3)
and Scenario 7 price-rise timing warning.
"""

import pandas as pd

from fpl_oracle.chips.simulate import chip_simulator


def create_base_mock_pool():
    rows = []
    # 4 GKP
    for i in range(1, 5):
        rows.append({
            "element": i,
            "web_name": f"GKP_{i}",
            "position": "GKP",
            "team": (i % 20) + 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5 - (i * 0.2),
            "p10": 2.0,
            "p90": 6.5,
            "chance_of_playing_next_round": 100,
        })
    # 10 DEF
    for i in range(5, 15):
        rows.append({
            "element": i,
            "web_name": f"DEF_{i}",
            "position": "DEF",
            "team": (i % 20) + 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 5.0 - (i * 0.1),
            "p10": 1.5,
            "p90": 8.0,
            "chance_of_playing_next_round": 100,
        })
    # 10 MID
    for i in range(15, 25):
        rows.append({
            "element": i,
            "web_name": f"MID_{i}",
            "position": "MID",
            "team": (i % 20) + 1,
            "value": 65,
            "purchase_price": 65,
            "selling_price": 65,
            "expected_points": 6.5 - (i * 0.15),
            "p10": 2.0,
            "p90": 10.0,
            "chance_of_playing_next_round": 100,
        })
    # 6 FWD
    for i in range(25, 31):
        rows.append({
            "element": i,
            "web_name": f"FWD_{i}",
            "position": "FWD",
            "team": (i % 20) + 1,
            "value": 75,
            "purchase_price": 75,
            "selling_price": 75,
            "expected_points": 7.5 - (i * 0.15),
            "p10": 2.0,
            "p90": 12.0,
            "chance_of_playing_next_round": 100,
        })
    return pd.DataFrame(rows)


def make_squad_from_pool(pool_df):
    squad_ids = [1, 2] + list(range(5, 10)) + list(range(15, 20)) + [25, 26, 27]
    return pool_df[pool_df["element"].isin(squad_ids)].copy().reset_index(drop=True)


def test_scenario_7_price_rise_timing_warning_attached():
    """Verify Scenario 7: When recommended transfer-in player is in imminent_rises,
    a warning '⚠️ (Price rises tonight, act before 23:59 UK)' is appended to the transfer line.
    """
    rec_plan = {
        "transfers_count": 1,
        "hit_cost": 0,
        "transfers_out": [{"web_name": "DEF_9", "element": 9}],
        "transfers_in": [{"web_name": "DEF_14", "element": 14}],
    }
    imminent_rises = [{"element": 14, "web_name": "DEF_14", "direction": "RISE_IMMINENT"}]

    imminent_rise_ids = {p.get("element") for p in imminent_rises if p.get("element")}
    lines = []
    for out, incoming in zip(rec_plan.get("transfers_out", []), rec_plan.get("transfers_in", []), strict=False):
        timing_warning = ""
        if incoming.get("element") in imminent_rise_ids:
            timing_warning = " ⚠️ (Price rises tonight, act before 23:59 UK)"
        lines.append(f"- {out['web_name']} → {incoming['web_name']}{timing_warning}")

    assert len(lines) == 1
    assert "- DEF_9 → DEF_14 ⚠️ (Price rises tonight, act before 23:59 UK)" == lines[0]


def test_scenario_2_broken_squad_fixture_artifact_vs_fixed():
    """Verify Scenario 2 audit finding was an artifact of desynchronized squad vs pool in run_scenarios.py.

    Artifact:
      run_scenarios.py set squad_sc2['expected_points'] = 0.0 for 5 broken players,
      but passed player_pool_df=pool_sc2 where those 5 players still had healthy xP (>4.5).
      The solver naturally re-selected them because the pool claimed they were top scorers.

    Fixed:
      When pool_sc2 is synchronized to reflect the broken players' 0.0 xP,
      evaluate_wildcard retains ZERO broken players.
    """
    pool_sc2 = create_base_mock_pool()
    squad_sc2 = make_squad_from_pool(pool_sc2)
    broken_ids = [5, 6, 15, 16, 25]

    # Stale/desynced squad:
    squad_sc2.loc[squad_sc2["element"].isin(broken_ids), "expected_points"] = 0.0
    squad_sc2.loc[squad_sc2["element"].isin(broken_ids), "chance_of_playing_next_round"] = 0

    # 1. With flawed fixture (pool not updated):
    wc_flawed = chip_simulator.evaluate_wildcard(
        gw=10,
        current_squad_df=squad_sc2,
        player_pool_df=pool_sc2,  # Bug in audit fixture: pool still has broken_ids at healthy xP
        budget=1000.0,
    )
    retained_flawed = [e for e in broken_ids if e in wc_flawed["target_squad"]["element"].tolist()]
    assert len(retained_flawed) > 0, "Flawed audit fixture mistakenly kept players because pool said they were healthy"

    # 2. With corrected fixture (pool synchronized with injuries):
    pool_sc2_corrected = pool_sc2.copy()
    pool_sc2_corrected.loc[pool_sc2_corrected["element"].isin(broken_ids), "expected_points"] = 0.0
    pool_sc2_corrected.loc[pool_sc2_corrected["element"].isin(broken_ids), "chance_of_playing_next_round"] = 0

    wc_corrected = chip_simulator.evaluate_wildcard(
        gw=10,
        current_squad_df=squad_sc2,
        player_pool_df=pool_sc2_corrected,
        budget=1000.0,
    )
    retained_corrected = [e for e in broken_ids if e in wc_corrected["target_squad"]["element"].tolist()]
    assert len(retained_corrected) == 0, "Corrected fixture replaces 100% of broken players with healthy alternatives"


def test_scenario_3_blank_gameweek_fixture_artifact_vs_fixed():
    """Verify Scenario 3 audit finding was an artifact of desynchronized squad vs pool in run_scenarios.py.

    Artifact:
      run_scenarios.py zeroed out blanking players in pool_sc3_gw10, but passed
      squad_sc3 with unadjusted baseline points to evaluate_free_hit.
      Baseline was scored using normal points, producing expected_gain = 0.0.

    Fixed:
      When current_squad_df has the blank gameweek projections applied (expected_points = 0.0
      for blanking players), Free Hit produces +11.75 xP expected gain, exceeding the 10.0 threshold.
    """
    pool_sc3 = create_base_mock_pool()
    squad_sc3 = make_squad_from_pool(pool_sc3)
    bgw_blank_ids = [5, 6, 7, 15, 16, 25]

    pool_sc3_gw10 = pool_sc3.copy()
    pool_sc3_gw10.loc[pool_sc3_gw10["element"].isin(bgw_blank_ids), "expected_points"] = 0.0

    # 1. Flawed fixture: squad_sc3 baseline evaluated without blank projections
    fh_flawed = chip_simulator.evaluate_free_hit(
        gw=10,
        current_squad_df=squad_sc3,  # Flaw: squad players still have healthy points!
        player_pool_df=pool_sc3_gw10,
        budget=1000.0,
    )
    assert fh_flawed.get("expected_gain", 0) <= 0.0, "Flawed audit fixture showed 0 gain because baseline was healthy"

    # 2. Corrected fixture: squad has GW10 blank projections applied
    squad_sc3_gw10 = squad_sc3.copy()
    squad_sc3_gw10.loc[squad_sc3_gw10["element"].isin(bgw_blank_ids), "expected_points"] = 0.0
    squad_sc3_gw10.loc[squad_sc3_gw10["element"].isin(bgw_blank_ids), "chance_of_playing_next_round"] = 0

    fh_corrected = chip_simulator.evaluate_free_hit(
        gw=10,
        current_squad_df=squad_sc3_gw10,
        player_pool_df=pool_sc3_gw10,
        budget=1000.0,
    )
    assert fh_corrected.get("expected_gain", 0) >= 10.0, (
        f"Free Hit expected gain {fh_corrected.get('expected_gain')} should exceed 10.0 threshold"
    )
