"""
Tests for WI-1.2 Ensemble Completeness and Decomposition Exactness.

Acceptance criteria:
1. Exact arithmetic decomposition:
   raw_expected_points == exp_appearance + exp_goals_pts + exp_assists_pts
   + exp_cs_pts + exp_defcon_pts + exp_bonus_pts + exp_saves_pts + exp_pen_saves_pts
   - exp_gc_deduction - exp_card_deduction - exp_og_deduction - exp_pen_miss_deduction
   holds exactly (0 error) across all rows.
2. In unblended mode, expected_points matches decomposition sum exactly.
3. Conceded goals deduction applies strictly to GKP and DEF only (0 for MID/FWD).
4. Clean sheet points require >= 60 minutes (0 when p_min60 is 0).
5. DefCon points never apply to GKP (0 for GKP regardless of p_defcon).
6. Own goals deduct -2 points per expected own goal.
7. Penalties missed deduct -2 points per expected penalty miss.
8. Historical data audit: GKP has 0 DefCon actions across historical records.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.ml.ensemble import ScoringEnsemble


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Multi-position feature dataset covering all roles and playing time variations."""
    return pd.DataFrame(
        [
            # GKP starting 90 mins
            {
                "element": 1,
                "position": "GKP",
                "pos_GKP": 1.0,
                "pos_DEF": 0.0,
                "pos_MID": 0.0,
                "pos_FWD": 0.0,
                "chance_of_playing": 100.0,
                "roll_own_goals_5": 0.0,
                "roll_penalties_missed_5": 0.0,
            },
            # DEF starting 90 mins with own goal history
            {
                "element": 2,
                "position": "DEF",
                "pos_GKP": 0.0,
                "pos_DEF": 1.0,
                "pos_MID": 0.0,
                "pos_FWD": 0.0,
                "chance_of_playing": 100.0,
                "roll_own_goals_5": 1.0,
                "roll_penalties_missed_5": 0.0,
            },
            # MID starting 90 mins with penalty taker duties
            {
                "element": 3,
                "position": "MID",
                "pos_GKP": 0.0,
                "pos_DEF": 0.0,
                "pos_MID": 1.0,
                "pos_FWD": 0.0,
                "chance_of_playing": 100.0,
                "roll_own_goals_5": 0.0,
                "roll_penalties_missed_5": 1.0,
            },
            # FWD sub (< 60 mins)
            {
                "element": 4,
                "position": "FWD",
                "pos_GKP": 0.0,
                "pos_DEF": 0.0,
                "pos_MID": 0.0,
                "pos_FWD": 1.0,
                "chance_of_playing": 100.0,
                "roll_own_goals_5": 0.0,
                "roll_penalties_missed_5": 0.0,
            },
            # Injured player (0 minutes)
            {
                "element": 5,
                "position": "MID",
                "pos_GKP": 0.0,
                "pos_DEF": 0.0,
                "pos_MID": 1.0,
                "pos_FWD": 0.0,
                "chance_of_playing": 0.0,
                "roll_own_goals_5": 0.0,
                "roll_penalties_missed_5": 0.0,
            },
        ]
    )


@pytest.fixture
def sample_components() -> dict[str, np.ndarray]:
    return {
        "p_starts": np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        "p_min60": np.array([0.95, 0.90, 0.85, 0.0, 0.0]),
        "expected_minutes": np.array([90.0, 88.0, 85.0, 25.0, 0.0]),
        "expected_goals": np.array([0.001, 0.08, 0.35, 0.50, 0.0]),
        "expected_assists": np.array([0.005, 0.15, 0.30, 0.20, 0.0]),
        "p_clean_sheet": np.array([0.45, 0.40, 0.35, 0.30, 0.0]),
        "expected_goals_conceded": np.array([1.10, 1.20, 1.50, 1.80, 0.0]),
        "expected_saves": np.array([3.50, 0.0, 0.0, 0.0, 0.0]),
        "p_defcon": np.array([0.50, 0.60, 0.40, 0.20, 0.0]),
        "expected_card_deduction": np.array([0.10, 0.25, 0.30, 0.15, 0.0]),
        "expected_bonus": np.array([0.40, 0.50, 0.80, 0.90, 0.0]),
        "expected_penalties_saved": np.array([0.05, 0.0, 0.0, 0.0, 0.0]),
        "expected_own_goals": np.array([0.005, 0.030, 0.005, 0.002, 0.0]),
        "expected_penalties_missed": np.array([0.0, 0.0, 0.040, 0.050, 0.0]),
    }


def test_decomposition_sums_exactly_to_final_xp(sample_features, sample_components):
    """Property test: unblended aggregate_components must equal decomposition sum with 0 discrepancy."""
    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(sample_components, sample_features)

    reconstructed = ScoringEnsemble.verify_decomposition(res_df)

    # 1. Reconstructed sum matches raw_expected_points identically
    assert np.allclose(reconstructed.values, res_df["raw_expected_points"].values, atol=1e-5), (
        f"Mismatch between reconstructed and raw: {reconstructed.values} vs {res_df['raw_expected_points'].values}"
    )

    # 2. In unblended mode, expected_points is identical to raw_expected_points
    assert np.allclose(res_df["expected_points"].values, res_df["raw_expected_points"].values, atol=1e-5), (
        f"Unblended expected_points differs from raw: {res_df['expected_points'].values} vs {res_df['raw_expected_points'].values}"
    )

    # 3. Injured player (row index 4) has exactly 0.0 across all components
    row_injured = res_df.iloc[4]
    assert row_injured["expected_points"] == 0.0
    assert row_injured["exp_appearance"] == 0.0
    assert row_injured["exp_goals_pts"] == 0.0
    assert row_injured["exp_cs_pts"] == 0.0
    assert row_injured["exp_gc_deduction"] == 0.0


def test_conceded_deduction_applies_to_gkp_and_def_only(sample_features, sample_components):
    """Conceded goals deduction (-1 per 2 conceded) must only penalise GKP and DEF."""
    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(sample_components, sample_features)

    # GKP (row 0) and DEF (row 1) must have positive conceded deduction
    assert res_df.loc[0, "exp_gc_deduction"] > 0.0, "GKP must have conceded deduction"
    assert res_df.loc[1, "exp_gc_deduction"] > 0.0, "DEF must have conceded deduction"

    # MID (row 2) and FWD (row 3) must have exactly 0.0 conceded deduction despite having expected_goals_conceded
    assert res_df.loc[2, "exp_gc_deduction"] == 0.0, "MID must not suffer conceded deduction"
    assert res_df.loc[3, "exp_gc_deduction"] == 0.0, "FWD must not suffer conceded deduction"


def test_clean_sheet_requires_minimum_60_minutes(sample_features, sample_components):
    """Clean sheet points require p_min60 > 0."""
    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(sample_components, sample_features)

    # GKP (row 0) and DEF (row 1) have p_min60 > 0 -> clean sheet points > 0
    assert res_df.loc[0, "exp_cs_pts"] > 0.0
    assert res_df.loc[1, "exp_cs_pts"] > 0.0

    # FWD (row 3) has p_min60 == 0.0 -> clean sheet points must be 0.0
    assert res_df.loc[3, "exp_cs_pts"] == 0.0


def test_defcon_never_applies_to_gkp(sample_features, sample_components):
    """Goalkeepers never receive DefCon points (+2) under 2026/27 rules."""
    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(sample_components, sample_features)

    # GKP (row 0) had p_defcon = 0.50, but must receive 0.0 DefCon points
    assert res_df.loc[0, "exp_defcon_pts"] == 0.0, "GKP must receive 0.0 DefCon points"

    # DEF (row 1) and MID (row 2) must receive positive DefCon points
    assert res_df.loc[1, "exp_defcon_pts"] > 0.0, "DEF must be eligible for DefCon"
    assert res_df.loc[2, "exp_defcon_pts"] > 0.0, "MID must be eligible for DefCon"


def test_own_goals_and_penalty_misses_deductions(sample_features, sample_components):
    """Own goals and missed penalties apply -2 points each."""
    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(sample_components, sample_features)

    # DEF (row 1): expected_own_goals = 0.030 -> deduction = 0.030 * 2 = 0.06
    assert res_df.loc[1, "exp_og_deduction"] == pytest.approx(0.06, abs=0.01)

    # MID (row 2): expected_penalties_missed = 0.040 -> deduction = 0.040 * 2 = 0.08
    assert res_df.loc[2, "exp_pen_miss_deduction"] == pytest.approx(0.08, abs=0.01)


def test_randomized_property_decomposition_exactness():
    """Stress test with 100 randomized player rows confirming 0 arithmetic deviation."""
    rng = np.random.default_rng(42)
    n = 100
    positions = rng.choice(["GKP", "DEF", "MID", "FWD"], size=n)
    pos_df = pd.get_dummies(positions, prefix="pos", dtype=float)
    for p in ["GKP", "DEF", "MID", "FWD"]:
        if f"pos_{p}" not in pos_df.columns:
            pos_df[f"pos_{p}"] = 0.0

    X = pd.DataFrame(
        {
            "position": positions,
            "pos_GKP": pos_df["pos_GKP"].values,
            "pos_DEF": pos_df["pos_DEF"].values,
            "pos_MID": pos_df["pos_MID"].values,
            "pos_FWD": pos_df["pos_FWD"].values,
            "chance_of_playing": rng.choice([0.0, 50.0, 75.0, 100.0], size=n),
            "roll_own_goals_5": rng.integers(0, 3, size=n).astype(float),
            "roll_penalties_missed_5": rng.integers(0, 2, size=n).astype(float),
        }
    )

    comps = {
        "p_starts": rng.uniform(0.0, 1.0, size=n),
        "p_min60": rng.uniform(0.0, 0.9, size=n),
        "expected_minutes": rng.uniform(0.0, 90.0, size=n),
        "expected_goals": rng.uniform(0.0, 1.2, size=n),
        "expected_assists": rng.uniform(0.0, 0.8, size=n),
        "p_clean_sheet": rng.uniform(0.1, 0.6, size=n),
        "expected_goals_conceded": rng.uniform(0.5, 3.0, size=n),
        "expected_saves": rng.uniform(0.0, 6.0, size=n),
        "p_defcon": rng.uniform(0.0, 0.8, size=n),
        "expected_card_deduction": rng.uniform(0.0, 0.5, size=n),
        "expected_bonus": rng.uniform(0.0, 1.5, size=n),
        "expected_penalties_saved": rng.uniform(0.0, 0.1, size=n),
        "expected_own_goals": rng.uniform(0.0, 0.05, size=n),
        "expected_penalties_missed": rng.uniform(0.0, 0.05, size=n),
    }

    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(comps, X)
    reconstructed = ScoringEnsemble.verify_decomposition(res_df)

    diff = np.abs(reconstructed.values - res_df["raw_expected_points"].values)
    max_diff = float(np.max(diff))
    assert max_diff == 0.0, f"Found non-zero deviation in decomposition property test: {max_diff}"


def test_golden_fixture_consistency():
    """Verify ensemble outputs against frozen golden fixture."""
    import json
    from pathlib import Path

    golden_path = Path(__file__).resolve().parent.parent / "fixtures" / "golden_gw" / "gw6_ensemble_golden.json"
    assert golden_path.exists(), f"Golden fixture missing at {golden_path}"

    with open(golden_path, encoding="utf-8") as f:
        golden_records = json.load(f)

    X = pd.DataFrame(
        [
            {"element": 1, "web_name": "Raya", "position": "GKP", "pos_GKP": 1.0, "pos_DEF": 0.0, "pos_MID": 0.0, "pos_FWD": 0.0, "chance_of_playing": 100.0, "roll_own_goals_5": 0.0, "roll_penalties_missed_5": 0.0},
            {"element": 2, "web_name": "Gabriel", "position": "DEF", "pos_GKP": 0.0, "pos_DEF": 1.0, "pos_MID": 0.0, "pos_FWD": 0.0, "chance_of_playing": 100.0, "roll_own_goals_5": 1.0, "roll_penalties_missed_5": 0.0},
            {"element": 3, "web_name": "Saka", "position": "MID", "pos_GKP": 0.0, "pos_DEF": 0.0, "pos_MID": 1.0, "pos_FWD": 0.0, "chance_of_playing": 100.0, "roll_own_goals_5": 0.0, "roll_penalties_missed_5": 1.0},
            {"element": 4, "web_name": "Haaland", "position": "FWD", "pos_GKP": 0.0, "pos_DEF": 0.0, "pos_MID": 0.0, "pos_FWD": 1.0, "chance_of_playing": 100.0, "roll_own_goals_5": 0.0, "roll_penalties_missed_5": 0.0},
            {"element": 5, "web_name": "Injured_Mid", "position": "MID", "pos_GKP": 0.0, "pos_DEF": 0.0, "pos_MID": 1.0, "pos_FWD": 0.0, "chance_of_playing": 0.0, "roll_own_goals_5": 0.0, "roll_penalties_missed_5": 0.0},
        ]
    )

    comps = {
        "p_starts": np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        "p_min60": np.array([0.95, 0.90, 0.85, 0.85, 0.0]),
        "expected_minutes": np.array([90.0, 88.0, 85.0, 85.0, 0.0]),
        "expected_goals": np.array([0.001, 0.08, 0.35, 0.75, 0.0]),
        "expected_assists": np.array([0.005, 0.15, 0.30, 0.15, 0.0]),
        "p_clean_sheet": np.array([0.45, 0.40, 0.35, 0.35, 0.0]),
        "expected_goals_conceded": np.array([1.10, 1.20, 1.50, 1.50, 0.0]),
        "expected_saves": np.array([3.50, 0.0, 0.0, 0.0, 0.0]),
        "p_defcon": np.array([0.50, 0.60, 0.40, 0.20, 0.0]),
        "expected_card_deduction": np.array([0.10, 0.25, 0.30, 0.15, 0.0]),
        "expected_bonus": np.array([0.40, 0.50, 0.80, 1.20, 0.0]),
        "expected_penalties_saved": np.array([0.05, 0.0, 0.0, 0.0, 0.0]),
        "expected_own_goals": np.array([0.005, 0.030, 0.005, 0.002, 0.0]),
        "expected_penalties_missed": np.array([0.0, 0.0, 0.040, 0.050, 0.0]),
    }

    ens = ScoringEnsemble(blend_weights=(1.0, 0.0, 0.0))
    res_df = ens.aggregate_components(comps, X)
    for i, golden in enumerate(golden_records):
        assert res_df.loc[i, "expected_points"] == golden["expected_points"]
        assert res_df.loc[i, "raw_expected_points"] == golden["raw_expected_points"]
        assert res_df.loc[i, "exp_og_deduction"] == golden["exp_og_deduction"]
        assert res_df.loc[i, "exp_pen_miss_deduction"] == golden["exp_pen_miss_deduction"]
