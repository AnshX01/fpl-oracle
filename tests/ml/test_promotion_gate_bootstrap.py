"""
Unit and Acceptance Test Suite for WI-3.9 (Paired Block-Bootstrap Promotion Gate)
and WI-3.10 (Position-Specific NNLS Stacking Weights).

Verifies:
1. Paired block-bootstrap over gameweeks with 10,000 resamples.
2. Promotion criteria:
   - 95% CI strictly negative -> PROMOTES
   - 95% CI includes 0, secondary metric improves, top-decile not worse -> PROMOTES
   - 95% CI includes 0, secondary degrades -> REJECTS
   - 95% CI strictly positive -> REJECTS
3. Logging of CI, resample count, and seed in manifest version entry.
4. Out-of-fold NNLS stacking weights per position (summing to 1, non-negative).
5. Positional stacking weights integration into ScoringEnsemble.
"""

import numpy as np

from fpl_oracle.ml.ensemble import ScoringEnsemble
from fpl_oracle.ml.model_registry import check_promotion_gate, paired_block_bootstrap_gw


def test_paired_block_bootstrap_ci_excludes_zero_good():
    """Verify candidate strictly superior across gameweeks is promoted with ci_excludes_zero_good."""
    rng = np.random.default_rng(42)
    n_gws = 38
    players_per_gw = 100
    gws = np.repeat(np.arange(1, n_gws + 1), players_per_gw)
    y_true = rng.poisson(2.0, size=len(gws)).astype(float)

    # Active predictions with RMSE ~ 2.0
    act_pred = y_true + rng.normal(0, 2.0, size=len(gws))
    # Candidate predictions with RMSE ~ 1.7 (significantly better)
    cand_pred = y_true + rng.normal(0, 1.7, size=len(gws))

    res = paired_block_bootstrap_gw(
        candidate_predictions=cand_pred,
        active_predictions=act_pred,
        targets=y_true,
        gameweeks=gws,
        n_resamples=10000,
        seed=42,
    )

    assert res["promoted"] is True
    assert res["decision_rule"] == "ci_excludes_zero_good"
    assert res["ci"][1] < 0.0  # upper bound strictly negative
    assert res["n_resamples"] == 10000
    assert res["seed"] == 42
    assert res["p_cand_better"] > 0.99


def test_paired_block_bootstrap_indistinguishable_with_secondary():
    """Verify candidate with indistinguishable RMSE is promoted if secondary metric improves and top decile not worse."""
    rng = np.random.default_rng(42)
    n_gws = 25
    players_per_gw = 80
    gws = np.repeat(np.arange(1, n_gws + 1), players_per_gw)
    y_true = rng.poisson(2.0, size=len(gws)).astype(float)

    noise = rng.normal(0, 1.5, size=len(gws))
    act_pred = y_true + noise
    # Candidate is nearly identical to active (CI will span 0)
    cand_pred = y_true + noise + rng.normal(0, 0.02, size=len(gws))

    # Case A: Secondary metric improves -> Promotes
    res_pass = paired_block_bootstrap_gw(
        candidate_predictions=cand_pred,
        active_predictions=act_pred,
        targets=y_true,
        gameweeks=gws,
        n_resamples=10000,
        seed=42,
        candidate_secondary=0.18,  # e.g. lower pinball loss
        active_secondary=0.22,
    )
    assert res_pass["promoted"] is True
    assert res_pass["decision_rule"] == "indistinguishable_with_secondary_and_top_decile"
    assert res_pass["ci"][0] <= 0.0 <= res_pass["ci"][1]

    # Case B: Secondary metric degrades -> Rejects
    res_fail = paired_block_bootstrap_gw(
        candidate_predictions=cand_pred,
        active_predictions=act_pred,
        targets=y_true,
        gameweeks=gws,
        n_resamples=10000,
        seed=42,
        candidate_secondary=0.25,  # worse secondary
        active_secondary=0.22,
    )
    assert res_fail["promoted"] is False
    assert res_fail["decision_rule"] == "rejected"


def test_paired_block_bootstrap_rejects_inferior_candidate():
    """Verify candidate strictly worse across gameweeks is rejected."""
    rng = np.random.default_rng(42)
    n_gws = 30
    players_per_gw = 100
    gws = np.repeat(np.arange(1, n_gws + 1), players_per_gw)
    y_true = rng.poisson(2.0, size=len(gws)).astype(float)

    act_pred = y_true + rng.normal(0, 1.5, size=len(gws))
    cand_pred = y_true + rng.normal(0, 2.2, size=len(gws))  # worse

    res = paired_block_bootstrap_gw(
        candidate_predictions=cand_pred,
        active_predictions=act_pred,
        targets=y_true,
        gameweeks=gws,
        n_resamples=10000,
        seed=42,
    )

    assert res["promoted"] is False
    assert res["ci"][0] > 0.0  # lower bound strictly positive (worse)


def test_check_promotion_gate_integration():
    """Verify check_promotion_gate handles paired bootstrap data and rejects appropriately."""
    rng = np.random.default_rng(42)
    gws = np.repeat(np.arange(1, 20), 50)
    y_true = rng.poisson(2.0, size=len(gws)).astype(float)
    act_pred = y_true + rng.normal(0, 1.5, size=len(gws))
    cand_pred_good = y_true + rng.normal(0, 1.2, size=len(gws))
    cand_pred_bad = y_true + rng.normal(0, 2.0, size=len(gws))

    # Good candidate passes
    cand_metrics_good = {
        "ml_mae": 1.1,
        "base_mae": 1.5,
        "candidate_predictions": cand_pred_good,
        "active_predictions": act_pred,
        "targets": y_true,
        "gameweeks": gws,
    }
    passed_good, reason_good = check_promotion_gate(cand_metrics_good)
    assert passed_good is True
    assert "bootstrap_result" in cand_metrics_good
    assert cand_metrics_good["bootstrap_result"]["n_resamples"] == 10000

    # Bad candidate fails
    cand_metrics_bad = {
        "ml_mae": 1.1,
        "base_mae": 1.5,
        "candidate_predictions": cand_pred_bad,
        "active_predictions": act_pred,
        "targets": y_true,
        "gameweeks": gws,
    }
    passed_bad, reason_bad = check_promotion_gate(cand_metrics_bad)
    assert passed_bad is False
    assert "bootstrap promotion gate" in reason_bad.lower()


def test_nnls_positional_stacking_weights_properties():
    """Verify NNLS stacking weights sum to 1, are non-negative, and apply positionally (WI-3.10)."""
    ens = ScoringEnsemble()

    rng = np.random.default_rng(42)
    n_samples = 2000
    positions = rng.choice(["GKP", "DEF", "MID", "FWD"], size=n_samples)
    y_true = rng.poisson(2.5, size=n_samples).astype(float)

    raw_xp = y_true * 0.9 + rng.normal(0, 0.5, size=n_samples)
    base_recent = y_true * 0.7 + rng.normal(0, 1.0, size=n_samples)
    base_season = y_true * 0.8 + rng.normal(0, 0.8, size=n_samples)

    weights = ens.fit_positional_stacking_weights(
        raw_xp=raw_xp,
        base_recent=base_recent,
        base_season=base_season,
        targets=y_true,
        positions=positions,
    )

    # 1. Weights structure per position
    assert set(weights.keys()) == {"GKP", "DEF", "MID", "FWD"}
    for _pos, (w_ml, w_rec, w_sea) in weights.items():
        assert w_ml >= 0.0
        assert w_rec >= 0.0
        assert w_sea >= 0.0
        assert abs((w_ml + w_rec + w_sea) - 1.0) < 1e-3

    # 2. Serialization and deserialization in calibration dict
    cal_dict = ens.get_calibration_dict()
    assert "positional_blend_weights" in cal_dict
    assert cal_dict["positional_blend_weights"]["MID"] == [round(x, 4) for x in weights["MID"]]
