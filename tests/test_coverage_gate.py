from fpl_oracle.ml.eval import ROLLING_ORIGIN_SCHEMA_KEYS
from fpl_oracle.ml.model_registry import check_promotion_gate

ORIGIN = "Holdout 2025-26 (trained on 2023-24, 2024-25)"


def origins(cov):
    base = {k: 0.0 for k in ROLLING_ORIGIN_SCHEMA_KEYS} | dict(mae=0.9, best_baseline_mae=1.0, gain_vs_baseline=0.1)
    seasons = ["Holdout 2024-25", ORIGIN, "Holdout 2026-27"]
    return [dict(base, season=n, interval_80_coverage_pct=cov if n == ORIGIN else 80.0) for n in seasons]


def gate(cov, active):
    metrics = dict(ml_mae=0.9, base_mae=1.0)
    return check_promotion_gate(
        metrics, rolling_origins=origins(cov), active_origins=origins(active) if active else None
    )


def test_outside_band_passes_only_when_no_worse_than_active_on_same_side():
    assert gate(85.04, 84.58)[0] and "no worse than the active" in gate(85.04, 84.58)[1]
    assert not gate(85.04, None)[0]
    assert not gate(86.0, 84.58)[0]  # more than 1 point further from 80% than the active model
    assert not gate(85.04, 77.0)[0]  # active model was on the other side
    assert gate(82.0, 84.58)[1] == "Candidate passed all promotion gate criteria."
