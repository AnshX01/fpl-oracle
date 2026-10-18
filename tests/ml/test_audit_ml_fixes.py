"""
Regression tests for the ML audit fixes: availability scaling, conditional components (no double minutes
discount), DefCon label handling, as-of Dixon-Coles, feature-builder parity, refit-after-gate promotion.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.data.features import FeatureEngineering, compute_player_rolling_stats
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.dixon_coles import DixonColesModel
from fpl_oracle.ml.ensemble import ScoringEnsemble
from fpl_oracle.ml.model_registry import ModelRegistry, check_promotion_gate
from fpl_oracle.news.reconcile import official_chance_of_playing

# --------------------------------------------------------------------------------------------------
# Ensemble: availability must scale every xP component, including the blended baselines
# --------------------------------------------------------------------------------------------------


def _mid_row(cop: float, limit: float | None = None) -> pd.DataFrame:
    row = {
        "pos_GKP": 0.0,
        "pos_DEF": 0.0,
        "pos_MID": 1.0,
        "pos_FWD": 0.0,
        "roll_points_5": 6.0,
        "roll_points_8": 6.0,
        "roll_starts_ratio_5": 1.0,
        "opponent_difficulty": 3.0,
        "chance_of_playing": cop,
    }
    if limit is not None:
        row["news_minutes_limit"] = limit
    return pd.DataFrame([row])


def _mid_components() -> dict[str, np.ndarray]:
    return {
        "p_min60": np.array([0.9]),
        "p_starts": np.array([0.92]),
        "p_play": np.array([0.97]),
        "expected_minutes": np.array([80.0]),
        "expected_goals": np.array([0.4]),
        "expected_assists": np.array([0.25]),
        "p_clean_sheet": np.array([0.35]),
        "expected_goals_conceded": np.array([1.2]),
        "expected_saves": np.array([0.0]),
        "p_defcon": np.array([0.15]),
        "expected_card_deduction": np.array([0.1]),
        "expected_bonus": np.array([0.4]),
    }


def test_blended_baselines_are_scaled_by_chance_of_playing():
    ens = ScoringEnsemble(z10=-0.8, z90=1.0, blend_weights=(0.70, 0.06, 0.24))
    xp = {c: float(ens.aggregate_components(_mid_components(), _mid_row(c))["expected_points"][0]) for c in (100, 50)}
    assert xp[50] == pytest.approx(0.5 * xp[100], abs=0.05)


def test_availability_is_applied_once_to_every_deduction():
    """chance_of_playing 50% halves each term exactly once (no cop^2 on own goals / missed penalties)."""
    ens = ScoringEnsemble(z10=-0.8, z90=1.0, blend_weights=(1.0, 0.0, 0.0))
    comps = {**_mid_components(), "expected_own_goals": np.array([0.05]), "expected_penalties_missed": np.array([0.05])}
    full = ens.aggregate_components(comps, _mid_row(100))
    half = ens.aggregate_components(comps, _mid_row(50))
    for col in ("exp_og_deduction", "exp_pen_miss_deduction", "exp_card_deduction", "exp_gc_deduction"):
        assert float(half[col][0]) == pytest.approx(0.5 * float(full[col][0]), abs=0.006), col


def test_news_minutes_limit_scales_blended_baselines_and_defcon():
    ens = ScoringEnsemble(z10=-0.8, z90=1.0, blend_weights=(0.70, 0.06, 0.24))
    free = ens.aggregate_components(_mid_components(), _mid_row(100))
    capped = ens.aggregate_components(_mid_components(), _mid_row(100, limit=40.0))
    # 40 of 80 expected minutes -> every unconditional component (and the baselines) halves
    assert float(capped["expected_points"][0]) < 0.7 * float(free["expected_points"][0])
    assert float(capped["exp_defcon_pts"][0]) == pytest.approx(0.5 * float(free["exp_defcon_pts"][0]), abs=0.01)


def test_interval_buckets_vectorised_and_fall_back_to_standardized_residuals():
    ens = ScoringEnsemble(z10=-0.8, z90=1.0, blend_weights=(0.72, 0.04, 0.24))
    ens.bucket_quantiles = {"MID_high": {"q10": -2.0, "q90": 5.0, "n": 100}}
    res = ens.aggregate_components(_mid_components(), _mid_row(100))
    xp = float(res["expected_points"][0])
    assert float(res["p10"][0]) == pytest.approx(max(0.0, xp - 2.0), abs=0.01)
    assert float(res["p90"][0]) == pytest.approx(xp + 5.0, abs=0.01)

    ens.bucket_quantiles = {}
    res_z = ens.aggregate_components(_mid_components(), _mid_row(100))
    assert float(res_z["p90"][0]) > float(res_z["expected_points"][0])


# --------------------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------------------


def _temporal_X(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "roll_minutes_5": rng.uniform(0, 90, size=n),
            "pos_DEF": 1.0,
            "pos_MID": 0.0,
            "pos_FWD": 0.0,
            "pos_GKP": 0.0,
            "is_2026_27": 0.0,
        }
    )
    X.attrs["meta"] = pd.DataFrame({"season": "2025-26", "round": np.repeat(np.arange(1, 21), n // 20)})
    return X


def test_clean_sheet_model_is_conditional_on_playing_60_minutes():
    """P(CS) is multiplied by P(60+) in the ensemble, so it must not already contain P(60+)."""
    n = 4000
    X = _temporal_X(n)
    rng = np.random.default_rng(1)
    min60 = (rng.uniform(size=n) < 0.5).astype(float)
    cs = min60 * (rng.uniform(size=n) < 0.30).astype(float)  # 30% CS when on the pitch, 0 otherwise
    Y = pd.DataFrame(
        {
            "target_min60": min60,
            "target_clean_sheet": cs,
            "target_goals_conceded": rng.poisson(1.2, size=n).astype(float),
        }
    )
    model = DefendingModel()
    model.fit(X, Y)
    p_cs = model.predict(X)["p_clean_sheet"]
    assert 0.25 < float(np.mean(p_cs)) < 0.35  # ~0.30, not the unconditional ~0.15


def _defcon_XY(n: int = 2400):
    X = _temporal_X(n, seed=3)
    rng = np.random.default_rng(4)
    actions = rng.poisson(6.0, size=n).astype(float)
    Y = pd.DataFrame(
        {
            "target_defcon": (actions >= 10).astype(float),
            "target_defcon_actions": actions,
            "target_minutes": np.full(n, 90.0),
        }
    )
    return X, Y


def test_defcon_fit_ignores_unlabelled_and_unplayed_rows():
    X, Y = _defcon_XY()
    half = len(X) // 2
    Y.loc[: half - 1, ["target_defcon", "target_defcon_actions"]] = np.nan  # season without DefCon
    Y.loc[half::2, "target_minutes"] = 0.0  # did not play
    model = DefConModel()
    model.fit(X, Y)
    assert model.is_fitted
    p = model.predict(X)["p_defcon"]
    assert np.isfinite(p).all()
    # trained on played rows only: P(DefCon | played), here Poisson(6) >= 10 ~ 8%
    assert 0.03 < float(p.mean()) < 0.2


def test_defcon_without_any_labelled_row_predicts_zero():
    X, Y = _defcon_XY(600)
    Y[["target_defcon", "target_defcon_actions"]] = np.nan
    model = DefConModel()
    model.fit(X, Y)
    assert float(model.predict(X)["p_defcon"].sum()) == 0.0


def test_team_goal_reconciliation_hands_absent_players_share_to_teammates():
    model = AttackingModel()
    goals = np.array([0.5, 0.3, 0.2, 0.2])
    assists = np.array([0.3, 0.2, 0.2, 0.1])
    X = pd.DataFrame({"implied_team_xG": np.full(4, 2.0)})
    avail = np.array([0.0, 1.0, 1.0, 1.0])  # the main scorer is out
    kwargs = {"team_ids": np.ones(4), "fixture_ids": np.ones(4)}

    g_plain, _, _, _ = model.reconcile_team_goals_and_assists(goals, assists, X, **kwargs)
    g_avail, a_avail, _, _ = model.reconcile_team_goals_and_assists(goals, assists, X, availability=avail, **kwargs)

    assert float(np.sum(g_avail * avail)) == pytest.approx(2.0)  # team total survives the availability step
    assert float(np.sum(g_plain * avail)) < 2.0  # old behaviour: the absent player's share is lost
    assert g_avail[1] > g_plain[1]
    assert float(np.sum(a_avail * avail)) == pytest.approx(2.0 * 0.85)


# --------------------------------------------------------------------------------------------------
# Official availability
# --------------------------------------------------------------------------------------------------


class _Elem:
    def __init__(self, status="a", next_round=None, this_round=None):
        self.status = status
        self.chance_of_playing_next_round = next_round
        self.chance_of_playing_this_round = this_round


@pytest.mark.parametrize("status", ["i", "s", "u", "n"])
def test_unavailable_statuses_are_zero_even_without_a_percentage(status):
    assert official_chance_of_playing(_Elem(status), target_gw=8, first_gw=8) == 0.0


def test_doubt_applies_to_next_round_then_decays_towards_fit():
    doubtful = _Elem("d", next_round=25)
    assert official_chance_of_playing(doubtful, target_gw=8, first_gw=8) == 25.0
    assert official_chance_of_playing(doubtful, target_gw=9, first_gw=8) == pytest.approx(62.5)
    assert official_chance_of_playing(doubtful, target_gw=12, first_gw=8) > 90.0
    assert official_chance_of_playing(_Elem("a"), target_gw=8, first_gw=8) == 100.0
    assert official_chance_of_playing(doubtful, target_gw=12) == 25.0  # unknown first GW: no ramp


# --------------------------------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------------------------------


def _hist_row(name, season, rnd, points=2, **extra):
    row = {
        "name": name,
        "season": season,
        "round": rnd,
        "position": "MID",
        "minutes": 90,
        "starts": 1,
        "total_points": points,
        "expected_goals": 0.2,
        "expected_assists": 0.1,
        "expected_goal_involvements": 0.3,
        "expected_goals_conceded": 1.0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "saves": 0,
        "defensive_contribution": 4.0,
        "ict_index": 5.0,
        "bps": 15,
        "bonus": 0,
        "was_home": True,
        "value": 70.0,
    }
    row.update(extra)
    return row


def test_defcon_label_is_unknown_for_seasons_that_never_recorded_it():
    rows = [_hist_row("A Player", "2024-25", r, defensive_contribution=0.0) for r in range(1, 6)]
    rows += [_hist_row("A Player", "2025-26", r, defensive_contribution=12.0) for r in range(1, 6)]
    X, Y, meta = FeatureEngineering().build_historical_features(pd.DataFrame(rows), return_meta=True)
    old = (meta["season"] == "2024-25").to_numpy()
    assert Y.loc[old, ["target_defcon", "target_defcon_actions"]].isna().all().all()
    assert Y.loc[~old, "target_defcon"].eq(1.0).all()
    assert np.isfinite(X["roll_defcon_5"]).all()


def test_card_label_excludes_own_goals_and_missed_penalties():
    row = _hist_row("A Player", "2025-26", 1, yellow_cards=1, red_cards=0, own_goals=1, penalties_missed=1)
    _, Y = FeatureEngineering().build_historical_features(pd.DataFrame([row]))
    assert float(Y["target_card_deduction"].iloc[0]) == 1.0


def test_double_gameweek_second_leg_does_not_see_first_leg():
    rows = [_hist_row("A Player", "2025-26", r, points=2) for r in (1, 2)]
    rows.append(_hist_row("A Player", "2025-26", 3, points=20))  # first leg of the double gameweek
    rows.append(_hist_row("A Player", "2025-26", 3, points=2))  # second leg
    X, _, meta = FeatureEngineering().build_historical_features(pd.DataFrame(rows), return_meta=True)
    leg1, leg2 = X[meta["round"] == 3].iloc[0], X[meta["round"] == 3].iloc[1]
    assert leg2["roll_points_3"] == leg1["roll_points_3"] == 2.0  # the 20-pointer is not pre-deadline info


def test_history_is_ordered_per_canonical_player_not_per_raw_spelling():
    """'Aarón' (2023-24) sorts after 'Aaron' (2024-25) by raw name; history must still be chronological."""
    rows = [_hist_row("Aarón Anselmino", "2023-24", r, points=9) for r in range(1, 4)]
    rows += [_hist_row("Aaron Anselmino", "2024-25", r, points=1) for r in range(1, 4)]
    X, _, meta = FeatureEngineering().build_historical_features(pd.DataFrame(rows), return_meta=True)
    first_2425 = X[(meta["season"] == "2024-25") & (meta["round"] == 1)].iloc[0]
    assert first_2425["roll_points_3"] == 9.0


def test_cold_start_rows_use_the_live_serving_priors():
    X, _ = FeatureEngineering().build_historical_features(pd.DataFrame([_hist_row("New Signing", "2025-26", 1)]))
    priors = compute_player_rolling_stats(None, pos_code="MID", val_m=7.0)
    assert {c: float(X.iloc[0][c]) for c in priors} == priors


def test_walk_forward_dixon_coles_never_sees_later_matches(monkeypatch):
    rows = []
    for rnd in range(1, 13):
        for k, (h, a) in enumerate([("A", "B"), ("C", "D")] if rnd % 2 else [("B", "C"), ("D", "A")]):
            late = rnd > 8
            rows.append(
                {
                    "season": "2025-26",
                    "round": rnd,
                    "home_team": h,
                    "away_team": a,
                    "home_goals": 7 if (late and h == "A") else 1,
                    "away_goals": 1,
                    "kickoff_dt": pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * rnd, hours=k),
                }
            )
    matches = pd.DataFrame(rows)
    monkeypatch.setattr(DixonColesModel, "_extract_matches_from_history", lambda self, df: matches)

    df = pd.DataFrame(  # only the columns the (monkeypatched) match extraction is gated on matter here
        {
            "season": "2025-26",
            "round": range(1, 13),
            "team": "A",
            "opponent_team": 1,
            "was_home": True,
            "kickoff_time": "",
            "goals_conceded": 0,
        }
    )
    models = FeatureEngineering()._walk_forward_dixon_coles(df)

    assert set(models) == {("2025-26", r) for r in range(1, 13)}
    early_only = DixonColesModel().fit(matches[matches["round"] <= 8])
    assert models[("2025-26", 12)].alphas["A"] == pytest.approx(early_only.alphas["A"])
    leaked = DixonColesModel().fit(matches)
    assert leaked.alphas["A"] > early_only.alphas["A"] + 0.2  # the 7-goal games would have shifted it
    assert models[("2025-26", 1)].alphas == {}  # nothing played before round 1: neutral priors, no global fit


# --------------------------------------------------------------------------------------------------
# Promotion: refit after the gate, metrics stay the held-out ones
# --------------------------------------------------------------------------------------------------


def _origins() -> list[dict]:
    return [
        {
            "season": f"Holdout {i}",
            "train_size": 1,
            "test_size": 1,
            "mae": 1.0,
            "rmse": 2.0,
            "spearman": 0.7,
            "best_baseline_mae": 1.1,
            "gain_vs_baseline": 0.1,
            "interval_80_coverage_pct": 80.0,
        }
        for i in range(3)
    ]


def test_gate_compares_rmse_when_reported_because_xp_is_a_mean():
    """An unbiased mean forecast can lose on MAE to a shrunk/median-like baseline yet win on RMSE and rank."""
    candidate = {
        "ml_mae": 1.023,
        "base_mae": 1.022,
        "ml_rmse": 1.965,
        "base_rmse": 2.131,
        "rolling_origins": _origins(),
    }
    active = {"ml_mae": 0.984, "ml_rmse": 1.993}
    assert check_promotion_gate(candidate, active)[0] is True
    assert check_promotion_gate({**candidate, "ml_rmse": 2.2}, active)[0] is False  # worse than baseline RMSE
    assert check_promotion_gate({**candidate, "ml_rmse": 2.0}, active)[0] is False  # worse than active RMSE
    mae_only = {k: v for k, v in candidate.items() if k not in ("ml_rmse", "base_rmse")}
    assert check_promotion_gate(mae_only, {"ml_mae": 0.984})[0] is False  # legacy MAE-only sets still gate on MAE


class _SavedModel:
    def __init__(self, tag):
        self.tag = tag

    def save(self, path):
        path.write_text(self.tag, encoding="utf-8")


def test_refit_runs_only_after_gate_passes_and_manifest_keeps_held_out_metrics(tmp_path):
    registry = ModelRegistry()
    registry.models_dir = tmp_path / "models"
    registry.versions_dir = registry.models_dir / "versions"
    registry.rejected_dir = registry.models_dir / "rejected"
    registry.manifest_path = registry.models_dir / "manifest.json"
    registry._ensure_dirs()
    registry._init_manifest()

    names = ["minutes_model", "attacking_model", "defending_model", "defcon_model", "bonus_model", "cards_saves_model"]
    candidates = {n: _SavedModel("candidate") for n in names}
    calls: list[str] = []

    def refit():
        calls.append("refit")
        return {n: _SavedModel("refit_on_all_rows") for n in names}

    bad = {"ml_mae": 2.5, "ml_spearman": 0.3, "base_mae": 1.74}
    res = registry.verify_and_promote(
        candidate_models=candidates,
        candidate_metrics=bad,
        active_metrics={"ml_mae": 1.48, "ml_spearman": 0.52, "base_mae": 1.74},
        tolerance=0.05,
        refit_final=refit,
    )
    assert res["promoted"] is False and calls == []

    origins = _origins()
    good = {"ml_mae": 1.35, "ml_spearman": 0.58, "base_mae": 1.74, "rolling_origins": origins}
    res = registry.verify_and_promote(
        candidate_models=candidates,
        candidate_metrics=good,
        active_metrics={"ml_mae": 1.48, "ml_spearman": 0.52, "base_mae": 1.74},
        new_version_tag="v-refit",
        calibration_data={"z10": -0.8, "z90": 1.0, "blend_weights": [0.7, 0.06, 0.24]},
        refit_final=refit,
    )
    assert res["promoted"] is True and calls == ["refit"]
    assert (registry.models_dir / "minutes_model.pkl").read_text(encoding="utf-8") == "refit_on_all_rows"
    entry = registry.get_active_version()
    assert entry["ml_mae"] == 1.35  # the held-out candidate metric, not a re-measured in-sample one
    assert entry["metric_applies_to"] == "held_out_candidate_before_final_refit"
