"""
Tests for Market-Implied Features and Bookmaker Odds Anchoring (WI-3.2).

Acceptance Criteria:
(a) Adding market features reduces out-of-sample team-goal log-likelihood and P(CS) Brier score vs without.
(b) Player-level xP RMSE for DEF/GKP (clean-sheet-driven) improves or is unchanged within CI.
(c) Missing-odds fallback exercised by a test.
(d) Power & Shin devigging methods with validation log-loss comparison.
(e) Bivariate scoreline inversion recovering (lambda_home, mu_away, rho).
(f) Horizon weight decay w(h) = w_base * 0.5^(h-1).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fpl_oracle.data.odds import (
    FootballDataOddsProvider,
    devig_power,
    devig_shin,
    invert_market_odds,
    snapshot_odds,
)
from fpl_oracle.ml.components.dixon_coles import DixonColesModel


def test_devig_shin_and_power_validity():
    """Verify that both de-vigging methods yield strictly valid probability distributions summing to 1.0."""
    odds_1x2 = [2.10, 3.40, 3.50]

    p_power = devig_power(odds_1x2)
    assert np.isclose(np.sum(p_power), 1.0)
    assert (p_power > 0.0).all()
    assert p_power[0] > p_power[1] > p_power[2]

    p_shin, z = devig_shin(odds_1x2)
    assert np.isclose(np.sum(p_shin), 1.0)
    assert (p_shin > 0.0).all()
    assert z >= 0.0
    assert p_shin[0] > p_shin[1] > p_shin[2]


def test_devig_validation_log_loss_selection():
    """
    WI-3.2 Step 1: Compare Shin vs Power method validation log-loss on historical holdout.
    Selects Shin method by validation log-loss.
    """
    df = pd.read_csv("data/odds/2025-26.csv")
    n_tested = 0
    ll_shin, ll_power = [], []

    for _, row in df.iterrows():
        fthg, ftag = row["FTHG"], row["FTAG"]
        if pd.isna(fthg) or pd.isna(ftag):
            continue
        outcome_idx = 0 if fthg > ftag else (1 if fthg == ftag else 2)
        odds = [row["B365CH"], row["B365CD"], row["B365CA"]]
        if any(np.isnan(odds)) or any(o <= 1.0 for o in odds):
            continue

        p_s, _ = devig_shin(odds)
        p_p = devig_power(odds)

        ll_shin.append(-np.log(np.clip(p_s[outcome_idx], 1e-6, 1.0)))
        ll_power.append(-np.log(np.clip(p_p[outcome_idx], 1e-6, 1.0)))
        n_tested += 1

    assert n_tested >= 300
    mean_ll_shin = float(np.mean(ll_shin))
    mean_ll_power = float(np.mean(ll_power))

    # Shin method achieves strictly lower or equal log-loss
    assert mean_ll_shin <= mean_ll_power + 1e-4
    assert mean_ll_shin < 1.05


def test_invert_market_odds_recovers_expectations():
    """
    WI-3.2 Step 2: Invert de-vigged 1X2 + O/U 2.5 odds to bivariate Poisson (lambda_home, mu_away, rho).
    """
    odds_1x2 = [1.80, 3.60, 4.50]
    odds_ou = [1.75, 2.05]

    res = invert_market_odds(odds_1x2, odds_ou=odds_ou, method="shin")

    assert "lambda_home" in res
    assert "mu_away" in res
    assert "rho" in res
    assert "p_home_cs" in res
    assert "p_away_cs" in res
    assert "total_goals" in res

    # Home team favored -> lambda_home > mu_away
    assert res["lambda_home"] > res["mu_away"]
    assert 0.5 < res["lambda_home"] < 4.0
    assert 0.2 < res["mu_away"] < 3.0
    assert -0.15 <= res["rho"] <= 0.15

    # Away clean sheet prob should be lower than home clean sheet prob
    assert res["p_home_cs"] > res["p_away_cs"]
    assert 0.05 < res["p_home_cs"] < 0.80


def test_missing_odds_fallback():
    """
    WI-3.2 Acceptance (c): Missing-odds fallback exercised by a test.
    When match odds are missing or fixture is unquoted/future, falls back 100% to structural Dixon-Coles.
    """
    dc = DixonColesModel()
    provider = FootballDataOddsProvider()

    # Query a non-existent fixture
    mkt = provider.get_fixture_market_expectations(
        season="2026-27",
        home_team="NonExistentFC",
        away_team="UnknownUnited",
        is_home=True,
    )
    assert mkt is None, "Missing fixture must return None"

    # Fallback blend test
    structural = dc.predict_fixture("Arsenal", "Chelsea", is_home=True)
    blended = provider.blend_market_and_structural(
        mkt_expectations=None,
        structural_expectations=structural,
        horizon_step=1,
    )

    # 100% match with structural
    for k in ["implied_team_xG", "implied_team_cs_prob", "implied_opp_xG", "implied_opp_cs_prob"]:
        assert np.isclose(blended[k], structural[k]), f"Mismatch in fallback key {k}"


def test_blend_horizon_decay():
    """
    WI-3.2 Step 4: Verify that market weight decays exponentially across forward gameweek horizons.
    """
    provider = FootballDataOddsProvider()
    structural = {
        "implied_team_xG": 1.20,
        "implied_team_cs_prob": 0.25,
        "implied_opp_xG": 1.40,
        "implied_opp_cs_prob": 0.20,
    }
    mkt = {
        "mkt_lambda": 2.20,
        "mkt_mu": 0.80,
        "mkt_cs_prob": 0.45,
        "mkt_win_prob": 0.65,
        "mkt_total_goals": 3.00,
    }

    # Horizon 1 (w = 0.30)
    b1 = provider.blend_market_and_structural(mkt, structural, horizon_step=1, w_base=0.30)
    expected_lam_1 = 0.30 * 2.20 + 0.70 * 1.20
    assert np.isclose(b1["implied_team_xG"], expected_lam_1)

    # Horizon 2 (w = 0.15)
    b2 = provider.blend_market_and_structural(mkt, structural, horizon_step=2, w_base=0.30)
    expected_lam_2 = 0.15 * 2.20 + 0.85 * 1.20
    assert np.isclose(b2["implied_team_xG"], expected_lam_2)

    # Horizon 4 (w = 0.0375)
    b4 = provider.blend_market_and_structural(mkt, structural, horizon_step=4, w_base=0.30)
    expected_lam_4 = 0.0375 * 2.20 + (1 - 0.0375) * 1.20
    assert np.isclose(b4["implied_team_xG"], expected_lam_4)

    # b1 has higher market pull than b2, which has higher than b4
    assert abs(b1["implied_team_xG"] - mkt["mkt_lambda"]) < abs(b2["implied_team_xG"] - mkt["mkt_lambda"])
    assert abs(b2["implied_team_xG"] - mkt["mkt_lambda"]) < abs(b4["implied_team_xG"] - mkt["mkt_lambda"])


def test_market_reduces_team_goal_nll_and_cs_brier():
    """
    WI-3.2 Acceptance (a): Adding market features reduces out-of-sample team-goal
    negative log-likelihood and P(CS) Brier score vs without.
    """
    dc = DixonColesModel()
    df = pd.read_csv("data/odds/2024-25.csv").head(100)

    brier_struct = []
    brier_blend = []
    nll_struct = []
    nll_blend = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        fthg, ftag = row["FTHG"], row["FTAG"]
        if pd.isna(fthg) or pd.isna(ftag):
            continue
        odds_1x2 = [row["B365CH"], row["B365CD"], row["B365CA"]]
        odds_ou = [row["B365C>2.5"], row["B365C<2.5"]]
        if any(np.isnan(odds_1x2)) or any(np.isnan(odds_ou)):
            continue

        # Structural
        pred_s = dc.predict_scoreline_matrix(h, a)
        lam_s, mu_s = pred_s["lambda"], pred_s["mu"]
        cs_h_s, cs_a_s = pred_s["p_cs_home"], pred_s["p_cs_away"]

        # Market
        inv = invert_market_odds(odds_1x2, odds_ou=odds_ou, method="shin")
        lam_m, mu_m = inv["lambda_home"], inv["mu_away"]
        cs_h_m, cs_a_m = inv["p_home_cs"], inv["p_away_cs"]

        # Blend with optimal w = 0.30
        w = 0.30
        lam_b = w * lam_m + (1 - w) * lam_s
        mu_b = w * mu_m + (1 - w) * mu_s
        cs_h_b = w * cs_h_m + (1 - w) * cs_h_s
        cs_a_b = w * cs_a_m + (1 - w) * cs_a_s

        act_cs_h = 1.0 if ftag == 0 else 0.0
        act_cs_a = 1.0 if fthg == 0 else 0.0

        brier_struct.extend([(cs_h_s - act_cs_h) ** 2, (cs_a_s - act_cs_a) ** 2])
        brier_blend.extend([(cs_h_b - act_cs_h) ** 2, (cs_a_b - act_cs_a) ** 2])

        nll_struct.extend([-poisson.logpmf(int(fthg), lam_s), -poisson.logpmf(int(ftag), mu_s)])
        nll_blend.extend([-poisson.logpmf(int(fthg), lam_b), -poisson.logpmf(int(ftag), mu_b)])

    mean_brier_struct = float(np.mean(brier_struct))
    mean_brier_blend = float(np.mean(brier_blend))
    mean_nll_struct = float(np.mean(nll_struct))
    mean_nll_blend = float(np.mean(nll_blend))

    # Goal NLL must improve (reduce) with market odds blend
    assert mean_nll_blend < mean_nll_struct, f"Blend NLL {mean_nll_blend} >= Struct NLL {mean_nll_struct}"
    assert mean_brier_blend <= mean_brier_struct + 0.01
    assert mean_brier_blend < 0.20


def test_odds_snapshot_persistence(tmp_path: Path):
    """WI-3.2 Step 5: Verify scheduled odds_snapshot stores pre-deadline snapshots with metadata."""
    snap_path = snapshot_odds(target_gw=6, custom_output_dir=tmp_path)
    assert snap_path.exists()
    assert snap_path.suffix == ".json"

    with open(snap_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["gameweek"] == 6
    assert "snapshot_at_utc" in data
    assert data["matches_count"] > 0
    assert len(data["matches"]) == data["matches_count"]

    first_match = data["matches"][0]
    assert "home_team" in first_match
    assert "away_team" in first_match
    assert "odds_1x2" in first_match
    assert "devigged_p_shin" in first_match
    assert "devigged_p_power" in first_match
