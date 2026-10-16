"""
Acceptance and property tests for WI-3.3:
Player share model and coherent team-goal reconciliation.

Criteria:
1. For every fixture the sum of player E[goals] is within 5% of team lambda (property test on live data).
2. For every fixture the sum of player E[assists] is within 5% of assist_rate * team lambda.
3. Player-level goal shares sum to 1.0 across the team.
4. Set-piece hierarchy boosts goal/assist shares for primary takers.
5. Single-player evaluation safely avoids allocating full team goals to a single player.
6. Player-level RMSE for MID/FWD not worse than before.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, Team
from fpl_oracle.data.features import FEATURE_COLUMNS, feature_engineering
from fpl_oracle.ml.components.attacking import AttackingModel


@pytest.fixture
def fitted_attacking_model():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({col: np.random.uniform(0.1, 1.0, size=n) for col in FEATURE_COLUMNS})
    X["pos_GKP"] = np.random.choice([0.0, 1.0], size=n, p=[0.85, 0.15])
    X["pos_DEF"] = np.random.choice([0.0, 1.0], size=n, p=[0.7, 0.3])
    X["pos_MID"] = np.random.choice([0.0, 1.0], size=n, p=[0.6, 0.4])
    X["pos_FWD"] = np.random.choice([0.0, 1.0], size=n, p=[0.8, 0.2])
    X["implied_team_xG"] = np.random.uniform(0.8, 2.5, size=n)
    X["chance_of_playing"] = np.full(n, 100.0)

    Y = pd.DataFrame(
        {
            "target_goals": np.random.poisson(0.25, size=n).astype(float),
            "target_assists": np.random.poisson(0.20, size=n).astype(float),
        }
    )

    model = AttackingModel()
    model.fit(X, Y)
    return model


def test_reconciliation_exact_sum_matches_team_lambda(fitted_attacking_model):
    """
    Property test: for each team in a fixture, sum of E[goals] equals implied_team_xG within 5%.
    """
    model = fitted_attacking_model
    n_players_per_team = 15
    teams = [1, 2, 3, 4]
    lambdas = [1.10, 2.35, 0.75, 3.10]

    rows = []
    team_ids = []
    fixture_ids = []

    for t, lam in zip(teams, lambdas, strict=True):
        for i in range(n_players_per_team):
            row = {col: 0.5 for col in FEATURE_COLUMNS}
            row["implied_team_xG"] = lam
            row["pos_GKP"] = 1.0 if i == 0 else 0.0
            row["pos_DEF"] = 1.0 if 1 <= i <= 5 else 0.0
            row["pos_MID"] = 1.0 if 6 <= i <= 10 else 0.0
            row["pos_FWD"] = 1.0 if 11 <= i <= 14 else 0.0
            row["roll_xG_5"] = 0.4 if i >= 11 else (0.2 if i >= 6 else 0.05)
            row["roll_xA_5"] = 0.3 if 6 <= i <= 10 else 0.05
            row["chance_of_playing"] = 100.0
            rows.append(row)
            team_ids.append(t)
            fixture_ids.append(100 + (t // 2))

    df_X = pd.DataFrame(rows)
    preds = model.predict(df_X, team_ids=team_ids, fixture_ids=fixture_ids, reconcile=True)

    exp_goals = preds["expected_goals"]
    exp_assists = preds["expected_assists"]
    goal_shares = preds["goal_shares"]

    # Verify per team
    for t, lam in zip(teams, lambdas, strict=True):
        idx = [i for i, tid in enumerate(team_ids) if tid == t]
        team_goal_sum = float(np.sum(exp_goals[idx]))
        team_assist_sum = float(np.sum(exp_assists[idx]))
        team_goal_share_sum = float(np.sum(goal_shares[idx]))

        # Main Acceptance Gate 1: sum within 5% of team lambda
        assert abs(team_goal_sum - lam) <= 0.05 * lam, (
            f"Team {t} goal sum {team_goal_sum:.4f} not within 5% of lambda {lam}"
        )
        # Sum of goal shares equals 1.0
        assert np.isclose(team_goal_share_sum, 1.0, atol=1e-4)

        # Main Acceptance Gate 2: assist sum within 5% of assist_rate * team lambda
        expected_assists = lam * 0.85
        assert abs(team_assist_sum - expected_assists) <= 0.05 * expected_assists, (
            f"Team {t} assist sum {team_assist_sum:.4f} not within 5% of {expected_assists}"
        )


def test_property_reconciliation_on_live_upcoming_pipeline(fitted_attacking_model):
    """
    Property test on live data fixture pipeline:
    Verifies that extract_live_features_for_upcoming generates coherent expectations.
    """
    teams = [
        Team(
            id=1,
            name="Arsenal",
            short_name="ARS",
            strength_attack_home=1200,
            strength_defence_home=1250,
            strength_attack_away=1180,
            strength_defence_away=1220,
        ),
        Team(
            id=2,
            name="Chelsea",
            short_name="CHE",
            strength_attack_home=1150,
            strength_defence_home=1100,
            strength_attack_away=1120,
            strength_defence_away=1080,
        ),
    ]
    elements = []
    # Build 15 players per team
    for tid in [1, 2]:
        for p in range(1, 16):
            elem_id = tid * 100 + p
            etype = 1 if p <= 2 else (2 if p <= 7 else (3 if p <= 12 else 4))
            pen_ord = 1 if p == 14 else None  # Player 14 is penalty taker
            elements.append(
                Element(
                    id=elem_id,
                    web_name=f"Player_{elem_id}",
                    team=tid,
                    element_type=etype,
                    now_cost=60,
                    status="a",
                    chance_of_playing_next_round=100,
                    penalties_order=pen_ord,
                )
            )

    bootstrap = BootstrapStatic(
        teams=teams,
        elements=elements,
        element_types=[
            ElementType(
                id=1,
                plural_name="Goalkeepers",
                plural_name_short="GKP",
                singular_name="Goalkeeper",
                singular_name_short="GKP",
            ),
            ElementType(
                id=2,
                plural_name="Defenders",
                plural_name_short="DEF",
                singular_name="Defender",
                singular_name_short="DEF",
            ),
            ElementType(
                id=3,
                plural_name="Midfielders",
                plural_name_short="MID",
                singular_name="Midfielder",
                singular_name_short="MID",
            ),
            ElementType(
                id=4,
                plural_name="Forwards",
                plural_name_short="FWD",
                singular_name="Forward",
                singular_name_short="FWD",
            ),
        ],
    )
    fixtures = [
        Fixture(id=101, code=101, event=7, team_h=1, team_a=2, kickoff_time="2026-10-18T15:00:00Z"),
    ]

    dummy_hist = pd.DataFrame(
        {
            "element": [99999],
            "name": ["Player 99999"],
            "round": [1],
            "season": ["2026-27"],
            "minutes": [90],
            "total_points": [2],
        }
    )
    features_df = feature_engineering.extract_live_features_for_upcoming(
        bootstrap=bootstrap,
        fixtures=fixtures,
        target_gw=7,
        history_df=dummy_hist,
    )

    assert not features_df.empty
    assert "penalties_order" in features_df.columns
    assert "implied_team_xG" in features_df.columns

    X = features_df[FEATURE_COLUMNS].copy()
    X_att = X.copy()
    for col in ("penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"):
        if col in features_df.columns:
            X_att[col] = features_df[col].values

    preds = fitted_attacking_model.predict(
        X_att,
        team_ids=features_df["team"].values,
        fixture_ids=features_df["fixture_id"].values,
        reconcile=True,
    )

    exp_goals = preds["expected_goals"]
    features_df["pred_goals"] = exp_goals

    # Check for each team in the fixture that sum of player E[goals] is within 5% of team lambda
    for tid, group in features_df.groupby("team"):
        lam = float(group["implied_team_xG"].iloc[0])
        total_p_goals = float(group["pred_goals"].sum())
        assert abs(total_p_goals - lam) <= 0.05 * lam, (
            f"Fixture team {tid} total predicted goals {total_p_goals:.4f} not within 5% of team lambda {lam:.4f}"
        )


def test_set_piece_hierarchy_boosts_shares(fitted_attacking_model):
    """
    Verifies that designated penalty and corner takers receive appropriately higher shares.
    """
    model = fitted_attacking_model
    n = 15
    rows = []
    for i in range(n):
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        row["implied_team_xG"] = 2.0
        row["pos_FWD"] = 1.0 if i in (11, 12) else 0.0
        row["pos_MID"] = 1.0 if 6 <= i <= 10 else 0.0
        row["roll_xG_5"] = 0.35 if i in (11, 12) else 0.1
        row["roll_xA_5"] = 0.30 if i in (7, 8) else 0.1
        row["chance_of_playing"] = 100.0
        rows.append(row)

    df_X = pd.DataFrame(rows)
    # Player 11 is primary penalty taker; Player 12 is identical forward without penalties
    df_X["penalties_order"] = None
    df_X.loc[11, "penalties_order"] = 1

    # Player 7 is corner taker; Player 8 is identical midfielder without corners
    df_X["corners_and_indirect_freekicks_order"] = None
    df_X.loc[7, "corners_and_indirect_freekicks_order"] = 1

    preds = model.predict(df_X, team_ids=[1] * n, fixture_ids=[101] * n, reconcile=True)

    # 1. Penalty taker receives strictly higher goal expectation and share
    assert preds["expected_goals"][11] > preds["expected_goals"][12], (
        f"Penalty taker {preds['expected_goals'][11]:.4f} should exceed non-taker {preds['expected_goals'][12]:.4f}"
    )
    assert preds["goal_shares"][11] > preds["goal_shares"][12]

    # 2. Corner taker receives strictly higher assist expectation and share
    assert preds["expected_assists"][7] > preds["expected_assists"][8], (
        f"Corner taker {preds['expected_assists'][7]:.4f} should exceed non-taker {preds['expected_assists'][8]:.4f}"
    )
    assert preds["assist_shares"][7] > preds["assist_shares"][8]

    # 3. Sum of team goals is still strictly equal to lambda
    assert np.isclose(np.sum(preds["expected_goals"]), 2.0, atol=1e-4)


def test_single_player_evaluation_safety(fitted_attacking_model):
    """
    Verifies that querying a single player does not assign 100% of team goals to that one player.
    """
    model = fitted_attacking_model
    single_row = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
    single_row["implied_team_xG"] = 2.5
    single_row["pos_FWD"] = 1.0

    preds = model.predict(single_row, reconcile=True)
    exp_g = float(preds["expected_goals"][0])
    # A single forward should have realistic individual goals (~0.3 to 0.7), never the whole 2.5 team goals!
    assert exp_g < 1.5, f"Single player query allocated excessive goals: {exp_g:.3f}"
    assert exp_g >= 0.0


def test_rmse_mid_fwd_not_worse_on_historical(fitted_attacking_model):
    """
    Verifies that player-level RMSE for MID and FWD does not regress with reconciliation.
    """
    np.random.seed(99)
    rows = []
    y_true_goals = []
    y_true_assists = []
    team_ids = []
    fixture_ids = []

    # 4 teams of 15 players
    for t in range(4):
        lam = 1.5 + 0.3 * t
        actual_team_goals = np.random.poisson(lam)
        for i in range(15):
            row = {col: 0.4 for col in FEATURE_COLUMNS}
            row["implied_team_xG"] = lam
            is_fwd = 1.0 if i >= 12 else 0.0
            is_mid = 1.0 if 7 <= i < 12 else 0.0
            row["pos_FWD"] = is_fwd
            row["pos_MID"] = is_mid
            row["pos_DEF"] = 1.0 if 2 <= i < 7 else 0.0
            row["pos_GKP"] = 1.0 if i < 2 else 0.0
            row["roll_xG_5"] = 0.4 if is_fwd else (0.2 if is_mid else 0.02)
            row["roll_xA_5"] = 0.3 if is_mid else (0.15 if is_fwd else 0.02)
            row["chance_of_playing"] = 100.0
            rows.append(row)
            team_ids.append(t)
            fixture_ids.append(50 + t)

            # True goals generated from multinomial conditioned on actual team goals
            p_goal = 0.25 if is_fwd else (0.12 if is_mid else 0.01)
            y_true_goals.append(float(np.random.binomial(actual_team_goals, min(0.4, p_goal))))
            y_true_assists.append(float(np.random.binomial(actual_team_goals, min(0.3, p_goal * 0.8))))

    df_val = pd.DataFrame(rows)
    y_true_g = np.array(y_true_goals)

    # Predictions without reconciliation
    preds_raw = fitted_attacking_model.predict(df_val, team_ids=team_ids, fixture_ids=fixture_ids, reconcile=False)
    # Predictions with reconciliation
    preds_rec = fitted_attacking_model.predict(df_val, team_ids=team_ids, fixture_ids=fixture_ids, reconcile=True)

    mid_fwd_mask = (df_val["pos_MID"] == 1.0) | (df_val["pos_FWD"] == 1.0)
    rmse_raw = float(np.sqrt(np.mean((preds_raw["expected_goals"][mid_fwd_mask] - y_true_g[mid_fwd_mask]) ** 2)))
    rmse_rec = float(np.sqrt(np.mean((preds_rec["expected_goals"][mid_fwd_mask] - y_true_g[mid_fwd_mask]) ** 2)))

    # Gate: RMSE not worse than before
    assert rmse_rec <= rmse_raw + 0.02, f"Reconciled RMSE {rmse_rec:.4f} regressed compared to raw RMSE {rmse_raw:.4f}"
