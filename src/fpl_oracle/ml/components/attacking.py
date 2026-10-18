"""
Attacking component model and player share reconciliation engine (WI-3.3).
Predicts expected goals (xG) and expected assists (xA), models player share
of team attacking output conditional on playing, incorporates set-piece hierarchy,
and strictly reconciles player expectations to match Dixon-Coles/market team lambda.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS
from fpl_oracle.ml.components.base import BaseComponent

logger = logging.getLogger("fpl_oracle.ml.components.attacking")


class AttackingModel(BaseComponent):
    """
    Player attacking expectation component with coherent team-goal reconciliation (WI-3.3).
    """

    def __init__(self):
        super().__init__("attacking_model")
        self.reg_goals = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.reg_assists = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        feat_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_train = X[feat_cols].copy()
        for c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)
        self.reg_goals.fit(X_train, Y["target_goals"].values)
        self.reg_assists.fit(X_train, Y["target_assists"].values)
        self.is_fitted = True

    def reconcile_team_goals_and_assists(
        self,
        exp_goals: np.ndarray,
        exp_assists: np.ndarray,
        X: pd.DataFrame,
        team_ids: pd.Series | np.ndarray | None = None,
        fixture_ids: pd.Series | np.ndarray | None = None,
        assist_rate: float = 0.85,
        availability: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        WI-3.3: Reconcile player-level expectations with team-level expectations.

        Rescales player goals and assists within each team-fixture so that:
            sum_{p in team} availability_p * E[goals_p] == implied_team_xG (within 5% tolerance)
            sum_{p in team} availability_p * E[assists_p] == assist_rate * implied_team_xG
        `availability` (chance of playing in [0, 1]) is applied later by the ensemble, so the team total only
        holds after that step; weighting by it here hands the goals of an absent player to his teammates
        instead of losing them. Returned expectations stay unweighted. Logs scaling factors for auditability.
        """
        n = len(exp_goals)
        if n == 0:
            empty = np.array([])
            return empty, empty, empty, empty

        reconciled_goals = exp_goals.copy()
        reconciled_assists = exp_assists.copy()
        avail = np.ones(n, dtype=float) if availability is None else np.asarray(availability, dtype=float)
        goal_shares = np.zeros(n, dtype=float)
        assist_shares = np.zeros(n, dtype=float)

        meta_df = X.attrs.get("meta") if hasattr(X, "attrs") else None
        if isinstance(meta_df, pd.DataFrame):
            if (
                len(meta_df) != n
                and X.index.is_unique
                and meta_df.index.is_unique
                and set(X.index).issubset(meta_df.index)
            ):
                meta_df = meta_df.loc[X.index]
            elif len(meta_df) != n:
                meta_df = None

        # 1. Resolve team grouping
        if team_ids is not None:
            teams = np.asarray(team_ids)
        elif "team" in X.columns:
            teams = X["team"].values
        elif meta_df is not None and "team" in meta_df.columns:
            teams = meta_df["team"].values
        else:
            surrogate_cols = [
                c for c in ["implied_team_xG", "was_home", "days_rest", "team_strength_attack"] if c in X.columns
            ]
            if surrogate_cols:
                teams = pd.factorize(X[surrogate_cols].apply(tuple, axis=1))[0]
            else:
                teams = np.zeros(n, dtype=int)

        # 2. Resolve fixture grouping
        if fixture_ids is not None:
            fixtures = np.asarray(fixture_ids)
        elif "fixture_id" in X.columns:
            fixtures = X["fixture_id"].values
        elif meta_df is not None and "fixture_id" in meta_df.columns:
            fixtures = meta_df["fixture_id"].values
        elif meta_df is not None and "round" in meta_df.columns:
            fixtures = meta_df["round"].values
        else:
            fixtures = np.zeros(n, dtype=int)

        # Group players by team-fixture
        group_keys = [f"{t}_{f}" for t, f in zip(teams, fixtures, strict=False)]
        s_keys = pd.Series(group_keys, index=np.arange(n))

        for key, idx_list in s_keys.groupby(s_keys).groups.items():
            idx = idx_list.values if hasattr(idx_list, "values") else np.array(idx_list)
            # Guard against single-player or tiny evaluation queries (<3 players)
            if len(idx) < 3:
                lam_fallback = (
                    float(np.mean(X["implied_team_xG"].iloc[idx])) if "implied_team_xG" in X.columns else 1.35
                )
                if lam_fallback > 0:
                    goal_shares[idx] = reconciled_goals[idx] / lam_fallback
                    assist_shares[idx] = reconciled_assists[idx] / (lam_fallback * assist_rate)
                continue

            # Target team lambda from Dixon-Coles / market blend
            if "implied_team_xG" in X.columns:
                lambda_team = float(np.mean(X["implied_team_xG"].iloc[idx]))
            else:
                lambda_team = 1.35

            if lambda_team <= 0.01:
                continue

            # Goals reconciliation
            s_goals = float(np.sum(reconciled_goals[idx] * avail[idx]))
            if s_goals > 1e-6:
                kappa_g = float(lambda_team / s_goals)
                reconciled_goals[idx] = reconciled_goals[idx] * kappa_g
                new_s_goals = float(np.sum(reconciled_goals[idx] * avail[idx]))
                goal_shares[idx] = reconciled_goals[idx] * avail[idx] / lambda_team
                logger.debug(
                    "WI-3.3: Reconciled goals for group %s: kappa=%.4f (lambda=%.3f, raw_sum=%.3f, reconciled_sum=%.3f)",
                    key,
                    kappa_g,
                    lambda_team,
                    s_goals,
                    new_s_goals,
                )
            else:
                goal_shares[idx] = 0.0

            # Assists reconciliation
            lambda_assists = lambda_team * assist_rate
            s_assists = float(np.sum(reconciled_assists[idx] * avail[idx]))
            if s_assists > 1e-6:
                kappa_a = float(lambda_assists / s_assists)
                reconciled_assists[idx] = reconciled_assists[idx] * kappa_a
                new_s_assists = float(np.sum(reconciled_assists[idx] * avail[idx]))
                assist_shares[idx] = reconciled_assists[idx] * avail[idx] / lambda_assists
                logger.debug(
                    "WI-3.3: Reconciled assists for group %s: kappa=%.4f (lambda=%.3f, raw_sum=%.3f, reconciled_sum=%.3f)",
                    key,
                    kappa_a,
                    lambda_assists,
                    s_assists,
                    new_s_assists,
                )
            else:
                assist_shares[idx] = 0.0

        return reconciled_goals, reconciled_assists, goal_shares, assist_shares

    def predict(
        self,
        X: pd.DataFrame,
        team_ids: pd.Series | np.ndarray | None = None,
        fixture_ids: pd.Series | np.ndarray | None = None,
        reconcile: bool = True,
        assist_rate: float = 0.85,
    ) -> dict[str, np.ndarray]:
        """
        Produce expected goals and assists, incorporating set-piece hierarchy and
        reconciling with team fixture expected goals (WI-3.3).
        """
        if not self.is_fitted:
            raise RuntimeError("AttackingModel is not fitted.")

        if X.empty:
            empty = np.array([])
            return {
                "expected_goals": empty,
                "expected_assists": empty,
                "goal_shares": empty,
                "assist_shares": empty,
            }

        # 1. Base individual predictions from Tweedie regressors
        # Strictly extract only numeric FEATURE_COLUMNS for booster evaluation
        feat_cols = getattr(self.reg_goals, "feature_name_", None)
        if feat_cols is not None:
            use_cols = [c for c in feat_cols if c in X.columns]
        else:
            use_cols = [c for c in FEATURE_COLUMNS if c in X.columns]

        X_feat = X[use_cols].copy()
        for c in X_feat.columns:
            X_feat[c] = pd.to_numeric(X_feat[c], errors="coerce").fillna(0.0)

        exp_goals = np.clip(self.reg_goals.predict(X_feat), 0.0, 3.5)
        exp_assists = np.clip(self.reg_assists.predict(X_feat), 0.0, 3.0)

        # 2. Suppress attacking expectations for GKP
        if "pos_GKP" in X.columns:
            is_gkp = pd.to_numeric(X["pos_GKP"], errors="coerce").fillna(0.0).values
            exp_goals = exp_goals * (1.0 - is_gkp * 0.98)
            exp_assists = exp_assists * (1.0 - is_gkp * 0.95)

        # 3. Set-piece hierarchy adjustments (WI-3.3). Availability is applied once, by the ensemble.
        # Penalty taker boost: ~0.12 goals/match for primary, ~0.03 for secondary
        if "penalties_order" in X.columns:
            p_ord = pd.to_numeric(X["penalties_order"], errors="coerce").fillna(99).values
            exp_goals = exp_goals + np.where(p_ord == 1, 0.12, np.where(p_ord == 2, 0.03, 0.0))

        # Direct free-kick boost: ~0.03 goals/match
        if "direct_freekicks_order" in X.columns:
            fk_ord = pd.to_numeric(X["direct_freekicks_order"], errors="coerce").fillna(99).values
            exp_goals = exp_goals + np.where(fk_ord == 1, 0.03, np.where(fk_ord == 2, 0.01, 0.0))

        # Corner / indirect free-kick boost: ~0.08 assists/match for primary
        if "corners_and_indirect_freekicks_order" in X.columns:
            cn_ord = pd.to_numeric(X["corners_and_indirect_freekicks_order"], errors="coerce").fillna(99).values
            exp_assists = exp_assists + np.where(cn_ord == 1, 0.08, np.where(cn_ord == 2, 0.03, 0.0))

        # 4. Team-level reconciliation (WI-3.3)
        if reconcile:
            availability = (
                np.clip(pd.to_numeric(X["chance_of_playing"], errors="coerce").fillna(100.0).values / 100.0, 0.0, 1.0)
                if "chance_of_playing" in X.columns
                else None
            )
            rec_g, rec_a, g_shares, a_shares = self.reconcile_team_goals_and_assists(
                exp_goals=exp_goals,
                exp_assists=exp_assists,
                X=X,
                team_ids=team_ids,
                fixture_ids=fixture_ids,
                assist_rate=assist_rate,
                availability=availability,
            )
            return {
                "expected_goals": rec_g,
                "expected_assists": rec_a,
                "goal_shares": g_shares,
                "assist_shares": a_shares,
            }

        return {
            "expected_goals": exp_goals,
            "expected_assists": exp_assists,
            "goal_shares": np.zeros(len(X)),
            "assist_shares": np.zeros(len(X)),
        }
