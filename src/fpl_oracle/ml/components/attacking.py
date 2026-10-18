"""
Attacking component model and player share reconciliation engine (WI-3.3).
Predicts expected goals and assists from each player's share of his team's chances given he plays, and
reconciles them so every team's players add up to the market (or Dixon-Coles) team goals.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS, LEAGUE_ASSISTS_PER_XA, SHARE_PRIOR
from fpl_oracle.ml.components.base import BaseComponent

logger = logging.getLogger("fpl_oracle.ml.components.attacking")

# Of every team goal, the share credited to one of its players and the assists recorded with it
# (2023-24 to 2026-27: own goals are 3.7% of goals and score for nobody; 0.88 assists per team goal).
PLAYER_GOALS_PER_TEAM_GOAL = 0.963
ASSISTS_PER_TEAM_GOAL = 0.879


# Goals per xG by position (2023-24 to 2026-27): xG over-rates defenders' set-piece headers
GOALS_PER_XG = {"GKP": 0.01, "DEF": 0.849, "MID": 1.005, "FWD": 0.978}
FINISHING_WEIGHT = 0.5  # how much of a player's own shrunk conversion record carries forward (held-out fit)


def _position(X: pd.DataFrame) -> np.ndarray:
    cols = [f"pos_{p}" for p in ("GKP", "DEF", "MID", "FWD")]
    if not set(cols) <= set(X.columns):
        return np.full(len(X), "MID")
    return np.array(["GKP", "DEF", "MID", "FWD"])[X[cols].to_numpy(dtype=float).argmax(axis=1)]


def share_offset(X: pd.DataFrame, minutes: np.ndarray, target: str) -> np.ndarray:
    """log E[goals (or assists) | plays] before learned corrections:
    team expected goals x the player's shrunk share of his team's xG (xA) on the pitch x minutes / 90,
    times the position's goals-per-xG and part of his own conversion record."""
    lam = np.clip(pd.to_numeric(X["implied_team_xG"], errors="coerce").fillna(1.35).to_numpy(float), 0.05, None)
    share_col, conv_col, norm = (
        ("xg_share_30", "finishing_30", 1.0)
        if target == "goals"
        else ("xa_share_30", "assist_conversion_30", LEAGUE_ASSISTS_PER_XA)
    )
    position = _position(X)
    prior = np.array([SHARE_PRIOR[p][0 if target == "goals" else 1] for p in position])
    share = pd.to_numeric(X[share_col], errors="coerce").to_numpy(float) if share_col in X else prior
    share = np.clip(np.where(np.isfinite(share), share, prior), 1e-4, None)
    conv = pd.to_numeric(X[conv_col], errors="coerce").to_numpy(float) if conv_col in X else np.full(len(X), norm)
    conv = np.clip(np.where(np.isfinite(conv), conv, norm) / norm, 0.2, 5.0)
    pos_rate = np.array([GOALS_PER_XG[p] for p in position]) if target == "goals" else 1.0
    return (
        np.log(lam * share * pos_rate)
        + FINISHING_WEIGHT * np.log(conv)
        + np.log(np.clip(np.asarray(minutes, dtype=float), 1.0, 90.0) / 90.0)
    )


class AttackingModel(BaseComponent):
    """
    Player attacking expectation component with coherent team-goal reconciliation (WI-3.3).
    E[goals | plays] is an explicit share model (share_offset) with LightGBM learning a multiplicative
    correction on top, so a correction shrinks towards "no change" rather than towards the average player.
    """

    def __init__(self):
        super().__init__("attacking_model")
        params = dict(
            objective="poisson",
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_estimators=4000,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.reg_goals = lgb.LGBMRegressor(**params)
        self.reg_assists = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # Fitted on matches the player played, with the minutes he played as exposure; the minutes model owns
        # whether and how long he plays. Trees stop when the latest 15% of matches stop improving.
        played = Y["target_minutes"].to_numpy() > 0
        feat_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_p = X.loc[played, feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Y_p = Y.loc[played]
        mins = Y_p["target_minutes"].to_numpy(float)
        cut = int(len(X_p) * 0.85)
        for reg, target, label in (
            (self.reg_goals, "goals", "target_goals"),
            (self.reg_assists, "assists", "target_assists"),
        ):
            offset = share_offset(X_p, mins, target)
            reg.fit(
                X_p.iloc[:cut],
                Y_p[label].iloc[:cut],
                init_score=offset[:cut],
                eval_X=(X_p.iloc[cut:],),
                eval_y=(Y_p[label].iloc[cut:],),
                eval_init_score=[offset[cut:]],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
        self.is_fitted = True

    @staticmethod
    def _with_market_scorers(
        goals: np.ndarray,
        p_play: np.ndarray,
        chance: np.ndarray,
        X: pd.DataFrame,
        team_ids: pd.Series | np.ndarray,
        fixture_ids: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Players with a bookmaker goalscorer price take the market's goals-if-he-plays. The prices carry a
        margin, most of it on long shots, so it is removed per team-match by the power method: chances p^k with
        k chosen so the priced players' goals, -ln(1 - p^k), add up to the team's market goals (own goals
        excluded) less what the model gives the unpriced players. The market sets the split, the match odds the
        total."""
        out = np.asarray(goals, dtype=float).copy()
        chance = np.asarray(chance, dtype=float)
        priced = np.isfinite(chance) & (chance > 0) & (chance < 1)
        if not priced.any():
            return out
        lam = pd.to_numeric(X["implied_team_xG"], errors="coerce").fillna(1.35).to_numpy(float)
        avail = (
            np.clip(pd.to_numeric(X["chance_of_playing"], errors="coerce").fillna(100.0).to_numpy(float) / 100.0, 0, 1)
            if "chance_of_playing" in X
            else np.ones(len(out))
        )
        keys = pd.Series([f"{t}_{f}" for t, f in zip(team_ids, fixture_ids, strict=True)])
        for idx in keys.groupby(keys).groups.values():
            idx = np.asarray(idx)
            listed = idx[priced[idx]]
            if len(listed) == 0:
                continue
            team_goals = PLAYER_GOALS_PER_TEAM_GOAL * float(np.mean(lam[idx]))
            unpriced = float(np.sum(avail[idx[~priced[idx]]] * out[idx[~priced[idx]]]))
            target = max(team_goals - unpriced, 0.25 * team_goals)
            w = avail[listed] * p_play[listed]

            def total(k: float, listed: np.ndarray = listed, w: np.ndarray = w) -> float:
                return float(np.sum(w * -np.log1p(-(chance[listed] ** k))))

            lo, hi = 1.0, 8.0
            if total(lo) <= target:  # no margin left to remove: keep the market's split, scale to the total
                rate = -np.log1p(-chance[listed])
                out[listed] = p_play[listed] * rate * target / max(total(lo), 1e-9)
                continue
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if total(mid) > target else (lo, mid)
            out[listed] = p_play[listed] * -np.log1p(-(chance[listed] ** hi))
        return out

    def reconcile_team_goals_and_assists(
        self,
        exp_goals: np.ndarray,
        exp_assists: np.ndarray,
        X: pd.DataFrame,
        team_ids: pd.Series | np.ndarray | None = None,
        fixture_ids: pd.Series | np.ndarray | None = None,
        assist_rate: float = ASSISTS_PER_TEAM_GOAL,
        availability: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        WI-3.3: Reconcile player-level expectations with team-level expectations.

        Rescales player goals and assists within each team-fixture so that:
            sum_{p in team} availability_p * E[goals_p] == PLAYER_GOALS_PER_TEAM_GOAL * implied_team_xG
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
            lambda_goals = lambda_team * PLAYER_GOALS_PER_TEAM_GOAL

            # Goals reconciliation
            s_goals = float(np.sum(reconciled_goals[idx] * avail[idx]))
            if s_goals > 1e-6:
                kappa_g = float(lambda_goals / s_goals)
                reconciled_goals[idx] = reconciled_goals[idx] * kappa_g
                new_s_goals = float(np.sum(reconciled_goals[idx] * avail[idx]))
                goal_shares[idx] = reconciled_goals[idx] * avail[idx] / lambda_goals
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

    def predict(  # type: ignore[override]  # also needs the minutes model's chance of playing
        self,
        X: pd.DataFrame,
        minutes: dict[str, np.ndarray],
        team_ids: pd.Series | np.ndarray | None = None,
        fixture_ids: pd.Series | np.ndarray | None = None,
        reconcile: bool = True,
        assist_rate: float = ASSISTS_PER_TEAM_GOAL,
        market_scoring_chance: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce expected goals and assists, reconciling with team fixture expected goals (WI-3.3).
        `minutes` is the minutes model's output: E[goals] = P(plays) * E[goals | plays], with E[goals | plays]
        from the share model at the minutes he is expected to play when he plays.
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

        p_play = np.clip(np.asarray(minutes["p_play"], dtype=float), 0.0, 1.0)
        mins_if_plays = np.asarray(minutes["expected_minutes"], dtype=float) / np.maximum(p_play, 1e-3)
        raw = {}
        for reg, target in ((self.reg_goals, "goals"), (self.reg_assists, "assists")):
            score = reg.predict(X_feat, raw_score=True, num_iteration=reg.best_iteration_)
            raw[target] = np.exp(score + share_offset(X, mins_if_plays, target)) * p_play
        if market_scoring_chance is not None and team_ids is not None and fixture_ids is not None:
            raw["goals"] = self._with_market_scorers(
                raw["goals"], p_play, market_scoring_chance, X, team_ids, fixture_ids
            )
        exp_goals = np.clip(raw["goals"], 0.0, 3.5)
        exp_assists = np.clip(raw["assists"], 0.0, 3.0)

        # 2. Suppress attacking expectations for GKP
        if "pos_GKP" in X.columns:
            is_gkp = pd.to_numeric(X["pos_GKP"], errors="coerce").fillna(0.0).values
            exp_goals = exp_goals * (1.0 - is_gkp * 0.98)
            exp_assists = exp_assists * (1.0 - is_gkp * 0.95)

        # Set pieces need no extra term: FPL xG / xA already count penalties and set-piece chances, so a
        # taker's rolling rates carry them (a flat boost on top double-counted them at serving time only).
        # 3. Team-level reconciliation (WI-3.3)
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
