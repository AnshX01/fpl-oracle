"""
Bivariate Dixon-Coles (1997) match scoreline and team strength model.
Fits attack (alpha), defence (delta), home advantage (gamma), and low-score dependency (rho).
Supports time-decay weighting (xi), ridge regularization, and promoted team priors.
Provides full bivariate scoreline probability grids (0..10 x 0..10), clean sheet probabilities,
expected goals conceded, and 1X2 match outcome distributions.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fpl_oracle.config import HISTORICAL_DIR

logger = logging.getLogger(__name__)

HISTORICAL_CSV_PATH = HISTORICAL_DIR / "master_history.csv"


class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model with low-score correlation adjustment tau_rho.
    """

    def __init__(
        self,
        xi: float = 0.001,
        ridge_lambda: float = 0.05,
        default_gamma: float = 0.20,
        default_rho: float = -0.10,
    ):
        self.xi = xi
        self.ridge_lambda = ridge_lambda
        self.default_gamma = default_gamma
        self.default_rho = default_rho

        self.alphas: dict[str, float] = {}
        self.deltas: dict[str, float] = {}
        self.gamma: float = default_gamma
        self.rho: float = default_rho

        # Pooled promoted team prior:
        # Sign convention: log lambda = alpha_h - delta_a + gamma, log mu = alpha_a - delta_h.
        # Higher delta decreases opponent scoring -> +delta represents stronger defence, -delta represents weaker defence.
        # Promoted clubs concede significantly more goals -> delta_promoted must be negative.
        # Fitted from historical 2023-26 promoted clubs (Luton, Sheffield Utd, Burnley, Ipswich, Leicester, Southampton):
        # mean alpha = -0.290, mean delta = -0.568.
        self.promoted_alpha: float = -0.29
        self.promoted_delta: float = -0.57
        self.cs_calibrator: Any = None

        self.teams: list[str] = []
        self._cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.is_fitted: bool = False

    def _extract_matches_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract pairwise match scorelines with kickoff dates from player-level historical rows.
        """
        from fpl_oracle.data.features import feature_engineering

        opp_map, _ = feature_engineering.build_team_id_maps(df)
        df_copy = df.copy()

        def opponent_name(season: str, opponent: Any) -> str:
            try:
                return opp_map.get((season, int(opponent)), str(opponent))
            except (ValueError, TypeError):  # already a team name
                return str(opponent)

        df_copy["opp_name"] = [
            opponent_name(s, o) for s, o in zip(df_copy["season"], df_copy["opponent_team"], strict=False)
        ]

        home_records = (
            df_copy[df_copy["was_home"] == True]  # noqa: E712
            .groupby(["season", "round", "team", "opp_name", "kickoff_time"])["goals_conceded"]
            .max()
            .reset_index()
        )
        home_records.rename(
            columns={"team": "home_team", "opp_name": "away_team", "goals_conceded": "away_goals"},
            inplace=True,
        )

        away_records = (
            df_copy[df_copy["was_home"] == False]  # noqa: E712
            .groupby(["season", "round", "team", "opp_name", "kickoff_time"])["goals_conceded"]
            .max()
            .reset_index()
        )
        away_records.rename(
            columns={"team": "away_team", "opp_name": "home_team", "goals_conceded": "home_goals"},
            inplace=True,
        )

        matches = home_records.merge(away_records, on=["season", "round", "home_team", "away_team", "kickoff_time"])
        matches["kickoff_dt"] = pd.to_datetime(matches["kickoff_time"])
        matches = matches.sort_values(by="kickoff_dt").reset_index(drop=True)
        return matches

    def fit(self, matches_df: pd.DataFrame | None = None, xi: float | None = None) -> "DixonColesModel":
        """
        Fit attack, defence, gamma, and rho via maximum weighted likelihood.
        """
        if xi is not None:
            self.xi = xi

        if matches_df is None:
            if HISTORICAL_CSV_PATH.exists():
                raw_df = pd.read_csv(HISTORICAL_CSV_PATH)
                matches_df = self._extract_matches_from_history(raw_df)
            else:
                logger.warning("Historical data file not found; initializing with default prior parameters.")
                self.is_fitted = True
                return self

        if len(matches_df) < 10:
            logger.warning("Fewer than 10 matches provided; retaining baseline parameters.")
            self.is_fitted = True
            return self

        matches = matches_df.copy()
        if "kickoff_dt" not in matches.columns:
            matches["kickoff_dt"] = pd.to_datetime(matches["kickoff_time"])
        matches = matches.sort_values(by="kickoff_dt").reset_index(drop=True)

        self.teams = sorted(list(set(matches["home_team"]).union(set(matches["away_team"]))))
        n_teams = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = np.array([team_idx[t] for t in matches["home_team"]], dtype=int)
        a_idx = np.array([team_idx[t] for t in matches["away_team"]], dtype=int)
        h_goals = matches["home_goals"].values.astype(int)
        a_goals = matches["away_goals"].values.astype(int)

        t_ref = matches["kickoff_dt"].max()
        days = (t_ref - matches["kickoff_dt"]).dt.total_seconds().values / 86400.0
        weights = np.exp(-self.xi * days)
        weights = weights / np.mean(weights)

        # Pre-compute match scoreline indicators for low scores
        m00 = (h_goals == 0) & (a_goals == 0)
        m01 = (h_goals == 0) & (a_goals == 1)
        m10 = (h_goals == 1) & (a_goals == 0)
        m11 = (h_goals == 1) & (a_goals == 1)
        n_matches = len(matches)

        def nll(params: np.ndarray) -> float:
            raw_alphas = params[:n_teams]
            deltas = params[n_teams : 2 * n_teams]
            gamma = params[2 * n_teams]
            rho = params[2 * n_teams + 1]

            # Zero-mean identifiability constraint on attack parameters
            alphas = raw_alphas - np.mean(raw_alphas)

            log_lam = alphas[h_idx] - deltas[a_idx] + gamma
            log_mu = alphas[a_idx] - deltas[h_idx]
            lam = np.exp(np.clip(log_lam, -3.0, 3.0))
            mu = np.exp(np.clip(log_mu, -3.0, 3.0))

            log_p = h_goals * np.log(lam) - lam + a_goals * np.log(mu) - mu

            tau_vals = np.ones(n_matches)
            tau_vals[m00] = np.maximum(1e-6, 1.0 - lam[m00] * mu[m00] * rho)
            tau_vals[m01] = np.maximum(1e-6, 1.0 + lam[m01] * rho)
            tau_vals[m10] = np.maximum(1e-6, 1.0 + mu[m10] * rho)
            tau_vals[m11] = np.maximum(1e-6, 1.0 - rho)

            log_lik = log_p + np.log(tau_vals)
            ridge = self.ridge_lambda * (np.sum(alphas**2) + np.sum(deltas**2))
            return float(-np.sum(weights * log_lik) + ridge)

        init_params = np.zeros(2 * n_teams + 2)
        init_params[2 * n_teams] = self.default_gamma
        init_params[2 * n_teams + 1] = self.default_rho

        bounds = [(-2.0, 2.0)] * (2 * n_teams) + [(0.0, 0.6), (-0.5, 0.4)]
        res = minimize(nll, init_params, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})

        fitted_alphas = res.x[:n_teams] - np.mean(res.x[:n_teams])
        fitted_deltas = res.x[n_teams : 2 * n_teams]
        self.gamma = float(res.x[2 * n_teams])
        self.rho = float(res.x[2 * n_teams + 1])

        self.alphas = {t: float(fitted_alphas[i]) for i, t in enumerate(self.teams)}
        self.deltas = {t: float(fitted_deltas[i]) for i, t in enumerate(self.teams)}
        self._cache.clear()

        # Update pooled promoted team prior from bottom tercile of established teams
        if self.alphas:
            alpha_vals = sorted(self.alphas.values())
            delta_vals = sorted(self.deltas.values())  # Ascending: lowest delta = weakest defence
            k = max(1, len(alpha_vals) // 5)
            self.promoted_alpha = float(np.mean(alpha_vals[:k]))
            self.promoted_delta = float(np.mean(delta_vals[:k]))

        # Fit isotonic clean sheet recalibration (WI-3.1)
        # Eliminates upper-bin overconfidence in clean sheet probabilities (>0.35 and >0.45)
        try:
            from fpl_oracle.ml.calibration import Calibrator

            cal_preds = []
            cal_acts = []
            for _, r in matches.iterrows():
                h_team = str(r["home_team"])
                a_team = str(r["away_team"])
                a_h = self.alphas.get(h_team, self.promoted_alpha)
                d_h = self.deltas.get(h_team, self.promoted_delta)
                a_a = self.alphas.get(a_team, self.promoted_alpha)
                d_a = self.deltas.get(a_team, self.promoted_delta)
                l_lam = float(np.exp(np.clip(a_h - d_a + self.gamma, -3.0, 2.0)))
                l_mu = float(np.exp(np.clip(a_a - d_h, -3.0, 2.0)))
                raw_h = float(np.clip(math.exp(-l_mu), 0.01, 0.95))
                raw_a = float(np.clip(math.exp(-l_lam), 0.01, 0.95))
                cal_preds.extend([raw_h, raw_a])
                cal_acts.extend([int(r["away_goals"] == 0), int(r["home_goals"] == 0)])

            cal = Calibrator("isotonic")
            cal.fit(np.array(cal_preds), np.array(cal_acts))
            self.cs_calibrator = cal
        except Exception as e:
            logger.warning("Could not fit CS calibrator: %s", e)

        self.is_fitted = True
        return self

    def get_team_parameters(self, team: str) -> tuple[float, float]:
        """
        Return (alpha, delta) with fallback shrinkage to pooled promoted prior.
        """
        if not self.is_fitted:
            self.fit()
        alpha = self.alphas.get(team, self.promoted_alpha)
        delta = self.deltas.get(team, self.promoted_delta)
        return alpha, delta

    def predict_scoreline_matrix(self, home_team: str, away_team: str, max_goals: int = 10) -> dict[str, Any]:
        """
        Compute bivariate scoreline joint probability matrix and derived match statistics.
        """
        if not self.is_fitted:
            self.fit()

        cache_key = (home_team, away_team, max_goals)
        if cache_key in self._cache:
            return self._cache[cache_key]

        alpha_h, delta_h = self.get_team_parameters(home_team)
        alpha_a, delta_a = self.get_team_parameters(away_team)

        log_lam = alpha_h - delta_a + self.gamma
        log_mu = alpha_a - delta_h
        lam = float(np.exp(np.clip(log_lam, -3.0, 2.0)))
        mu = float(np.exp(np.clip(log_mu, -3.0, 2.0)))

        # Build raw bivariate grid
        grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for x in range(max_goals + 1):
            px = math.exp(-lam) * (lam**x) / math.factorial(x)
            for y in range(max_goals + 1):
                py = math.exp(-mu) * (mu**y) / math.factorial(y)
                tau = 1.0
                if x == 0 and y == 0:
                    tau = max(1e-6, 1.0 - lam * mu * self.rho)
                elif x == 0 and y == 1:
                    tau = max(1e-6, 1.0 + lam * self.rho)
                elif x == 1 and y == 0:
                    tau = max(1e-6, 1.0 + mu * self.rho)
                elif x == 1 and y == 1:
                    tau = max(1e-6, 1.0 - self.rho)
                grid[x, y] = tau * px * py

        total_prob = np.sum(grid)
        if total_prob > 0:
            grid /= total_prob

        # Clean sheet probabilities (opponent scores 0)
        p_cs_home_raw = float(np.sum(grid[:, 0]))  # Away goals = 0
        p_cs_away_raw = float(np.sum(grid[0, :]))  # Home goals = 0

        # Calibrate clean sheet probabilities with isotonic recalibration & empirical shrinkage
        if self.cs_calibrator is not None:
            p_cs_h_iso = float(self.cs_calibrator.calibrate(np.array([p_cs_home_raw]))[0])
            p_cs_a_iso = float(self.cs_calibrator.calibrate(np.array([p_cs_away_raw]))[0])
            p_cs_home = float(np.clip(0.85 * p_cs_h_iso + 0.15 * 0.28, 0.02, 0.75))
            p_cs_away = float(np.clip(0.85 * p_cs_a_iso + 0.15 * 0.28, 0.02, 0.75))
        else:
            p_cs_home = float(np.clip(0.85 * p_cs_home_raw + 0.15 * 0.28, 0.02, 0.75))
            p_cs_away = float(np.clip(0.85 * p_cs_away_raw + 0.15 * 0.28, 0.02, 0.75))

        # Expected goals conceded
        exp_gc_home = float(np.sum([y * np.sum(grid[:, y]) for y in range(max_goals + 1)]))
        exp_gc_away = float(np.sum([x * np.sum(grid[x, :]) for x in range(max_goals + 1)]))

        # 1X2 Match outcomes
        p_home_win = float(np.sum(np.tril(grid, -1)))
        p_draw = float(np.sum(np.diag(grid)))
        p_away_win = float(np.sum(np.triu(grid, 1)))

        res = {
            "lambda": lam,
            "mu": mu,
            "grid": grid,
            "p_cs_home": p_cs_home,
            "p_cs_away": p_cs_away,
            "exp_gc_home": exp_gc_home,
            "exp_gc_away": exp_gc_away,
            "p_home_win": p_home_win,
            "p_draw": p_draw,
            "p_away_win": p_away_win,
        }
        self._cache[cache_key] = res
        return res

    def predict_fixture(self, home_team: str, away_team: str, is_home: bool) -> dict[str, float]:
        """
        Helper returning the implied continuous features for a player's team in this fixture.
        Purely structural: training features use the same recipe (fitted as-of each gameweek), and unplayed
        fixtures have no market odds at serve time, so a market blend would break train/serve parity.
        """
        pred = self.predict_scoreline_matrix(home_team, away_team)
        if is_home:
            return {
                "implied_team_xG": pred["lambda"],
                "implied_team_cs_prob": pred["p_cs_home"],
                "implied_opp_xG": pred["mu"],
                "implied_opp_cs_prob": pred["p_cs_away"],
            }
        return {
            "implied_team_xG": pred["mu"],
            "implied_team_cs_prob": pred["p_cs_away"],
            "implied_opp_xG": pred["lambda"],
            "implied_opp_cs_prob": pred["p_cs_home"],
        }


# Singleton model instance
dixon_coles_model = DixonColesModel()
