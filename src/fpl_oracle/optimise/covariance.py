"""
Squad Portfolio Covariance & Correlation Engine (Markowitz Modern Portfolio Theory).
Quantifies lineup risk, backline double-up variance, attacking stack upside, and fixture hedges
using empirical covariance matrices from Monte Carlo match simulation.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.sim.match_sim import MatchSimulator, SimulationResult

logger = logging.getLogger("fpl_oracle.optimise.covariance")


@dataclass
class PlayerPortfolioMetric:
    element: int
    web_name: str
    team: int | str
    position: str
    is_starter: bool
    is_captain: bool
    is_vice_captain: bool
    weight: float
    expected_points: float
    volatility: float  # std dev
    marginal_risk_contribution: float  # (Sigma @ w)_i / sigma_p


@dataclass
class CorrelatedPairMetric:
    element_1: int
    name_1: str
    element_2: int
    name_2: str
    relationship: str  # "DEFENSIVE_DOUBLE_UP", "ATTACKING_STACK", "OPPONENT_HEDGE", "NEUTRAL"
    covariance: float
    correlation: float
    variance_contribution: float  # 2 * w_1 * w_2 * cov


@dataclass
class SquadCovarianceResult:
    player_metrics: list[PlayerPortfolioMetric]
    covariance_matrix: list[list[float]]  # (N, N) rounded
    correlation_matrix: list[list[float]]  # (N, N) rounded
    portfolio_expected_points: float
    portfolio_variance: float
    portfolio_volatility: float  # sigma_p
    independent_volatility: float  # sqrt(sum w_i^2 sigma_i^2)
    covariance_delta_variance: float  # sigma_p^2 - sigma_ind^2
    diversification_ratio: float  # sum(w_i sigma_i) / sigma_p
    portfolio_type: str  # "CONCENTRATED", "BALANCED", "HEDGED"
    defensive_stack_risk: float  # variance added by same-club backline double/triple ups
    attacking_stack_variance: float  # variance added by same-club attacker pairs
    opponent_hedge_benefit: float  # variance reduced by opposing attacker vs keeper/defender
    sharpe_ratio: float  # portfolio_xp / portfolio_volatility
    top_correlated_pairs: list[CorrelatedPairMetric]
    top_hedged_pairs: list[CorrelatedPairMetric]
    summary: str
    shrinkage_intensity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ledoit_wolf_constant_correlation(squad_draws: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes the Ledoit-Wolf (2004) linear shrinkage covariance matrix towards
    a constant correlation target to ensure well-conditioned, strictly positive-definite matrices:
    Sigma_shrunk = delta * F + (1 - delta) * S.
    """
    n, t = squad_draws.shape
    if n <= 1 or t <= 1:
        cov = np.cov(squad_draws)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        return cov, 0.0

    mean = np.mean(squad_draws, axis=1, keepdims=True)
    y = squad_draws - mean
    sample_cov = (y @ y.T) / max(1, t - 1)

    var = np.diag(sample_cov).reshape(-1, 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    unit_cor_var = std @ std.T

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(unit_cor_var > 1e-12, sample_cov / unit_cor_var, 0.0)
    np.fill_diagonal(corr, 1.0)

    sum_corr = np.sum(corr) - n
    avg_corr = float(np.clip(sum_corr / max(1, n * (n - 1)), -0.999, 0.999))

    # Shrinkage target F
    prior = avg_corr * unit_cor_var
    np.fill_diagonal(prior, var.flatten())

    # Asymptotic variance components
    y2 = y**2
    phi_mat = (y2 @ y2.T) / t - sample_cov**2
    phi = float(np.sum(phi_mat))

    theta_mat = ((y**3) @ y.T) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_std_prod = np.where(unit_cor_var > 1e-12, 1.0 / unit_cor_var, 0.0)

    rho = float(np.sum(np.diag(phi_mat)) + avg_corr * np.sum(inv_std_prod * theta_mat))
    gamma = float(np.linalg.norm(sample_cov - prior, "fro") ** 2)

    if gamma < 1e-12:
        shrink = 0.0
    else:
        kappa = (phi - rho) / gamma
        shrink = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk_cov = shrink * prior + (1.0 - shrink) * sample_cov
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)

    # Ensure strictly positive-definite: eigenvalues >= 1e-8
    eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
    if np.any(eigvals < 1e-8):
        eigvals = np.maximum(eigvals, 1e-8)
        shrunk_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)

    return shrunk_cov, shrink


class SquadPortfolioOptimizer:
    """
    Computes closed-form Markowitz covariance and correlation matrices for an FPL squad.
    """

    def __init__(self, simulator: MatchSimulator | None = None):
        self.simulator = simulator or MatchSimulator(n_sims=2000, seed=42)

    def analyze_squad_covariance(
        self,
        squad_df: pd.DataFrame,
        sim_result: SimulationResult | None = None,
        projections_df: pd.DataFrame | None = None,
        captain_id: int | None = None,
        vice_captain_id: int | None = None,
        is_triple_captain: bool = False,
        opp_team: dict[Any, Any] | None = None,
    ) -> SquadCovarianceResult:
        """
        Calculates the complete 15x15 (or NxN) covariance and correlation matrices.

        squad_df columns:
        - element: int
        - web_name: str
        - position: str ("GKP", "DEF", "MID", "FWD")
        - team: int or str
        - is_starter: bool (optional, defaults to position <= 11)
        - is_captain: bool (optional)
        - is_vice_captain: bool (optional)
        - expected_points: float (optional)
        """
        df = squad_df.copy().reset_index(drop=True)
        n = len(df)
        if n == 0:
            raise ValueError("squad_df cannot be empty.")

        # Resolve starter / captain / weights
        starter_flags = []
        cap_flags = []
        vc_flags = []
        weights = []

        cap_mult = 3.0 if is_triple_captain else 2.0

        for i, row in enumerate(df.to_dict(orient="records")):
            eid = int(row["element"])
            is_st = bool(row.get("is_starter", i < 11))
            is_cap = bool(row.get("is_captain", False) or (captain_id is not None and eid == captain_id))
            is_vc = bool(row.get("is_vice_captain", False) or (vice_captain_id is not None and eid == vice_captain_id))

            w = 0.0
            if is_st:
                w = cap_mult if is_cap else 1.0

            starter_flags.append(is_st)
            cap_flags.append(is_cap)
            vc_flags.append(is_vc)
            weights.append(w)

        df["is_starter"] = starter_flags
        df["is_captain"] = cap_flags
        df["is_vice_captain"] = vc_flags
        df["weight"] = weights

        w_vec = np.array(weights, dtype=float)

        # Obtain simulation draws for squad elements
        element_ids = df["element"].to_numpy(dtype=int)

        if sim_result is None:
            if projections_df is not None:
                sim_result = self.simulator.simulate_gameweek(projections_df)
            else:
                sim_result = self.simulator.simulate_gameweek(df)

        # Extract draws for each squad element
        squad_draws = np.zeros((n, sim_result.n_sims), dtype=float)
        sim_elem_map = {elem: idx for idx, elem in enumerate(sim_result.player_ids)}

        for i, eid in enumerate(element_ids):
            if eid in sim_elem_map:
                squad_draws[i] = sim_result.scores[sim_elem_map[eid]]
            else:
                # If player not in simulation, generate baseline distribution from expected_points
                xp = float(df.iloc[i].get("expected_points", 2.0))
                squad_draws[i] = np.random.default_rng(eid).poisson(max(0.1, xp), size=sim_result.n_sims)

        # Expected points and empirical covariance matrix with Ledoit-Wolf shrinkage
        mu_vec = np.mean(squad_draws, axis=1)
        cov_matrix, shrinkage_delta = ledoit_wolf_constant_correlation(squad_draws)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])

        # Standard deviations
        volatilities = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)))

        # Correlation matrix with safe zero-division
        corr_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif volatilities[i] > 1e-6 and volatilities[j] > 1e-6:
                    corr_matrix[i, j] = float(
                        np.clip(cov_matrix[i, j] / (volatilities[i] * volatilities[j]), -1.0, 1.0)
                    )
                else:
                    corr_matrix[i, j] = 0.0

        # Portfolio metrics under weights w_vec
        portfolio_xp = float(np.dot(w_vec, mu_vec))
        portfolio_variance = float(w_vec @ cov_matrix @ w_vec)
        portfolio_volatility = float(np.sqrt(max(0.0, portfolio_variance)))

        independent_variance = float(np.sum((w_vec**2) * (volatilities**2)))
        independent_volatility = float(np.sqrt(max(0.0, independent_variance)))

        cov_delta = float(portfolio_variance - independent_variance)

        # Diversification ratio
        weighted_vol_sum = float(np.sum(w_vec * volatilities))
        diversification_ratio = (
            float(round(weighted_vol_sum / portfolio_volatility, 3)) if portfolio_volatility > 1e-6 else 1.0
        )

        # Portfolio type classification
        if cov_delta > 5.0:
            p_type = "CONCENTRATED"
        elif cov_delta < -5.0:
            p_type = "HEDGED"
        else:
            p_type = "BALANCED"

        # Marginal risk contribution: (Sigma @ w)_i / sigma_p
        marginal_risk: list[float] = []
        if portfolio_volatility > 1e-6:
            sigma_w = cov_matrix @ w_vec
            marginal_risk = [float(round(sw / portfolio_volatility, 3)) for sw in sigma_w]
        else:
            marginal_risk = [0.0] * n

        player_metrics: list[PlayerPortfolioMetric] = []
        for i, row in enumerate(df.to_dict(orient="records")):
            player_metrics.append(
                PlayerPortfolioMetric(
                    element=int(row["element"]),
                    web_name=str(row.get("web_name", f"Player_{row['element']}")),
                    team=row.get("team", 0),
                    position=str(row.get("position", "MID")),
                    is_starter=bool(row["is_starter"]),
                    is_captain=bool(row["is_captain"]),
                    is_vice_captain=bool(row["is_vice_captain"]),
                    weight=float(row["weight"]),
                    expected_points=float(round(mu_vec[i], 2)),
                    volatility=float(round(volatilities[i], 2)),
                    marginal_risk_contribution=marginal_risk[i],
                )
            )

        # Categorize pairwise relationships & covariance contributions
        defensive_stack_risk = 0.0
        attacking_stack_variance = 0.0
        opponent_hedge_benefit = 0.0

        # Resolve gameweek fixture opponent mapping to gate opponent hedge detection
        opp_map: dict[Any, Any] = dict(opp_team) if opp_team else {}
        if not opp_map:
            for source in [df, projections_df]:
                if source is not None:
                    col = (
                        "opponent_team"
                        if "opponent_team" in source.columns
                        else ("opp_team" if "opp_team" in source.columns else None)
                    )
                    if col and "team" in source.columns:
                        for r in source.dropna(subset=["team", col]).to_dict(orient="records"):
                            tm = r["team"]
                            op = r[col]
                            if op not in (0, "0", ""):
                                opp_map[tm] = op
                                opp_map[op] = tm
                                try:
                                    opp_map[int(tm)] = int(op)
                                    opp_map[int(op)] = int(tm)
                                except (ValueError, TypeError):
                                    pass
        opp_team = opp_map

        pair_metrics: list[CorrelatedPairMetric] = []

        for i in range(n):
            for j in range(i + 1, n):
                w_ij = 2.0 * w_vec[i] * w_vec[j]
                cov_ij = float(cov_matrix[i, j])
                corr_ij = float(corr_matrix[i, j])
                var_contrib = w_ij * cov_ij

                pos_i = df.iloc[i].get("position", "MID")
                pos_j = df.iloc[j].get("position", "MID")
                t_i = df.iloc[i].get("team", 0)
                t_j = df.iloc[j].get("team", 1)

                rel = "NEUTRAL"
                if t_i == t_j:
                    if pos_i in ("GKP", "DEF") and pos_j in ("GKP", "DEF"):
                        rel = "DEFENSIVE_DOUBLE_UP"
                        defensive_stack_risk += var_contrib
                    elif pos_i in ("MID", "FWD") and pos_j in ("MID", "FWD"):
                        rel = "ATTACKING_STACK"
                        attacking_stack_variance += var_contrib
                else:
                    # Check for opposing fixture hedge (attack vs defense in opponent club)
                    # Gated so attacker and defender are only classified as OPPONENT_HEDGE if their teams actually face each other
                    if opp_team.get(t_i) == t_j and (
                        (pos_i in ("MID", "FWD") and pos_j in ("GKP", "DEF"))
                        or (pos_j in ("MID", "FWD") and pos_i in ("GKP", "DEF"))
                    ):
                        if cov_ij < -0.05:
                            rel = "OPPONENT_HEDGE"
                            opponent_hedge_benefit += abs(var_contrib)

                if abs(corr_ij) > 0.01:
                    pair_metrics.append(
                        CorrelatedPairMetric(
                            element_1=int(df.iloc[i]["element"]),
                            name_1=str(df.iloc[i].get("web_name", f"Player_{df.iloc[i]['element']}")),
                            element_2=int(df.iloc[j]["element"]),
                            name_2=str(df.iloc[j].get("web_name", f"Player_{df.iloc[j]['element']}")),
                            relationship=rel,
                            covariance=float(round(cov_ij, 3)),
                            correlation=float(round(corr_ij, 3)),
                            variance_contribution=float(round(var_contrib, 3)),
                        )
                    )

        # Sort pairs
        sorted_by_corr = sorted(pair_metrics, key=lambda p: p.correlation, reverse=True)
        top_correlated = sorted_by_corr[:5]
        top_hedged = sorted(pair_metrics, key=lambda p: p.correlation)[:5]

        sharpe = float(round(portfolio_xp / portfolio_volatility, 2)) if portfolio_volatility > 1e-6 else 0.0

        summary = (
            f"Portfolio xP: {portfolio_xp:.1f} | Volatility: {portfolio_volatility:.2f} pts "
            f"(Independent: {independent_volatility:.2f}) | Structure: {p_type} "
            f"(Delta Cov: {cov_delta:+.2f} pts²). "
            f"Backline double-up risk: {defensive_stack_risk:+.2f} pts²; "
            f"Attacking stack variance: {attacking_stack_variance:+.2f} pts²; "
            f"Sharpe ratio: {sharpe:.2f}."
        )

        return SquadCovarianceResult(
            player_metrics=player_metrics,
            covariance_matrix=[[float(round(val, 3)) for val in row] for row in cov_matrix],
            correlation_matrix=[[float(round(val, 3)) for val in row] for row in corr_matrix],
            portfolio_expected_points=float(round(portfolio_xp, 2)),
            portfolio_variance=float(round(portfolio_variance, 2)),
            portfolio_volatility=float(round(portfolio_volatility, 2)),
            independent_volatility=float(round(independent_volatility, 2)),
            covariance_delta_variance=float(round(cov_delta, 2)),
            diversification_ratio=diversification_ratio,
            portfolio_type=p_type,
            defensive_stack_risk=float(round(defensive_stack_risk, 2)),
            attacking_stack_variance=float(round(attacking_stack_variance, 2)),
            opponent_hedge_benefit=float(round(opponent_hedge_benefit, 2)),
            sharpe_ratio=sharpe,
            top_correlated_pairs=top_correlated,
            top_hedged_pairs=top_hedged,
            summary=summary,
            shrinkage_intensity=float(round(shrinkage_delta, 4)),
        )


squad_portfolio_optimizer = SquadPortfolioOptimizer()
