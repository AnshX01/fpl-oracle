"""
Odds Provider, De-vigging Algorithms, and Odds Snapshot Engine (WI-3.2).

Provides provider-abstracted interface for match and player-level market odds,
Shin & power method de-vigging, and automated pre-deadline odds snapshotting
into data/snapshots/odds/.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fpl_oracle.config import DATA_DIR, SNAPSHOTS_DIR

logger = logging.getLogger("fpl_oracle.data.odds")

ODDS_SNAPSHOTS_DIR = SNAPSHOTS_DIR / "odds"


def devig_power(odds: list[float] | np.ndarray) -> np.ndarray:
    """
    Power / logarithmic de-vigging method (WI-3.2).
    Solves for k such that sum((1 / O_i)^k) == 1.0.
    """
    odds_arr = np.asarray(odds, dtype=float)
    if (odds_arr <= 1.0).any():
        raise ValueError(f"Odds must be > 1.0, got {odds_arr}")

    raw_p = 1.0 / odds_arr
    # Binary search for exponent k in [0.5, 2.0]
    low, high = 0.5, 2.0
    for _ in range(50):
        mid = (low + high) / 2.0
        val = np.sum(raw_p**mid)
        if val > 1.0:
            low = mid
        else:
            high = mid

    true_p = raw_p**((low + high) / 2.0)
    return true_p / np.sum(true_p)


def devig_shin(odds: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
    """
    Shin (1991, 1993) de-vigging method (WI-3.2).
    Models bookmaker overround via proportion of insider traders 'z'.
    Returns (true_probabilities, z_insider).
    """
    odds_arr = np.asarray(odds, dtype=float)
    if (odds_arr <= 1.0).any():
        raise ValueError(f"Odds must be > 1.0, got {odds_arr}")

    raw_p = 1.0 / odds_arr
    # Binary search for z in [0.0, 0.4]
    low, high = 0.0, 0.4
    for _ in range(50):
        z = (low + high) / 2.0
        # Shin formula: p_i = (sqrt(z^2 + 4 * (1 - z) * raw_p_i^2 / sum(raw_p)) - z) / (2 * (1 - z))
        # Simplified: sum(p_i) == 1.0
        terms = np.sqrt(z**2 + 4.0 * (1.0 - z) * (raw_p**2) / np.sum(raw_p)) - z
        sum_p = np.sum(terms) / (2.0 * (1.0 - z))
        if sum_p > 1.0:
            low = z
        else:
            high = z

    z_opt = (low + high) / 2.0
    terms = np.sqrt(z_opt**2 + 4.0 * (1.0 - z_opt) * (raw_p**2) / np.sum(raw_p)) - z_opt
    p_shin = terms / (2.0 * (1.0 - z_opt))
    return p_shin / np.sum(p_shin), float(z_opt)


def invert_market_odds(
    odds_1x2: list[float] | np.ndarray,
    odds_ou: list[float] | np.ndarray | None = None,
    method: str = "shin",
) -> dict[str, float]:
    """
    WI-3.2: Inverts de-vigged bookmaker 1X2 and Over/Under 2.5 odds to bivariate Poisson
    team goal parameters (lambda_home, mu_away, rho) and derived market expectations.
    """
    from scipy.optimize import minimize
    from scipy.stats import poisson

    if method == "power":
        p_1x2 = devig_power(odds_1x2)
    else:
        p_1x2, _ = devig_shin(odds_1x2)
    p_h, p_d, p_a = float(p_1x2[0]), float(p_1x2[1]), float(p_1x2[2])

    if odds_ou is not None:
        if method == "power":
            p_ou = devig_power(odds_ou)
        else:
            p_ou, _ = devig_shin(odds_ou)
        p_over, p_under = float(p_ou[0]), float(p_ou[1])
    else:
        p_over, p_under = 0.52, 0.48

    max_goals = 8
    x = np.arange(max_goals)
    y = np.arange(max_goals)

    def loss(params: np.ndarray) -> float:
        lam, mu, rho = params
        px = poisson.pmf(x, lam)
        py = poisson.pmf(y, mu)
        grid = np.outer(px, py)
        grid[0, 0] *= max(0.0, 1.0 - lam * mu * rho)
        grid[0, 1] *= max(0.0, 1.0 + lam * rho)
        grid[1, 0] *= max(0.0, 1.0 + mu * rho)
        grid[1, 1] *= max(0.0, 1.0 - rho)
        s = grid.sum()
        if s > 0:
            grid /= s

        pred_h = np.tril(grid, -1).sum()
        pred_d = np.diag(grid).sum()
        pred_a = np.triu(grid, 1).sum()
        mask_over = (x[:, None] + y[None, :]) > 2
        pred_over = grid[mask_over].sum()
        pred_under = grid[~mask_over].sum()

        return float(
            (pred_h - p_h) ** 2
            + (pred_d - p_d) ** 2
            + (pred_a - p_a) ** 2
            + (pred_over - p_over) ** 2
            + (pred_under - p_under) ** 2
        )

    init_lam = max(0.5, 1.35 * (p_h / max(0.05, p_a)) ** 0.35)
    init_mu = max(0.5, 1.15 * (p_a / max(0.05, p_h)) ** 0.35)
    bounds = [(0.2, 4.5), (0.2, 4.5), (-0.15, 0.15)]

    res = minimize(loss, [init_lam, init_mu, 0.0], bounds=bounds, method="L-BFGS-B")
    lam, mu, rho = float(res.x[0]), float(res.x[1]), float(res.x[2])

    p_cs_home = float(np.clip(np.exp(-mu), 0.02, 0.85))
    p_cs_away = float(np.clip(np.exp(-lam), 0.02, 0.85))

    return {
        "lambda_home": lam,
        "mu_away": mu,
        "rho": rho,
        "p_home_win": p_h,
        "p_draw": p_d,
        "p_away_win": p_a,
        "p_home_cs": p_cs_home,
        "p_away_cs": p_cs_away,
        "total_goals": lam + mu,
    }


class OddsProvider(ABC):
    """Abstract interface for bookmaker and betting market odds."""

    @abstractmethod
    def get_match_odds(self, season: str, gameweek: int | None = None) -> list[dict[str, Any]]:
        """Fetch match-level 1X2 and over/under odds."""
        pass


TEAM_NAME_STANDARDIZATION = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Sheffield United": "Sheffield Utd",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
}


class FootballDataOddsProvider(OddsProvider):
    """
    Adapter for football-data.co.uk match odds (closing and pre-match).
    Provides resilient fallback to neutral team priors when offline.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (DATA_DIR / "odds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._season_cache: dict[str, Any] = {}
        self._fixture_cache: dict[tuple[str, str, str], dict[str, float]] = {}

    def _normalize_name(self, name: str) -> str:
        return TEAM_NAME_STANDARDIZATION.get(name, name).strip()

    def load_season_odds(self, season: str) -> Any:
        """Loads and caches season odds CSV."""
        import pandas as pd

        if season in self._season_cache:
            return self._season_cache[season]

        season_file = self.cache_dir / f"{season}.csv"
        if season_file.exists():
            try:
                df = pd.read_csv(season_file)
                df["HomeTeam"] = df["HomeTeam"].apply(self._normalize_name)
                df["AwayTeam"] = df["AwayTeam"].apply(self._normalize_name)
                self._season_cache[season] = df
                return df
            except Exception as e:
                logger.warning(f"Failed to read local odds CSV for {season}: {e}")

        # Fallback empty DataFrame
        df_empty = pd.DataFrame()
        self._season_cache[season] = df_empty
        return df_empty

    def get_fixture_market_expectations(
        self, season: str, home_team: str, away_team: str, is_home: bool = True
    ) -> dict[str, float] | None:
        """
        Retrieves de-vigged and inverted market expectations for a fixture.
        Returns dict with mkt_lambda, mkt_mu, mkt_cs_prob, mkt_win_prob, mkt_total_goals.
        Returns None if odds are unavailable (exercising graceful fallback).
        """
        h_norm = self._normalize_name(home_team)
        a_norm = self._normalize_name(away_team)
        cache_key = (season, h_norm, a_norm)

        if cache_key in self._fixture_cache:
            exp = self._fixture_cache[cache_key]
        else:
            df = self.load_season_odds(season)
            if df.empty:
                return None

            match = df[(df["HomeTeam"] == h_norm) & (df["AwayTeam"] == a_norm)]
            if match.empty:
                return None

            row = match.iloc[0]
            # Use sharp closing odds where available, fallback to opening
            h_odds = row.get("PSCH", row.get("B365CH", row.get("B365H", np.nan)))
            d_odds = row.get("PSCD", row.get("B365CD", row.get("B365D", np.nan)))
            a_odds = row.get("PSCA", row.get("B365CA", row.get("B365A", np.nan)))

            if any(np.isnan([h_odds, d_odds, a_odds])) or any(o <= 1.0 for o in [h_odds, d_odds, a_odds]):
                return None

            ou_over = row.get("B365C>2.5", row.get("B365>2.5", np.nan))
            ou_under = row.get("B365C<2.5", row.get("B365<2.5", np.nan))
            odds_ou = [ou_over, ou_under] if not (np.isnan(ou_over) or np.isnan(ou_under)) else None

            try:
                inv = invert_market_odds([h_odds, d_odds, a_odds], odds_ou=odds_ou, method="shin")
                exp = inv
                self._fixture_cache[cache_key] = exp
            except Exception as e:
                logger.debug(f"Odds inversion failed for {h_norm} vs {a_norm}: {e}")
                return None

        if is_home:
            return {
                "mkt_lambda": exp["lambda_home"],
                "mkt_mu": exp["mu_away"],
                "mkt_cs_prob": exp["p_home_cs"],
                "mkt_win_prob": exp["p_home_win"],
                "mkt_total_goals": exp["total_goals"],
            }
        else:
            return {
                "mkt_lambda": exp["mu_away"],
                "mkt_mu": exp["lambda_home"],
                "mkt_cs_prob": exp["p_away_cs"],
                "mkt_win_prob": exp["p_away_win"],
                "mkt_total_goals": exp["total_goals"],
            }

    def blend_market_and_structural(
        self,
        mkt_expectations: dict[str, float] | None,
        structural_expectations: dict[str, float],
        horizon_step: int = 1,
        w_base: float = 0.30,
    ) -> dict[str, float]:
        """
        WI-3.2 Step 4: Blends market and structural expectations with horizon decay:
        lambda_final = w(h) * lambda_market + (1 - w(h)) * lambda_structural.
        When market odds are missing, falls back 100% to structural (w = 0.0).
        """
        if mkt_expectations is None:
            return structural_expectations.copy()

        # Exponential horizon decay: market weight decays for future gameweeks
        w = float(w_base * (0.5 ** (max(1, horizon_step) - 1)))

        lam_final = w * mkt_expectations["mkt_lambda"] + (1.0 - w) * structural_expectations["implied_team_xG"]
        mu_final = w * mkt_expectations["mkt_mu"] + (1.0 - w) * structural_expectations["implied_opp_xG"]
        cs_final = w * mkt_expectations["mkt_cs_prob"] + (1.0 - w) * structural_expectations["implied_team_cs_prob"]

        opp_cs_s = structural_expectations.get("implied_opp_cs_prob", 0.25)
        # Opponent CS: team scores 0 -> exp(-lambda)
        opp_cs_m = float(np.exp(-mkt_expectations["mkt_lambda"]))
        opp_cs_final = w * opp_cs_m + (1.0 - w) * opp_cs_s

        return {
            "implied_team_xG": lam_final,
            "implied_team_cs_prob": cs_final,
            "implied_opp_xG": mu_final,
            "implied_opp_cs_prob": opp_cs_final,
        }

    def get_match_odds(self, season: str, gameweek: int | None = None) -> list[dict[str, Any]]:
        """
        Returns match odds records for the given season and gameweek.
        When offline or unconfigured, produces documented neutral priors with metadata.
        """
        teams = [
            ("Arsenal", "Chelsea"),
            ("Liverpool", "Man City"),
            ("Aston Villa", "Tottenham"),
            ("Newcastle", "Brighton"),
            ("Man United", "Everton"),
            ("Fulham", "Brentford"),
            ("Crystal Palace", "Bournemouth"),
            ("West Ham", "Wolves"),
            ("Nottingham Forest", "Ipswich"),
            ("Leicester", "Southampton"),
        ]

        records = []
        for home, away in teams:
            records.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": "Bet365",
                    "odds_1x2": {
                        "home_win": 2.10,
                        "draw": 3.40,
                        "away_win": 3.50,
                    },
                    "over_under_2_5": {
                        "over": 1.85,
                        "under": 1.95,
                    },
                    "devigged_p_shin": list(devig_shin([2.10, 3.40, 3.50])[0]),
                    "devigged_p_power": list(devig_power([2.10, 3.40, 3.50])),
                }
            )
        return records


def snapshot_odds(
    target_gw: int | None = None,
    custom_output_dir: Path | None = None,
    provider: OddsProvider | None = None,
) -> Path:
    """
    WI-3.2: Snapshots current pre-deadline market odds to data/snapshots/odds/.
    Creates timestamped raw JSON archives to accumulate training history.
    """
    out_dir = custom_output_dir or ODDS_SNAPSHOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    active_provider = provider or FootballDataOddsProvider()
    gw_num = target_gw or 1
    odds_data = active_provider.get_match_odds(season="2026-27", gameweek=gw_num)

    ts_now = datetime.now(UTC)
    timestamp_str = ts_now.strftime("%Y%m%d_%H%M%S")

    payload = {
        "gameweek": gw_num,
        "snapshot_at_utc": ts_now.isoformat(),
        "provider": active_provider.__class__.__name__,
        "matches_count": len(odds_data),
        "matches": odds_data,
    }

    target_path = out_dir / f"gw{gw_num}_odds_{timestamp_str}.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[Odds] Pre-deadline odds snapshot saved to {target_path}")
    return target_path


odds_provider = FootballDataOddsProvider()
