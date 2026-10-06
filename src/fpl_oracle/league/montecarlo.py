"""
Monte Carlo Mini-League Win-Probability Simulator.
Simulates player point distributions across rivals to estimate
championship probability P(1st) and rank distributions.
Features:
- Local seeded RNG (np.random.default_rng)
- Common-player draw across all managers (no independent noise for the same player)
- Honest handling when rivals data is not loaded (no fake 100% win rate)
"""

from typing import Any

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 500):
        self.n_simulations = n_simulations

    def simulate_league(
        self,
        user_points: float,
        user_squad_df: pd.DataFrame,
        rival_squads: list[dict[str, Any]],
        projections_df: pd.DataFrame,
        horizon_gws: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo simulations across user and rivals over the horizon.
        Uses common-player draws so shared players produce identical scores for all managers.
        """
        rng = np.random.default_rng(seed)

        if not rival_squads:
            return {
                "status": "NO_RIVALS_FOUND",
                "user_win_probability_pct": 0.0,
                "user_top3_probability_pct": 0.0,
                "expected_final_rank": 1.0,
                "simulations_count": 0,
                "rank_distribution": {"rank_1": 0.0, "top_3": 0.0},
                "message": "No mini-league rivals loaded. Enter a valid target league ID or sync league standings to simulate.",
            }

        # Map player projections (mean, std)
        proj_map = {}
        for _, row in projections_df.iterrows():
            elem_id = int(row["element"])
            xp = float(row.get("expected_points", 3.0))
            var = float(row.get("variance", 4.0))
            sigma = max(0.5, np.sqrt(var))
            proj_map[elem_id] = (xp, sigma)

        # Collect user starting elements
        if "is_starter" in user_squad_df.columns:
            user_starters = [int(e) for e in user_squad_df[user_squad_df["is_starter"] == True]["element"].tolist()[:11]]
        else:
            user_starters = [int(e) for e in user_squad_df["element"].tolist()[:11]]

        # Identify user captain
        if "is_captain" in user_squad_df.columns and (user_squad_df["is_captain"] == True).any():
            user_cap = int(user_squad_df[user_squad_df["is_captain"] == True].iloc[0]["element"])
        elif "expected_points" in user_squad_df.columns:
            user_cap = int(user_squad_df.sort_values(by="expected_points", ascending=False).iloc[0]["element"])
        elif user_starters:
            user_cap = user_starters[0]
        else:
            user_cap = None

        # Prepare rival entries
        rival_entries = []
        for r in rival_squads:
            starters = [int(p["element"]) for p in r.get("squad", []) if p.get("is_starter", True)][:11]
            cap = r.get("captain_element") or (starters[0] if starters else None)
            rival_entries.append(
                {
                    "entry_id": r["entry_id"],
                    "name": r.get("player_name", ""),
                    "current_points": float(r.get("total_points", 0.0)),
                    "starters": starters,
                    "captain": cap,
                }
            )

        # Identify all unique players in user + rival teams
        all_unique_elements = set(user_starters)
        for r in rival_entries:
            all_unique_elements.update(r["starters"])

        # Run simulations
        user_wins = 0
        user_top3 = 0
        user_ranks = []

        for _ in range(self.n_simulations):
            trial_user_pts = user_points
            trial_rival_pts = [r["current_points"] for r in rival_entries]

            for _ in range(horizon_gws):
                # Common player draw: draw each player's score ONCE per matchday
                player_draws: dict[int, float] = {}
                for elem in all_unique_elements:
                    mu, sig = proj_map.get(elem, (3.5, 2.0))
                    player_draws[elem] = max(0.0, float(rng.normal(mu, sig)))

                # User points this GW
                gw_user = sum(
                    player_draws.get(elem, 0.0) * (2.0 if elem == user_cap else 1.0)
                    for elem in user_starters
                )
                trial_user_pts += gw_user

                # Rivals points this GW
                for i, r in enumerate(rival_entries):
                    gw_rival = sum(
                        player_draws.get(elem, 0.0) * (2.0 if elem == r["captain"] else 1.0)
                        for elem in r["starters"]
                    )
                    trial_rival_pts[i] += gw_rival

            # Determine rank
            all_scores = [trial_user_pts] + trial_rival_pts
            user_score = trial_user_pts
            rank = sum(1 for s in all_scores if s > user_score) + 1
            user_ranks.append(rank)

            if rank == 1:
                user_wins += 1
            if rank <= 3:
                user_top3 += 1

        win_prob = round((user_wins / self.n_simulations) * 100.0, 1)
        top3_prob = round((user_top3 / self.n_simulations) * 100.0, 1)
        avg_rank = round(float(np.mean(user_ranks)), 2)

        return {
            "status": "SIMULATION_SUCCESS",
            "user_win_probability_pct": win_prob,
            "user_top3_probability_pct": top3_prob,
            "expected_final_rank": avg_rank,
            "simulations_count": self.n_simulations,
            "rank_distribution": {"rank_1": round(user_wins / self.n_simulations * 100.0, 1), "top_3": top3_prob},
        }


monte_carlo_simulator = MonteCarloSimulator()
