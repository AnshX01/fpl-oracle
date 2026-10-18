"""
Stateful Transfer Optimizer Routing Module.

Transfer options for every caller come from the Multi-Period MILP (src/fpl_oracle/optimise/multiperiod.py),
the same engine as the main advice. The beam search class is inherited only for its chip-plan fallback.
"""

from typing import Any

import pandas as pd

from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
from fpl_oracle.optimise.transfers_beam import (
    TransferOptimizer as BeamSearchTransferOptimizer,
)


class TransferOptimizer(BeamSearchTransferOptimizer):
    """
    Transfer options (no chips unless named) from the MILP; chip-plan fallback inherited from the beam search.
    """

    def evaluate_transfer_options(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int = 1,
        horizon_projections: dict[int, pd.DataFrame] | None = None,
        current_gw: int = 1,
        target_gw: int = 1,
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        risk_preference: str = "balanced",
        include_price_gain: bool = False,
        stability_threshold: float = 0.3,
        num_mc_scenarios: int = 0,  # beam-search signature; the MILP gate uses the fitted estimation error instead
        horizon_len: int | None = None,
        chips_available: list[str] | None = None,
        previous_chip: str | None = None,
    ) -> dict[str, Any]:
        """
        Transfer options only, whichever engine runs: the beam search plans no chips, so the MILP is given
        only the chips the caller names (none by default). It must never pick a chip the manager has used.
        """
        projections = horizon_projections if horizon_projections is not None else {target_gw: player_pool_df}

        # Every surface plans with the same MILP as the main advice, so a backup plan can never contradict it.
        return multiperiod_optimizer.evaluate_transfer_options(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=projections,
            current_gw=current_gw,
            target_gw=target_gw,
            locked_in_ids=locked_in_ids,
            locked_out_ids=locked_out_ids,
            excluded_team_ids=excluded_team_ids,
            chips_available=chips_available or [],
            previous_chip=previous_chip,
            risk_preference=risk_preference,
            include_price_gain=include_price_gain,
            stability_threshold=stability_threshold,
            horizon_len=horizon_len,
            xp_estimation_sd=scoring_ensemble.xp_estimation_sd,
        )


transfer_optimizer = TransferOptimizer()
