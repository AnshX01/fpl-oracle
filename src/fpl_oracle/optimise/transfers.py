"""
Stateful Transfer Optimizer Routing Module.

Routes callers between the Multi-Period Mixed-Integer Linear Programming (MILP)
optimizer (src/fpl_oracle/optimise/multiperiod.py) and the preserved legacy
beam search optimizer (src/fpl_oracle/optimise/transfers_beam.py).
"""

from typing import Any

import pandas as pd

from fpl_oracle.config import is_flag_enabled
from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
from fpl_oracle.optimise.transfers_beam import (
    VALID_FORMATIONS,
    _fast_eval_squad_full,
    _safe_team,
    compute_dynamic_chip_retention_values,
)
from fpl_oracle.optimise.transfers_beam import (
    TransferOptimizer as BeamSearchTransferOptimizer,
)
from fpl_oracle.rules.transfers import (
    next_free_transfers,
    sell_price,
    validate_chip_legality,
)


class TransferOptimizer(BeamSearchTransferOptimizer):
    """
    Unified Transfer Optimizer router.
    Routes to Multi-Period MILP when flag is active or when 3+ FTs are banked (F-005 fix).
    Otherwise, executes beam search trajectory engine.
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
        num_mc_scenarios: int = 50,
        horizon_len: int | None = None,
    ) -> dict[str, Any]:
        projections = horizon_projections if horizon_projections is not None else {target_gw: player_pool_df}

        if is_flag_enabled("multiperiod_milp", default=False) or free_transfers >= 3:
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
                risk_preference=risk_preference,
                include_price_gain=include_price_gain,
                stability_threshold=stability_threshold,
                num_mc_scenarios=num_mc_scenarios,
                horizon_len=horizon_len,
            )

        return super().evaluate_transfer_options(
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
            risk_preference=risk_preference,
            include_price_gain=include_price_gain,
            stability_threshold=stability_threshold,
            num_mc_scenarios=num_mc_scenarios,
            horizon_len=horizon_len,
        )


transfer_optimizer = TransferOptimizer()

__all__ = [
    "VALID_FORMATIONS",
    "TransferOptimizer",
    "_fast_eval_squad_full",
    "_safe_team",
    "compute_dynamic_chip_retention_values",
    "next_free_transfers",
    "sell_price",
    "transfer_optimizer",
    "validate_chip_legality",
]
