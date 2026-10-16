import asyncio
import time

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import ManagerStateService
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


async def test_horizons():
    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elements_map = {e.id: e for e in bootstrap.elements}

    purchase_prices = {
        496: 45,
        109: 45,
        8: 55,
        418: 50,
        87: 45,
        10: 55,
        259: 40,
        427: 80,
        154: 97,
        368: 70,
        286: 51,
        124: 55,
        411: 155,
        346: 60,
        165: 77,
    }
    gw5_market_costs = {
        496: 45,
        109: 45,
        8: 58,
        418: 49,
        87: 45,
        10: 55,
        259: 40,
        427: 79,
        154: 97,
        368: 70,
        286: 51,
        124: 57,
        411: 156,
        346: 60,
        165: 78,
    }
    squad_eids = [496, 8, 418, 87, 427, 154, 368, 286, 124, 411, 346, 109, 165, 10, 259]
    projs = projection_engine.predict_multi_gameweeks(start_gw=6, horizon=8, bootstrap=bootstrap, fixtures=fixtures)
    gw6_pool = projs[6].copy()

    squad_rows = []
    for eid in squad_eids:
        elem = elements_map[eid]
        pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[elem.element_type]
        buy_p = purchase_prices[eid]
        now_c = gw5_market_costs[eid]
        sell_p = ManagerStateService.calculate_selling_price(buy_p, now_c)
        squad_rows.append(
            {
                "element": eid,
                "web_name": elem.web_name,
                "position": pos_str,
                "team": elem.team,
                "value": now_c,
                "purchase_price": buy_p,
                "selling_price": sell_p,
                "expected_points": float(gw6_pool[gw6_pool["element"] == eid]["expected_points"].iloc[0]),
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    for H in [1, 3, 5, 8]:
        cfg = MultiPeriodConfig(horizon_len=H, solver_name="highs", time_limit_sec=30)
        opt = MultiPeriodOptimizer(cfg)
        t0 = time.perf_counter()
        plan = opt.optimize(
            current_squad_df=squad_df,
            player_pool_df=gw6_pool,
            bank=18.0,
            free_transfers=1,
            horizon_projections=projs,
            current_gw=5,
            target_gw=6,
            horizon_len=H,
            forced_w1_transfers=0,
        )
        elapsed = time.perf_counter() - t0
        act = plan.recommended_action
        starters = [elements_map[p].web_name for p in act.starters]
        cap = elements_map[act.captain].web_name
        print(
            f"H={H}: status={plan.solver_status}, time={elapsed:.2f}s, obj={plan.total_objective:.2f}, starters={len(starters)}, cap={cap}"
        )


asyncio.run(test_horizons())
