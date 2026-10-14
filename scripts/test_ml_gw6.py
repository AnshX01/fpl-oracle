import asyncio

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.ml.predict import projection_engine


async def test_proj():
    b, _ = await fpl_client.get_bootstrap_static()
    f, _ = await fpl_client.get_fixtures()
    projs = projection_engine.predict_multi_gameweeks(start_gw=6, horizon=8, bootstrap=b, fixtures=f)
    print("Generated projections for GWs:", list(projs.keys()))
    for gw in list(projs.keys())[:3]:
        print(f"GW{gw}: {len(projs[gw])} players, mean xP: {projs[gw]['expected_points'].mean():.2f}")

asyncio.run(test_proj())
