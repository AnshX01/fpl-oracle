from fpl_oracle.ml.predict import projection_engine

projs = projection_engine.predict_horizon(start_gw=6, horizon_len=8)
print("Projections generated for GWs:", list(projs.keys()))
for gw in list(projs.keys())[:3]:
    df = projs[gw]
    print(f"GW{gw}: count={len(df)}, mean={df['expected_points'].mean():.2f}")
