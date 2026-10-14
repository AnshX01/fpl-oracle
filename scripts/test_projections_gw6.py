from fpl_oracle.ml.predict import projection_engine

projs = projection_engine.predict_horizon(start_gw=6, horizon_len=6)
print("Horizon keys:", list(projs.keys()))
for gw, df in projs.items():
    print(f"GW{gw}: {len(df)} players, mean xP={df['expected_points'].mean():.2f}, top 3:")
    top3 = df.nlargest(3, "expected_points")[["element", "web_name", "expected_points", "now_cost"]]
    for _, r in top3.iterrows():
        print(f"  {r['web_name']} ({r['element']}): {r['expected_points']} xP, cost={r['now_cost']}")
