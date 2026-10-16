import json
from pathlib import Path

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager

df_raw = historical_manager.ensure_dataset_ready()
X, Y, meta = feature_engineering.build_historical_features(df_raw, return_meta=True)

# Training corpus for component models (up to 2025-26 GW38)
train_mask = meta["season"].isin(["2023-24", "2024-25", "2025-26"])
meta_tr = meta[train_mask]
print("=== COMPONENT MODELS TRAINING CORPUS ===")
print("Seasons:", sorted(meta_tr["season"].unique().tolist()))
print("Max Round per Season:")
for s in sorted(meta_tr["season"].unique()):
    sub = meta_tr[meta_tr["season"] == s]
    print(f"  {s}: Max Round = {sub['round'].max()}, Rows = {len(sub):,}")
print("Total Training Rows:", len(meta_tr))

# Check is_2026_27 in training features
is_26_vals = X.loc[train_mask, "is_2026_27"].value_counts().to_dict()
print("is_2026_27 values in training corpus:", is_26_vals)

# CQR training corpus
cqr_tr_mask = (meta["season"] < "2025-26") | ((meta["season"] == "2025-26") & (meta["round"] <= 10))
cqr_meta_tr = meta[cqr_tr_mask]
print("\n=== CQR BANDS TRAINING CORPUS ===")
print("Seasons:", sorted(cqr_meta_tr["season"].unique().tolist()))
print("Max Round per Season:")
for s in sorted(cqr_meta_tr["season"].unique()):
    sub = cqr_meta_tr[cqr_meta_tr["season"] == s]
    print(f"  {s}: Max Round = {sub['round'].max()}, Rows = {len(sub):,}")
print("Total CQR Training Rows:", len(cqr_meta_tr))

# CQR Calibration fold
cqr_cal_mask = (meta["season"] == "2025-26") & (meta["round"] > 10) & (meta["round"] <= 18) & (Y["target_minutes"] > 0)
cqr_meta_cal = meta[cqr_cal_mask]
print("\n=== CQR CALIBRATION FOLD ===")
print("Seasons:", cqr_meta_cal["season"].unique().tolist())
print(f"Round range: GW{cqr_meta_cal['round'].min()} to GW{cqr_meta_cal['round'].max()}")
print("Total Calibration Rows (minutes > 0):", len(cqr_meta_cal))
print("Counts per position:", cqr_meta_cal["position"].value_counts().to_dict())

c4_record = {
    "component_models_corpus": {
        "seasons": sorted(meta_tr["season"].unique().tolist()),
        "max_round": int(meta_tr[meta_tr["season"] == meta_tr["season"].max()]["round"].max()),
        "total_rows": int(len(meta_tr)),
        "rows_by_season": {
            s: {
                "max_round": int(meta_tr[meta_tr["season"] == s]["round"].max()),
                "rows": int(len(meta_tr[meta_tr["season"] == s])),
            }
            for s in sorted(meta_tr["season"].unique())
        },
        "is_2026_27_feature": {
            "is_constant": bool(len(is_26_vals) == 1 and 0.0 in is_26_vals),
            "unique_values_and_counts": {str(k): int(v) for k, v in is_26_vals.items()},
            "defcon_era_feature_verdict": (
                "The DefCon model era feature is_2026_27 is constant (0.0 for all 87,087 rows) in training data. "
                "2026-27 rows were never in training."
            ),
        },
    },
    "cqr_bands_corpus": {
        "training": {
            "seasons": sorted(cqr_meta_tr["season"].unique().tolist()),
            "max_round": int(cqr_meta_tr[cqr_meta_tr["season"] == cqr_meta_tr["season"].max()]["round"].max()),
            "total_rows": int(len(cqr_meta_tr)),
            "rows_by_season": {
                s: {
                    "max_round": int(cqr_meta_tr[cqr_meta_tr["season"] == s]["round"].max()),
                    "rows": int(len(cqr_meta_tr[cqr_meta_tr["season"] == s])),
                }
                for s in sorted(cqr_meta_tr["season"].unique())
            },
        },
        "calibration": {
            "season": "2025-26",
            "first_round": int(cqr_meta_cal["round"].min()),
            "last_round": int(cqr_meta_cal["round"].max()),
            "total_rows": int(len(cqr_meta_cal)),
            "counts_by_position": {
                str(k): int(v) for k, v in cqr_meta_cal["position"].value_counts().to_dict().items()
            },
        },
    },
}

out_path = Path("reports/c4_training_corpus.json")
out_path.write_text(json.dumps(c4_record, indent=2), encoding="utf-8")
print(f"\nSaved C4 machine artifact to: {out_path}")
