# Active Production Model Evaluation & Integrity Lock

**Active Version**: `v2026.10.06.1420`
**Updated**: 2026-10-06T14:20:40.205756+00:00
**Target Commit**: `36210ceaba4d5b902787fc9fa8e1a7e77d75a79d`
**Feature Schema Hash**: `ec91cae9cf9b75c88d81b133cb6b80907bf112a126ab3e90b9e6916792140e84`

---

## 1. Production Model Performance

- **Production Served MAE**: **1.019 pts**
- **Spearman Rank Correlation ($\\rho$)**: **0.704**
- **Naive Baseline MAE**: 1.078 pts
- **Provenance Protocol**: Zero refit after promotion. All metrics strictly evaluate the served binary weights below.

---

## 2. Component Weights SHA256 Integrity Locks

| Component File | SHA256 Checksum |
|:---|:---|
| `minutes_model.pkl` | `40641ea9ba80da3f5b8d29c1c656999789ce5496950957e7ec2e21965150ca8d` |
| `attacking_model.pkl` | `ec5e0d0c732a0c7915c8425bfc676c06690117b264f18d958607744fb647bcbe` |
| `defending_model.pkl` | `abd363499a38c41503ee6bf5ae5454a4e77d405947235ae02e54938de1ead419` |
| `defcon_model.pkl` | `c5a243b73b70d19b461eba15673675d00af8b62458f53797fdcd7dc9b8d07c1a` |
| `bonus_model.pkl` | `b1850bb840cb67a7c684800a01f09f39870f010248121b8bf08481c8492d4af7` |
| `cards_saves_model.pkl` | `e6e6fb87dd5a528ffe6464ef9af241550a9eae020d2aaaee770b1e642d161a95` |
| `calibration.json` | `180cf122fd3c5614f02b15e8ca6853063046f5402f07f2406086dba078f67ba1` |

*Verified at server startup by `fpl_oracle.ml.predict.verify_model_manifest_integrity`.*
