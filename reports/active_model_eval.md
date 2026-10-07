# Active Production Model Evaluation & Integrity Lock

**Active Version**: `v2026.10.06.1713`
**Updated**: 2026-10-06T17:13:06.071703+00:00
**Target Commit**: `4b6db3cc641e670c31ab8ac85446452acecdf7f7`
**Feature Schema Hash**: `ec91cae9cf9b75c88d81b133cb6b80907bf112a126ab3e90b9e6916792140e84`

---

## 1. Production Model Performance

- **Production Served MAE**: **1.016 pts**
- **Spearman Rank Correlation ($\\rho$)**: **0.704**
- **Naive Baseline MAE**: 1.078 pts
- **Provenance Protocol**: Zero refit after promotion. All metrics strictly evaluate the served binary weights below.

---

## 2. Component Weights SHA256 Integrity Locks

| Component File | SHA256 Checksum |
|:---|:---|
| `minutes_model.pkl` | `6fcb69189755b730618a983c3197c22c8ffbc1756af537f9dfa365ebe0ad3669` |
| `attacking_model.pkl` | `611091e62a65c6b5e1765c71713f69e1b5a77f4694494d1f0a84cc929f24597d` |
| `defending_model.pkl` | `7a07a777d0cc8c8bf144b5a1225430b5703ac8e4427da04b9b76b7bf8b303b8e` |
| `defcon_model.pkl` | `c0298907536f73e5f743013c120d15775fe970c600ef52973d450cc9a79e87f8` |
| `bonus_model.pkl` | `abcb23681970ae376b3188848ec5188abba7d2a4098bb4d3d43cc3fb7682e2a8` |
| `cards_saves_model.pkl` | `d96518e17d97341729a99246cdb428ed459dda2f32f04d692c9c5a49ac979940` |
| `calibration.json` | `4251387c9dad9ed85b5e6ab9edd8b5306c2059bfbc4dc7e730acb3a1c9806995` |

*Verified at server startup by `fpl_oracle.ml.predict.verify_model_manifest_integrity`.*
