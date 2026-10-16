# Active Production Model Evaluation & Integrity Lock

**Active Version**: `v2026.10.16.1307`
**Updated**: 2026-10-16T07:36:33.329985+00:00
**Target Commit**: `64b6f81739e21791cdb1bf010ddc417fb854cd2b`
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
| `minutes_model.pkl` | `136f9d1d4085105f01061bab04b5b332d576ea29dd56f5458ed58cb296a5a79b` |
| `attacking_model.pkl` | `bc7adba50e0bb401d6f5d96140ed55f31c0dde657951de9745107656ab0bc98e` |
| `defending_model.pkl` | `c31b59a85d587230fab5c06eb11a8c7e4d1efa8e8c3061f02066224c3bdba4d9` |
| `defcon_model.pkl` | `ae9e1abf0df089f9e2c63043c9c763d00af3e977cb1b41b6355c60f32691bf8a` |
| `bonus_model.pkl` | `5a3f066d7e3e9a4531492897f9ac4811854334d2ad39cfbe3ecb1df6393b80b1` |
| `cards_saves_model.pkl` | `5a970c4195d6dd5f275cb3e8e220ca050d1d79620803cbac6ddee1e671447fcc` |
| `calibration.json` | `4251387c9dad9ed85b5e6ab9edd8b5306c2059bfbc4dc7e730acb3a1c9806995` |
| `cqr/cqr_DEF_q10.txt` | `5131ea0e7ad7dedff0b87b6ce68b822db8a64ffa36939d3decf2e635a8995502` |
| `cqr/cqr_DEF_q90.txt` | `ab05536a92d9eb36dfe2a3c4f06191003793cc8c2d6a0bc8970aaf4080b21e85` |
| `cqr/cqr_FWD_q10.txt` | `ff4c7dc6a0773c3c459e55614256cb119864894f7cc7d2805470cc5d93f3c5e7` |
| `cqr/cqr_FWD_q90.txt` | `f53fef6cb96b7a67010d3414bee00ca083ad476dc60626e52a82597f911690a3` |
| `cqr/cqr_GKP_q10.txt` | `d16aa0c263e223b253099e4fa9609e2ff35fdf16270b6d99446cc73a5d2c3401` |
| `cqr/cqr_GKP_q90.txt` | `23a08fda547a9b61f11f9e257ad943111ec3827e01d6f1bc4e46672818751aba` |
| `cqr/cqr_MID_q10.txt` | `ee612b1d8d8aadf79cf53b2ac1c03a4c037c87c8582fee5dec319778718520d5` |
| `cqr/cqr_MID_q90.txt` | `40d21bd7059c0b4bf3e1331f8743523cdb9752e3180a59188a7d3700d4502ca4` |
| `cqr/cqr_meta.json` | `9529d90bb4179fd8a25581fb214bce8b443435b7f0a209b32bdb6ec669305254` |

*Verified at server startup by `fpl_oracle.ml.predict.verify_model_manifest_integrity`.*
