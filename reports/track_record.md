# Prediction track record

Predictions frozen before each deadline, scored after FPL finalises the gameweek. Same players for every
source. Lower error is better; 'Top 20' is the average points actually scored by each source's 20
highest-predicted players (higher is better). Buckets follow the OpenFPL paper (arXiv 2508.09992).

## Gameweek 7 (model v2026.10.16.1307, 667 players)

| Source | Error (RMSE) | Did not play | 1-2 pts | 3-4 pts | 5+ pts | Top 20 scored |
|---|---|---|---|---|---|---|
| FPL Oracle | 2.039 | 0.61 | 1.534 | 1.549 | 5.532 | 4.65 |
| FPL ep_next | 2.469 | 0.97 | 2.647 | 2.424 | 5.58 | 3.15 |
| Last-5 average | 2.195 | 1.01 | 2.219 | 1.492 | 5.133 | 3.5 |

