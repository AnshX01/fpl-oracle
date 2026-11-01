# Exploratory historical replay

Not forward evidence or production-readiness certification.

```json
{
  "rounds_evaluated": [
    1,
    2,
    3
  ],
  "oracle_total": 152.0,
  "user_total": 237.0,
  "baseline_total": 118.0,
  "average_manager_total": 158.0,
  "hindsight_total": 428.0,
  "oracle_uplift_over_baseline": 34.0,
  "oracle_uplift_over_average": -6.0,
  "oracle_captain_success_rate": 0.0,
  "baseline_captain_success_rate": 33.3,
  "mean_spearman": 0.619,
  "mean_pearson": 0.5,
  "mean_mae": 1.34,
  "per_gw_results": [
    {
      "gw": 1,
      "oracle": 26.0,
      "user": 81.0,
      "baseline": 23.0,
      "average": 57.0,
      "hindsight": 145.0,
      "captain": "Bruno Borges Fernandes",
      "captain_pts": 2,
      "spearman": 0.449,
      "pearson": 0.34,
      "mae": 1.64,
      "starters": [
        {
          "name": "Bruno Borges Fernandes",
          "pos": "MID",
          "xP": 6.41,
          "actual": 2,
          "is_cap": true
        },
        {
          "name": "Harry Maguire",
          "pos": "DEF",
          "xP": 4.42,
          "actual": 1,
          "is_cap": false
        },
        {
          "name": "Karl Darlow",
          "pos": "GKP",
          "xP": 3.91,
          "actual": 0,
          "is_cap": false
        },
        {
          "name": "James Tarkowski",
          "pos": "DEF",
          "xP": 4.44,
          "actual": 6,
          "is_cap": false
        },
        {
          "name": "Morgan Gibbs-White",
          "pos": "MID",
          "xP": 4.74,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Igor Thiago Nascimento Rodrigues",
          "pos": "FWD",
          "xP": 4.28,
          "actual": 0,
          "is_cap": false
        },
        {
          "name": "Antoine Semenyo",
          "pos": "MID",
          "xP": 4.63,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Ollie Watkins",
          "pos": "FWD",
          "xP": 4.42,
          "actual": 0,
          "is_cap": false
        },
        {
          "name": "Dominik Szoboszlai",
          "pos": "MID",
          "xP": 5.49,
          "actual": 8,
          "is_cap": false
        },
        {
          "name": "Virgil van Dijk",
          "pos": "DEF",
          "xP": 4.57,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Enzo Fern\u00e1ndez",
          "pos": "MID",
          "xP": 4.75,
          "actual": 1,
          "is_cap": false
        }
      ]
    },
    {
      "gw": 2,
      "oracle": 80.0,
      "user": 104.0,
      "baseline": 74.0,
      "average": 52.0,
      "hindsight": 160.0,
      "captain": "Dominik Szoboszlai",
      "captain_pts": 4,
      "spearman": 0.676,
      "pearson": 0.58,
      "mae": 1.23,
      "starters": [
        {
          "name": "Dominik Szoboszlai",
          "pos": "MID",
          "xP": 5.67,
          "actual": 4,
          "is_cap": true
        },
        {
          "name": "Virgil van Dijk",
          "pos": "DEF",
          "xP": 4.83,
          "actual": 1,
          "is_cap": false
        },
        {
          "name": "Adrien Truffert",
          "pos": "DEF",
          "xP": 4.21,
          "actual": 4,
          "is_cap": false
        },
        {
          "name": "Caoimh\u00edn Kelleher",
          "pos": "GKP",
          "xP": 3.77,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Cole Palmer",
          "pos": "MID",
          "xP": 4.79,
          "actual": 7,
          "is_cap": false
        },
        {
          "name": "Dominic Calvert-Lewin",
          "pos": "FWD",
          "xP": 4.79,
          "actual": 8,
          "is_cap": false
        },
        {
          "name": "Jo\u00e3o Pedro Junqueira de Jesus",
          "pos": "FWD",
          "xP": 4.45,
          "actual": 9,
          "is_cap": false
        },
        {
          "name": "Morgan Rogers",
          "pos": "MID",
          "xP": 4.48,
          "actual": 5,
          "is_cap": false
        },
        {
          "name": "Bruno Borges Fernandes",
          "pos": "MID",
          "xP": 4.94,
          "actual": 23,
          "is_cap": false
        },
        {
          "name": "Harry Maguire",
          "pos": "DEF",
          "xP": 4.43,
          "actual": 4,
          "is_cap": false
        },
        {
          "name": "Matheus Santos Carneiro da Cunha",
          "pos": "MID",
          "xP": 4.45,
          "actual": 9,
          "is_cap": false
        }
      ]
    },
    {
      "gw": 3,
      "oracle": 46.0,
      "user": 52.0,
      "baseline": 21.0,
      "average": 49.0,
      "hindsight": 123.0,
      "captain": "Dominik Szoboszlai",
      "captain_pts": 3,
      "spearman": 0.733,
      "pearson": 0.58,
      "mae": 1.16,
      "starters": [
        {
          "name": "Dominik Szoboszlai",
          "pos": "MID",
          "xP": 5.75,
          "actual": 3,
          "is_cap": true
        },
        {
          "name": "Virgil van Dijk",
          "pos": "DEF",
          "xP": 4.6,
          "actual": 6,
          "is_cap": false
        },
        {
          "name": "Antoine Semenyo",
          "pos": "MID",
          "xP": 4.75,
          "actual": 6,
          "is_cap": false
        },
        {
          "name": "Caoimh\u00edn Kelleher",
          "pos": "GKP",
          "xP": 3.9,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Erling Haaland",
          "pos": "FWD",
          "xP": 5.37,
          "actual": 9,
          "is_cap": false
        },
        {
          "name": "Igor Thiago Nascimento Rodrigues",
          "pos": "FWD",
          "xP": 4.37,
          "actual": 2,
          "is_cap": false
        },
        {
          "name": "Maxim De Cuyper",
          "pos": "DEF",
          "xP": 4.11,
          "actual": 4,
          "is_cap": false
        },
        {
          "name": "Nathan Collins",
          "pos": "DEF",
          "xP": 4.81,
          "actual": 1,
          "is_cap": false
        },
        {
          "name": "Pascal Gro\u00df",
          "pos": "MID",
          "xP": 4.42,
          "actual": 1,
          "is_cap": false
        },
        {
          "name": "Phil Foden",
          "pos": "MID",
          "xP": 4.8,
          "actual": 1,
          "is_cap": false
        },
        {
          "name": "Bryan Mbeumo",
          "pos": "MID",
          "xP": 4.31,
          "actual": 8,
          "is_cap": false
        }
      ]
    }
  ]
}
```
