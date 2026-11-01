"""
Scoring Ensemble domain module.
Re-exports ScoringEnsemble and singleton instance scoring_ensemble
for domain-level projection aggregation and quantile calibrated inference.
"""

from fpl_oracle.ml.ensemble import ScoringEnsemble, scoring_ensemble

__all__ = ["ScoringEnsemble", "scoring_ensemble"]
