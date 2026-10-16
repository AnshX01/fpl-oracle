"""
Team goals and match scoreline modelling module (WI-3.1).
Exposes DixonColesModel for bivariate scoreline estimation and match prediction.
"""

from fpl_oracle.ml.components.dixon_coles import DixonColesModel, dixon_coles_model

__all__ = ["DixonColesModel", "dixon_coles_model"]
