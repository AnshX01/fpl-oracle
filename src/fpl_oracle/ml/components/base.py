"""
Base class for ML component models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

class BaseComponent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        pass

    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "BaseComponent":
        with open(filepath, "rb") as f:
            return pickle.load(f)
