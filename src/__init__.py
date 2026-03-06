"""
Fake News Detection System - Source Package
"""

from .preprocessor import TextPreprocessor
from .models import FakeNewsClassifier, ModelMetrics
from .explainer import FakeNewsExplainer
from .dataset import FakeNewsDataset

__version__ = "1.0.0"
__author__ = "PranayMahendrakar"

__all__ = [
    "TextPreprocessor",
    "FakeNewsClassifier",
    "ModelMetrics",
    "FakeNewsExplainer",
    "FakeNewsDataset",
]
