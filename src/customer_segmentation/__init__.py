"""Production utilities for customer segmentation."""

from .model import load_model, save_model, score_customers, train_model

__all__ = ["load_model", "save_model", "score_customers", "train_model"]
__version__ = "0.1.0"
