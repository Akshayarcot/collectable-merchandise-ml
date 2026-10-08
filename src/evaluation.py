"""
evaluation.py
Comprehensive evaluation metrics, cross-validation harnesses, and reporting utilities.
Tracks:
- Weighted Precision, Weighted Recall, Weighted F1-score
- Per-class precision for each price bin (Bin 1 to Bin 5)
- Train vs Validation performance to observe overfitting and variance reduction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int] = [1, 2, 3, 4, 5]) -> Dict[str, Any]:
    """
    Computes weighted average metrics and per-class precision across all price bins.
    """
    w_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    w_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    per_class_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1": float(w_f1),
        "per_class_precision": {f"Bin {lbl}": float(p) for lbl, p in zip(labels, per_class_prec)},
        "per_class_recall": {f"Bin {lbl}": float(r) for lbl, r in zip(labels, per_class_rec)},
        "confusion_matrix": cm.tolist()
    }
    return metrics


def format_metrics_table(train_metrics: Dict[str, Any], valid_metrics: Dict[str, Any], model_name: str) -> str:
    """
    Formats model results into clean readable comparison markdown tables matching the paper's format.
    """
    lines = [
        f"### Model Evaluation: {model_name}",
        "",
        "| Split | Weighted Precision | Weighted Recall | Weighted F1 |",
        "| :--- | :---: | :---: | :---: |",
        f"| Train | {train_metrics['weighted_precision']:.3f} | {train_metrics['weighted_recall']:.3f} | {train_metrics['weighted_f1']:.3f} |",
        f"| Valid | {valid_metrics['weighted_precision']:.3f} | {valid_metrics['weighted_recall']:.3f} | {valid_metrics['weighted_f1']:.3f} |",
        "",
        "**Per-Class Precision:**",
        "",
        "| Split | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| Train | {train_metrics['per_class_precision']['Bin 1']:.3f} | {train_metrics['per_class_precision']['Bin 2']:.3f} | {train_metrics['per_class_precision']['Bin 3']:.3f} | {train_metrics['per_class_precision']['Bin 4']:.3f} | {train_metrics['per_class_precision']['Bin 5']:.3f} |",
        f"| Valid | {valid_metrics['per_class_precision']['Bin 1']:.3f} | {valid_metrics['per_class_precision']['Bin 2']:.3f} | {valid_metrics['per_class_precision']['Bin 3']:.3f} | {valid_metrics['per_class_precision']['Bin 4']:.3f} | {valid_metrics['per_class_precision']['Bin 5']:.3f} |",
        ""
    ]
    return "\n".join(lines)
