"""Evaluation metrics.

Provides:
  * classification_metrics - accuracy / precision / recall / F1 / ROC-AUC for the
    binary real-vs-AI task given true labels and predicted probabilities.
  * robustness_score - summarises per-transformation accuracies into a single
    robustness score (mean accuracy across the 15 transforms by default).
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def classification_metrics(y_true, y_prob, threshold=0.5):
    # Convert probabilities -> hard predictions using the decision threshold.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Standard multi-metric report on the hard predictions.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # ROC-AUC needs both classes present; otherwise report NaN.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics, y_pred


def robustness_score(per_transform_acc, weight_by_severity=True,
                     severity_weights=None):
    # Aggregate per-transformation accuracies into a single robustness score.
    # By default (no weights) this is simply the mean accuracy across all transforms.
    # When severity_weights are supplied, a weighted mean is used instead.
    names = list(per_transform_acc.keys())
    accs = np.array([per_transform_acc[n] for n in names], dtype=float)
    if weight_by_severity and severity_weights is not None:
        weights = np.array([severity_weights.get(n, 1.0) for n in names], dtype=float)
        weights = weights / weights.sum()
        score = float(np.sum(accs * weights))
    else:
        score = float(np.mean(accs))
    return {
        "robustness_score": score,
        "mean_accuracy": float(np.mean(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        "per_transform": per_transform_acc,
    }
