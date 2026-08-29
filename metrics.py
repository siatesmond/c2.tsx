"""Evaluation metrics.

Provides:
  * classification_metrics - accuracy / precision / recall / F1 / ROC-AUC for the
    binary real-vs-AI task given true labels and predicted probabilities.
  * robustness_score - summarises per-transformation ROC-AUCs into a single
    robustness score (mean AUC across the 15 transforms by default).
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)


def optimal_threshold(y_true, y_prob, method="youden"):
    # Pick the decision threshold that maximizes the ROC-based separation, so
    # any threshold-dependent metric (accuracy/F1/...) is reported at the
    # ROC-optimal operating point rather than an arbitrary fixed value.
    #   youden -> maximize Youden's J = TPR - FPR (best balance of sensitivity/specificity)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5  # ROC is undefined with a single class; fall back to default.
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if method == "youden":
        scores = tpr - fpr
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(thr[int(np.argmax(scores))])


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


def robustness_score(per_transform_auc, weight_by_severity=True,
                      severity_weights=None):
    # Aggregate per-transformation ROC-AUCs into a single robustness score.
    # By default (no weights) this is simply the mean AUC across all transforms.
    # When severity_weights are supplied, a weighted mean is used instead.
    names = list(per_transform_auc.keys())
    aucs = np.array([per_transform_auc[n] for n in names], dtype=float)
    if weight_by_severity and severity_weights is not None:
        weights = np.array([severity_weights.get(n, 1.0) for n in names], dtype=float)
        weights = weights / weights.sum()
        score = float(np.sum(aucs * weights))
    else:
        score = float(np.mean(aucs))
    return {
        "robustness_score": score,
        "mean_auc": float(np.mean(aucs)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "per_transform": per_transform_auc,
    }


def final_score(clean_auc, robust_auc):
    # Final leaderboard score: equal-weight blend of clean and robust AUC.
    return 0.5 * float(clean_auc) + 0.5 * float(robust_auc)
