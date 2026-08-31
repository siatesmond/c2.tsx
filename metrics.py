"""Evaluation metrics.

Provides:
  * classification_metrics - accuracy / precision / recall / F1 / ROC-AUC for the
    binary real-vs-AI task given true labels and predicted probabilities.
  * optimal_threshold - the ROC-optimal decision threshold (Youden's J), used so
    the threshold-dependent metrics report the model's best operating point
    rather than an arbitrary 0.5.
  * robustness_score - summarises the per-transformation scores (ROC-AUC per
    transform) into a single robustness number (their mean by default).
  * final_score - the headline number: mean of clean ROC-AUC and robust ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)


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


def optimal_threshold(y_true, y_prob):
    # ROC-optimal decision threshold via Youden's J (max of tpr - fpr).
    # Falls back to 0.5 when only one class is present.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = float(thr[int(np.argmax(tpr - fpr))])
    if not np.isfinite(best):  # sklearn>=1.3 prepends inf to `thr`
        best = float(np.max(y_prob))
    return float(np.clip(best, 0.0, 1.0))


def robustness_score(per_transform_score, weight_by_severity=True,
                     severity_weights=None):
    # Aggregate the per-transformation scores (ROC-AUC per transform) into a
    # single robustness number. By default this is the plain mean across all
    # transforms; when severity_weights are supplied, a weighted mean is used.
    names = list(per_transform_score.keys())
    vals = np.array([per_transform_score[n] for n in names], dtype=float)
    if weight_by_severity and severity_weights is not None:
        weights = np.array([severity_weights.get(n, 1.0) for n in names], dtype=float)
        weights = weights / weights.sum()
        score = float(np.sum(vals * weights))
    else:
        score = float(np.mean(vals))
    return {
        "robustness_score": score,
        "mean_auc": float(np.mean(vals)),
        "min_auc": float(np.min(vals)),
        "max_auc": float(np.max(vals)),
        "per_transform": per_transform_score,
    }


def final_score(auc_clean, auc_robust, w_clean=0.5, w_robust=0.5):
    # Headline number: balanced mean of clean-data ROC-AUC and mean robust ROC-AUC.
    return float(w_clean * auc_clean + w_robust * auc_robust)
