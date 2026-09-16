"""
Reusable analysis/05 logic: training-only cross-validation probability
generation, confusion-matrix-derived classification metrics, and the
deterministic Balanced-Accuracy threshold search with its pre-declared
tie-break rule.

This module intentionally contains no scientific narrative -- that lives in
`analysis/05_threshold_selection_and_model_freeze.py`. Nothing here is fit
at import time: every function only computes a quantity the caller supplies
data for. No function in this module ever receives, or needs, a held-out
test-set argument -- threshold selection here is a training-only procedure
by construction, not by discipline alone.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def compute_training_cv_probabilities(pipeline, X, y, cv: StratifiedKFold) -> np.ndarray:
    """Training-only cross-validation positive-class probabilities.

    Every training patient receives exactly ONE probability, produced by a
    copy of `pipeline` fitted on the four folds that did NOT include that
    patient (this is technically an out-of-fold prediction, but is referred
    to throughout this project as a "training-only cross-validation
    prediction" -- it is a routine input to threshold selection here, not a
    separate methodological centerpiece).

    `cv` is a required argument, not constructed internally, so the caller
    always supplies the SAME fixed splitter used everywhere else in this
    project (see `modeling.build_cv_splitter`).
    """
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    return np.asarray(proba)[:, 1]


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Hard 0/1 predictions at `score >= threshold` (the same `>=` decision
    rule `classification_metrics` and the threshold search below assume
    throughout)."""
    return (np.asarray(scores) >= threshold).astype(int)


class ClassificationMetrics(TypedDict):
    """The full set of confusion-matrix-derived metrics `classification_metrics`
    returns for one hard decision. `tp`/`tn`/`fp`/`fn` are counts (`int`);
    every ratio is a `float`."""

    tp: int
    tn: int
    fp: int
    fn: int
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    predicted_positive_rate: float


class ThresholdSelectionResult(TypedDict):
    """The shape `select_balanced_accuracy_threshold` returns: the
    selected threshold and its Balanced Accuracy, how many candidates
    tied at that maximum, and the full metric set at that threshold."""

    threshold: float
    balanced_accuracy: float
    n_tied_candidates: int
    metrics: ClassificationMetrics


def classification_metrics(y_true, y_pred) -> ClassificationMetrics:
    """Standard confusion-matrix-derived metrics for one hard decision.

    Computed from plain confusion counts (no `sklearn.metrics` calls) so
    that a ~2000-candidate threshold search never raises
    `UndefinedMetricWarning` at a degenerate all-one-class threshold, and
    so a small toy example is trivial to hand-verify.

    Recall (positive class) and Sensitivity are the SAME statistic
    (TP / (TP+FN)) -- both are returned under their own key because both
    names are in common use, not because they are independent
    measurements; `recall == sensitivity` always holds exactly.

    Every ratio is guarded against division by zero: a metric whose
    denominator is 0 (e.g. Specificity when there are no actual negatives,
    or Precision when nothing is predicted positive) is defined as 0.0
    rather than raising or returning NaN.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity  # identical definition for the positive class
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    predicted_positive_rate = float(np.mean(y_pred == 1)) if len(y_pred) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "predicted_positive_rate": predicted_positive_rate,
    }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """The complete, exact set of distinct classification operating points
    reachable by the `score >= threshold` decision rule.

    `np.unique(scores)` alone reaches every operating point from
    "everyone positive" (threshold = min score) up to "only the single
    highest-scoring patient positive" (threshold = max score) -- but NOT
    the further, equally valid "everyone negative" operating point, which
    requires a threshold strictly greater than the maximum score. One
    deterministic sentinel just above the maximum score
    (`np.nextafter(max, inf)`, the smallest representable float strictly
    greater than the maximum) is appended so that extreme is reachable
    too. This is an exact enumeration of distinct operating points, not an
    arbitrary coarse grid.
    """
    unique_scores = np.unique(np.asarray(scores))
    sentinel = np.nextafter(unique_scores[-1], np.inf)
    return np.append(unique_scores, sentinel)


def select_balanced_accuracy_threshold(y_true, scores) -> ThresholdSelectionResult:
    """Deterministic threshold search maximizing Balanced Accuracy.

    Searches the exact candidate set from `_candidate_thresholds` (every
    distinct achievable operating point, including both the all-positive
    and all-negative extremes) -- an exact search, not a coarse grid.

    Pre-declared tie-break rule, applied deterministically (never a
    post-hoc manual choice):
      1. If multiple thresholds tie on the maximum Balanced Accuracy,
         select the one closest to 0.5.
      2. If there is still an exact tie in distance to 0.5, select the
         LOWER threshold.

    Implementation note: candidates are visited in ascending order, and a
    running best is updated only on a STRICT improvement (higher Balanced
    Accuracy, or equal Balanced Accuracy with strictly smaller distance to
    0.5). Because of that, an exact tie in both Balanced Accuracy and
    distance to 0.5 is automatically resolved in favour of whichever
    candidate was reached first in ascending order -- i.e. the lower one --
    which is exactly tie-break rule 2, without a separate branch for it.

    Takes only `(y_true, scores)`: no held-out test-set argument exists in
    this function's signature, because none is needed -- threshold
    selection here is a training-only procedure by construction.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = _candidate_thresholds(scores)

    best_threshold: float | None = None
    best_ba = -1.0
    best_dist = np.inf
    n_tied_candidates = 0

    for threshold in candidates:
        pred = apply_threshold(scores, threshold)
        ba = classification_metrics(y_true, pred)["balanced_accuracy"]
        dist = abs(threshold - 0.5)

        if ba > best_ba:
            best_threshold, best_ba, best_dist = threshold, ba, dist
            n_tied_candidates = 1
        elif ba == best_ba:
            n_tied_candidates += 1
            if dist < best_dist:
                best_threshold, best_dist = threshold, dist
            # else: equal or larger distance -> keep the existing (earlier,
            # i.e. lower) best_threshold, implementing tie-break rule 2.

    # `_candidate_thresholds` always returns at least one candidate (it is
    # `np.unique(...)`, itself never empty for a non-empty `scores`, plus
    # one appended sentinel), and `best_ba` starts below every achievable
    # Balanced Accuracy (`-1.0`), so the loop's first iteration always
    # assigns `best_threshold` -- it is never `None` once the loop has
    # run. This is a runtime invariant a type checker cannot see through a
    # `for` loop alone, so it is asserted explicitly.
    assert best_threshold is not None, "candidates must be non-empty"
    final_pred = apply_threshold(scores, best_threshold)
    return {
        "threshold": float(best_threshold),
        "balanced_accuracy": float(best_ba),
        "n_tied_candidates": n_tied_candidates,
        "metrics": classification_metrics(y_true, final_pred),
    }
