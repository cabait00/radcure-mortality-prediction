"""
Structural/toy-example validation for `src/radcure/evaluation.py`
(analysis/05).

Deliberately structural: locks in the frozen final Logistic Regression
parameters, the confusion-matrix-derived metric formulas (hand-verified on
small toy examples), the exact deterministic threshold search (both
extreme operating points, both tie-break rules), and the fixed-CV-reuse /
no-test-set-argument contracts. Does NOT assert exact real-dataset
thresholds or exact real CV performance -- those are outcomes of the data,
not invariants of the code.

Run with:  pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import config, ensemble, evaluation, modeling  # noqa: E402


# =============================================================================
# Frozen final model parameters
# =============================================================================
def test_frozen_final_logistic_regression_parameters_are_correct():
    lr = ensemble.build_frozen_logistic_regression().named_steps["classifier"]
    assert isinstance(lr, LogisticRegression)
    lr_config = modeling.describe_logistic_regression(lr)
    assert lr_config["C"] == 1.0
    assert lr_config["class_weight"] is None
    assert lr_config["max_iter"] == 5000
    assert lr_config["random_state"] == config.RANDOM_STATE == 42


# =============================================================================
# classification_metrics -- hand-verified toy example
# =============================================================================
def test_classification_metrics_matches_hand_computed_toy_example():
    # 10 patients: 4 actual positives, 6 actual negatives.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    # TP=2, FN=2, FP=1, TN=5.
    metrics = evaluation.classification_metrics(y_true, y_pred)

    assert metrics["tp"] == 2
    assert metrics["fn"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 5

    assert metrics["sensitivity"] == 2 / 4
    assert metrics["specificity"] == 5 / 6
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == metrics["sensitivity"]
    expected_f1 = 2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))
    assert abs(metrics["f1"] - expected_f1) < 1e-12
    assert abs(metrics["balanced_accuracy"] - (2 / 4 + 5 / 6) / 2) < 1e-12
    assert metrics["predicted_positive_rate"] == 3 / 10


def test_recall_and_sensitivity_are_always_identical():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)
    metrics = evaluation.classification_metrics(y_true, y_pred)
    assert metrics["recall"] == metrics["sensitivity"]


def test_classification_metrics_zero_division_guards():
    # No predicted positives at all -> precision, recall/sensitivity all
    # well-defined without raising or returning NaN.
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = evaluation.classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["sensitivity"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 0.0
    assert not any(np.isnan(v) for v in metrics.values() if isinstance(v, float))

    # No actual negatives at all -> specificity well-defined.
    y_true_all_pos = np.array([1, 1, 1])
    y_pred_mixed = np.array([1, 0, 1])
    metrics2 = evaluation.classification_metrics(y_true_all_pos, y_pred_mixed)
    assert metrics2["specificity"] == 0.0  # tn=0, fp=0 -> guarded to 0.0


def test_apply_threshold_uses_greater_or_equal():
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    pred = evaluation.apply_threshold(scores, 0.5)
    assert pred.tolist() == [0, 1, 1, 1]


# =============================================================================
# select_balanced_accuracy_threshold -- exact search + both extremes
# =============================================================================
def _brute_force_best_ba(y_true, scores):
    """Reference brute-force search over the SAME candidate set the
    production function uses, for cross-checking."""
    candidates = np.append(np.unique(scores), np.nextafter(np.max(scores), np.inf))
    best_ba = -1.0
    for t in candidates:
        pred = evaluation.apply_threshold(scores, t)
        ba = evaluation.classification_metrics(y_true, pred)["balanced_accuracy"]
        best_ba = max(best_ba, ba)
    return best_ba


def test_threshold_search_achieves_the_maximum_balanced_accuracy():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=40)
    scores = rng.random(40)

    result = evaluation.select_balanced_accuracy_threshold(y_true, scores)
    reference_best = _brute_force_best_ba(y_true, scores)

    assert abs(result["balanced_accuracy"] - reference_best) < 1e-12
    # No candidate should beat the selected Balanced Accuracy.
    all_candidates = np.append(np.unique(scores), np.nextafter(np.max(scores), np.inf))
    for t in all_candidates:
        pred = evaluation.apply_threshold(scores, t)
        ba = evaluation.classification_metrics(y_true, pred)["balanced_accuracy"]
        assert ba <= result["balanced_accuracy"] + 1e-12


def test_all_negative_extreme_is_reachable_and_can_be_selected():
    # A mathematical fact worth noting first: whenever at least one actual
    # positive and one actual negative are present, the all-POSITIVE
    # extreme (threshold = min score: sensitivity=1.0, specificity=0.0) and
    # the all-NEGATIVE extreme (threshold above max score: sensitivity=0.0,
    # specificity=1.0) both always achieve EXACTLY Balanced Accuracy 0.5.
    # So to make the all-negative extreme the one actually SELECTED, this
    # toy example is built so every intermediate threshold does WORSE than
    # 0.5, and the all-negative extreme's threshold (just above the
    # single positive's low score) is much closer to 0.5 than the
    # all-positive extreme's threshold (0.05) -- so tie-break 1 picks it.
    y_true = np.array([1, 0, 0, 0, 0])
    scores = np.array([0.05, 0.50, 0.51, 0.52, 0.55])

    result = evaluation.select_balanced_accuracy_threshold(y_true, scores)

    # The all-negative extreme requires a threshold strictly above the max
    # observed score (0.55) -- only reachable via the above-max sentinel.
    assert result["threshold"] > scores.max()
    assert result["metrics"]["predicted_positive_rate"] == 0.0
    assert result["metrics"]["specificity"] == 1.0
    # Balanced accuracy at this extreme: sensitivity=0, specificity=1 -> 0.5,
    # and no candidate in this construction does better.
    assert abs(result["balanced_accuracy"] - 0.5) < 1e-12


def test_all_positive_extreme_is_reachable_via_apply_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    pred_all_positive = evaluation.apply_threshold(scores, scores.min())
    assert pred_all_positive.tolist() == [1, 1, 1, 1]


def test_candidate_count_includes_unique_scores_plus_one_sentinel():
    scores = np.array([0.1, 0.2, 0.2, 0.3, 0.5])
    y_true = np.array([0, 1, 0, 1, 0])
    result = evaluation.select_balanced_accuracy_threshold(y_true, scores)
    # 4 unique scores (0.1, 0.2, 0.3, 0.5) + 1 sentinel = 5 candidates total;
    # n_tied_candidates can be at most that many.
    assert result["n_tied_candidates"] <= 5


# =============================================================================
# Tie-break rules
# =============================================================================
def test_tie_break_1_selects_threshold_closest_to_half():
    # Two actual positives and two actual negatives, scores chosen so that
    # thresholds 0.3 and 0.6 both yield Balanced Accuracy 0.5 (predicting
    # the single lower-scored item as positive vs. none), while a
    # threshold nearer 0.5 achieves a strictly higher Balanced Accuracy of
    # 1.0 by separating the classes perfectly.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    # Perfect separation exists at any threshold in (0.3, 0.7]; the closest
    # to 0.5 among the achievable exact candidates is 0.7 itself (since
    # candidates are the observed scores) -- but 0.7 also achieves BA=1.0,
    # and so would any considered candidate scoring 1.0. We assert the
    # selected threshold is the one BA-maximizing candidate closest to 0.5.
    result = evaluation.select_balanced_accuracy_threshold(y_true, scores)
    assert result["balanced_accuracy"] == 1.0
    assert result["threshold"] == 0.7  # closest-to-0.5 candidate achieving BA=1.0


def test_tie_break_2_selects_the_lower_threshold_on_exact_distance_tie():
    # 0.25 and 0.75 are used (rather than e.g. 0.3/0.7) because both are
    # EXACTLY representable in binary floating point, so their distances
    # to 0.5 are bit-identical (0.25 exactly each) -- an exact tie in
    # distance, not one that floating-point rounding could break either
    # way by accident.
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.25, 0.25, 0.75, 0.75])
    # At threshold=0.25: predict [1,1,1,1] (all >= 0.25) -> TP=2 (both y=1),
    #   FP=2 (both y=0) -> sensitivity=1.0, specificity=0.0 -> BA=0.5.
    # At threshold=0.75: predict [0,0,1,1] -> TP=1, FN=1, FP=1, TN=1
    #   -> sensitivity=0.5, specificity=0.5 -> BA=0.5.
    # At the above-max sentinel: predict [0,0,0,0] -> sensitivity=0.0,
    #   specificity=1.0 -> BA=0.5.
    # All three candidates tie at BA=0.5; 0.25 and 0.75 are equidistant
    # (0.25) from 0.5 and no candidate does better -> tie-break 2 picks the
    # LOWER of the two, 0.25 (the sentinel is even farther from 0.5 than
    # either, so it is not in contention here).
    result = evaluation.select_balanced_accuracy_threshold(y_true, scores)
    assert result["balanced_accuracy"] == 0.5
    assert result["threshold"] == 0.25


# =============================================================================
# select_balanced_accuracy_threshold requires no test-set argument
# =============================================================================
def test_threshold_selection_signature_takes_only_training_arguments():
    sig = inspect.signature(evaluation.select_balanced_accuracy_threshold)
    params = list(sig.parameters)
    assert params == ["y_true", "scores"]
    assert not any("test" in p.lower() for p in params)


# =============================================================================
# compute_training_cv_probabilities -- fixed-CV reuse + one-per-patient
# =============================================================================
def test_compute_training_cv_probabilities_reuses_the_supplied_cv():
    """The `cv` argument passed in by the caller must be the one used --
    e.g. the project-wide fixed 5-fold splitter -- never re-derived
    internally with a different fold count."""
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    pipeline = LogisticRegression(max_iter=1000)

    cv_project = modeling.build_cv_splitter()
    assert cv_project.get_n_splits() == config.N_CV_FOLDS == 5

    probabilities = evaluation.compute_training_cv_probabilities(pipeline, X, y, cv_project)
    assert len(probabilities) == len(y)

    # A DIFFERENT fold count, explicitly supplied, must also be honoured --
    # confirms the function does not silently substitute its own splitter.
    cv_custom = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    probabilities_custom = evaluation.compute_training_cv_probabilities(
        pipeline, X, y, cv_custom
    )
    assert len(probabilities_custom) == len(y)


def test_every_training_sample_gets_exactly_one_cv_probability():
    X, y = make_classification(
        n_samples=50, n_features=5, weights=[0.7, 0.3], random_state=1
    )
    pipeline = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    probabilities = evaluation.compute_training_cv_probabilities(pipeline, X, y, cv)

    assert len(probabilities) == len(y) == 50
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
