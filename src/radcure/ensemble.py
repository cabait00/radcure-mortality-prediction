"""
Reusable analysis/04 logic: the three FROZEN tuned base pipelines, the
training-only out-of-fold (OOF) score machinery used for the
complementarity analysis, and the single controlled stacking
configuration.

Separated from `tuning.py` because the concerns are genuinely different:
`tuning.py` SEARCHES hyperparameters, whereas this module only
CONSTRUCTS models whose hyperparameters are already fixed, and computes
OOF quantities from them. Nothing here searches, tunes, or selects
anything.

Nothing in this module is fit at import time, and no function here ever
receives or touches a held-out test set -- every function is called with
training data only.

Score-scale warning (important for every consumer of this module):
    Logistic Regression and Random Forest produce PROBABILITIES in [0, 1].
    The RBF SVC produces `decision_function` margins on an unbounded,
    differently-scaled axis, because `probability=True` is deliberately
    never enabled anywhere in this project. Continuous scores from
    different models are therefore NOT on a common scale and must be
    compared with RANK-based statistics (Spearman) as the primary
    measure. Pearson correlation on these mixed scales is supplementary
    information only.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .modeling import build_cv_splitter, build_pipeline

# =============================================================================
# FROZEN tuned hyperparameters (analysis/03 + the Section-6b refinement)
# =============================================================================
# These are RESULTS of the completed searches, recorded here so the frozen
# models can be rebuilt deterministically without re-running any search.
# Phase B of analysis/04: these three families are now frozen -- this
# module never tunes them further.
#
# Each dict below is precisely typed as a TypedDict (rather than
# `dict[str, object]`) so that `**FROZEN_..._PARAMS` unpacking into its
# constructor is statically checkable (PEP 692 TypedDict-unpacking) -- a
# `Literal` is used wherever the frozen value is a single fixed choice
# (e.g. `max_features="sqrt"`, the SVC's `class_weight="balanced"`) rather
# than the open `str`/`float` the underlying sklearn parameter accepts in
# general.


class FrozenLogisticRegressionParams(TypedDict):
    C: float
    class_weight: None


class FrozenRandomForestParams(TypedDict):
    n_estimators: int
    max_depth: int
    max_features: Literal["sqrt"]
    min_samples_leaf: int
    class_weight: None


class FrozenRBFSVCParams(TypedDict):
    C: float
    gamma: float
    class_weight: Literal["balanced"]


FROZEN_LOGISTIC_REGRESSION_PARAMS: FrozenLogisticRegressionParams = {
    "C": 1.0,
    "class_weight": None,
}

FROZEN_RANDOM_FOREST_PARAMS: FrozenRandomForestParams = {
    "n_estimators": 500,
    "max_depth": 10,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": None,
}

# Selected by the SECOND-stage (final) SVC refinement; see
# `analysis/03_hyperparameter_tuning.py` Section 6b. The refinement
# re-selected exactly the first-stage configuration: extending C upward to
# 3000 found nothing better, so C=100 is an INTERIOR optimum with respect
# to the union of both searched grids (0.01 ... 3000) and the first-stage
# upper-boundary flag is resolved.
FROZEN_RBF_SVC_PARAMS: FrozenRBFSVCParams = {
    "C": 100.0,
    "gamma": 0.001,
    "class_weight": "balanced",
}

# The two continuous-score extraction methods this project ever uses,
# named once so `compute_oof_scores`/`hard_predictions`/`OOF_SCORE_METHODS`
# share a single precise type instead of a plain `str`.
OOFScoreMethod = Literal["predict_proba", "decision_function"]

# Which continuous OOF score each frozen model exposes. The SVC entry is
# `decision_function` specifically so that `probability=True` never has to
# be enabled -- see the module docstring's score-scale warning.
OOF_SCORE_METHODS: dict[str, OOFScoreMethod] = {
    "Logistic Regression": "predict_proba",
    "Random Forest": "predict_proba",
    "RBF SVC": "decision_function",
}

# The native hard-decision threshold that corresponds to each score type.
_NATIVE_THRESHOLDS: dict[OOFScoreMethod, float] = {
    "predict_proba": 0.5,
    "decision_function": 0.0,
}


# =============================================================================
# Frozen base pipelines (nothing fit here)
# =============================================================================
def build_frozen_logistic_regression() -> Pipeline:
    """Frozen tuned Logistic Regression in a full pipeline with a FRESH
    preprocessor. Note that the search selected exactly the analysis/02
    default cell (C=1.0, class_weight=None)."""
    return build_pipeline(
        LogisticRegression(
            max_iter=5000,
            random_state=config.RANDOM_STATE,
            **FROZEN_LOGISTIC_REGRESSION_PARAMS,
        )
    )


def build_frozen_random_forest() -> Pipeline:
    """Frozen tuned Random Forest in a full pipeline with a FRESH
    preprocessor."""
    return build_pipeline(
        RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            **FROZEN_RANDOM_FOREST_PARAMS,
        )
    )


def build_frozen_rbf_svc() -> Pipeline:
    """Frozen REFINED RBF SVC in a full pipeline with a FRESH
    preprocessor. `probability` is never set, so this estimator exposes
    `decision_function` and not `predict_proba`."""
    return build_pipeline(SVC(kernel="rbf", **FROZEN_RBF_SVC_PARAMS))


def build_frozen_base_pipelines() -> dict[str, Pipeline]:
    """The three frozen tuned base models, each with its OWN fresh,
    unfitted preprocessor. Insertion order is the order used in every
    report table."""
    return {
        "Logistic Regression": build_frozen_logistic_regression(),
        "Random Forest": build_frozen_random_forest(),
        "RBF SVC": build_frozen_rbf_svc(),
    }


# =============================================================================
# Training-only out-of-fold scores
# =============================================================================
def compute_oof_scores(pipeline, X, y, method: OOFScoreMethod) -> np.ndarray:
    """Out-of-fold continuous scores for class 1, computed on TRAINING
    data only under the fixed shared CV splitter.

    Each observation's score comes from a model fitted on the other four
    folds, so no observation contributes to the model that scores it.

    `method` is "predict_proba" (returns the class-1 column) or
    "decision_function" (returns the margin as-is). Scores from different
    methods are NOT on a comparable scale -- see the module docstring.
    """
    if method not in _NATIVE_THRESHOLDS:
        raise ValueError(f"Unsupported OOF score method: {method!r}")
    raw = cross_val_predict(pipeline, X, y, cv=build_cv_splitter(), method=method)
    if method == "predict_proba":
        return np.asarray(raw)[:, 1]
    return np.asarray(raw)


def hard_predictions(scores: np.ndarray, method: OOFScoreMethod) -> np.ndarray:
    """Apply the estimator's NATIVE decision rule to continuous OOF
    scores: probability >= 0.5, or decision_function >= 0.

    No threshold is tuned or selected anywhere -- these are the built-in
    defaults each estimator would use itself, and are used only to
    describe where the models disagree.
    """
    if method not in _NATIVE_THRESHOLDS:
        raise ValueError(f"Unsupported OOF score method: {method!r}")
    return (np.asarray(scores) >= _NATIVE_THRESHOLDS[method]).astype(int)


def error_overlap(y_true, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Descriptive 2x2 breakdown of where two models' hard OOF predictions
    are right and wrong, plus their false-negative / false-positive
    overlap. Counts and rates only -- no test statistic, no p-value.
    """
    y_true = np.asarray(y_true)
    wrong_a = pred_a != y_true
    wrong_b = pred_b != y_true
    n = len(y_true)

    both_wrong = int(np.sum(wrong_a & wrong_b))
    only_a_wrong = int(np.sum(wrong_a & ~wrong_b))
    only_b_wrong = int(np.sum(~wrong_a & wrong_b))
    both_correct = int(np.sum(~wrong_a & ~wrong_b))

    positives = y_true == 1
    negatives = y_true == 0
    fn_a, fn_b = wrong_a & positives, wrong_b & positives
    fp_a, fp_b = wrong_a & negatives, wrong_b & negatives

    return {
        "n": n,
        "disagreement_rate": float(np.mean(pred_a != pred_b)),
        "both_wrong": both_wrong,
        "both_wrong_rate": both_wrong / n,
        "only_a_wrong": only_a_wrong,
        "only_b_wrong": only_b_wrong,
        "both_correct": both_correct,
        "both_correct_rate": both_correct / n,
        "fn_both": int(np.sum(fn_a & fn_b)),
        "fn_only_a": int(np.sum(fn_a & ~fn_b)),
        "fn_only_b": int(np.sum(~fn_a & fn_b)),
        "fp_both": int(np.sum(fp_a & fp_b)),
        "fp_only_a": int(np.sum(fp_a & ~fp_b)),
        "fp_only_b": int(np.sum(~fp_a & fp_b)),
    }


# =============================================================================
# The single controlled stacking configuration
# =============================================================================
def build_stacking_classifier(n_jobs: int = 1) -> StackingClassifier:
    """The ONE stacking configuration permitted for analysis/04.

    Leakage control: `cv` is an explicit fixed stratified 5-fold splitter,
    so `StackingClassifier` builds the meta-features with
    `cross_val_predict` internally -- every meta-feature for an observation
    comes from a base learner that did NOT see that observation. Base
    learners are never fitted on the full training set and then asked for
    in-sample predictions to train the meta-learner.

    `stack_method="auto"` resolves to `predict_proba` for Logistic
    Regression and Random Forest and to `decision_function` for the SVC
    (which has no `predict_proba`, by design). The meta-learner therefore
    receives columns on DIFFERENT scales -- a probability-like score from
    LR and RF, and an unbounded `decision_function` margin from the SVC.
    Because the final estimator is an L2-regularised LogisticRegression,
    and L2 regularisation is scale-dependent, the final estimator is a
    small `Pipeline([("scaler", StandardScaler()), ("classifier",
    LogisticRegression(...))])` rather than a bare LogisticRegression, so
    the meta-features are standardised before the penalty is applied. This
    scaling step is fit only on the meta-features produced by the inner CV
    (never on the held-out test set, which this module never touches), and
    is part of the SAME single stacking configuration -- it is a
    correction to how the existing experiment's final estimator handles
    its inputs, not an additional stacking variant.

    Fixed, not searched: no alternative meta-learner, no `passthrough`
    variant, no tuning of the stack.
    """
    final_estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE),
            ),
        ]
    )
    return StackingClassifier(
        estimators=list(build_frozen_base_pipelines().items()),
        final_estimator=final_estimator,
        cv=build_cv_splitter(),
        stack_method="auto",
        passthrough=False,
        n_jobs=n_jobs,
    )
