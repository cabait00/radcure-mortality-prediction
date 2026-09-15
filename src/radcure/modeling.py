"""
Reusable Milestone-2 modelling utilities: the fixed cross-validation
splitter, the fixed baseline candidate-model registry, and small helpers to
wrap each candidate in a fresh full Pipeline and evaluate it under the
pre-specified CV metrics.

This module intentionally contains no scientific narrative -- that lives in
``analysis/02_baseline_modeling.py``. Nothing here fits anything at import
time: every function only CONSTRUCTS objects (a splitter, a pipeline) or
runs cross-validation when explicitly called by the caller on data the
caller supplies. No function in this module ever touches a held-out test
set; that discipline is the analysis script's responsibility, not this
module's.

Milestone-2 scope (confirmed, not to be silently extended):
    - five DEFAULT-hyperparameter candidates, no tuning
    - no ``class_weight``, no resampling, no probability calibration
    - three pre-specified CV metrics: roc_auc (primary), average_precision
      (secondary), balanced_accuracy (supplementary)
    - plain ``cross_val_score`` per metric, matching the course's own
      convention, rather than a more elaborate evaluation framework
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config, preprocessing

# =============================================================================
# Cross-validation design (Milestone 2, Section 3 -- pre-specified)
# =============================================================================
# The SAME fixed splitter is used for every model and every metric.
CV_SCORING: dict[str, str] = {
    "roc_auc": "roc_auc",  # PRIMARY
    "average_precision": "average_precision",  # SECONDARY
    "balanced_accuracy": "balanced_accuracy",  # SUPPLEMENTARY
}
# Ordinary accuracy is deliberately NOT included: it is not used for model
# selection on an ~18.9%-prevalence outcome.


def build_cv_splitter() -> StratifiedKFold:
    """The single fixed cross-validation splitter for Milestone 2."""
    return StratifiedKFold(
        n_splits=config.N_CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )


# =============================================================================
# Fixed baseline candidate-model registry (Milestone 2, Section 2C)
# =============================================================================
def build_model_registry() -> dict[str, object]:
    """The fixed Milestone-2 baseline candidate-model registry.

    Exactly five default-hyperparameter estimators -- a default-family
    comparison, not tuning. No ``class_weight``, no resampling, no
    probability calibration, no ``SVC(probability=True)``: ROC-AUC and
    Average Precision are scored directly off the RBF SVC's
    ``decision_function``, which scikit-learn's ``"roc_auc"`` and
    ``"average_precision"`` scorers use automatically when
    ``predict_proba`` is unavailable.

    Returns bare, unfitted ESTIMATORS (not yet wrapped in a Pipeline);
    use `build_pipeline` to wrap each one with its own fresh preprocessor.
    """
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=config.RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "RBF SVC": SVC(kernel="rbf"),
    }


def build_pipeline(estimator: object) -> Pipeline:
    """Wrap one candidate estimator in a full Pipeline with a FRESH,
    unfitted preprocessor.

    Every model receives its own `preprocessing.build_preprocessor()`
    instance -- fitted preprocessing state is never shared across models,
    and nothing is fitted by this function itself.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """The full registry, with each candidate already wrapped in its own
    fresh Pipeline (fresh preprocessor + fresh estimator instance)."""
    return {name: build_pipeline(estimator) for name, estimator in build_model_registry().items()}


# =============================================================================
# CV evaluation helper
# =============================================================================
def evaluate_candidate(
    pipeline: Pipeline,
    X_train,
    y_train,
    cv: StratifiedKFold,
    scoring: dict[str, str] = CV_SCORING,
) -> dict[str, object]:
    """Run the pre-specified CV metrics for one candidate pipeline via
    separate `cross_val_score` calls (one per metric), matching the
    course's own convention rather than a combined multi-metric framework.

    TRAINING DATA ONLY: this function does not know about, and must never
    be called with, the held-out test set.

    Returns, for each metric, the raw per-fold scores plus their mean and
    standard deviation across folds.
    """
    result: dict[str, object] = {}
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
        result[f"{metric_name}_scores"] = scores
        result[f"{metric_name}_mean"] = float(scores.mean())
        result[f"{metric_name}_std"] = float(scores.std())
    return result
