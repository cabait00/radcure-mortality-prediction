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

from typing import Protocol, TypedDict, cast

import numpy as np
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


class _StratifiedKFoldAttributes(Protocol):
    """Structural description of the specific `StratifiedKFold` instance
    attributes this project inspects for reporting (`shuffle`,
    `random_state`).

    This exists only because scikit-learn's `StratifiedKFold.__init__`
    forwards to `_BaseKFold.__init__(self, n_splits, *, shuffle,
    random_state)`, whose parameters carry no type annotations (a
    scikit-learn source limitation, not a gap in this project's own
    typing) -- a static type checker therefore cannot infer that
    `self.shuffle`/`self.random_state` are `bool`/`int | None` from
    scikit-learn's source alone. `build_cv_splitter()` above already
    returns the precise, correct `StratifiedKFold` type; this Protocol
    only lets a type checker verify attribute access on the SAME object
    for the handful of places this project prints those two values.
    """

    shuffle: bool
    random_state: int | None


def describe_cv_splitter(cv: StratifiedKFold) -> _StratifiedKFoldAttributes:
    """Narrow, reporting-only view of a `StratifiedKFold` built by
    `build_cv_splitter()`, exposing `shuffle`/`random_state` with a
    precise static type. Does not alter `cv` or any of its runtime
    values -- see `_StratifiedKFoldAttributes` for why this cast exists.
    """
    return cast(_StratifiedKFoldAttributes, cv)


# =============================================================================
# Estimator-parameter inspection helpers
# =============================================================================
# Every scikit-learn estimator's `__init__` in this project's dependencies
# is unannotated (no type hints on its parameters), so a constructor-
# assigned attribute (`self.x = x`) has no static type a checker can rely
# on -- the same scikit-learn source limitation as `StratifiedKFold` above,
# just repeated across every estimator family this project inspects
# (Dummy, Logistic Regression, Decision Tree, Random Forest, RBF SVC). Each
# helper below reads the estimator's OWN parameters via `get_params
# (deep=False)` -- scikit-learn's own inspection API, which never mutates,
# refits, or otherwise alters the estimator -- and casts the result to a
# precise `TypedDict`. One small helper per estimator family (not one
# generic `describe_params(estimator, params_type)` taking a `TypedDict`
# class as a runtime argument, which would be needlessly clever for what
# is, in every case, a single fixed field list).


class DummyClassifierConfig(TypedDict):
    strategy: str


class LogisticRegressionConfig(TypedDict):
    C: float
    class_weight: None
    max_iter: int
    random_state: int | None
    # In scikit-learn 1.9, the constructor's own default for `penalty` is
    # the sentinel string "deprecated" rather than the literal "l2".
    penalty: str
    l1_ratio: float


class DecisionTreeConfig(TypedDict):
    random_state: int | None


class RandomForestConfig(TypedDict):
    n_estimators: int
    max_depth: int | None
    max_features: str
    min_samples_leaf: int
    class_weight: None
    random_state: int | None
    n_jobs: int | None


class RBFSVCConfig(TypedDict):
    C: float
    gamma: float | str
    class_weight: str | None
    kernel: str
    # In scikit-learn 1.9, the constructor's own default for `probability`
    # is the sentinel string "deprecated" rather than the literal `False`
    # -- both are possible values, so the field is typed as the union
    # rather than a bare `bool`.
    probability: bool | str


def describe_dummy_classifier(estimator: DummyClassifier) -> DummyClassifierConfig:
    """Read `estimator`'s own constructor parameters (see module-level
    note above for why this cast is needed and safe)."""
    return cast(DummyClassifierConfig, estimator.get_params(deep=False))


def describe_logistic_regression(estimator: LogisticRegression) -> LogisticRegressionConfig:
    """Read `estimator`'s own constructor parameters (see module-level
    note above for why this cast is needed and safe)."""
    return cast(LogisticRegressionConfig, estimator.get_params(deep=False))


def describe_decision_tree(estimator: DecisionTreeClassifier) -> DecisionTreeConfig:
    """Read `estimator`'s own constructor parameters (see module-level
    note above for why this cast is needed and safe)."""
    return cast(DecisionTreeConfig, estimator.get_params(deep=False))


def describe_random_forest(estimator: RandomForestClassifier) -> RandomForestConfig:
    """Read `estimator`'s own constructor parameters (see module-level
    note above for why this cast is needed and safe)."""
    return cast(RandomForestConfig, estimator.get_params(deep=False))


def describe_rbf_svc(estimator: SVC) -> RBFSVCConfig:
    """Read `estimator`'s own constructor parameters (see module-level
    note above for why this cast is needed and safe)."""
    return cast(RBFSVCConfig, estimator.get_params(deep=False))


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
) -> dict[str, np.ndarray | float]:
    """Run the pre-specified CV metrics for one candidate pipeline via
    separate `cross_val_score` calls (one per metric), matching the
    course's own convention rather than a combined multi-metric framework.

    TRAINING DATA ONLY: this function does not know about, and must never
    be called with, the held-out test set.

    Returns, for each metric, the raw per-fold scores (an `ndarray`, under
    the `<metric>_scores` key) plus their mean and standard deviation
    across folds (`float`, under `<metric>_mean`/`<metric>_std`) -- the
    two value types that genuinely coexist in this flat dict; there is no
    third kind of value here, so `dict[str, np.ndarray | float]` describes
    it exactly (not `dict[str, object]`).
    """
    result: dict[str, np.ndarray | float] = {}
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
        result[f"{metric_name}_scores"] = scores
        result[f"{metric_name}_mean"] = float(scores.mean())
        result[f"{metric_name}_std"] = float(scores.std())
    return result
