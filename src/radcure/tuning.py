"""
Reusable Milestone-3A/3B hyperparameter-search utilities: the fixed search
spaces for the three Milestone-2 shortlisted model families (Logistic
Regression, Random Forest, RBF SVC), and helpers to construct each as a
fresh, unfitted `GridSearchCV` wrapping a fresh preprocessor.

This module intentionally contains no scientific narrative -- that lives in
`analysis/03_hyperparameter_tuning.py`. Nothing here is fit at import time:
every function only CONSTRUCTS a search object or reads results from a
search object supplied by the caller. No function in this module ever
touches a held-out test set.

Reuses (does not duplicate) `modeling.CV_SCORING` and
`modeling.build_cv_splitter()` -- the fixed CV design is identical to
Milestone 2's, so it is imported, not re-implemented. `src/radcure/modeling.py`
itself is not modified anywhere in Milestone 3.

Milestone-3A/3B scope (confirmed, not to be silently extended):
    - exactly three model families, each searched over a FIXED grid
    - hyperparameter selection uses refit="roc_auc" (PRIMARY metric);
      Average Precision and Balanced Accuracy are read off the SAME
      selected row, never independently re-optimised
    - no threshold tuning, no probability calibration, no stacking
    - SVC(probability=True) is never set
"""

from __future__ import annotations

from typing import Protocol, TypedDict, cast

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config, preprocessing
from .modeling import CV_SCORING, build_cv_splitter

REFIT_METRIC = "roc_auc"

# =============================================================================
# Fixed search spaces (Milestone 3, confirmed) -- sizes are asserted in the
# analysis script and in tests, never silently changed.
# =============================================================================
LOGISTIC_REGRESSION_PARAM_GRID: dict[str, list] = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__class_weight": [None, "balanced"],
}  # 5 x 2 = 10

RANDOM_FOREST_PARAM_GRID: dict[str, list] = {
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 5, 10],
    "classifier__max_features": ["sqrt", 0.5, 1.0],
    "classifier__class_weight": [None, "balanced"],
}  # 4 x 4 x 3 x 2 = 96

RBF_SVC_PARAM_GRID: dict[str, list] = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, "scale"],
    "classifier__class_weight": [None, "balanced"],
}  # 5 x 6 x 2 = 60

# Second-stage (and FINAL) RBF SVC grid. The first-stage search selected
# C=100.0, which was the UPPER edge of its grid, so `C` is extended upward
# here and `gamma` is refined locally around the selected 0.001. This is
# the ONLY refinement that will be run: if C=3000.0 is selected it is
# flagged and the search STOPS regardless -- there is deliberately no third
# stage, to keep the number of times the training folds are consulted for
# model selection bounded and pre-declared.
RBF_SVC_REFINEMENT_PARAM_GRID: dict[str, list] = {
    "classifier__C": [100.0, 300.0, 1000.0, 3000.0],
    "classifier__gamma": [0.0003, 0.001, 0.003],
    "classifier__class_weight": [None, "balanced"],
}  # 4 x 3 x 2 = 24


def grid_size(param_grid: dict[str, list]) -> int:
    """Number of parameter combinations a grid enumerates (product of the
    per-parameter list lengths) -- used to assert the confirmed grid sizes
    without hand-counting them twice.
    """
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


# =============================================================================
# Search construction (nothing fit here)
# =============================================================================
def build_logistic_regression_search(n_jobs: int = -1) -> GridSearchCV:
    """Logistic Regression search: L2 regularisation and the scikit-learn
    default solver are kept unchanged from Milestone 2 -- only `C` and
    `class_weight` are searched.

    Parallelism: LogisticRegression's default solver is single-threaded
    per fit, so parallelism is taken at the GridSearchCV level.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=LOGISTIC_REGRESSION_PARAM_GRID,
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


def build_random_forest_search(n_jobs: int = 1) -> GridSearchCV:
    """Random Forest search: `n_estimators`, `random_state` and the
    estimator's own `n_jobs=-1` are fixed; `max_depth`, `min_samples_leaf`,
    `max_features` and `class_weight` are searched.

    Parallelism (documented choice, Section 7 of the Milestone-3 design):
    the RandomForestClassifier itself is constructed with `n_jobs=-1` and
    already parallelises tree-building across every core on EACH fit. If
    GridSearchCV ALSO used `n_jobs=-1`, up to `n_candidates x n_folds`
    independent fully-parallel forest fits would contend for the same
    cores simultaneously (oversubscription), which typically slows the
    search down rather than speeding it up. Parallelism is therefore taken
    at the ESTIMATOR level only for this search; GridSearchCV itself
    defaults to `n_jobs=1` (sequential over the 96 x 5 = 480 fits).
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=RANDOM_FOREST_PARAM_GRID,
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


def build_rbf_svc_search(n_jobs: int = -1) -> GridSearchCV:
    """RBF SVC search: `C`, `gamma` and `class_weight` are searched.
    `probability` is never set (stays at its scikit-learn default), so
    ROC-AUC and Average Precision are scored via `decision_function`, not
    `predict_proba` -- matching Milestone 2's baseline configuration.

    Parallelism: SVC is single-threaded per fit, so parallelism is taken
    at the GridSearchCV level.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("classifier", SVC(kernel="rbf")),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=RBF_SVC_PARAM_GRID,
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


def build_rbf_svc_refinement_search(n_jobs: int = -1) -> GridSearchCV:
    """Second-stage (and final) RBF SVC search over
    `RBF_SVC_REFINEMENT_PARAM_GRID`.

    Identical in every other respect to `build_rbf_svc_search()` -- same
    fresh preprocessor, same fixed CV splitter, same multi-metric scoring,
    same `refit="roc_auc"`, and `probability` again never set. Only the
    searched grid differs, so the refined result stays directly comparable
    to the first-stage result and to the other tuned models.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            ("classifier", SVC(kernel="rbf")),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=RBF_SVC_REFINEMENT_PARAM_GRID,
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


def get_search_pipeline(search: GridSearchCV) -> Pipeline:
    """Narrow, reporting-only accessor for `search.estimator` as the
    `Pipeline` every `build_*_search()` function above always constructs.

    `GridSearchCV.__init__`'s `estimator` parameter has no default and no
    type annotation, so a static type checker cannot rely on
    `search.estimator`'s type -- a scikit-learn source limitation, not a
    gap in this project's own typing. Reading it back through sklearn's
    own `get_params(deep=False)` inspection API, then narrowing with
    `isinstance`, avoids relying on that attribute's static type. Does
    not alter `search` or its `estimator`.
    """
    estimator = search.get_params(deep=False)["estimator"]
    assert isinstance(estimator, Pipeline)
    return estimator


def get_search_best_pipeline(search: GridSearchCV) -> Pipeline:
    """Narrow, reporting-only accessor for `search.best_estimator_` as the
    `Pipeline` it always is after `.fit(...)`. `best_estimator_` is
    assigned dynamically inside `.fit()` (not a constructor parameter, so
    `get_params()` does not carry it) -- `getattr` avoids relying on its
    static type, and `isinstance` confirms it is the `Pipeline` every
    `build_*_search()` function above always constructs. Must only be
    called after `search.fit(...)`. Does not alter `search` or its
    `best_estimator_`.
    """
    best_estimator = getattr(search, "best_estimator_", None)
    assert isinstance(best_estimator, Pipeline)
    return best_estimator


# =============================================================================
# Exploratory XGBoost challenger (Milestone 3C, Phase E)
# =============================================================================
# Scientific status, stated once and not softened elsewhere: XGBoost is a
# POST-HOC EXPLORATORY CHALLENGER, added AFTER the prespecified
# Logistic Regression / Random Forest / RBF SVC comparison had been run and
# inspected. It was not part of the original Milestone-2 shortlist. It is
# reported under a separate "EXPLORATORY" label and must not be presented
# as though it had been prespecified.
#
# The import is guarded so that the rest of the module -- and every other
# milestone -- keeps working in an environment without xgboost installed.
try:  # pragma: no cover - availability depends on the environment
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(exc)


class _XGBClassifierFactory(Protocol):
    """Structural description of the ONE `XGBClassifier(...)` call shape
    used anywhere in this project (the fixed keyword set both search
    builders below pass; `n_estimators` is optional because the
    refinement search deliberately omits it -- see
    `build_xgboost_refinement_search`).

    Needed because the guarded import above makes the `XGBClassifier`
    symbol's type a union that includes `None` (the `except` branch's
    assignment), so calling it directly is not staticaly verifiable even
    though, by construction, every call site is reached only after
    `XGBOOST_AVAILABLE` has already confirmed the real import succeeded.
    """

    def __call__(
        self,
        *,
        objective: str,
        eval_metric: str,
        n_estimators: int = ...,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        n_jobs: int,
        verbosity: int,
        tree_method: str,
    ) -> BaseEstimator: ...


def _get_xgboost_factory() -> _XGBClassifierFactory:
    """The real `XGBClassifier` class, cast to `_XGBClassifierFactory`.

    Callers must only invoke this after their own `XGBOOST_AVAILABLE`
    check has already confirmed the import succeeded (both search
    builders below do this before calling it) -- the assertion here is a
    second, explicit confirmation of that same runtime invariant, not a
    new one.
    """
    assert XGBClassifier is not None, "xgboost is not installed"
    return cast(_XGBClassifierFactory, XGBClassifier)


def build_xgboost_param_grid(scale_pos_weight: float) -> dict[str, list]:
    """Deliberately small exploratory grid: 3 x 2 x 2 = 12 combinations.

    `scale_pos_weight` is passed in (computed by the caller as
    negatives/positives on the TRAINING split only) rather than hardcoded,
    so the imbalance ratio always matches the actual confirmed training
    counts. 1.0 -- i.e. no reweighting -- is searched alongside it.

    No other XGBoost parameter is tuned: no early stopping, no Optuna, no
    randomised search.
    """
    return {
        "classifier__max_depth": [2, 4, 6],
        "classifier__learning_rate": [0.03, 0.1],
        "classifier__scale_pos_weight": [1.0, scale_pos_weight],
    }


def build_xgboost_search(scale_pos_weight: float, n_jobs: int = -1) -> GridSearchCV:
    """Exploratory XGBoost search using the SAME frozen preprocessor, the
    SAME fixed CV splitter, the SAME multi-metric scoring and the SAME
    `refit="roc_auc"` as every other search in this project, so the result
    is directly comparable to the prespecified models.

    Parallelism: the estimator is pinned to `n_jobs=1` and parallelism is
    taken at the GridSearchCV level, to avoid the nested oversubscription
    documented for the Random Forest search above.
    """
    if not XGBOOST_AVAILABLE:  # pragma: no cover
        raise RuntimeError(f"xgboost is not installed: {XGBOOST_IMPORT_ERROR}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            (
                "classifier",
                _get_xgboost_factory()(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=build_xgboost_param_grid(scale_pos_weight),
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


# =============================================================================
# Targeted second-stage XGBoost refinement (Milestone 3C, Phase E continued)
# =============================================================================
# Scientific motivation: the first-stage search selected `learning_rate` and
# `max_depth` at the LOWER edge of their tested ranges, while `n_estimators`
# had been held fixed at 300. Because `learning_rate` and `n_estimators`
# trade off against each other in gradient boosting (more, smaller steps can
# reach a similar or better fit than fewer, larger ones), the first stage
# could not distinguish "this region is genuinely best" from "a smaller
# learning rate needs more rounds to be given a fair chance". This ONE
# targeted refinement adds `n_estimators` as a third searched axis and
# extends `learning_rate`/`max_depth` slightly below their first-stage
# minimums, without turning into a broad re-optimisation.
#
# This is the FINAL XGBoost search: no third stage follows, regardless of
# where the selected values fall in this grid (see the boundary-flag
# handling in the analysis script).
def build_xgboost_refinement_param_grid(scale_pos_weight: float) -> dict[str, list]:
    """Targeted second-stage exploratory grid: 3 x 3 x 3 x 2 = 54
    combinations. Does NOT alter or overwrite `build_xgboost_param_grid`
    (the first-stage grid) -- both are preserved so the two stages remain
    separately reportable.

    Only `max_depth`, `learning_rate`, `n_estimators` and
    `scale_pos_weight` are searched -- no `min_child_weight`, `gamma`,
    `reg_alpha`, `reg_lambda`, additional `subsample`/`colsample_bytree`
    values, alternative boosters, early stopping, Optuna or randomised
    search.
    """
    return {
        "classifier__max_depth": [1, 2, 3],
        "classifier__learning_rate": [0.01, 0.03, 0.05],
        "classifier__n_estimators": [300, 600, 1000],
        "classifier__scale_pos_weight": [1.0, scale_pos_weight],
    }


def build_xgboost_refinement_search(scale_pos_weight: float, n_jobs: int = -1) -> GridSearchCV:
    """Second-stage (and FINAL) XGBoost search over
    `build_xgboost_refinement_param_grid`.

    `n_estimators` is searched here rather than fixed on the estimator, so
    it is deliberately NOT passed to the `XGBClassifier` constructor
    (GridSearchCV overrides it per candidate via `classifier__n_estimators`
    regardless, but omitting it here avoids stating a value that is not
    actually used). Every other fixed setting is identical to the
    first-stage search: same objective/eval_metric, same
    subsample/colsample_bytree, same `random_state`, same fresh
    preprocessor, same fixed CV splitter, same multi-metric scoring, same
    `refit="roc_auc"`.

    Parallelism: the estimator remains pinned to `n_jobs=1`; parallelism is
    taken at the GridSearchCV level, exactly as in the first-stage search.
    """
    if not XGBOOST_AVAILABLE:  # pragma: no cover
        raise RuntimeError(f"xgboost is not installed: {XGBOOST_IMPORT_ERROR}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.build_preprocessor()),
            (
                "classifier",
                _get_xgboost_factory()(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=build_xgboost_refinement_param_grid(scale_pos_weight),
        cv=build_cv_splitter(),
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        return_train_score=False,
        n_jobs=n_jobs,
    )


# =============================================================================
# Result extraction
# =============================================================================
class CVMetricSummary(TypedDict):
    """Mean/std for each of the three fixed `CV_SCORING` metrics -- the
    shape shared by every training-CV result table in this project
    (frozen defaults, tuned, exploratory)."""

    roc_auc_mean: float
    roc_auc_std: float
    average_precision_mean: float
    average_precision_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float


class SearchSummary(CVMetricSummary):
    """`CVMetricSummary` plus the selected hyperparameters and candidate
    count -- the exact shape `summarize_best()` returns."""

    best_params: dict[str, object]
    n_candidates: int


def as_float_view(summary: CVMetricSummary) -> dict[str, float]:
    """Narrow view of a `CVMetricSummary`/`SearchSummary` for DYNAMIC-key
    lookups such as `f"{metric}_mean"` (`metric` a loop variable, not a
    string literal). A `TypedDict` only supports statically-checked
    LITERAL-key access, so this plain `dict[str, float]` view is used
    instead for the handful of places this project looks a metric value
    up by a computed key. Every key ever built this way
    (`f"{metric}_{stat}"` for `metric` in `CV_SCORING` and `stat` in
    {"mean", "std"}) is one of `CVMetricSummary`'s six float fields, so
    this cast changes nothing about the underlying values -- it only
    gives a type checker a shape it can index dynamically.
    """
    return cast(dict[str, float], summary)


def summarize_best(search: GridSearchCV) -> SearchSummary:
    """Extract the selected best parameters plus the mean/std of ALL three
    CV_SCORING metrics from the SAME row (`search.best_index_`, chosen by
    `refit="roc_auc"`). Average Precision and Balanced Accuracy are never
    independently re-selected -- they are read off exactly the parameter
    combination ROC-AUC picked.

    Must be called AFTER `search.fit(X_train, y_train)`.
    """
    idx = search.best_index_
    results = search.cv_results_
    summary: dict[str, object] = {
        "best_params": search.best_params_,
        "n_candidates": len(results["params"]),
    }
    for metric in CV_SCORING:
        summary[f"{metric}_mean"] = float(results[f"mean_test_{metric}"][idx])
        summary[f"{metric}_std"] = float(results[f"std_test_{metric}"][idx])
    # Built via dynamic keys above (a TypedDict only supports statically
    # checked literal-key construction), so the completed dict is cast to
    # the precise SearchSummary shape at this single return point -- the
    # loop above guarantees every SearchSummary key is present.
    return cast(SearchSummary, summary)


def flag_boundary_hits(
    best_params: dict[str, object],
    param_grid: dict[str, list],
    keys: tuple[str, ...],
) -> list[str]:
    """For each of `keys` (e.g. "classifier__C", "classifier__gamma")
    present in both `best_params` and `param_grid`, check whether the
    selected value equals the numeric minimum or maximum of its grid.

    Non-numeric grid entries (e.g. `"scale"` for `gamma`) are ignored when
    computing the numeric min/max, and a non-numeric selected value is
    never flagged as a boundary hit (there is no "edge" to a category).

    Returns a list of human-readable flags (empty if nothing hit an edge).
    Does NOT launch, suggest sizes for, or otherwise construct a follow-up
    search -- flagging only.
    """
    flags: list[str] = []
    for key in keys:
        if key not in best_params or key not in param_grid:
            continue
        selected = best_params[key]
        numeric_values = [v for v in param_grid[key] if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numeric_values or not isinstance(selected, (int, float)) or isinstance(selected, bool):
            continue
        lo, hi = min(numeric_values), max(numeric_values)
        if selected == lo:
            flags.append(f"{key}={selected!r} is at the LOWER edge of its searched grid ({numeric_values}).")
        elif selected == hi:
            flags.append(f"{key}={selected!r} is at the UPPER edge of its searched grid ({numeric_values}).")
    return flags
