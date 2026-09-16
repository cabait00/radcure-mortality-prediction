"""
Lightweight validation for `src/radcure/tuning.py` (analysis/03).

Deliberately small and structural: locks in the fixed search-space sizes,
the selection rule (`refit="roc_auc"`), the shared fixed CV design, and the
fresh-preprocessor-per-search contract. Does NOT test exact CV scores --
those are outcomes of the data and the search, not invariants of the code.

Run with:  pytest tests/test_tuning.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import config, modeling, tuning  # noqa: E402


def test_logistic_regression_grid_size_is_10():
    assert tuning.grid_size(tuning.LOGISTIC_REGRESSION_PARAM_GRID) == 10


def test_random_forest_grid_size_is_96():
    assert tuning.grid_size(tuning.RANDOM_FOREST_PARAM_GRID) == 96


def test_rbf_svc_grid_size_is_60():
    assert tuning.grid_size(tuning.RBF_SVC_PARAM_GRID) == 60


def test_tuning_reuses_modeling_cv_scoring_and_splitter():
    """The fixed CV design must be identical to analysis/02's, not
    re-implemented -- tuning.py imports it rather than duplicating it."""
    assert tuning.CV_SCORING is modeling.CV_SCORING
    assert set(tuning.CV_SCORING) == {"roc_auc", "average_precision", "balanced_accuracy"}


def _all_searches():
    return {
        "Logistic Regression": tuning.build_logistic_regression_search(),
        "Random Forest": tuning.build_random_forest_search(),
        "RBF SVC": tuning.build_rbf_svc_search(),
    }


def test_every_search_uses_refit_roc_auc():
    for name, search in _all_searches().items():
        assert search.get_params(deep=False)["refit"] == "roc_auc", name


def test_every_search_uses_the_fixed_five_fold_stratified_cv():
    for name, search in _all_searches().items():
        cv = search.get_params(deep=False)["cv"]
        assert isinstance(cv, StratifiedKFold), name
        assert cv.get_n_splits() == 5 == config.N_CV_FOLDS, name
        cv_attrs = modeling.describe_cv_splitter(cv)
        assert cv_attrs.shuffle is True, name
        assert cv_attrs.random_state == 42 == config.RANDOM_STATE, name


def test_every_search_uses_the_shared_scoring_dict():
    for name, search in _all_searches().items():
        scoring = search.get_params(deep=False)["scoring"]
        assert scoring == tuning.CV_SCORING, name
        assert scoring is modeling.CV_SCORING, name


def test_every_search_wraps_a_pipeline_with_fresh_preprocessor():
    searches = _all_searches()
    preprocessors = [tuning.get_search_pipeline(s).named_steps["preprocessor"] for s in searches.values()]
    ids = {id(p) for p in preprocessors}
    assert len(ids) == len(preprocessors), "Two or more searches share the same preprocessor object."

    p1 = tuning.get_search_pipeline(tuning.build_logistic_regression_search()).named_steps["preprocessor"]
    p2 = tuning.get_search_pipeline(tuning.build_logistic_regression_search()).named_steps["preprocessor"]
    assert p1 is not p2


def test_estimators_match_the_confirmed_fixed_hyperparameters():
    searches = _all_searches()

    lr = tuning.get_search_pipeline(searches["Logistic Regression"]).named_steps["classifier"]
    assert isinstance(lr, LogisticRegression)
    lr_config = modeling.describe_logistic_regression(lr)
    assert lr_config["max_iter"] == 5000
    assert lr_config["random_state"] == config.RANDOM_STATE
    # In scikit-learn 1.9, LogisticRegression's own constructor default for
    # `penalty` is the sentinel string "deprecated" rather than the literal
    # "l2", so the functional contract (L2-equivalent regularisation, never
    # explicitly overridden to L1) is checked instead of the raw attribute.
    assert lr_config["penalty"] != "l1"
    assert lr_config["l1_ratio"] == 0.0  # 0.0 = pure L2 in the unified elastic-net framing

    rf = tuning.get_search_pipeline(searches["Random Forest"]).named_steps["classifier"]
    assert isinstance(rf, RandomForestClassifier)
    rf_config = modeling.describe_random_forest(rf)
    assert rf_config["n_estimators"] == 500
    assert rf_config["random_state"] == config.RANDOM_STATE
    assert rf_config["n_jobs"] == -1

    svc = tuning.get_search_pipeline(searches["RBF SVC"]).named_steps["classifier"]
    assert isinstance(svc, SVC)
    svc_config = modeling.describe_rbf_svc(svc)
    assert svc_config["kernel"] == "rbf"


def test_svc_search_never_requires_probability_true():
    svc = tuning.get_search_pipeline(tuning.build_rbf_svc_search()).named_steps["classifier"]
    assert isinstance(svc, SVC)
    svc_config = modeling.describe_rbf_svc(svc)
    assert svc_config["probability"] is not True
    assert not hasattr(svc, "predict_proba")


def test_random_forest_gridsearch_uses_sequential_n_jobs_to_avoid_oversubscription():
    """Documented parallelism choice: the RF estimator itself is n_jobs=-1,
    so GridSearchCV must not ALSO be -1 (see analysis/03 Section 2,
    design) -- this is a structural regression guard, not a performance test.
    """
    search = tuning.build_random_forest_search()
    assert search.get_params(deep=False)["n_jobs"] in (1, None)
    rf = tuning.get_search_pipeline(search).named_steps["classifier"]
    assert isinstance(rf, RandomForestClassifier)
    assert modeling.describe_random_forest(rf)["n_jobs"] == -1


def test_param_grids_use_classifier_prefixed_keys():
    for grid in (
        tuning.LOGISTIC_REGRESSION_PARAM_GRID,
        tuning.RANDOM_FOREST_PARAM_GRID,
        tuning.RBF_SVC_PARAM_GRID,
    ):
        assert all(key.startswith("classifier__") for key in grid)


def test_flag_boundary_hits_detects_edges_and_ignores_non_numeric():
    grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    assert tuning.flag_boundary_hits({"classifier__C": 0.01}, grid, ("classifier__C",))
    assert tuning.flag_boundary_hits({"classifier__C": 100.0}, grid, ("classifier__C",))
    assert not tuning.flag_boundary_hits({"classifier__C": 1.0}, grid, ("classifier__C",))

    gamma_grid = {"classifier__gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, "scale"]}
    assert not tuning.flag_boundary_hits({"classifier__gamma": "scale"}, gamma_grid, ("classifier__gamma",))
    assert tuning.flag_boundary_hits({"classifier__gamma": 0.0001}, gamma_grid, ("classifier__gamma",))


def test_grid_size_helper_matches_manual_count():
    grid = {"a": [1, 2, 3], "b": ["x", "y"]}
    assert tuning.grid_size(grid) == 6


def test_search_construction_does_not_fit_anything():
    """Constructing a search must not fit it -- fitting happens only when
    the caller explicitly calls .fit()."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    for search in _all_searches().values():
        try:
            check_is_fitted(search)
            raise AssertionError("GridSearchCV unexpectedly reports as fitted before .fit() was called.")
        except NotFittedError:
            pass
