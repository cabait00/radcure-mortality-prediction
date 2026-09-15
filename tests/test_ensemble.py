"""
Structural validation for `src/radcure/ensemble.py` and the Milestone-3C
additions to `src/radcure/tuning.py`.

Deliberately structural, never performance-based: these tests lock in the
frozen configuration, the fixed CV design, the fresh-preprocessor contract,
the no-probability contract, and the search-space sizes. They assert NO CV
scores -- those are outcomes of the data, not invariants of the code.

Run with:  pytest tests/test_ensemble.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import config, ensemble, tuning  # noqa: E402


# =============================================================================
# Phase A -- SVC refinement search
# =============================================================================
def test_svc_refinement_grid_size_is_24():
    assert tuning.grid_size(tuning.RBF_SVC_REFINEMENT_PARAM_GRID) == 24


def test_svc_refinement_grid_has_the_confirmed_values():
    grid = tuning.RBF_SVC_REFINEMENT_PARAM_GRID
    assert grid["classifier__C"] == [100.0, 300.0, 1000.0, 3000.0]
    assert grid["classifier__gamma"] == [0.0003, 0.001, 0.003]
    assert grid["classifier__class_weight"] == [None, "balanced"]


def test_svc_refinement_search_preserves_the_fixed_design():
    search = tuning.build_rbf_svc_refinement_search()
    assert search.refit == "roc_auc"
    assert search.scoring is tuning.CV_SCORING
    cv = search.cv
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == config.N_CV_FOLDS == 5
    assert cv.shuffle is True
    assert cv.random_state == config.RANDOM_STATE == 42


def test_svc_refinement_never_enables_probability():
    svc = tuning.build_rbf_svc_refinement_search().estimator.named_steps["classifier"]
    assert isinstance(svc, SVC)
    assert svc.kernel == "rbf"
    assert svc.probability is not True
    assert not hasattr(svc, "predict_proba")


def test_svc_refinement_uses_a_fresh_preprocessor_each_call():
    a = tuning.build_rbf_svc_refinement_search().estimator.named_steps["preprocessor"]
    b = tuning.build_rbf_svc_refinement_search().estimator.named_steps["preprocessor"]
    assert a is not b


# =============================================================================
# Phase B -- frozen base pipelines
# =============================================================================
def test_three_frozen_base_pipelines_in_the_expected_order():
    pipelines = ensemble.build_frozen_base_pipelines()
    assert list(pipelines) == ["Logistic Regression", "Random Forest", "RBF SVC"]


def test_frozen_pipelines_carry_the_recorded_tuned_hyperparameters():
    pipelines = ensemble.build_frozen_base_pipelines()

    lr = pipelines["Logistic Regression"].named_steps["classifier"]
    assert isinstance(lr, LogisticRegression)
    assert lr.C == ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS["C"]
    assert lr.class_weight == ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS["class_weight"]
    assert lr.random_state == config.RANDOM_STATE

    rf = pipelines["Random Forest"].named_steps["classifier"]
    assert isinstance(rf, RandomForestClassifier)
    for key, value in ensemble.FROZEN_RANDOM_FOREST_PARAMS.items():
        assert getattr(rf, key) == value, key
    assert rf.random_state == config.RANDOM_STATE

    svc = pipelines["RBF SVC"].named_steps["classifier"]
    assert isinstance(svc, SVC)
    assert svc.kernel == "rbf"
    for key, value in ensemble.FROZEN_RBF_SVC_PARAMS.items():
        assert getattr(svc, key) == value, key


def test_frozen_svc_never_enables_probability():
    svc = ensemble.build_frozen_rbf_svc().named_steps["classifier"]
    assert svc.probability is not True
    assert not hasattr(svc, "predict_proba")


def test_each_frozen_pipeline_gets_its_own_fresh_preprocessor():
    pipelines = ensemble.build_frozen_base_pipelines()
    ids = {id(p.named_steps["preprocessor"]) for p in pipelines.values()}
    assert len(ids) == 3

    again = ensemble.build_frozen_base_pipelines()
    for name in pipelines:
        assert (
            pipelines[name].named_steps["preprocessor"]
            is not again[name].named_steps["preprocessor"]
        )


def test_frozen_svc_params_match_the_refinement_grid():
    """The frozen SVC configuration must be a value the refinement search
    could actually have selected -- guards against a transcription slip
    between the search result and the recorded constant."""
    grid = tuning.RBF_SVC_REFINEMENT_PARAM_GRID
    assert ensemble.FROZEN_RBF_SVC_PARAMS["C"] in grid["classifier__C"]
    assert ensemble.FROZEN_RBF_SVC_PARAMS["gamma"] in grid["classifier__gamma"]
    assert ensemble.FROZEN_RBF_SVC_PARAMS["class_weight"] in grid["classifier__class_weight"]


# =============================================================================
# Phase C -- OOF helpers
# =============================================================================
def test_oof_score_methods_keep_svc_on_decision_function():
    assert ensemble.OOF_SCORE_METHODS["Logistic Regression"] == "predict_proba"
    assert ensemble.OOF_SCORE_METHODS["Random Forest"] == "predict_proba"
    assert ensemble.OOF_SCORE_METHODS["RBF SVC"] == "decision_function"


def test_hard_predictions_apply_the_native_thresholds():
    probs = np.array([0.0, 0.49, 0.5, 0.51, 1.0])
    assert ensemble.hard_predictions(probs, "predict_proba").tolist() == [0, 0, 1, 1, 1]

    margins = np.array([-2.0, -0.01, 0.0, 0.01, 2.0])
    assert ensemble.hard_predictions(margins, "decision_function").tolist() == [0, 0, 1, 1, 1]


def test_unsupported_oof_method_is_rejected():
    with pytest.raises(ValueError):
        ensemble.hard_predictions(np.array([0.1]), "predict")


def test_error_overlap_partitions_every_observation():
    y = np.array([0, 0, 1, 1, 1, 0])
    a = np.array([0, 1, 1, 0, 1, 0])
    b = np.array([0, 0, 0, 0, 1, 1])
    stats = ensemble.error_overlap(y, a, b)

    assert (
        stats["both_wrong"] + stats["only_a_wrong"]
        + stats["only_b_wrong"] + stats["both_correct"]
    ) == len(y)
    assert stats["n"] == len(y)
    assert 0.0 <= stats["disagreement_rate"] <= 1.0
    # False negatives and false positives are disjoint subsets of the errors.
    assert stats["fn_both"] + stats["fp_both"] <= stats["both_wrong"]


def test_error_overlap_is_symmetric_in_its_shared_counts():
    y = np.array([0, 1, 1, 0, 1])
    a = np.array([1, 1, 0, 0, 1])
    b = np.array([0, 1, 1, 1, 0])
    ab = ensemble.error_overlap(y, a, b)
    ba = ensemble.error_overlap(y, b, a)

    assert ab["both_wrong"] == ba["both_wrong"]
    assert ab["both_correct"] == ba["both_correct"]
    assert ab["disagreement_rate"] == ba["disagreement_rate"]
    assert ab["only_a_wrong"] == ba["only_b_wrong"]
    assert ab["only_b_wrong"] == ba["only_a_wrong"]


# =============================================================================
# Phase D -- the single stacking configuration
# =============================================================================
def test_stacking_has_exactly_three_base_estimators():
    stack = ensemble.build_stacking_classifier()
    assert isinstance(stack, StackingClassifier)
    assert len(stack.estimators) == 3
    assert [name for name, _ in stack.estimators] == [
        "Logistic Regression", "Random Forest", "RBF SVC",
    ]


def test_stacking_final_estimator_is_a_scaled_logistic_regression():
    """The final estimator standardises meta-features before the
    L2-regularised LogisticRegression, since stack_method="auto" mixes
    probability-like scores (LR, RF) with an unbounded decision_function
    score (SVC), and L2 regularisation is scale-dependent."""
    stack = ensemble.build_stacking_classifier()
    assert isinstance(stack.final_estimator, Pipeline)
    step_names = list(stack.final_estimator.named_steps)
    assert step_names == ["scaler", "classifier"]
    assert isinstance(stack.final_estimator.named_steps["scaler"], StandardScaler)
    final_lr = stack.final_estimator.named_steps["classifier"]
    assert isinstance(final_lr, LogisticRegression)
    assert final_lr.random_state == config.RANDOM_STATE
    assert final_lr.max_iter == 5000


def test_stacking_passthrough_is_false_and_stack_method_is_auto():
    stack = ensemble.build_stacking_classifier()
    assert stack.passthrough is False
    assert stack.stack_method == "auto"


def test_stacking_uses_an_explicit_fixed_inner_stratified_cv():
    """The inner CV is what keeps the meta-learner from seeing in-sample
    base predictions -- it must be an explicit fixed stratified splitter,
    never left at None."""
    stack = ensemble.build_stacking_classifier()
    assert isinstance(stack.cv, StratifiedKFold)
    assert stack.cv.get_n_splits() == config.N_CV_FOLDS == 5
    assert stack.cv.shuffle is True
    assert stack.cv.random_state == config.RANDOM_STATE == 42


def test_stacking_base_estimators_are_full_pipelines_with_own_preprocessors():
    stack = ensemble.build_stacking_classifier()
    ids = {id(est.named_steps["preprocessor"]) for _, est in stack.estimators}
    assert len(ids) == 3


# =============================================================================
# Phase E -- exploratory XGBoost
# =============================================================================
def test_xgboost_availability_flag_is_consistent_with_importability():
    if tuning.XGBOOST_AVAILABLE:
        assert tuning.XGBClassifier is not None
        assert tuning.XGBOOST_IMPORT_ERROR is None
    else:
        assert tuning.XGBClassifier is None
        assert tuning.XGBOOST_IMPORT_ERROR


def test_xgboost_grid_size_is_12():
    grid = tuning.build_xgboost_param_grid(4.295045045045045)
    assert tuning.grid_size(grid) == 12


def test_xgboost_grid_searches_only_the_three_permitted_parameters():
    grid = tuning.build_xgboost_param_grid(4.0)
    assert set(grid) == {
        "classifier__max_depth",
        "classifier__learning_rate",
        "classifier__scale_pos_weight",
    }
    assert grid["classifier__max_depth"] == [2, 4, 6]
    assert grid["classifier__learning_rate"] == [0.03, 0.1]


def test_xgboost_grid_uses_the_supplied_class_ratio_not_a_hardcoded_value():
    grid = tuning.build_xgboost_param_grid(7.5)
    assert grid["classifier__scale_pos_weight"] == [1.0, 7.5]


@pytest.mark.skipif(not tuning.XGBOOST_AVAILABLE, reason="xgboost is not installed")
def test_xgboost_search_preserves_the_fixed_design_and_parallelism_choice():
    search = tuning.build_xgboost_search(4.0)
    assert search.refit == "roc_auc"
    assert search.scoring is tuning.CV_SCORING
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.random_state == config.RANDOM_STATE
    # Estimator pinned to 1 thread; parallelism taken at the search level.
    assert search.estimator.named_steps["classifier"].n_jobs == 1
    assert search.n_jobs == -1


# =============================================================================
# Phase E continued -- targeted second-stage XGBoost refinement
# =============================================================================
def test_xgboost_refinement_grid_size_is_54():
    grid = tuning.build_xgboost_refinement_param_grid(4.295045045045045)
    assert tuning.grid_size(grid) == 54


def test_xgboost_refinement_grid_has_the_confirmed_values():
    grid = tuning.build_xgboost_refinement_param_grid(4.295045045045045)
    assert grid["classifier__max_depth"] == [1, 2, 3]
    assert grid["classifier__learning_rate"] == [0.01, 0.03, 0.05]
    assert grid["classifier__n_estimators"] == [300, 600, 1000]
    assert grid["classifier__scale_pos_weight"] == [1.0, 4.295045045045045]


def test_xgboost_refinement_grid_searches_exactly_four_parameters():
    grid = tuning.build_xgboost_refinement_param_grid(4.0)
    assert set(grid) == {
        "classifier__max_depth",
        "classifier__learning_rate",
        "classifier__n_estimators",
        "classifier__scale_pos_weight",
    }


def test_xgboost_refinement_grid_uses_the_supplied_class_ratio():
    grid = tuning.build_xgboost_refinement_param_grid(7.5)
    assert grid["classifier__scale_pos_weight"] == [1.0, 7.5]


def test_xgboost_refinement_does_not_alter_the_first_stage_grid():
    """Constructing the refinement grid must not mutate or overwrite the
    first-stage exploratory grid -- both stages stay separately
    reportable."""
    first_stage = tuning.build_xgboost_param_grid(4.295045045045045)
    tuning.build_xgboost_refinement_param_grid(4.295045045045045)
    assert tuning.grid_size(first_stage) == 12
    assert set(first_stage) == {
        "classifier__max_depth",
        "classifier__learning_rate",
        "classifier__scale_pos_weight",
    }


@pytest.mark.skipif(not tuning.XGBOOST_AVAILABLE, reason="xgboost is not installed")
def test_xgboost_refinement_search_preserves_the_fixed_design():
    search = tuning.build_xgboost_refinement_search(4.0)
    assert search.refit == "roc_auc"
    assert search.scoring is tuning.CV_SCORING
    cv = search.cv
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == config.N_CV_FOLDS == 5
    assert cv.shuffle is True
    assert cv.random_state == config.RANDOM_STATE == 42


@pytest.mark.skipif(not tuning.XGBOOST_AVAILABLE, reason="xgboost is not installed")
def test_xgboost_refinement_estimator_n_jobs_is_1_and_search_n_jobs_is_minus_1():
    search = tuning.build_xgboost_refinement_search(4.0)
    assert search.estimator.named_steps["classifier"].n_jobs == 1
    assert search.n_jobs == -1


@pytest.mark.skipif(not tuning.XGBOOST_AVAILABLE, reason="xgboost is not installed")
def test_xgboost_refinement_uses_a_fresh_preprocessor_each_call():
    a = tuning.build_xgboost_refinement_search(4.0).estimator.named_steps["preprocessor"]
    b = tuning.build_xgboost_refinement_search(4.0).estimator.named_steps["preprocessor"]
    assert a is not b


@pytest.mark.skipif(not tuning.XGBOOST_AVAILABLE, reason="xgboost is not installed")
def test_xgboost_refinement_does_not_fix_n_estimators_on_the_estimator():
    """n_estimators is searched via the grid (`classifier__n_estimators`),
    not fixed on the constructor -- unlike the first-stage estimator, which
    pins it to 300."""
    first_stage_estimator = tuning.build_xgboost_search(4.0).estimator.named_steps["classifier"]
    refinement_estimator = tuning.build_xgboost_refinement_search(4.0).estimator.named_steps["classifier"]
    assert first_stage_estimator.n_estimators == 300
    assert refinement_estimator.n_estimators is None
    assert "classifier__n_estimators" in tuning.build_xgboost_refinement_param_grid(4.0)
