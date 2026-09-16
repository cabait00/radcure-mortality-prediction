"""
Lightweight validation for `src/radcure/modeling.py` (used by analysis/02).

Deliberately small: locks in the fixed CV design and the fixed
five-candidate registry as hard contracts, so a future edit cannot
silently change the cross-validation splitter, add/remove a candidate, or
let two models share a fitted preprocessor -- without also touching these
tests. Does not test cross-validated performance values, which are
outcomes of the data, not invariants of the code.

Run with:  pytest tests/test_modeling.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import config, modeling  # noqa: E402


def test_cv_splitter_has_expected_configuration():
    cv = modeling.build_cv_splitter()
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == 5 == config.N_CV_FOLDS
    cv_attrs = modeling.describe_cv_splitter(cv)
    assert cv_attrs.shuffle is True
    assert cv_attrs.random_state == 42 == config.RANDOM_STATE


def test_cv_scoring_is_exactly_the_three_pre_specified_metrics():
    assert set(modeling.CV_SCORING) == {"roc_auc", "average_precision", "balanced_accuracy"}
    assert "accuracy" not in modeling.CV_SCORING


def test_model_registry_has_exactly_the_five_intended_candidates():
    registry = modeling.build_model_registry()
    assert set(registry) == {
        "Dummy (prior)", "Logistic Regression", "Decision Tree", "Random Forest", "RBF SVC",
    }
    assert len(registry) == 5

    dummy = registry["Dummy (prior)"]
    assert isinstance(dummy, DummyClassifier)
    dummy_config = modeling.describe_dummy_classifier(dummy)
    assert dummy_config["strategy"] == "prior"

    logreg = registry["Logistic Regression"]
    assert isinstance(logreg, LogisticRegression)
    logreg_config = modeling.describe_logistic_regression(logreg)
    assert logreg_config["max_iter"] == 5000
    assert logreg_config["random_state"] == config.RANDOM_STATE
    assert logreg_config["class_weight"] is None  # not applied in analysis/02

    tree = registry["Decision Tree"]
    assert isinstance(tree, DecisionTreeClassifier)
    tree_config = modeling.describe_decision_tree(tree)
    assert tree_config["random_state"] == config.RANDOM_STATE

    forest = registry["Random Forest"]
    assert isinstance(forest, RandomForestClassifier)
    forest_config = modeling.describe_random_forest(forest)
    assert forest_config["random_state"] == config.RANDOM_STATE
    assert forest_config["n_jobs"] == -1
    assert forest_config["class_weight"] is None

    svc = registry["RBF SVC"]
    assert isinstance(svc, SVC)
    svc_config = modeling.describe_rbf_svc(svc)
    assert svc_config["kernel"] == "rbf"
    # SVC(probability=True) is explicitly disallowed. In scikit-learn 1.9,
    # the constructor's own default for `probability` is the sentinel
    # string "deprecated" rather than the literal `False`, so the
    # functional contract (no explicit True, and predict_proba genuinely
    # unavailable) is checked instead of the raw attribute value.
    assert svc_config["probability"] is not True
    assert not hasattr(svc, "predict_proba")


def test_every_candidate_is_wrapped_in_a_pipeline_with_expected_steps():
    pipelines = modeling.build_candidate_pipelines()
    assert set(pipelines) == set(modeling.build_model_registry())
    for name, pipeline in pipelines.items():
        assert isinstance(pipeline, Pipeline), name
        assert [step for step, _ in pipeline.steps] == ["preprocessor", "classifier"], name


def test_each_pipeline_receives_a_fresh_preprocessor_instance():
    """No two candidates may share the same (fitted or unfitted)
    preprocessor object -- each must be constructed fresh."""
    pipelines = modeling.build_candidate_pipelines()
    preprocessors = [pipeline.named_steps["preprocessor"] for pipeline in pipelines.values()]
    ids = {id(p) for p in preprocessors}
    assert len(ids) == len(preprocessors), "Two or more candidates share the same preprocessor object."

    # Also true for two independent calls to build_pipeline on the same estimator:
    registry = modeling.build_model_registry()
    p1 = modeling.build_pipeline(registry["Logistic Regression"])
    p2 = modeling.build_pipeline(registry["Logistic Regression"])
    assert p1.named_steps["preprocessor"] is not p2.named_steps["preprocessor"]


def test_evaluate_candidate_runs_on_a_tiny_synthetic_training_set():
    """Smoke test only -- not a check on any specific performance value."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "Age": rng.uniform(20, 90, n),
            "Smoking PY": rng.uniform(0, 100, n),
            "Sex": rng.choice(["Male", "Female"], n),
            "ECOG PS": rng.choice(["ECOG 0", "ECOG 1", "Missing"], n),
            "Smoking Status": rng.choice(["Current", "Ex-smoker", "Non-smoker"], n),
            "Ds Site": rng.choice(["Oropharynx", "Larynx"], n),
            "T": rng.choice(["T1", "T2", "T3"], n),
            "N": rng.choice(["N0", "N1"], n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipeline = modeling.build_pipeline(DummyClassifier(strategy="prior"))
    result = modeling.evaluate_candidate(pipeline, X, y, cv)

    for metric in modeling.CV_SCORING:
        assert f"{metric}_mean" in result
        assert f"{metric}_std" in result
        fold_scores = result[f"{metric}_scores"]
        assert isinstance(fold_scores, np.ndarray)
        assert len(fold_scores) == 3


@pytest.mark.parametrize(
    "strategy_attr,value",
    [("Decision Tree", "class_weight"), ("Random Forest", "class_weight")],
)
def test_no_class_weight_applied_in_baseline_comparison(strategy_attr, value):
    registry = modeling.build_model_registry()
    estimator = registry[strategy_attr]
    assert getattr(estimator, value) is None
