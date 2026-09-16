"""
Structural and scientific invariants of the sequential analysis/01 -> 07
artefact pipeline.

Two kinds of check live here:

  1. ARCHITECTURE -- the split is generated in exactly one place, downstream
     scripts consume artefacts instead of recreating upstream work, and the
     final model is loaded rather than refitted.
  2. COHORT / SPLIT INVARIANTS -- the frozen counts, the disjointness of the
     partitions, the predictor set, and the exactness of the artefact
     round-trip.

These run against the raw workbook, so they are slower than the pure unit
tests but still well under a second per session (the workbook is loaded once,
via a module-scoped fixture).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import artifacts, cleaning, cohort, config  # noqa: E402

ANALYSIS_FILES = sorted((PROJECT_ROOT / "analysis").glob("*.py"))
SRC_FILES = sorted((PROJECT_ROOT / "src" / "radcure").glob("*.py"))


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return cleaning.load_raw_clinical()


@pytest.fixture(scope="module")
def cohort_df(raw_df) -> pd.DataFrame:
    return cohort.build_modeling_cohort(raw_df)


@pytest.fixture(scope="module")
def split_df(cohort_df) -> pd.DataFrame:
    return cohort.assign_frozen_split(cohort_df)


# =============================================================================
# Architecture: one split-generation site, and no upstream work redone
# =============================================================================
#: An import of, or a call to, `train_test_split` -- as opposed to a passing
#: mention of the name in prose, which several docstrings legitimately make.
_SPLIT_USE = re.compile(r"train_test_split\s*\(|import[^\n]*\btrain_test_split\b")


def test_train_test_split_is_used_in_exactly_one_module():
    """`train_test_split` must be imported and called only in `radcure.cohort`.
    If an analysis script splits again, the frozen assignment stops being
    authoritative.
    """
    users = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in ANALYSIS_FILES + SRC_FILES
        if _SPLIT_USE.search(path.read_text(encoding="utf-8"))
    ]
    assert users == ["src/radcure/cohort.py"], (
        f"train_test_split must be used only in src/radcure/cohort.py, found in: {users}"
    )


def test_downstream_scripts_load_the_frozen_cohort():
    """analysis/02-07 must consume the persisted partition, not rebuild it."""
    for path in ANALYSIS_FILES:
        if path.name.startswith("01_"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "artifacts.load_training_partition()" in text or (
            "artifacts.load_heldout_partition()" in text
        ), f"{path.name} does not load a frozen partition from the cohort artefact."


def test_only_analysis_01_builds_the_cohort_from_raw_data():
    """Deterministic cleaning and target construction happen once, in 01."""
    for path in ANALYSIS_FILES:
        text = path.read_text(encoding="utf-8")
        rebuilds = "load_raw_clinical()" in text
        if path.name.startswith("01_"):
            assert rebuilds, "analysis/01 must load the raw workbook."
        else:
            assert not rebuilds, (
                f"{path.name} reloads the raw workbook; it should consume the frozen "
                f"cohort artefact instead."
            )


def test_only_analysis_06_loads_the_heldout_partition():
    """The held-out partition must not be loaded during model development."""
    loaders = [
        path.name for path in ANALYSIS_FILES
        if "load_heldout_partition()" in path.read_text(encoding="utf-8")
    ]
    assert loaders == ["06_final_heldout_test_evaluation.py"], (
        f"Only analysis/06 may load the held-out partition, found: {loaders}"
    )


def test_analysis_06_and_07_load_the_fitted_model_without_refitting():
    """The final model is fitted once, in 05, and reused thereafter."""
    for name in ["06_final_heldout_test_evaluation.py", "07_final_model_interpretation.py"]:
        text = (PROJECT_ROOT / "analysis" / name).read_text(encoding="utf-8")
        assert "artifacts.load_final_model()" in text, f"{name} must load the fitted model."
        assert not re.search(r"^\s*\w*pipeline\.fit\(", text, flags=re.MULTILINE), (
            f"{name} appears to fit a pipeline; it must reuse the model fitted in analysis/05."
        )


def test_analysis_05_fits_and_persists_the_final_model():
    text = (PROJECT_ROOT / "analysis" / "05_threshold_selection_and_model_freeze.py").read_text(
        encoding="utf-8"
    )
    assert "final_pipeline.fit(X_train, y_train)" in text
    assert "artifacts.save_final_model(" in text


# =============================================================================
# Cohort and split invariants
# =============================================================================
#: Any expression that would write to the raw workbook or its directory.
_RAW_WRITE = re.compile(
    r"\.to_excel\s*\(|"
    r"\.to_csv\s*\(\s*config\.RAW_DATA_PATH|"
    r"open\s*\(\s*config\.RAW_DATA_PATH\s*,\s*[\"'][wax]|"
    r"RAW_DATA_PATH\.(write_text|write_bytes|unlink|rename)"
)


def test_no_module_writes_to_the_raw_workbook():
    """Raw data is immutable: nothing in the project may write to it."""
    offenders = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in ANALYSIS_FILES + SRC_FILES
        if _RAW_WRITE.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"These modules would write to the raw workbook: {offenders}"


def test_loading_the_raw_workbook_leaves_it_unmodified(raw_df):
    """A functional counterpart to the structural check above: loading the
    workbook must not touch the file on disk."""
    path = config.RAW_DATA_PATH
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    cleaning.load_raw_clinical()
    cleaning.load_data_dictionary()
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    assert before == after, "Reading the raw workbook changed it on disk."


def test_modeling_cohort_matches_the_confirmed_counts(cohort_df):
    assert len(cohort_df) == config.EXPECTED_ELIGIBLE_N == 2939
    assert int((cohort_df[config.TARGET_COLUMN] == 1).sum()) == config.EXPECTED_EVENTS_N == 555
    assert int((cohort_df[config.TARGET_COLUMN] == 0).sum()) == config.EXPECTED_NONEVENTS_N == 2384
    assert config.EXPECTED_RAW_ROWS - len(cohort_df) == config.EXPECTED_EXCLUDED_N == 407


def test_cohort_carries_exactly_the_frozen_predictors(cohort_df):
    expected = [config.PATIENT_ID_COLUMN, *config.PRIMARY_FEATURES, config.TARGET_COLUMN]
    assert list(cohort_df.columns) == expected
    assert len(config.PRIMARY_FEATURES) == 8
    assert config.PRIMARY_FEATURES == [
        "Age", "Smoking PY", "Sex", "ECOG PS", "Smoking Status", "Ds Site", "T", "N"
    ]


def test_target_is_not_among_the_predictors():
    assert config.TARGET_COLUMN not in config.PRIMARY_FEATURES
    assert config.PATIENT_ID_COLUMN not in config.PRIMARY_FEATURES
    assert config.TIME_FROM_LANDMARK_COLUMN not in config.PRIMARY_FEATURES
    assert config.ELIGIBILITY_COLUMN not in config.PRIMARY_FEATURES


def test_split_counts_and_integrity(split_df):
    cohort.assert_split_integrity(split_df)
    summary = cohort.split_summary(split_df)
    train_n = int(cast(int, summary.at["train", "n"]))
    train_events = int(cast(int, summary.at["train", "events"]))
    test_n = int(cast(int, summary.at["test", "n"]))
    test_events = int(cast(int, summary.at["test", "events"]))
    assert train_n == config.EXPECTED_TRAIN_N == 2351
    assert train_events == config.EXPECTED_TRAIN_EVENTS == 444
    assert test_n == config.EXPECTED_TEST_N == 588
    assert test_events == config.EXPECTED_TEST_EVENTS == 111


def test_train_and_test_ids_are_disjoint_and_cover_every_patient(split_df):
    train_ids = set(split_df.loc[split_df["split"] == "train", config.PATIENT_ID_COLUMN])
    test_ids = set(split_df.loc[split_df["split"] == "test", config.PATIENT_ID_COLUMN])
    assert not (train_ids & test_ids)
    assert len(train_ids) + len(test_ids) == config.EXPECTED_ELIGIBLE_N


def test_event_proportions_are_comparable_across_partitions(split_df):
    summary = cohort.split_summary(split_df)
    train_prevalence = float(cast(float, summary.at["train", "prevalence_pct"]))
    test_prevalence = float(cast(float, summary.at["test", "prevalence_pct"]))
    assert abs(train_prevalence - test_prevalence) < 0.5, (
        "Stratification should keep the two partitions' prevalence within a fraction of a point."
    )


def test_split_assignment_is_deterministic(cohort_df):
    first = cohort.assign_frozen_split(cohort_df)
    second = cohort.assign_frozen_split(cohort_df)
    assert first.equals(second)


def test_partition_round_trips_exactly_through_csv(split_df, tmp_path):
    """Values, dtypes, row order AND index must survive persistence.

    Row order matters: StratifiedKFold assigns folds by position, so a cohort
    reloaded in a different order would yield different folds and different
    cross-validated results.
    """
    X_direct, y_direct, id_direct = cohort.partition(split_df, cohort.TRAIN_SPLIT)

    path = tmp_path / "cohort.csv"
    split_df.to_csv(path, index=False)
    reloaded = pd.read_csv(path, dtype={config.PATIENT_ID_COLUMN: "string"})
    X_back, y_back, id_back = cohort.partition(reloaded, cohort.TRAIN_SPLIT)

    assert X_back.equals(X_direct)
    assert y_back.equals(y_direct)
    assert id_back.equals(id_direct)
    assert list(X_back.index) == list(X_direct.index)


def test_partition_rejects_an_unknown_split_name(split_df):
    with pytest.raises(ValueError, match="Unknown split"):
        cohort.partition(split_df, "validation")


# =============================================================================
# Frozen scientific results are recorded as executable expectations
# =============================================================================
def test_frozen_threshold_constant():
    assert config.EXPECTED_FINAL_THRESHOLD == pytest.approx(0.16929818782414302, abs=1e-15)


def test_frozen_heldout_confusion_matrix_constant():
    assert config.EXPECTED_HELDOUT_CONFUSION == {"tn": 331, "fp": 146, "fn": 30, "tp": 81}
    total = sum(config.EXPECTED_HELDOUT_CONFUSION.values())
    assert total == config.EXPECTED_TEST_N
    assert (
        config.EXPECTED_HELDOUT_CONFUSION["tp"] + config.EXPECTED_HELDOUT_CONFUSION["fn"]
        == config.EXPECTED_TEST_EVENTS
    )


def test_frozen_heldout_metrics_are_internally_consistent():
    """Sensitivity, specificity, precision, F1 and Balanced Accuracy must be
    exactly what the recorded confusion matrix implies."""
    counts = config.EXPECTED_HELDOUT_CONFUSION
    expected = config.EXPECTED_HELDOUT_METRICS
    sensitivity = counts["tp"] / (counts["tp"] + counts["fn"])
    specificity = counts["tn"] / (counts["tn"] + counts["fp"])
    precision = counts["tp"] / (counts["tp"] + counts["fp"])
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    assert sensitivity == pytest.approx(expected["sensitivity"], abs=config.METRIC_TOLERANCE)
    assert specificity == pytest.approx(expected["specificity"], abs=config.METRIC_TOLERANCE)
    assert precision == pytest.approx(expected["precision"], abs=config.METRIC_TOLERANCE)
    assert f1 == pytest.approx(expected["f1"], abs=config.METRIC_TOLERANCE)
    assert (sensitivity + specificity) / 2 == pytest.approx(
        expected["balanced_accuracy"], abs=config.METRIC_TOLERANCE
    )


def test_training_oof_confusion_matrices_are_consistent_with_the_training_partition():
    for counts in (
        config.EXPECTED_TRAIN_OOF_DEFAULT_CONFUSION,
        config.EXPECTED_TRAIN_OOF_SELECTED_CONFUSION,
    ):
        assert sum(counts.values()) == config.EXPECTED_TRAIN_N
        assert counts["tp"] + counts["fn"] == config.EXPECTED_TRAIN_EVENTS
        assert counts["tn"] + counts["fp"] == config.EXPECTED_TRAIN_NONEVENTS


def test_balanced_accuracy_equals_the_youden_j_identity():
    """The identity analysis/05 relies on: BA = (1 + (TPR - FPR)) / 2."""
    counts = config.EXPECTED_TRAIN_OOF_SELECTED_CONFUSION
    tpr = counts["tp"] / (counts["tp"] + counts["fn"])
    fpr = counts["fp"] / (counts["fp"] + counts["tn"])
    balanced_accuracy = (tpr + (1 - fpr)) / 2
    assert balanced_accuracy == pytest.approx((1 + (tpr - fpr)) / 2, abs=1e-12)
    assert balanced_accuracy == pytest.approx(
        config.EXPECTED_TRAIN_OOF_SELECTED_BALANCED_ACCURACY, abs=config.METRIC_TOLERANCE
    )


# =============================================================================
# Artefact-backed checks (skipped when the pipeline has not been run here)
# =============================================================================
def _load_or_skip(name: str):
    try:
        return artifacts.load_json(name)
    except FileNotFoundError:
        pytest.skip(f"{name} not present -- run the analysis pipeline to generate it.")


def test_persisted_threshold_matches_the_frozen_value():
    payload = _load_or_skip(artifacts.THRESHOLD)
    assert payload["threshold"] == pytest.approx(config.EXPECTED_FINAL_THRESHOLD, abs=1e-12)
    assert payload["selected_after_model_choice"] is True
    assert "test" not in payload["selected_on"].lower() or "no test-set" in payload["selected_on"]


def test_persisted_heldout_metrics_match_the_frozen_values():
    payload = _load_or_skip(artifacts.HELDOUT_METRICS)
    assert payload["confusion_matrix"] == config.EXPECTED_HELDOUT_CONFUSION
    for name, expected in config.EXPECTED_HELDOUT_METRICS.items():
        assert payload["metrics"][name] == pytest.approx(expected, abs=config.METRIC_TOLERANCE)
    assert payload["threshold"] == pytest.approx(config.EXPECTED_FINAL_THRESHOLD, abs=1e-12)
    assert payload["n_test"] == config.EXPECTED_TEST_N


def test_persisted_final_specification_records_the_selected_family():
    payload = _load_or_skip(artifacts.FINAL_MODEL_SPECIFICATION)
    assert payload["model_family"] == "Logistic Regression"
    assert payload["hyperparameters"]["C"] == 1.0
    assert payload["hyperparameters"]["class_weight"] is None
    assert payload["predictors"] == config.PRIMARY_FEATURES
    assert payload["selected_before_threshold_selection"] is True


def test_persisted_tuned_results_use_the_frozen_cv_design():
    payload = _load_or_skip(artifacts.TUNED_RESULTS)
    assert payload["cv"]["n_splits"] == config.N_CV_FOLDS
    assert payload["cv"]["shuffle"] is True
    assert payload["cv"]["random_state"] == config.RANDOM_STATE
    assert payload["refit_metric"] == "roc_auc"


def test_persisted_cohort_reproduces_the_frozen_split():
    try:
        persisted = artifacts.load_modeling_cohort()
    except FileNotFoundError:
        pytest.skip("modeling_cohort.csv not present -- run analysis/01 to generate it.")
    cohort.assert_split_integrity(persisted)
