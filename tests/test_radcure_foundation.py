"""
Lightweight validation for `src/radcure/` (the project foundation through
Section 4). Deliberately compact: this locks in the key data contracts and
deterministic-cleaning rules that the analysis script's own inline
assertions already check end-to-end -- it does not re-implement a large
test suite for trivial code.

Run with:  pytest tests/test_radcure_foundation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import cleaning, config, leakage, preprocessing, target  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return cleaning.load_raw_clinical()


# =============================================================================
# Raw data contract
# =============================================================================
def test_raw_shape(raw_df):
    assert raw_df.shape == (config.EXPECTED_RAW_ROWS, config.EXPECTED_RAW_COLUMNS)


def test_patient_id_unique(raw_df):
    assert raw_df[config.PATIENT_ID_COLUMN].duplicated().sum() == 0
    assert raw_df.duplicated().sum() == 0


# =============================================================================
# Leakage audit
# =============================================================================
def test_leakage_audit_covers_every_raw_column_exactly_once(raw_df):
    leakage.assert_full_coverage(raw_df.columns)


def test_leakage_audit_row_count():
    assert len(leakage.LEAKAGE_AUDIT_TABLE) == config.EXPECTED_RAW_COLUMNS


def test_candidate_predictors_match_expected():
    assert set(leakage.CANDIDATE_PREDICTORS) == config.EXPECTED_CANDIDATE_PREDICTORS
    assert len(leakage.CANDIDATE_PREDICTORS) == 13


def test_candidate_and_excluded_partition_all_columns():
    assert len(leakage.CANDIDATE_PREDICTORS) + len(leakage.EXCLUDED_VARIABLES) == config.EXPECTED_RAW_COLUMNS
    assert set(leakage.CANDIDATE_PREDICTORS).isdisjoint(leakage.EXCLUDED_VARIABLES)


@pytest.mark.parametrize(
    "variable,expected_category",
    [
        ("RT Start", "CALENDAR_ERA_PROXY"),
        ("RADCURE-challenge", "DATASET_MEMBERSHIP_PROXY"),
        ("ContrastEnhanced", "TECHNICAL_PROTOCOL_PROXY"),
    ],
)
def test_proxy_variables_are_not_labelled_direct_target_leakage(variable, expected_category):
    row = leakage.LEAKAGE_AUDIT_TABLE.loc[leakage.LEAKAGE_AUDIT_TABLE["variable"] == variable].iloc[0]
    assert row["category"] == expected_category
    assert row["category"] != "DIRECT_TARGET_LEAKAGE"
    assert row["decision"] == "EXCLUDE"


def test_assert_no_excluded_column_in_frame_catches_violation():
    with pytest.raises(AssertionError):
        leakage.assert_no_excluded_column_in_frame(["Age", "Status"])
    # Should not raise for a clean PRIMARY-only column set:
    leakage.assert_no_excluded_column_in_frame(config.PRIMARY_FEATURES)


# =============================================================================
# Primary predictor set (Section 3.12)
# =============================================================================
def test_primary_features_are_a_subset_of_candidates():
    assert set(config.PRIMARY_FEATURES) <= set(leakage.CANDIDATE_PREDICTORS)


def test_primary_features_exact_list():
    assert config.PRIMARY_FEATURES == [
        "Age", "Smoking PY", "Sex", "ECOG PS", "Smoking Status", "Ds Site", "T", "N",
    ]
    assert len(config.PRIMARY_FEATURES) == 8


# =============================================================================
# Deterministic semantic cleaning (Section 4) -- pure-function unit tests
# =============================================================================
def test_clean_ecog_ps_exact_corrections_and_ambiguous_range_to_missing():
    raw = pd.Series(["ECOG-Pt 0", "ECOG-Pt 1", "ECOG-Pt 2", "Unknown", "ECOG 0-1", None, "ECOG 3"])
    cleaned = cleaning.clean_ecog_ps(raw)
    assert list(cleaned) == ["ECOG 0", "ECOG 1", "ECOG 2", "Missing", "Missing", "Missing", "ECOG 3"]


def test_clean_ecog_ps_rejects_unexpected_level():
    with pytest.raises(ValueError):
        cleaning.clean_ecog_ps(pd.Series(["ECOG 5"]))


def test_clean_ds_site_case_harmonisation_only():
    raw = pd.Series(["nasal cavity", "Nasal Cavity", "esophagus", "Esophagus", "Sarcoma"])
    cleaned = cleaning.clean_ds_site(raw)
    assert set(cleaned) == {"Nasal Cavity", "Esophagus", "Sarcoma"}
    assert cleaned.nunique() == 3  # 5 raw values collapse to 3 after merging the two duplicate pairs


def test_clean_smoking_py_tokens_and_missing_become_nan_never_invented():
    raw = pd.Series(["20", "na", "<20", ">50", "37.5", None, "0"])
    cleaned = cleaning.clean_smoking_py(raw)
    assert cleaned.tolist()[0] == 20.0
    assert pd.isna(cleaned.iloc[1])  # 'na'
    assert pd.isna(cleaned.iloc[2])  # '<20' -- a bound, never coerced to 20
    assert pd.isna(cleaned.iloc[3])  # '>50' -- a bound, never coerced to 50
    assert cleaned.iloc[4] == 37.5
    assert pd.isna(cleaned.iloc[5])  # true NaN
    assert cleaned.iloc[6] == 0.0
    assert cleaned.dtype == "float64"


def test_clean_smoking_py_rejects_unparsable_value():
    with pytest.raises(ValueError):
        cleaning.clean_smoking_py(pd.Series(["twenty"]))


def test_clean_smoking_status_case_only_unknown_never_imputed():
    raw = pd.Series(["Current", "Ex-smoker", "Non-smoker", "unknown"])
    cleaned = cleaning.clean_smoking_status(raw)
    assert list(cleaned) == ["Current", "Ex-smoker", "Non-smoker", "Unknown"]


def test_clean_t_category_unresolved_and_unassessable_codes_to_missing():
    raw = pd.Series(["T1", "T1a", "TX", "T1 (2)", "T2 (2)", "T3 (2)", "rT0", None])
    cleaned = cleaning.clean_t_category(raw)
    assert cleaned.iloc[0] == "T1"
    assert cleaned.iloc[1] == "T1a"
    assert all(v == "Missing" for v in cleaned.iloc[2:])


def test_clean_t_category_does_not_collapse_subcategories():
    raw = pd.Series(["T1a", "T1b", "T2a", "T2b", "T4a", "T4b"])
    cleaned = cleaning.clean_t_category(raw)
    assert list(cleaned) == list(raw)  # every sub-category passes through unchanged


def test_clean_n_category_nx_to_missing_n2_not_reassigned():
    raw = pd.Series(["N0", "N2", "N2a", "N2b", "N2c", "NX", None])
    cleaned = cleaning.clean_n_category(raw)
    assert cleaned.iloc[1] == "N2"  # never reassigned to a sub-letter
    assert cleaned.iloc[5] == "Missing"
    assert cleaned.iloc[6] == "Missing"


def test_clean_hpv_labels_and_missing_wording():
    raw = pd.Series(["Yes, positive", "Yes, Negative", None])
    cleaned = cleaning.clean_hpv(raw)
    assert list(cleaned) == ["Positive", "Negative", config.HPV_MISSING_LABEL]
    assert "not tested" not in config.HPV_MISSING_LABEL.lower()


def test_missingness_summary_invariant_after_never_less_than_before():
    raw = pd.DataFrame({"T": ["T1", "TX", None]})
    cleaned = pd.DataFrame({"T": cleaning.clean_t_category(raw["T"])})
    table = cleaning.missingness_summary(raw, cleaned, {"T": config.MISSING_LABEL})
    row = table.iloc[0]
    assert row["missing_after_cleaning"] >= row["missing_raw"]
    assert row["missing_raw"] == 1
    assert row["missing_after_cleaning"] == 2


# =============================================================================
# Target construction (Sections 3.8-3.9)
# =============================================================================
def test_build_target_full_cohort_matches_confirmed_design(raw_df):
    target_df = target.build_target(raw_df)
    summary = target.target_summary(target_df)
    assert summary["n_total"] == config.EXPECTED_RAW_ROWS
    assert summary["n_eligible"] == config.EXPECTED_ELIGIBLE_N
    assert summary["n_events"] == config.EXPECTED_EVENTS_N
    assert summary["n_nonevents"] == config.EXPECTED_NONEVENTS_N
    assert summary["n_excluded"] == config.EXPECTED_EXCLUDED_N


def test_no_deaths_before_landmark(raw_df):
    target.assert_no_deaths_before_landmark(raw_df)  # must not raise


def test_target_partition_is_exhaustive_and_never_treats_excluded_as_nonevent(raw_df):
    target_df = target.build_target(raw_df)
    excluded = ~target_df[config.ELIGIBILITY_COLUMN]
    # Excluded rows must carry <NA>, never 0 or 1.
    assert target_df.loc[excluded, config.TARGET_COLUMN].isna().all()
    assert target_df.loc[~excluded, config.TARGET_COLUMN].notna().all()


def test_reconstructed_diagnosis_date_corroborates_challenge_temporal_separation(raw_df):
    """The reconstruction is corroborated by, not proven by, this check --
    see `target.reconstruct_diagnosis_date`'s docstring caution."""
    diagnosis_date = target.reconstruct_diagnosis_date(raw_df)
    challenge = raw_df["RADCURE-challenge"].astype(str)
    train_max = diagnosis_date[challenge == "training"].max()
    test_min = diagnosis_date[challenge == "test"].min()
    n_overlap = int(
        (diagnosis_date[challenge == "training"] >= test_min).sum()
        + (diagnosis_date[challenge == "test"] <= train_max).sum()
    )
    assert n_overlap == 0


# =============================================================================
# Section 3.8 design comparison: diagnosis-anchored vs. RT-Start-anchored
# =============================================================================
def test_diagnosis_anchored_target_matches_confirmed_comparison_counts(raw_df):
    diagnosis_target_df = target.build_target_diagnosis_anchored(raw_df)
    summary = target.target_summary(
        diagnosis_target_df,
        eligibility_col="diagnosis_anchored_eligible",
        target_col="diagnosis_anchored_mortality_2y",
    )
    assert summary["n_eligible"] == config.EXPECTED_DIAGNOSIS_ELIGIBLE_N
    assert summary["n_events"] == config.EXPECTED_DIAGNOSIS_EVENTS_N
    assert summary["n_nonevents"] == config.EXPECTED_DIAGNOSIS_NONEVENTS_N
    assert summary["n_excluded"] == config.EXPECTED_DIAGNOSIS_EXCLUDED_N


def test_diagnosis_to_rt_start_transitions_match_confirmed_findings(raw_df):
    """Reproduces the four patient-level transition counts that justify
    the RT-Start anchor over a diagnosis anchor (Section 3.8)."""
    is_dead = raw_df["Status"].eq("Dead")
    horizon = config.OUTCOME_HORIZON_DAYS

    diagnosis_target_df = target.build_target_diagnosis_anchored(raw_df)
    rt_target_df = target.build_target(raw_df)
    t_diag = diagnosis_target_df["days_from_diagnosis"]
    t_rt = rt_target_df[config.TIME_FROM_LANDMARK_COLUMN]

    newly_event_mask = is_dead & (t_diag > horizon) & (t_rt <= horizon)
    assert int(newly_event_mask.sum()) == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_N
    assert int(t_rt[newly_event_mask].min()) == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_MIN_DAYS

    newly_excluded_mask = (~is_dead) & (t_diag >= horizon) & (t_rt < horizon)
    assert int(newly_excluded_mask.sum()) == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EXCLUDED_N

    diagnosis_date = target.reconstruct_diagnosis_date(raw_df)
    gap_diagnosis_to_rt = (raw_df["RT Start"] - diagnosis_date).dt.days
    n_horizon_elapsed_before_landmark = int((gap_diagnosis_to_rt >= horizon).sum())
    assert n_horizon_elapsed_before_landmark == config.EXPECTED_HORIZON_ELAPSED_BEFORE_LANDMARK_N


# =============================================================================
# Section 3.8 fixed-730-day boundary
# =============================================================================
def test_fixed_730_day_boundary_case_is_eligible_nonevent(raw_df):
    is_dead = raw_df["Status"].eq("Dead")
    rt_target_df = target.build_target(raw_df)
    t_rt = rt_target_df[config.TIME_FROM_LANDMARK_COLUMN]

    boundary_mask = (~is_dead) & (t_rt == config.OUTCOME_HORIZON_DAYS)
    assert int(boundary_mask.sum()) == 1

    # Select the single boundary ROW first (`.loc[bool_mask]` alone is an
    # unambiguous DataFrame-returning call), then read column values off
    # that one-row Series by label. This avoids the combined
    # `.loc[bool_mask, dynamic_column_name]` indexer, whose overloads
    # Pylance/pandas-stubs cannot resolve unambiguously when the column
    # key is a `str` variable rather than a literal.
    boundary_row_raw = raw_df.loc[boundary_mask].iloc[0]
    boundary_row_target = rt_target_df.loc[boundary_mask].iloc[0]

    boundary_patient_id = boundary_row_raw["patient_id"]
    assert boundary_patient_id == config.EXPECTED_BOUNDARY_PATIENT_ID

    assert bool(boundary_row_target[config.ELIGIBILITY_COLUMN]) is True
    assert int(boundary_row_target[config.TARGET_COLUMN]) == 0


def test_fixed_730_days_equivalent_to_calendar_two_years_in_this_dataset(raw_df):
    is_dead = raw_df["Status"].eq("Dead")
    rt_target_df = target.build_target(raw_df)
    t_rt = rt_target_df[config.TIME_FROM_LANDMARK_COLUMN]

    calendar_horizon_days = (
        (raw_df["RT Start"] + pd.DateOffset(years=2)) - raw_df["RT Start"]
    ).dt.days

    fixed_groups = target.classify_two_year_outcome(is_dead, t_rt, config.OUTCOME_HORIZON_DAYS)
    calendar_groups = target.classify_two_year_outcome(is_dead, t_rt, calendar_horizon_days)

    def _label(groups):
        label = pd.Series(pd.NA, index=t_rt.index, dtype=object)
        for name, mask in groups.items():
            label = label.mask(mask, name)
        return label

    n_reclassified = int((_label(fixed_groups) != _label(calendar_groups)).sum())
    assert n_reclassified == 0


def test_classify_two_year_outcome_partitions_exhaustively():
    is_dead = pd.Series([True, True, False, False])
    t = pd.Series([100.0, 800.0, 900.0, 50.0])
    groups = target.classify_two_year_outcome(is_dead, t, 730)
    assert groups["event"].tolist() == [True, False, False, False]
    assert groups["nonevent_died_later"].tolist() == [False, True, False, False]
    assert groups["nonevent_alive"].tolist() == [False, False, True, False]
    assert groups["excluded"].tolist() == [False, False, False, True]


# =============================================================================
# Preprocessing objects (construction only -- never fitted here)
# =============================================================================
def test_preprocessor_builds_and_is_unfitted():
    preprocessor = preprocessing.build_preprocessor()
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)
    with pytest.raises(NotFittedError):
        preprocessor.transform(pd.DataFrame({
            "Age": [50.0], "Smoking PY": [10.0], "Sex": ["Male"],
            "ECOG PS": ["ECOG 0"], "Smoking Status": ["Current"],
            "Ds Site": ["Larynx"], "T": ["T1"], "N": ["N0"],
        }))


def test_preprocessor_column_assignment_matches_primary_features():
    preprocessor = preprocessing.build_preprocessor()
    transformers = preprocessor.get_params(deep=False)["transformers"]
    assert isinstance(transformers, list)
    numeric_cols = transformers[0][2]
    categorical_cols = transformers[1][2]
    assert list(numeric_cols) == config.PRIMARY_NUMERIC_FEATURES
    assert list(categorical_cols) == config.PRIMARY_CATEGORICAL_FEATURES
