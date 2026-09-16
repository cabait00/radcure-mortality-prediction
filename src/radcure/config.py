"""
Project-wide configuration for the RADCURE two-year mortality project.

This module contains only STATIC configuration:

  - file paths and sheet names for the immutable raw workbook,
  - the confirmed study-design parameters (prediction landmark semantics,
    outcome horizon, random seed, split/CV parameters),
  - the deterministic semantic-cleaning dictionaries approved in the
    Section 4 methodology review, and
  - the confirmed PRIMARY predictor set from Section 3.5.

No statistical parameter is defined here (no imputation value, no scaler
mean/variance, no learned frequency threshold, no category-pooling rule).
Those are estimated only inside the training pipeline / cross-validation
(see ``preprocessing.py``) and must never appear in this file.

Every mapping below is a SEMANTIC correction only: it repairs how a value
was written (case, spelling, an explicit unassessable/unknown code), never
an assertion about what an unrecorded value probably was.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# PROJECT_ROOT = .../RADCURE_ML  (this file lives at RADCURE_ML/src/radcure/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "RADCURE_Clinical_v04_20241219.xlsx"

CLINICAL_SHEET_NAME = "RADCURE_TCIA_Clinical_r2_offset"
DATA_DICTIONARY_SHEET_NAME = "Data Dictionary"

# Output locations of the sequential 01 -> 07 artefact pipeline. Each has
# exactly one job; see `artifacts.py` for the save/load helpers and
# `artifacts/README.md` for what each file contains.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODELING_COHORT_FILENAME = "modeling_cohort.csv"
FINAL_MODEL_FILENAME = "final_logistic_regression.joblib"

EXPECTED_RAW_ROWS = 3346
EXPECTED_RAW_COLUMNS = 34

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# Confirmed study design (Sections 3.1-3.3 methodology reviews)
# =============================================================================
# Prediction landmark: immediately before the first radiotherapy fraction,
# operationalised by the observed `RT Start` date. The outcome clock also
# starts at `RT Start` (Section 3.1, "Design B").
#
# Horizon is a FIXED 730-day window, not a calendar `RT Start + 2 years`
# offset -- Section 3.1 (Task 5) showed the two definitions reclassify
# zero patients here, and the fixed-day rule avoids leap-year/implementation
# ambiguity in the definition of the study's primary endpoint.
OUTCOME_HORIZON_DAYS = 730

# Boundary rule (Section 3.1, Task 4): event if t <= horizon; non-event if
# deceased with t > horizon, or alive with t >= horizon; excluded if alive
# with t < horizon (target unobservable).

TARGET_COLUMN = "mortality_2y"
TIME_FROM_LANDMARK_COLUMN = "days_from_rt_start"
ELIGIBILITY_COLUMN = "target_eligible"

# Primary evaluation design (Section 3.3): a stratified random 80/20
# holdout is primary; a temporal holdout was evaluated and rejected as a
# PRIMARY design because of differential censoring selection bias between
# the historical and later periods (documented, not re-derived, in the
# analysis script using simple censoring counts/rates only -- no
# Kaplan-Meier or other survival-analysis methodology is used anywhere in
# this project).
TEST_SIZE = 0.20
N_CV_FOLDS = 5

# Expected counts, reproduced as executable assertions in the analysis
# script rather than merely asserted in prose (full RADCURE file, n=3346):
EXPECTED_ELIGIBLE_N = 2939
EXPECTED_EVENTS_N = 555
EXPECTED_NONEVENTS_N = 2384
EXPECTED_EXCLUDED_N = 407

# Diagnosis-anchored comparison target (Section 3.1 design justification
# ONLY -- this is NOT the project's outcome definition and its counts are
# never used to build X/y; see `target.build_target_diagnosis_anchored`
# and the Section 3.1 cell of the analysis script for how these are used
# to justify the confirmed RT-Start anchor instead).
EXPECTED_DIAGNOSIS_ELIGIBLE_N = 2968
EXPECTED_DIAGNOSIS_EVENTS_N = 532
EXPECTED_DIAGNOSIS_NONEVENTS_N = 2436
EXPECTED_DIAGNOSIS_EXCLUDED_N = 378

# Patient-level transitions between the diagnosis-anchored comparison
# target and the confirmed RT-Start-anchored target (Section 3.1):
#   - patients the diagnosis anchor calls a non-event ("died after two
#     years from diagnosis") who in fact die within 730 days of the
#     actual prediction landmark (RT Start);
#   - the minimum RT-Start-to-death time among that group;
#   - patients the diagnosis anchor calls a verified non-event ("alive
#     with >=730 days of follow-up from diagnosis") who must be excluded
#     under the RT-Start anchor because their observable follow-up AFTER
#     the actual landmark is under 730 days;
#   - patients whose diagnosis-anchored 730-day horizon had already
#     fully elapsed before the actual prediction landmark was reached.
EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_N = 23
EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_MIN_DAYS = 99
EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EXCLUDED_N = 29
EXPECTED_HORIZON_ELAPSED_BEFORE_LANDMARK_N = 3

# The single alive patient whose observable follow-up after RT Start is
# exactly 730 days (Section 3.1, fixed-730-day boundary check).
EXPECTED_BOUNDARY_PATIENT_ID = "RADCURE-0727"

EXPECTED_TRAIN_N = 2351
EXPECTED_TRAIN_EVENTS = 444
EXPECTED_TRAIN_NONEVENTS = 1907
EXPECTED_TEST_N = 588
EXPECTED_TEST_EVENTS = 111
EXPECTED_TEST_NONEVENTS = 477

# =============================================================================
# Frozen final results (analysis/05 and analysis/06)
# =============================================================================
# The completed model-development cycle produced these values. They are
# recorded here so every re-execution asserts against them instead of
# silently accepting a different number: a change beyond floating-point
# noise means something in the pipeline moved and must be investigated,
# never that a new value should be adopted.
#
# Decision threshold, selected on TRAINING-ONLY out-of-fold predictions by
# maximising Balanced Accuracy (equivalently Youden's J -- see
# analysis/05). It is a training-derived operating point under equal
# sensitivity/specificity weighting, not a clinically optimal threshold.
EXPECTED_FINAL_THRESHOLD = 0.16929818782414302

# Training-only out-of-fold operating points either side of the selection.
EXPECTED_TRAIN_OOF_DEFAULT_CONFUSION = {"tn": 1826, "fp": 81, "fn": 347, "tp": 97}
EXPECTED_TRAIN_OOF_SELECTED_CONFUSION = {"tn": 1316, "fp": 591, "fn": 107, "tp": 337}
EXPECTED_TRAIN_OOF_SELECTED_BALANCED_ACCURACY = 0.7245

# Final held-out evaluation of the frozen model at the frozen threshold.
EXPECTED_HELDOUT_CONFUSION = {"tn": 331, "fp": 146, "fn": 30, "tp": 81}
EXPECTED_HELDOUT_METRICS = {
    "roc_auc": 0.8055,
    "average_precision": 0.5222,
    "sensitivity": 0.7297,
    "specificity": 0.6939,
    "precision": 0.3568,
    "f1": 0.4793,
    "balanced_accuracy": 0.7118,
}
# Tolerance for comparing a recomputed metric against the recorded value.
# Wide enough for the 4-decimal rounding above plus platform-level
# floating-point drift, tight enough that any real change is caught.
METRIC_TOLERANCE = 5e-4

# =============================================================================
# Deterministic semantic-cleaning maps (Section 4)
# =============================================================================
# A single label used for every categorical PRIMARY predictor's explicit
# "value not usable" category. Using one label across ECOG/T/N keeps the
# rule uniform and auditable; Smoking Status uses its own native "Unknown"
# level instead (see below), and HPV uses a longer, deliberately more
# specific label (see HPV_MISSING_LABEL) because the Data Dictionary's
# wording for HPV is more specific than a bare "no data".
MISSING_LABEL = "Missing"

# --- ECOG PS -----------------------------------------------------------------
# Exact corrections: the numeral is unambiguous, "-Pt" is a data-entry
# prefix carrying no grade information.
ECOG_PS_EXACT_CORRECTIONS = {
    "ECOG-Pt 0": "ECOG 0",
    "ECOG-Pt 1": "ECOG 1",
    "ECOG-Pt 2": "ECOG 2",
}
# Explicit not-documented code, and an ambiguous range that must NOT be
# arbitrarily assigned to either grade (doing so would invent a clinical
# judgement that was never made).
ECOG_PS_TO_MISSING = frozenset({"Unknown", "ECOG 0-1"})
ECOG_PS_VALID_LEVELS = frozenset(
    {"ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"}
)

# --- Ds Site ------------------------------------------------------------------
# Case/spelling duplicates only. No clinically distinct or rare-but-valid
# site is merged (Section 3.2: rarity is not grounds to pool patients).
DS_SITE_CASE_HARMONISATION = {
    "nasal cavity": "Nasal Cavity",
    "esophagus": "Esophagus",
}
# Level count within the ELIGIBLE modelling cohort (n=2939) specifically --
# NOT the full raw file (n=3346, which has 17: it additionally contains
# Orbit and Lacrimal gland, two single-patient sites that happen to fall
# entirely within the 407 patients excluded for target-unobservability).
EXPECTED_DS_SITE_LEVELS = 15

# --- Smoking PY -----------------------------------------------------------
# `na` and every bound/range token become missing. A bound is not a
# measurement; no numeric value is invented for it.
SMOKING_PY_MISSING_TOKENS = frozenset({"na", ">50", "<20", ">20", "<10", "<5", ">25"})

# --- Smoking Status ---------------------------------------------------------
# Case harmonisation only. `Unknown` remains an explicit, un-imputed level.
SMOKING_STATUS_CASE_HARMONISATION = {"unknown": "Unknown"}

# --- T category --------------------------------------------------------------
# `TX` = explicit "cannot be assessed". The four `(2)`-suffixed codes and
# `rT0` have NO documented meaning in the Data Dictionary; evidence (a
# `2nd Ca` flag shortly before `RT Start` in 4 of 5 records) is consistent
# with -- but does not prove -- indexing a synchronous second primary.
# Because the reference tumour cannot be established, these become missing
# rather than being guessed at.
T_TO_MISSING_CODES = frozenset({"TX", "T1 (2)", "T2 (2)", "T3 (2)", "rT0"})
T_VALID_LEVELS = frozenset(
    {"T0", "Tis", "T1", "T1a", "T1b", "T2", "T2a", "T2b", "T3", "T4", "T4a", "T4b"}
)

# --- N category --------------------------------------------------------------
# `NX` = explicit "cannot be assessed".
N_TO_MISSING_CODES = frozenset({"NX"})
N_VALID_LEVELS = frozenset(
    {"N0", "N1", "N2", "N2a", "N2b", "N2c", "N3", "N3a", "N3b"}
)

# --- HPV (sensitivity-only; NOT part of the PRIMARY model) --------------------
# The Data Dictionary states a blank cell means "no data available" -- a
# statement about the record, not proof the assay was never performed.
# The missing category is therefore deliberately NOT labelled "not tested".
HPV_VALUE_MAP = {
    "Yes, positive": "Positive",
    "Yes, Negative": "Negative",
}
HPV_MISSING_LABEL = "Missing / not documented"

# =============================================================================
# Leakage-audit outcome and PRIMARY predictor set (Sections 3.4-3.5)
# =============================================================================
# The 13 candidates below are those that PASSED the temporal-availability
# leakage audit (see leakage.py for the full 34-variable table and the
# programmatic derivation of this list -- it is repeated here only as a
# fixed expectation to assert against, not as the source of truth).
EXPECTED_CANDIDATE_PREDICTORS = frozenset(
    {
        "Age", "Sex", "ECOG PS", "Smoking PY", "Smoking Status",
        "Ds Site", "Subsite", "T", "N", "M ", "Stage", "Path", "HPV",
    }
)

# PRIMARY predictors (Section 3.5): a domain-informed subset of the 13
# candidates, chosen on clinical relevance, non-redundancy and structural
# stability -- NOT on predictive performance, and NOT by an automatic
# feature-selection procedure.
PRIMARY_NUMERIC_FEATURES = ["Age", "Smoking PY"]
PRIMARY_CATEGORICAL_FEATURES = ["Sex", "ECOG PS", "Smoking Status", "Ds Site", "T", "N"]
PRIMARY_FEATURES = PRIMARY_NUMERIC_FEATURES + PRIMARY_CATEGORICAL_FEATURES

# Retained in the working dataframe for cohort description only -- never
# admitted to the model matrix X:
DESCRIPTIVE_ONLY_CANDIDATES = ["M ", "Stage", "Subsite", "Path"]

# Candidate, but reserved for one pre-specified PRIMARY + HPV sensitivity
# comparison -- not part of the PRIMARY model, and not modelled yet:
SENSITIVITY_ONLY_FEATURE = "HPV"

PATIENT_ID_COLUMN = "patient_id"
