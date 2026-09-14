# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — Project Foundation
Two-Year All-Cause Mortality Prediction in Head and Neck Tumours
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset
Data source: The Cancer Imaging Archive (TCIA), Collection RADCURE,
             Clinical Data v4, updated 2024-12-19.
             File: data/raw/RADCURE_Clinical_v04_20241219.xlsx (immutable).

--------------------------------------------------------------------------------
RESEARCH QUESTION
--------------------------------------------------------------------------------
How reliably can two-year all-cause mortality following the start of
radiotherapy be predicted in patients with head and neck tumours, using
clinical, demographic and tumour-related characteristics available before
treatment initiation, with classical machine learning methods?

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
This script implements the confirmed project FOUNDATION only -- everything
through Section 4 of the methodology (data preparation and preprocessing
specification). It does NOT fit, tune or evaluate any predictive model.
It stops immediately after constructing (but not fitting) the preprocessing
pipeline objects, per the explicit scope of this implementation phase.

Sections, mirroring the completed methodology reviews:

    2.  Data loading and compact ML-relevant data verification
    3.8 Prediction landmark and RT-Start-anchored outcome definition
    3.9 Cohort definition (target observability only; no clinical restriction)
    3.10 Split-strategy rationale (stratified random primary; naive
         temporal holdout rejected on documented censoring-selection grounds)
    3.11 Feature leakage audit (all 34 original variables, exactly once)
    3.12 Primary predictor set definition
    4.  Deterministic semantic cleaning of the PRIMARY predictors
    5.  Assembly of X / y / patient_id, and the primary train/test split
    6.  Construction (NOT fitting) of the PRIMARY preprocessing pipeline

--------------------------------------------------------------------------------
GOVERNING METHODOLOGICAL RULES (carried through every section below)
--------------------------------------------------------------------------------
  - Raw data is immutable: this script only READS
    data/raw/RADCURE_Clinical_v04_20241219.xlsx and never writes to it.
  - No survival-analysis methodology (Kaplan-Meier, log-rank, Cox models,
    concordance) is used anywhere in this script. The rejection of a naive
    temporal holdout is documented using simple censoring counts and rates
    only.
  - Two kinds of operations are strictly separated throughout:
        (A) deterministic semantic cleaning -- estimates NO statistical
            parameter, and may be applied before the train/test split
            (implemented in `src/radcure/cleaning.py`);
        (B) learned / data-dependent preprocessing -- estimates a
            statistical parameter (an imputation median, a scaler's
            mean/variance, an encoder's category list) and MUST be fitted
            only inside the training pipeline / cross-validation, never
            before the split, never on the held-out test set
            (implemented, but NOT fitted, in `src/radcure/preprocessing.py`).
  - No predictor-outcome association, feature selection, hyperparameter
    tuning or model fitting is performed in this script.

Confirmed project environment: `ml` conda environment
(/home/c/miniconda3/envs/ml/bin/python) -- pandas 3.0.5, numpy 2.4.6,
scikit-learn 1.9.0, openpyxl 3.1.5, pytest 9.1.1.
"""

# %%
# =============================================================================
# SECTION 1 — Imports, environment, reusable narrative helpers
# =============================================================================
# Objective:
#   Make the `radcure` package (src/radcure/) importable from this script,
#   and define two tiny display helpers used throughout the narrative below.
#   No modelling library (no estimator of any kind) is imported here or
#   anywhere else in this script -- this phase stops before model fitting.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import cleaning, config, leakage, preprocessing, target  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 120)


def print_value_counts(series: pd.Series, title: str) -> None:
    """Print every raw level of a series with its count, making whitespace
    and case variants visible (via repr()) rather than hiding them behind
    a summary statistic.
    """
    n_missing = int(series.isna().sum())
    n_levels = int(series.nunique(dropna=True))
    print(f"\n{title}  (n={len(series)}, missing={n_missing}, levels={n_levels})")
    for value, count in series.value_counts(dropna=False).items():
        is_missing = bool(pd.Index([value]).isna()[0])
        label = "<NaN>" if is_missing else repr(value)
        print(f"   {label:32s} {count:5d}")


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


print("Imports OK. `radcure` package resolved from:", SRC_DIR)


# %%
# =============================================================================
# SECTION 2 — Data loading and compact ML-relevant data verification
# =============================================================================
# Objective:
#   Load the immutable raw workbook and verify, in code, the structural
#   facts the entire downstream design depends on -- rather than merely
#   asserting them in prose.
#
# Rationale:
#   The project raw-data policy requires the source workbook to remain
#   immutable. `cleaning.load_raw_clinical` only reads the file. The
#   compact verification below is deliberately limited to what is needed
#   to proceed safely (shape, identifier uniqueness, presence of the Data
#   Dictionary) -- a comprehensive exploratory analysis of this dataset
#   had already been completed separately, so only compact, ML-relevant
#   checks are repeated here.
#
# Assumptions:
#   The workbook contains exactly two sheets: the clinical data sheet and
#   a `Data Dictionary` sheet documenting every column's meaning.

section("SECTION 2 — Data loading and verification")

raw_df = cleaning.load_raw_clinical()
data_dictionary = cleaning.load_data_dictionary()

# Captured now, before `raw_df` is later joined with target-construction
# columns (Section 3.9) -- this is the list the Section 3.11 leakage-audit
# coverage check must match, i.e. exactly the 34 ORIGINAL raw columns.
original_raw_columns = list(raw_df.columns)

print(f"Raw file: {config.RAW_DATA_PATH}")
print(f"Loaded sheet '{config.CLINICAL_SHEET_NAME}': shape = {raw_df.shape}")

assert raw_df.shape == (config.EXPECTED_RAW_ROWS, config.EXPECTED_RAW_COLUMNS), (
    f"Unexpected raw shape {raw_df.shape}; expected "
    f"({config.EXPECTED_RAW_ROWS}, {config.EXPECTED_RAW_COLUMNS})."
)

n_duplicate_ids = int(raw_df[config.PATIENT_ID_COLUMN].duplicated().sum())
n_duplicate_rows = int(raw_df.duplicated().sum())
assert n_duplicate_ids == 0, f"{n_duplicate_ids} duplicate patient_id value(s) found."
assert n_duplicate_rows == 0, f"{n_duplicate_rows} fully duplicated row(s) found."

print(f"Duplicate patient_id: {n_duplicate_ids}   Fully duplicated rows: {n_duplicate_rows}")
print(f"\nData Dictionary sheet: {data_dictionary.shape[0]} variable definitions.")

# Result / Interpretation:
#   3346 rows x 34 columns, one row per patient, patient_id unique, no
#   duplicated rows -- consistent with the previous EDA project's findings.
# Decision:
#   Proceed with the full 3346-row file; no row is dropped at this stage.
# Next step:
#   Construct the RT-Start-anchored outcome (Section 3.8) before any
#   predictor is touched, since eligibility for modelling depends on the
#   outcome, not the reverse.


# %%
# =============================================================================
# SECTION 3.8 — Prediction landmark and RT-Start-anchored outcome
# =============================================================================
# Objective:
#   Fix the prediction landmark and the outcome time origin, and verify
#   the structural facts that justify anchoring both at `RT Start`
#   (confirmed design, Section 3.8 methodology review).
#
# Rationale:
#   The landmark is "immediately before the first radiotherapy fraction",
#   operationalised by the observed `RT Start` date -- the only exactly
#   dated event in the file that corresponds to this clinical moment. The
#   outcome clock also starts at `RT Start` (not at diagnosis), so every
#   patient is given the IDENTICAL 730-day prospective horizon measured
#   from the moment the prediction is actually made. A diagnosis-anchored
#   alternative was evaluated and is demonstrated below (with executable
#   counts, not merely asserted) to be unsuitable: it produces
#   PATIENT-SPECIFIC REMAINING PREDICTION HORIZONS after the actual
#   prediction landmark, because a variable amount of the diagnosis-based
#   730-day window has already elapsed by the time RT actually starts --
#   and therefore does not align with the intended PROSPECTIVE prediction
#   question.
#
# Assumptions:
#   `RT Start`, `Date of Death` and `Last FU` are all directly OBSERVED
#   date columns (no reconstruction is involved in the confirmed outcome
#   definition itself). A separate DERIVED/RECONSTRUCTED diagnosis date is
#   used below ONLY for (a) the design comparison that justifies the
#   RT-Start anchor and (b) a corroborating check against the
#   RADCURE-challenge cohort structure -- never for the confirmed outcome,
#   and never as a predictor.

section("SECTION 3.8 — Prediction landmark and outcome definition")

print("Prediction landmark : immediately before the first RT fraction, i.e. `RT Start`.")
print("Outcome time origin  : `RT Start` (same clock as the landmark).")
print(f"Outcome horizon      : a FIXED {config.OUTCOME_HORIZON_DAYS}-day window.")

# --- Structural check 1: no death precedes the landmark ---------------------
target.assert_no_deaths_before_landmark(raw_df)
print("\n[OK] No deceased patient has Date of Death before RT Start.")

days_from_rt_start_full = target.compute_days_from_rt_start(raw_df)
is_dead_full = raw_df["Status"].eq("Dead")
min_days_to_death = days_from_rt_start_full[is_dead_full].min()
print(f"Minimum days from RT Start to death (n={int(is_dead_full.sum())} deceased): {min_days_to_death:.0f}")

# --- Structural check 2: diagnosis-to-RT-Start gap (DERIVED, diagnostic only)
# The diagnosis date is RECONSTRUCTED from documented follow-up duration
# (`Length FU`), using the dataset's apparent 365-day encoding -- it is
# NOT a directly observed date. It is used here ONLY to (a) report the
# typical workup interval as a cohort characteristic, and (b) corroborate
# the reconstruction against the RADCURE-challenge cohort's OBSERVED
# temporal separation (the challenge's exact assignment dates are not
# documented in any locally available source; the separation itself is
# inferred from this dataset, not published). `RADCURE-challenge` is used
# here EXCLUSIVELY for this one-off corroboration -- it is never used
# again below, and it never becomes a predictor (Section 3.11: it is a
# DATASET_MEMBERSHIP_PROXY, excluded from X).
diagnosis_date = target.reconstruct_diagnosis_date(raw_df)
gap_diagnosis_to_rt = (raw_df["RT Start"] - diagnosis_date).dt.days

print("\nDiagnosis-to-RT-Start gap (DERIVED/RECONSTRUCTED diagnosis date; diagnostic only, not a predictor):")
print(
    gap_diagnosis_to_rt.describe(percentiles=[0.25, 0.5, 0.75]).round(1).to_string()
)

challenge = raw_df["RADCURE-challenge"].astype(str)
diag_train_max = diagnosis_date[challenge == "training"].max()
diag_test_min = diagnosis_date[challenge == "test"].min()
n_overlap = int(
    (diagnosis_date[challenge == "training"] >= diag_test_min).sum()
    + (diagnosis_date[challenge == "test"] <= diag_train_max).sum()
)
print("\nRADCURE-challenge temporal-separation corroboration check "
      "(does NOT prove the reconstruction exact; see caution below):")
print(f"   training: reconstructed diagnosis <= {diag_train_max.date()}")
print(f"   test    : reconstructed diagnosis >= {diag_test_min.date()}")
print(f"   overlapping patients: {n_overlap}")
assert n_overlap == 0, (
    "The reconstructed diagnosis date does not reproduce the RADCURE-challenge cohort's "
    f"observed temporal separation cleanly ({n_overlap} overlapping patients) -- the "
    "reconstruction assumption (an apparent strict 365-day year in `Length FU`) should be "
    "re-examined."
)
print("[OK] Reconstructed diagnosis dates reproduce the observed RADCURE-challenge "
      "training/test temporal separation with zero overlap.")

# --- Structural check 3: design comparison -- diagnosis-anchored vs.
# RT-Start-anchored 730-day endpoint (Section 3.8 design justification;
# `diagnosis_target_df` is a COMPARISON-ONLY construct, never used for
# X/y). `rt_target_df_preview` recomputes the confirmed RT-Start-anchored
# target here (a pure, side-effect-free function call) purely so this
# comparison is self-contained; Section 3.9 below performs the
# authoritative construction of `target_df` used by the rest of the script.
diagnosis_target_df = target.build_target_diagnosis_anchored(raw_df)
rt_target_df_preview = target.build_target(raw_df)

diag_summary = target.target_summary(
    diagnosis_target_df,
    eligibility_col="diagnosis_anchored_eligible",
    target_col="diagnosis_anchored_mortality_2y",
)
rt_summary_preview = target.target_summary(rt_target_df_preview)

print("\nDesign comparison -- diagnosis-anchored vs. RT-Start-anchored 730-day endpoint:")
print("   diagnosis-anchored (COMPARISON ONLY -- not the confirmed design, never used for X/y):")
for k, v in diag_summary.items():
    print(f"      {k:16s} {v}")
print("   RT-Start-anchored (CONFIRMED design):")
for k, v in rt_summary_preview.items():
    print(f"      {k:16s} {v}")

assert diag_summary["n_eligible"] == config.EXPECTED_DIAGNOSIS_ELIGIBLE_N
assert diag_summary["n_events"] == config.EXPECTED_DIAGNOSIS_EVENTS_N
assert diag_summary["n_nonevents"] == config.EXPECTED_DIAGNOSIS_NONEVENTS_N
assert diag_summary["n_excluded"] == config.EXPECTED_DIAGNOSIS_EXCLUDED_N
assert rt_summary_preview["n_eligible"] == config.EXPECTED_ELIGIBLE_N
assert rt_summary_preview["n_events"] == config.EXPECTED_EVENTS_N
assert rt_summary_preview["n_nonevents"] == config.EXPECTED_NONEVENTS_N
assert rt_summary_preview["n_excluded"] == config.EXPECTED_EXCLUDED_N
print("[OK] Both design variants reproduce their confirmed counts.")

# Patient-level transitions between the two designs -- the concrete
# evidence that justifies the confirmed RT-Start anchor.
t_diag = diagnosis_target_df["days_from_diagnosis"]
t_rt = rt_target_df_preview[config.TIME_FROM_LANDMARK_COLUMN]
horizon = config.OUTCOME_HORIZON_DAYS

newly_event_mask = is_dead_full & (t_diag > horizon) & (t_rt <= horizon)
n_newly_event = int(newly_event_mask.sum())
min_days_among_newly_event = t_rt[newly_event_mask].min()

newly_excluded_mask = (~is_dead_full) & (t_diag >= horizon) & (t_rt < horizon)
n_newly_excluded = int(newly_excluded_mask.sum())

n_horizon_elapsed_before_landmark = int((gap_diagnosis_to_rt >= horizon).sum())

print("\nPatient-level transitions, diagnosis-anchored -> RT-Start-anchored design:")
print(f"   diagnosis-non-event (died > {horizon}d from diagnosis) but RT-Start-event "
      f"(died <= {horizon}d after the actual landmark): {n_newly_event}")
print(f"      minimum RT-Start-to-death time among them: {min_days_among_newly_event:.0f} days")
print(f"   diagnosis-verified non-event (alive, >= {horizon}d from diagnosis) but RT-Start-"
      f"excluded (< {horizon}d OBSERVABLE follow-up after the actual landmark): {n_newly_excluded}")
print(f"   patients whose diagnosis-anchored {horizon}-day horizon had already elapsed "
      f"BEFORE the actual landmark was reached: {n_horizon_elapsed_before_landmark}")

assert n_newly_event == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_N
assert int(min_days_among_newly_event) == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EVENT_MIN_DAYS
assert n_newly_excluded == config.EXPECTED_DIAGNOSIS_TO_RT_NEWLY_EXCLUDED_N
assert n_horizon_elapsed_before_landmark == config.EXPECTED_HORIZON_ELAPSED_BEFORE_LANDMARK_N
print("[OK] All four transition counts match the confirmed Section 3.8 design-comparison findings.")

print(
    "\nInterpretation: the diagnosis-anchored endpoint measures a fixed 730-day window from "
    "diagnosis, so the fraction of that window remaining AFTER the actual prediction landmark "
    "(RT Start) varies patient-by-patient. This produces PATIENT-SPECIFIC REMAINING PREDICTION "
    "HORIZONS after the actual landmark and therefore does not align with the intended "
    "PROSPECTIVE prediction question -- exactly what anchoring the outcome clock at RT Start "
    "avoids, by giving every patient the identical 730-day prospective horizon."
)

# --- Structural check 4: fixed 730-day boundary case, and fixed-vs-
# calendar equivalence (Section 3.8 fixed-horizon design justification).
alive_at_boundary_mask = (~is_dead_full) & (t_rt == horizon)
n_at_boundary = int(alive_at_boundary_mask.sum())
assert n_at_boundary == 1, f"Expected exactly 1 patient at the fixed {horizon}-day boundary, found {n_at_boundary}."
boundary_patient_id = raw_df.loc[alive_at_boundary_mask, "patient_id"].iloc[0]
assert boundary_patient_id == config.EXPECTED_BOUNDARY_PATIENT_ID
boundary_eligible = bool(rt_target_df_preview.loc[alive_at_boundary_mask, config.ELIGIBILITY_COLUMN].iloc[0])
boundary_label = int(rt_target_df_preview.loc[alive_at_boundary_mask, config.TARGET_COLUMN].iloc[0])
assert boundary_eligible and boundary_label == 0

print(f"\nFixed-{horizon}-day boundary case: patient {boundary_patient_id!r} has exactly "
      f"{horizon} days of documented follow-up after RT Start -> ELIGIBLE, class 0.")
print(f"Boundary rule applied throughout: event if t <= {horizon} (deaths); "
      f"alive non-event if t >= {horizon}.")

# Fixed 730 days vs. a calendar `RT Start + 2 years` offset: does the
# choice of definition reclassify any patient in THIS dataset?
calendar_horizon_days = ((raw_df["RT Start"] + pd.DateOffset(years=2)) - raw_df["RT Start"]).dt.days
fixed_groups = target.classify_two_year_outcome(is_dead_full, t_rt, horizon)
calendar_groups = target.classify_two_year_outcome(is_dead_full, t_rt, calendar_horizon_days)


def _group_label(groups: dict) -> pd.Series:
    label = pd.Series(pd.NA, index=t_rt.index, dtype=object)
    for name, mask in groups.items():
        label = label.mask(mask, name)
    return label


n_reclassified = int((_group_label(fixed_groups) != _group_label(calendar_groups)).sum())
print(f"\nFixed {horizon} days vs. calendar `RT Start + 2 years`: {n_reclassified} patient(s) reclassified.")
assert n_reclassified == 0, (
    f"{n_reclassified} patient(s) classify differently under a calendar two-year offset -- "
    "the fixed-day and calendar definitions are not equivalent in this dataset after all."
)
print("[OK] A fixed 730-day window and a calendar two-year offset classify every patient "
      "identically here. The fixed-day definition is retained as the PRIMARY rule regardless, "
      "for its unambiguous, leap-year-independent specification.")

# Result / Interpretation:
#   Every deceased patient died after RT Start (minimum 23 days). The
#   diagnosis-to-RT-Start gap has a median of ~43 days. Reconstructed
#   diagnosis dates reproduce the RADCURE-challenge cohort's OBSERVED
#   training/test temporal separation with zero overlapping patients --
#   this strongly CORROBORATES the reconstruction, but does not make the
#   diagnosis date directly observed or prove it exact. The diagnosis-
#   anchored design comparison reproduces its confirmed counts exactly
#   (2968 eligible / 532 events / 2436 non-events / 378 excluded) against
#   the confirmed RT-Start design (2939 / 555 / 2384 / 407), and the four
#   patient-level transition counts (23 / 99 days / 29 / 3) demonstrate
#   concretely that the diagnosis anchor gives patients unequal remaining
#   prediction horizons after the actual landmark. The fixed-730-day
#   boundary is exercised by exactly one patient (RADCURE-0727) and is
#   shown to be equivalent to a calendar two-year offset for every patient
#   in this dataset.
# Decision:
#   Anchor both the landmark and the outcome clock at `RT Start`, using a
#   FIXED 730-day horizon. Use only `RT Start`, `Date of Death` and
#   `Last FU` (all observed) to build the confirmed target in the next
#   section; the derived diagnosis date and the comparison-only target are
#   not used again below.
# Next step:
#   Construct the confirmed binary two-year mortality target and its
#   eligibility partition (Section 3.9).


# %%
# =============================================================================
# SECTION 3.9 — Target construction and cohort definition
# =============================================================================
# Objective:
#   Build the RT-Start-anchored binary two-year mortality target with an
#   explicit eligibility partition, and confirm the resulting counts
#   against the values fixed by the confirmed study design.
#
# Rationale:
#   A patient's two-year status is genuinely unobservable if they are
#   alive with less than 730 days of documented follow-up after `RT Start`
#   -- such patients must never be silently assigned to class 0. This is a
#   hard requirement of the project design and is enforced here as an
#   explicit, separately tracked eligibility flag rather than a
#   silent row drop.
#
# Assumptions:
#   The 730-day horizon is a fixed number of days (not a calendar 2-year
#   offset); Section 3.8 established these two definitions reclassify zero
#   patients in this cohort, so the simpler, unambiguous fixed-day rule is
#   used throughout.

section("SECTION 3.9 — Target construction and eligibility")

target_df = target.build_target(raw_df)
raw_df = raw_df.join(target_df)

summary = target.target_summary(target_df)
print("Target summary (full RADCURE file, n = {}):".format(summary["n_total"]))
for k, v in summary.items():
    print(f"   {k:16s} {v}")

assert summary["n_eligible"] == config.EXPECTED_ELIGIBLE_N
assert summary["n_events"] == config.EXPECTED_EVENTS_N
assert summary["n_nonevents"] == config.EXPECTED_NONEVENTS_N
assert summary["n_excluded"] == config.EXPECTED_EXCLUDED_N
print("\n[OK] Target counts match the confirmed study design "
      f"(eligible={config.EXPECTED_ELIGIBLE_N}, events={config.EXPECTED_EVENTS_N}, "
      f"non-events={config.EXPECTED_NONEVENTS_N}, excluded={config.EXPECTED_EXCLUDED_N}).")

# --- Cohort definition: target observability is the ONLY restriction ------
# Section 3.9 established that rarity, unusual histology, or an uncommon
# disease site is NOT grounds to exclude a patient. The full RADCURE
# population is retained; the eligibility flag above is the sole filter
# ever applied to define the modelling cohort. No disease site is removed
# by a CLINICAL eligibility rule -- but a site can still end up with zero
# eligible patients if its only occurrence(s) happen to be
# target-unobservable, which is a property of that patient's follow-up,
# not of the site. The table below shows this precisely rather than
# asserting "every site is retained".
eligible_mask = target_df[config.ELIGIBILITY_COLUMN]
site_counts_full = raw_df["Ds Site"].str.strip().value_counts()
site_counts_eligible = raw_df.loc[eligible_mask, "Ds Site"].str.strip().value_counts()
cohort_composition = pd.DataFrame(
    {"n_full_cohort": site_counts_full, "n_eligible_cohort": site_counts_eligible}
).fillna(0).astype(int)
print("\nDisease-site composition, full vs. eligible cohort "
      "(no site is excluded by a CLINICAL eligibility rule; "
      "a site reaching zero here reflects its patients' follow-up, not the site):")
print(cohort_composition.sort_values("n_full_cohort", ascending=False).to_string())

sites_with_zero_eligible = cohort_composition.loc[cohort_composition["n_eligible_cohort"] == 0]
if not sites_with_zero_eligible.empty:
    print(f"\nSite(s) with zero eligible patients: {list(sites_with_zero_eligible.index)}")
    print("These are absent from the eligible modelling cohort only because their sole raw "
          "patient(s) are target-unobservable under the fixed 730-day outcome definition "
          "(alive with < 730 days of documented follow-up after RT Start) -- NOT because of "
          "any clinical or cohort-definition rule targeting the site itself.")

# Result / Interpretation:
#   2939 of 3346 patients (87.8%) are eligible; 555 events, 2384
#   non-events, 555/2939 = 18.88% prevalence -- matching every figure
#   fixed by the confirmed study design. No disease site was excluded by a
#   clinical eligibility rule. `Orbit` and `Lacrimal gland` are absent from
#   the eligible modelling cohort only because their sole patients are
#   target-unobservable under the fixed 730-day outcome definition -- every
#   other site retains at least one eligible patient.
# Decision:
#   The modelling cohort is the full RADCURE population restricted ONLY by
#   target observability. No clinical, histological, or frequency-based
#   restriction is applied.
# Next step:
#   Before selecting predictors, motivate the primary evaluation design
#   (Section 3.10): a stratified random holdout, not a naive temporal one.


# %%
# =============================================================================
# SECTION 3.10 — Split-strategy rationale
# =============================================================================
# Objective:
#   Document, with simple counts and rates only (no survival-analysis
#   methodology of any kind), why a naive chronological holdout is
#   rejected as the PRIMARY evaluation design in favour of a stratified
#   random 80/20 split.
#
# Rationale:
#   A temporal holdout evaluates patients treated more recently -- but in
#   this file, more-recently-treated patients are disproportionately
#   likely to still be within their 730-day follow-up window at the
#   apparent follow-up cutoff in the offset date fields (`RT Start` /
#   `Last FU` are documented as date-offset values, not literal
#   real-world calendar dates), i.e. disproportionately excluded from the
#   eligible cohort. This makes a naive temporal test set both much
#   smaller and differently censored than its training period, confounding
#   any observed performance difference between "temporal shift" and
#   "differential censoring selection".
#
# Assumptions:
#   The chronological boundary is defined on the FULL file (n=3346),
#   before any eligibility filtering -- computing it after filtering would
#   itself be a censoring-dependent (and therefore circular) procedure.

section("SECTION 3.10 — Split-strategy rationale (temporal holdout rejected)")


def chronological_cutoff(dates: pd.Series, train_fraction: float) -> pd.Timestamp:
    """Return the smallest date D such that the INCLUSIVE cumulative count
    of `dates <= D` reaches at least `train_fraction` of all records,
    computed on the FULL cohort (never on an already-filtered one).

    The caller then splits with `historical = dates < D` (strict) and
    `later = dates >= D`. Because D itself is chosen from an inclusive
    count but the split excludes D from "historical", every record
    sharing exactly date D is kept together in the "later" group rather
    than being split across the boundary -- so the realised historical
    share can be marginally below `train_fraction` whenever multiple
    records share date D.
    """
    cumulative = dates.value_counts().sort_index().cumsum()
    target_n = int(np.ceil(train_fraction * len(dates)))
    return cumulative[cumulative >= target_n].index[0]


cutoff_date = chronological_cutoff(raw_df["RT Start"], train_fraction=1 - config.TEST_SIZE)
historical_mask = raw_df["RT Start"] < cutoff_date  # boundary date D itself goes to "later" (see docstring)
temporal_test_mask = ~historical_mask

# `== 1` on the nullable Int64 target column yields pandas' nullable
# boolean dtype, with <NA> for the (excluded) rows whose outcome is
# unobservable; `.fillna(False)` turns that into an ordinary boolean mask
# without ever treating "unobservable" as "non-event".
is_event_full = (raw_df[config.TARGET_COLUMN] == 1).fillna(False)

for label, mask in [("historical (train-like)", historical_mask), ("temporal test-like", temporal_test_mask)]:
    n = int(mask.sum())
    n_excl = int((~eligible_mask & mask).sum())
    n_elig = n - n_excl
    n_ev = int((is_event_full & mask).sum())
    print(
        f"   {label:26s} n={n:5d}  censored={n_excl:4d} ({n_excl/n*100:5.1f}%)  "
        f"eligible={n_elig:5d}  events={n_ev:4d}  prevalence={n_ev/n_elig*100:5.2f}%"
    )

print(f"\nChronological 80/20 boundary in the offset RT-Start dates: {cutoff_date.date()}")
print("The temporal test-like period loses a disproportionate share of patients to censoring")
print("relative to the historical period -- confirming the Section 3.10 methodology finding.")

# Result / Interpretation:
#   The later (temporal test-like) period shows a far higher censoring
#   rate than the historical period, because patients treated close to the
#   latest observed offset follow-up date in this clinical extract have
#   not had time to accrue 730 days of follow-up. A test set built this
#   way is not just smaller but
#   systematically different in composition from its own training period
#   -- a confound a temporal holdout cannot resolve on its own, and one
#   that a stratified random split (which mixes all treatment eras into
#   both sets) does not create. No Kaplan-Meier or other survival-analysis
#   method was used to reach this conclusion -- only the counts above.
# Decision:
#   Use a stratified random 80/20 holdout as the PRIMARY evaluation split
#   (implemented in Section 5 below). A temporal holdout is not
#   implemented in this project.
# Next step:
#   Run the feature leakage audit (Section 3.11) before any predictor is
#   admitted to X.


# %%
# =============================================================================
# SECTION 3.11 — Feature leakage audit
# =============================================================================
# Objective:
#   Classify every one of the 34 original columns exactly once against the
#   admission test: "If a new patient is immediately about to receive
#   their first radiotherapy fraction, could this variable legitimately be
#   known and used at that moment?" -- and derive the candidate/excluded
#   predictor lists PROGRAMMATICALLY from that classification, so the
#   audit's conclusions and the code's behaviour cannot silently diverge.
#
# Rationale:
#   The full table, its taxonomy, and the reasoning behind every row are
#   documented in `src/radcure/leakage.py`. This section only verifies its
#   coverage and prints the resulting candidate/excluded lists.

section("SECTION 3.11 — Feature leakage audit")

leakage.assert_full_coverage(original_raw_columns)
print(f"[OK] Leakage audit covers all {config.EXPECTED_RAW_COLUMNS} original raw columns exactly once.")

print(f"\nCANDIDATE (temporally admissible) predictors: {len(leakage.CANDIDATE_PREDICTORS)}")
for v in leakage.CANDIDATE_PREDICTORS:
    print(f"   {v!r}")

print(f"\nEXCLUDED (leakage / proxy) variables: {len(leakage.EXCLUDED_VARIABLES)}")
print(pd.DataFrame(
    {"variable": leakage.EXCLUDED_VARIABLES}
).merge(
    leakage.LEAKAGE_AUDIT_TABLE[["variable", "category"]], on="variable", how="left"
).to_string(index=False))

assert set(leakage.CANDIDATE_PREDICTORS) == config.EXPECTED_CANDIDATE_PREDICTORS, (
    "Derived candidate-predictor set no longer matches the Section 3.11 methodology review."
)
print("\n[OK] Candidate-predictor set matches the confirmed 13-variable Section 3.11 outcome.")

# Explicit statement (not merely an omission): RT Start, RADCURE-challenge
# and ContrastEnhanced are each excluded as a PROXY, never as direct
# target leakage.
for proxy_var, expected_category in [
    ("RT Start", "CALENDAR_ERA_PROXY"),
    ("RADCURE-challenge", "DATASET_MEMBERSHIP_PROXY"),
    ("ContrastEnhanced", "TECHNICAL_PROTOCOL_PROXY"),
]:
    row = leakage.LEAKAGE_AUDIT_TABLE.loc[leakage.LEAKAGE_AUDIT_TABLE["variable"] == proxy_var].iloc[0]
    assert row["category"] == expected_category and row["decision"] == "EXCLUDE"
    assert row["category"] != "DIRECT_TARGET_LEAKAGE"
print("[OK] RT Start / RADCURE-challenge / ContrastEnhanced are documented as proxies, not target leakage.")

# Result / Interpretation:
#   13 of 34 variables pass the temporal-availability audit; 21 are
#   excluded for one of: identifier, direct target leakage, follow-up
#   information, a post-landmark clinical event, post-landmark treatment
#   information, mixed temporal information, a dataset-membership proxy,
#   a calendar-era proxy, or a technical-protocol proxy.
# Decision:
#   Only the 13 CANDIDATE variables may be considered for X. Passing this
#   audit does not imply inclusion -- Section 3.12 selects a parsimonious
#   subset on clinical, not statistical, grounds.
# Next step:
#   Define the PRIMARY predictor set (Section 3.12).


# %%
# =============================================================================
# SECTION 3.12 — Primary predictor set definition
# =============================================================================
# Objective:
#   Narrow the 13 leakage-safe candidates to the confirmed 8-variable
#   PRIMARY predictor set, and document -- for each of the 5 candidates
#   NOT admitted -- the specific, non-leakage scientific reason.
#
# Rationale (Section 3.12 methodology review; reasons corrected per that
# review, not merely repeated from an earlier, less precise draft):
#   - `M`       : near-zero variation -- in the eligible cohort (n=2939),
#                 2927 patients are M0, 10 are missing and 2 are MX, with
#                 no M1 patient at all.
#   - `Stage`   : redundant given Ds Site + T + N together. Stage is NOT
#                 claimed to be a deterministic function of TNM alone --
#                 it is a SITE-SPECIFIC recombination (confirmed
#                 empirically: 24 of 70 TNM combinations map to more than
#                 one Stage, and the ambiguity resolves along AJCC-7
#                 site-specific rules, e.g. nasopharyngeal T2N1M0 = Stage
#                 II vs. Stage III everywhere else). Using Stage alone
#                 would also lose substantial T/N granularity (Stage IVA
#                 alone spans ten T categories).
#   - `Subsite` : high cardinality (59 levels) and heterogeneous
#                 missingness: for carcinoma of unknown primary it is a
#                 STRUCTURAL absence (100% missing -- no identifiable
#                 primary subsite exists by definition of the entity);
#                 for nasopharynx it is substantial, site-dependent
#                 missing data with limited routine subdivision (~52%
#                 missing), not a universal structural impossibility.
#   - `Path`    : excluded primarily because its dominant clinical
#                 distinction (SCC vs. nasopharyngeal carcinoma) is highly
#                 redundant with `Ds Site` (307 of 308 NPC-histology
#                 records are nasopharyngeal, and zero nasopharyngeal
#                 records are SCC-like) -- NOT because of rarity alone.
#   - `HPV`     : availability at the prediction landmark cannot reliably
#                 be established for all historical patients (48.6%
#                 missing, and documentation is both site-dependent and
#                 era-dependent, with evidence some historical results
#                 were determined retrospectively). Reserved as a
#                 pre-specified sensitivity-only variable (Section 4,
#                 Task 13) -- NOT modelled in this phase.

section("SECTION 3.12 — Primary predictor set")

assert set(config.PRIMARY_FEATURES) <= set(leakage.CANDIDATE_PREDICTORS), (
    "A PRIMARY feature is not among the leakage-audit CANDIDATE predictors."
)
assert not (set(config.PRIMARY_FEATURES) & set(leakage.EXCLUDED_VARIABLES)), (
    "A PRIMARY feature is among the leakage-audit EXCLUDED variables."
)
print("[OK] Every PRIMARY feature is a leakage-audit CANDIDATE, and none is an EXCLUDED variable.")

print(f"\nPRIMARY predictors ({len(config.PRIMARY_FEATURES)}):")
print(f"   numeric     : {config.PRIMARY_NUMERIC_FEATURES}")
print(f"   categorical : {config.PRIMARY_CATEGORICAL_FEATURES}")

not_admitted = sorted(set(leakage.CANDIDATE_PREDICTORS) - set(config.PRIMARY_FEATURES))
primary_exclusion_reasons = {
    "M ": "Near-zero variation: 2927/2939 eligible patients are M0; the remaining "
          "records are 10 missing and 2 MX, with no M1 patient in the eligible "
          "modelling cohort.",
    "Stage": "Redundant given Ds Site + T + N together (a site-specific TNM "
             "recombination, not a simple deterministic function of TNM alone); "
             "Stage-only would lose substantial T/N granularity.",
    "Subsite": "High cardinality (59 levels); structurally absent for carcinoma of "
               "unknown primary (no identifiable subsite exists) and substantially "
               "site-dependent missing/under-subdivided for nasopharynx (~52%).",
    "Path": "Its dominant clinical distinction is highly redundant with Ds Site "
            "(not excluded because of rarity alone).",
    "HPV": "Availability at the prediction landmark cannot reliably be established "
           "for all historical patients; missing documentation is site- and "
           "era-dependent. Reserved as a pre-specified sensitivity-only variable.",
}
assert set(not_admitted) == set(primary_exclusion_reasons), "Exclusion-reason table is out of sync with the predictor lists."
print(f"\nCandidate predictors NOT admitted to PRIMARY ({len(not_admitted)}):")
for v in not_admitted:
    status = "SENSITIVITY_ONLY" if v == config.SENSITIVITY_ONLY_FEATURE else "descriptive-only"
    print(f"   {v!r:12s} [{status:17s}] {primary_exclusion_reasons[v]}")

# Result / Interpretation:
#   8 of 13 candidates form the PRIMARY predictor set: Age, Sex, ECOG PS,
#   Smoking PY, Smoking Status, Ds Site, T, N. M, Stage, Subsite and Path
#   are retained in the working data for cohort description only. HPV is
#   reserved for one pre-specified PRIMARY + HPV sensitivity comparison,
#   not implemented in this phase.
# Decision:
#   X will contain exactly the 8 PRIMARY features. No automatic feature
#   selection is applied because the PRIMARY model already contains only
#   eight pre-specified, domain-informed predictors (Section 3.12). The
#   encoded representation remains manageable and can be handled through
#   regularisation and cross-validation.
# Next step:
#   Apply deterministic semantic cleaning to every PRIMARY predictor
#   (Section 4).


# %%
# =============================================================================
# SECTION 4 — Deterministic semantic cleaning of the PRIMARY predictors
# =============================================================================
# Objective:
#   Apply the approved, purely semantic cleaning rules to each PRIMARY
#   predictor (plus HPV, for the later sensitivity representation), and
#   make the raw -> cleaned effect visible: every raw level, every cleaned
#   level, and the resulting change in "not usable" counts.
#
# Rationale:
#   Every rule below repairs how a value was WRITTEN (case, an explicit
#   unknown/unassessable code, a non-numeric token), based on the Data
#   Dictionary together with the findings of the completed data-quality
#   audit (Sections 3.8-3.12). These are fixed semantic/coding rules; they
#   do not estimate data-dependent preprocessing parameters such as
#   imputation values, scaling statistics or category-frequency
#   thresholds, which is why all of it may run on the full file, before
#   the train/test split.
#
# Assumptions:
#   The exact mapping dictionaries are fixed constants in
#   `src/radcure/config.py`, not re-derived here, so this section is a
#   transparent APPLICATION and VERIFICATION of already-approved rules,
#   not a place where new cleaning decisions are made.

section("SECTION 4 — Deterministic semantic cleaning")

clinical_df = raw_df.copy()

# --- Age: no cleaning needed (Section 4, Task 1) -----------------------------
# Age is continuous -- a full value_counts dump would be uninformative
# noise, so a compact numeric summary is used instead (consistent with
# keeping new EDA compact).
print("\nAge (n=3346, missing=0) — modelled as continuous, no binning:")
print(raw_df["Age"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_string())
n_top_coded_90 = int((raw_df["Age"] == 90.0).sum())
n_under_18 = int((raw_df["Age"] < 18).sum())
print("\nAge: 0 missing, no deterministic cleaning required.")
print(f"   {n_top_coded_90} patients have Age == 90.0, consistent with age top-coding / "
      f"de-identification; values are retained without clipping.")
print(f"   Patients under 18 retained without exclusion: {n_under_18}.")
assert raw_df["Age"].isna().sum() == 0

# --- Sex: no cleaning needed --------------------------------------------------
print_value_counts(raw_df["Sex"], "Sex")
assert set(raw_df["Sex"].dropna().unique()) == {"Male", "Female"}

# --- ECOG PS ------------------------------------------------------------------
print_value_counts(raw_df["ECOG PS"], "ECOG PS (RAW)")
clinical_df["ECOG PS"] = cleaning.clean_ecog_ps(raw_df["ECOG PS"])
print_value_counts(clinical_df["ECOG PS"], "ECOG PS (CLEANED)")

# --- Smoking PY ---------------------------------------------------------------
print_value_counts(raw_df["Smoking PY"], "Smoking PY (RAW, showing every non-numeric token)")
clinical_df["Smoking PY"] = cleaning.clean_smoking_py(raw_df["Smoking PY"])
print(
    "\nSmoking PY (CLEANED, numeric): "
    f"n_valid={int(clinical_df['Smoking PY'].notna().sum())}, "
    f"missing={int(clinical_df['Smoking PY'].isna().sum())}, "
    f"min={clinical_df['Smoking PY'].min():.1f}, "
    f"median={clinical_df['Smoking PY'].median():.1f}, "
    f"max={clinical_df['Smoking PY'].max():.1f}"
)
assert pd.api.types.is_float_dtype(clinical_df["Smoking PY"]), "Smoking PY must be numeric after cleaning."

# --- Smoking Status -------------------------------------------------------
print_value_counts(raw_df["Smoking Status"], "Smoking Status (RAW)")
clinical_df["Smoking Status"] = cleaning.clean_smoking_status(raw_df["Smoking Status"])
print_value_counts(clinical_df["Smoking Status"], "Smoking Status (CLEANED)")

# --- Ds Site --------------------------------------------------------------
# Note on level counts, so the two independent effects are not conflated:
#   19 raw labels (full file)
#   -> 17 levels after deterministic case/spelling harmonisation
#      ('nasal cavity'->'Nasal Cavity', 'esophagus'->'Esophagus') -- this
#      is the ONLY change cleaning makes here.
#   -> 15 levels observed in the ELIGIBLE cohort (Section 5) -- this
#      further reduction is NOT cleaning: `Orbit` and `Lacrimal gland`
#      (one raw patient each) happen to fall entirely within the 407
#      patients excluded for target-unobservability, so they are simply
#      absent from the eligible cohort's distribution, not removed by any
#      cleaning or cohort-definition rule.
n_ds_site_raw = raw_df["Ds Site"].str.strip().nunique()
print_value_counts(raw_df["Ds Site"], "Ds Site (RAW, full cohort n=3346)")
clinical_df["Ds Site"] = cleaning.clean_ds_site(raw_df["Ds Site"])
print_value_counts(clinical_df["Ds Site"], "Ds Site (CLEANED, full cohort — case duplicates merged only)")
assert clinical_df["Ds Site"].nunique() == n_ds_site_raw - 2, (
    "Expected exactly the two known case-duplicate pairs "
    "('nasal cavity'/'Nasal Cavity', 'esophagus'/'Esophagus') to merge; "
    f"raw levels={n_ds_site_raw}, cleaned levels={clinical_df['Ds Site'].nunique()}."
)

# --- T category -------------------------------------------------------------
print_value_counts(raw_df["T"], "T (RAW)")
clinical_df["T"] = cleaning.clean_t_category(raw_df["T"])
print_value_counts(clinical_df["T"], "T (CLEANED)")

# --- N category -------------------------------------------------------------
print_value_counts(raw_df["N"], "N (RAW)")
clinical_df["N"] = cleaning.clean_n_category(raw_df["N"])
print_value_counts(clinical_df["N"], "N (CLEANED)")

# --- HPV (sensitivity-only representation; not part of X in this phase) -----
print_value_counts(raw_df["HPV"], "HPV (RAW)")
clinical_df["HPV"] = cleaning.clean_hpv(raw_df["HPV"])
print_value_counts(clinical_df["HPV"], "HPV (CLEANED — deliberately NOT labelled 'not tested')")

# --- Missingness before/after, on the ELIGIBLE cohort (the modelling frame) --
missing_spec = {
    "Age": None,
    "Sex": None,
    "ECOG PS": config.MISSING_LABEL,
    "Smoking PY": None,
    "Smoking Status": "Unknown",
    "Ds Site": None,
    "T": config.MISSING_LABEL,
    "N": config.MISSING_LABEL,
    "HPV": config.HPV_MISSING_LABEL,
}
missingness_table = cleaning.missingness_summary(
    raw_df=raw_df.loc[eligible_mask],
    cleaned_df=clinical_df.loc[eligible_mask],
    missing_token_by_column=missing_spec,
)
print(f"\nMissingness before vs. after deterministic cleaning (ELIGIBLE cohort, n = {int(eligible_mask.sum())}):")
print(missingness_table.to_string(index=False))

# Result / Interpretation:
#   Deterministic cleaning increases recorded "not usable" counts for
#   ECOG PS, T, N and Smoking PY (as expected: it converts values that
#   only LOOKED present -- `Unknown`, `TX`, `na`, a bound token -- into
#   honest missing values). Every PRIMARY predictor remains under 1.6%
#   missing after cleaning. `Ds Site` goes from 19 raw labels to 17
#   cleaned levels via the two known case-duplicate merges; the further
#   17-to-15 reduction observed in the eligible cohort is a consequence of
#   target observability (Orbit/Lacrimal gland), not of cleaning -- no
#   clinically distinct or rare-but-valid category was pooled or removed.
# Decision:
#   `clinical_df` now holds the semantically cleaned version of every
#   PRIMARY predictor (plus HPV). It is safe to select X/y and split, since
#   nothing above estimated a statistical parameter from the sample.
# Next step:
#   Assemble X, y, and patient_id from the ELIGIBLE cohort, then create
#   and freeze the primary train/test split (Section 5).


# %%
# =============================================================================
# SECTION 5 — Assemble X / y / patient_id, and the primary train/test split
# =============================================================================
# Objective:
#   Build the modelling frame from the eligible cohort only, guard it
#   against any leakage-excluded column, and create the frozen stratified
#   80/20 train/test split.
#
# Rationale:
#   The order of operations matters: eligibility and predictor selection
#   are both determined from data that is available BEFORE the split
#   (target construction and the leakage audit / predictor-set definition
#   estimate no statistical parameter), so both may safely precede the
#   split. The split is stratified on y to keep the ~18.9% prevalence
#   comparable between train and test.

section("SECTION 5 — X / y / patient_id assembly and the primary split")

cohort_df = clinical_df.loc[eligible_mask].reset_index(drop=True)

X = cohort_df[config.PRIMARY_FEATURES].copy()
y = cohort_df[config.TARGET_COLUMN].astype(int)
patient_id = cohort_df[config.PATIENT_ID_COLUMN].copy()

# HPV sensitivity representation is carried alongside X (not inside it) so
# it can be reused later without recomputation -- it is NOT modelled here.
hpv_sensitivity_feature = cohort_df["HPV"].copy()

print(f"X shape: {X.shape}   y length: {len(y)}   patient_id length: {len(patient_id)}")
assert X.shape == (config.EXPECTED_ELIGIBLE_N, len(config.PRIMARY_FEATURES))
assert list(X.columns) == config.PRIMARY_FEATURES

# --- Guard: no leakage-excluded column may be present in X -------------------
leakage.assert_no_excluded_column_in_frame(X.columns)
print("[OK] No leakage-excluded variable is present in X.")

# --- Guard: patient_id uniqueness within the modelling cohort ----------------
assert patient_id.is_unique, "patient_id is not unique within the eligible cohort."
print(f"[OK] patient_id is unique across all {len(patient_id)} eligible patients.")

# --- Guard: final category sets match the approved cleaning specification ---
assert set(X["ECOG PS"].unique()) <= (config.ECOG_PS_VALID_LEVELS | {config.MISSING_LABEL})
assert set(X["T"].unique()) <= (config.T_VALID_LEVELS | {config.MISSING_LABEL})
assert set(X["N"].unique()) <= (config.N_VALID_LEVELS | {config.MISSING_LABEL})
assert X["Ds Site"].nunique() == config.EXPECTED_DS_SITE_LEVELS
assert pd.api.types.is_float_dtype(X["Smoking PY"]) and pd.api.types.is_float_dtype(X["Age"])
print("[OK] Final category sets for ECOG PS / T / N / Ds Site match the approved cleaning specification.")
print("[OK] Age and Smoking PY are numeric with no residual non-numeric tokens.")

# --- Primary stratified 80/20 split ------------------------------------------
(
    X_train, X_test,
    y_train, y_test,
    id_train, id_test,
) = train_test_split(
    X, y, patient_id,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE,
)

print(f"\nTrain: n={len(X_train)}  events={int(y_train.sum())}  non-events={int((y_train == 0).sum())}  "
      f"prevalence={y_train.mean()*100:.2f}%")
print(f"Test : n={len(X_test)}  events={int(y_test.sum())}  non-events={int((y_test == 0).sum())}  "
      f"prevalence={y_test.mean()*100:.2f}%")

assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
assert int((y_train == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS
assert len(X_test) == config.EXPECTED_TEST_N
assert int(y_test.sum()) == config.EXPECTED_TEST_EVENTS
assert int((y_test == 0).sum()) == config.EXPECTED_TEST_NONEVENTS
print("[OK] Train/test sizes and event counts match the confirmed study design.")

assert set(id_train) & set(id_test) == set(), "Train and test patient_id sets are not disjoint!"
print(f"[OK] Train and test patient_id sets are disjoint ({len(id_train)} + {len(id_test)} = "
      f"{len(id_train) + len(id_test)} unique patients).")

# ------------------------------------------------------------------------------
# >>> THE HELD-OUT TEST SET IS NOW FROZEN. <<<
# X_test / y_test / id_test must not be inspected, transformed, or used to
# inform any decision (imputation, scaling, encoding, model choice,
# hyperparameters, threshold) until the single, final evaluation of the
# finished PRIMARY model. Nothing below this line touches them again.
# ------------------------------------------------------------------------------

# Result / Interpretation:
#   2351 training / 588 test patients, 444 / 111 events, prevalence 18.89%
#   / 18.88% -- balanced by construction. Every structural guard (no
#   excluded column in X, unique patient_id, valid final category sets,
#   disjoint train/test identifiers) passes.
# Decision:
#   The split is frozen. All subsequent preprocessing-parameter estimation
#   must be fitted on X_train (or inside CV folds drawn from it) only.
# Next step:
#   Construct (but do NOT fit) the PRIMARY preprocessing pipeline objects
#   (Section 6).


# %%
# =============================================================================
# SECTION 6 — PRIMARY preprocessing pipeline objects (construction only)
# =============================================================================
# Objective:
#   Build the ColumnTransformer that will later preprocess the PRIMARY
#   features, and verify -- by attempting to use it -- that it truly holds
#   no fitted state yet.
#
# Rationale:
#   Building an (unfitted) scikit-learn object estimates nothing and is
#   therefore leakage-safe regardless of the split. Fitting it is a
#   learned-preprocessing operation and belongs strictly inside the next
#   phase (model training / cross-validation), applied only to X_train or
#   to CV folds drawn from X_train -- never here, and never to X_test.

section("SECTION 6 — PRIMARY preprocessing pipeline (construction only, NOT fitted)")

preprocessor = preprocessing.build_preprocessor()
print("Constructed (unfitted) ColumnTransformer:")
print(preprocessor)

try:
    check_is_fitted(preprocessor)
    raise RuntimeError("Preprocessor unexpectedly reports as fitted -- investigate before proceeding.")
except NotFittedError:
    print("\n[OK] Preprocessor is confirmed UNFITTED (NotFittedError raised by check_is_fitted).")

try:
    preprocessor.transform(X_train.head())
    raise RuntimeError("Preprocessor unexpectedly transformed data without being fitted.")
except NotFittedError:
    print("[OK] Calling .transform() before .fit() correctly raises NotFittedError.")

print(
    "\nNumeric branch    : SimpleImputer(strategy='median', add_indicator=False) -> StandardScaler()"
    f"\n                    columns = {config.PRIMARY_NUMERIC_FEATURES}"
    "\nCategorical branch: OneHotEncoder(handle_unknown='ignore', drop=None)"
    f"\n                    columns = {config.PRIMARY_CATEGORICAL_FEATURES}"
)
print(
    "\nNote (drop=None): no reference level is omitted from any categorical column, so a "
    "future fitted coefficient must NOT be read as a simple 'category vs. one omitted "
    "reference category' effect -- every level, including the most frequent one, carries "
    "its own coefficient."
)

# Result / Interpretation:
#   The PRIMARY preprocessing pipeline is fully specified and demonstrably
#   unfitted. No rare-category pooling is included: all clinically valid
#   categories are retained, relying on `handle_unknown="ignore"` plus
#   later model regularisation (Section 4, Task 12 -- the EPV>=10 pooling
#   rationale from an earlier draft was withdrawn on methodological review
#   and is not implemented).
# Decision:
#   This pipeline object is "ready for later modelling" in the sense
#   required by this phase: constructed, verified, not fitted. Fitting it
#   -- inside a training Pipeline together with an estimator, under
#   StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on X_train
#   only -- is explicitly OUT OF SCOPE for this phase.
# Next step (NOT implemented in this script):
#   Model training and cross-validation on X_train / y_train, with X_test
#   / y_test remaining untouched until the single final evaluation.


# %%
# =============================================================================
# SECTION 7 — Foundation summary and explicit stop
# =============================================================================
section("SECTION 7 — Foundation complete: STOP before model fitting")

print("""
Completed in this script:
  [x] Immutable raw data loaded and structurally verified (3346 x 34).
  [x] Prediction landmark and outcome time origin fixed at RT Start;
      zero deaths precede the landmark; the reconstructed diagnosis date
      corroborates against the RADCURE-challenge cohort's observed
      temporal separation (zero overlap); the diagnosis-anchored vs.
      RT-Start-anchored design comparison and the fixed-730-day boundary
      check both reproduce their confirmed counts.
  [x] RT-Start-anchored 730-day mortality target built and eligibility
      partition confirmed (2939 eligible / 555 events / 2384 non-events /
      407 excluded as target-unobservable).
  [x] Cohort definition restricted ONLY by target observability -- no
      site, histology, or frequency-based exclusion.
  [x] Naive chronological holdout rejected as PRIMARY design, documented
      with simple censoring counts/rates only (no survival analysis).
  [x] Feature leakage audit covering all 34 original columns exactly once;
      13 candidates derived programmatically; RT Start / RADCURE-challenge
      / ContrastEnhanced confirmed as proxies, not direct target leakage.
  [x] PRIMARY predictor set fixed at 8 variables (Age, Sex, ECOG PS,
      Smoking PY, Smoking Status, Ds Site, T, N); 5 non-admitted
      candidates (M, Stage, Subsite, Path, HPV) documented with their
      specific reasons; HPV reserved sensitivity-only.
  [x] Deterministic semantic cleaning applied and verified for every
      PRIMARY predictor (plus HPV).
  [x] X / y / patient_id assembled from the eligible cohort; guarded
      against excluded columns, duplicate IDs and unexpected categories.
  [x] Stratified 80/20 train/test split created and FROZEN
      (2351 / 588 patients, 444 / 111 events).
  [x] PRIMARY preprocessing pipeline constructed and confirmed UNFITTED.

Explicitly NOT done in this script (out of scope for this phase):
  [ ] No model has been fitted.
  [ ] No hyperparameter has been tuned.
  [ ] No cross-validation has been run.
  [ ] The held-out test set has not been inspected beyond its size and
      class counts.
  [ ] The PRIMARY + HPV sensitivity model has not been built.

STOP. Awaiting review before proceeding to model training.
""")
