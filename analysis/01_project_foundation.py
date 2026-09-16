# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 01 Project Foundation
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

This is fixed-horizon binary classification; time-to-event modelling is
outside the scope of this project.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT PRODUCES
--------------------------------------------------------------------------------
This is the first stage of a sequential pipeline (01 -> 07). It turns the
immutable raw workbook into the project's single source of truth:

    data/processed/modeling_cohort.csv

one row per eligible patient, carrying the patient identifier, the eight
semantically cleaned PRIMARY predictors, the binary target, and the frozen
train/test assignment. Every later script LOADS that artefact instead of
re-deriving the cohort, so the split is created exactly once, here.

Sections:

    2.   Data loading and compact ML-relevant verification
    3.1  Prediction landmark and outcome definition
    3.2  Target construction and cohort definition
    3.3  Split-strategy rationale
    3.4  Feature leakage audit
    3.5  Primary predictor set definition
    4.   Deterministic semantic cleaning of the PRIMARY predictors
    5.   Modelling cohort, the frozen split, and the persisted artefact
    6.   Construction (NOT fitting) of the PRIMARY preprocessing pipeline

--------------------------------------------------------------------------------
GOVERNING METHODOLOGICAL RULES
--------------------------------------------------------------------------------
  - Raw data is immutable: this script only READS the source workbook.
  - Two kinds of operation are kept strictly separate:
        (A) deterministic semantic cleaning — estimates NO statistical
            parameter, so it may run before the train/test split
            (`src/radcure/cleaning.py`);
        (B) learned preprocessing — estimates a statistical parameter (an
            imputation median, a scaler's mean/variance, an encoder's
            category list) and is fitted only inside the training pipeline /
            cross-validation, never before the split and never on the
            held-out test set (`src/radcure/preprocessing.py`).
  - No model is fitted and no predictor-outcome association is examined here.

A comprehensive exploratory data analysis of this dataset was completed in a
separate project (https://github.com/cabait00/radcure-eda) and is not
repeated here; only compact checks needed for the modelling decisions below
are re-run.

Confirmed project environment: `ml` conda environment
(/home/c/miniconda3/envs/ml/bin/python) -- pandas 3.0.5, numpy 2.4.6,
scikit-learn 1.9.0, openpyxl 3.1.5, pytest 9.1.1.
"""

# %%
# =============================================================================
# SECTION 1 — Imports, environment, reusable narrative helpers
# =============================================================================
# Objective:
#   Make the `radcure` package (src/radcure/) importable from this script and
#   define the small display helpers the narrative below uses.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import (  # noqa: E402
    artifacts,
    cleaning,
    cohort,
    config,
    leakage,
    plots,
    preprocessing,
    target,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 120)
plots.apply_project_style()


def print_value_counts(series: pd.Series, title: str) -> None:
    """Print every raw level of a series with its count, making whitespace
    and case variants visible (via repr()) rather than hiding them behind a
    summary statistic.
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
# SECTION 2 — Data loading and compact ML-relevant verification
# =============================================================================
# Objective:
#   Load the immutable raw workbook and verify, in code, the structural facts
#   the downstream design depends on.
#
# Rationale:
#   `cleaning.load_raw_clinical` only reads the file; the raw workbook is
#   never written to. The verification is deliberately limited to what is
#   needed to proceed safely -- shape, identifier uniqueness, presence of the
#   Data Dictionary -- because the full exploratory analysis belongs to the
#   separate EDA project.

section("SECTION 2 — Data loading and verification")

raw_df = cleaning.load_raw_clinical()
data_dictionary = cleaning.load_data_dictionary()

# Captured before `raw_df` is joined with target columns below: this is the
# list the Section 3.4 leakage audit must cover, i.e. the 34 ORIGINAL columns.
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

# Result:
#   3346 rows x 34 columns, one row per patient, patient_id unique, no
#   duplicated rows -- consistent with the EDA project's findings.
# Next:
#   Fix the prediction landmark and outcome definition (Section 3.1) before
#   any predictor is touched, since modelling eligibility depends on the
#   outcome, not the reverse.


# %%
# =============================================================================
# SECTION 3.1 — Prediction landmark and outcome definition
# =============================================================================
# Objective:
#   Fix the prediction landmark and the outcome time origin, and verify the
#   structural facts that justify anchoring both at `RT Start`.
#
# Rationale:
#   The landmark is "immediately before the first radiotherapy fraction",
#   operationalised by the observed `RT Start` date -- the only exactly dated
#   event in the file corresponding to that clinical moment. The outcome
#   clock starts at the same point, so every patient is given the IDENTICAL
#   730-day prospective horizon measured from the moment the prediction is
#   actually made. A diagnosis-anchored alternative is evaluated below (with
#   executable counts, not merely asserted) and rejected: it leaves a
#   patient-specific, variable remainder of the 730-day window after the
#   actual landmark, so it does not match the intended prospective question.
#
# Why two years:
#   One year was considered relatively short for the intended prediction
#   problem. Horizons substantially beyond two years materially reduce
#   outcome observability in this dataset and increase the number of patients
#   excluded for insufficient follow-up (Section 3.3 shows how steeply that
#   bites for later-treated patients). Two years is therefore a practical
#   compromise here between a meaningful prediction horizon and adequate
#   outcome observability -- not a universal or clinically optimal mortality
#   horizon.
#
# Assumptions:
#   `RT Start`, `Date of Death` and `Last FU` are directly OBSERVED date
#   columns. A separate DERIVED diagnosis date is used below only for the
#   design comparison and one corroborating check -- never for the confirmed
#   outcome, and never as a predictor.

section("SECTION 3.1 — Prediction landmark and outcome definition")

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
# (`Length FU`) using the dataset's apparent 365-day encoding -- it is NOT a
# directly observed date. It is used only to report the typical workup
# interval and to corroborate the reconstruction against the
# RADCURE-challenge cohort's observed temporal separation. `RADCURE-challenge`
# is used exclusively for that one check and never becomes a predictor
# (Section 3.4 classifies it as a DATASET_MEMBERSHIP_PROXY).
diagnosis_date = target.reconstruct_diagnosis_date(raw_df)
gap_diagnosis_to_rt = (raw_df["RT Start"] - diagnosis_date).dt.days

print("\nDiagnosis-to-RT-Start gap (DERIVED diagnosis date; diagnostic only, not a predictor):")
print(gap_diagnosis_to_rt.describe(percentiles=[0.25, 0.5, 0.75]).round(1).to_string())

challenge = raw_df["RADCURE-challenge"].astype(str)
diag_train_max = diagnosis_date[challenge == "training"].max()
diag_test_min = diagnosis_date[challenge == "test"].min()
n_overlap = int(
    (diagnosis_date[challenge == "training"] >= diag_test_min).sum()
    + (diagnosis_date[challenge == "test"] <= diag_train_max).sum()
)
print("\nRADCURE-challenge temporal-separation corroboration check:")
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
      "training/test temporal separation with zero overlap. This CORROBORATES the "
      "reconstruction; it does not prove it exact.")

# --- Structural check 3: diagnosis-anchored vs. RT-Start-anchored endpoint ---
# `diagnosis_target_df` is a COMPARISON-ONLY construct, never used for X/y.
diagnosis_target_df = target.build_target_diagnosis_anchored(raw_df)
rt_target_df_preview = target.build_target(raw_df)

diag_summary = target.target_summary(
    diagnosis_target_df,
    eligibility_col="diagnosis_anchored_eligible",
    target_col="diagnosis_anchored_mortality_2y",
)
rt_summary_preview = target.target_summary(rt_target_df_preview)

print("\nDesign comparison -- diagnosis-anchored vs. RT-Start-anchored 730-day endpoint:")
print("   diagnosis-anchored (COMPARISON ONLY -- never used for X/y):")
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

# Patient-level transitions between the two designs -- the concrete evidence
# that justifies the RT-Start anchor.
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
print("[OK] All four transition counts match the confirmed design comparison.")

print(
    "\nInterpretation: the diagnosis-anchored endpoint measures a fixed 730-day window from "
    "diagnosis, so the fraction of that window still remaining AFTER the actual prediction "
    "landmark varies patient by patient. Anchoring the outcome clock at RT Start instead "
    "gives every patient the identical 730-day prospective horizon."
)

# --- Structural check 4: the fixed 730-day boundary, and fixed vs. calendar --
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
      "identically here; the fixed-day rule is retained for its unambiguous, "
      "leap-year-independent specification.")

# Result:
#   Every deceased patient died after RT Start (minimum 23 days); the median
#   diagnosis-to-RT gap is ~43 days. The diagnosis-anchored comparison
#   reproduces its counts (2968/532/2436/378) against the confirmed RT-Start
#   design (2939/555/2384/407), and the four transition counts
#   (23 / 99 days / 29 / 3) show concretely that the diagnosis anchor gives
#   unequal remaining horizons after the actual landmark.
# Decision:
#   Anchor both the landmark and the outcome clock at `RT Start`, with a
#   fixed 730-day horizon, using only observed dates.
# Next:
#   Build the confirmed target and its eligibility partition (Section 3.2).


# %%
# =============================================================================
# SECTION 3.2 — Target construction and cohort definition
# =============================================================================
# Objective:
#   Build the RT-Start-anchored binary two-year mortality target with an
#   explicit eligibility partition, and confirm the resulting counts.
#
# Rationale:
#   A patient's two-year status is genuinely unobservable if they are alive
#   with less than 730 days of documented follow-up after `RT Start`. Such
#   patients must never be silently assigned to class 0, so eligibility is
#   tracked as an explicit flag rather than a silent row drop.
#
#   Target observability is the ONLY restriction applied to the cohort:
#   rarity, unusual histology and uncommon disease site are not grounds to
#   exclude a patient.

section("SECTION 3.2 — Target construction and cohort definition")

target_df = target.build_target(raw_df)

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

eligible_mask = target_df[config.ELIGIBILITY_COLUMN]

# A site can still end up with zero eligible patients if its only
# occurrence(s) happen to be target-unobservable -- a property of those
# patients' follow-up, not of the site. The table shows this precisely.
site_counts_full = raw_df["Ds Site"].str.strip().value_counts()
site_counts_eligible = raw_df.loc[eligible_mask, "Ds Site"].str.strip().value_counts()
cohort_composition = pd.DataFrame(
    {"n_full_cohort": site_counts_full, "n_eligible_cohort": site_counts_eligible}
).fillna(0).astype(int)
print("\nDisease-site composition, full vs. eligible cohort:")
print(cohort_composition.sort_values("n_full_cohort", ascending=False).to_string())

sites_with_zero_eligible = cohort_composition.loc[cohort_composition["n_eligible_cohort"] == 0]
if not sites_with_zero_eligible.empty:
    print(f"\nSite(s) with zero eligible patients: {list(sites_with_zero_eligible.index)}")
    print("Absent from the eligible cohort only because their sole raw patient(s) are "
          "target-unobservable, NOT because of any cohort-definition rule targeting the site.")

# %%
# --- FIGURE: cohort and target flow ------------------------------------------
# Every number below is read from the computed summary, not hardcoded.
fig, (ax_flow, ax_class) = plt.subplots(
    1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]}
)

flow_labels = [
    f"Raw cohort\n(n = {summary['n_total']})",
    f"Eligible for modelling\n(n = {summary['n_eligible']})",
    f"Excluded: outcome not\nobservable (n = {summary['n_excluded']})",
]
flow_values = [summary["n_total"], summary["n_eligible"], summary["n_excluded"]]
flow_colors = [plots.NEUTRAL_COLOR, plots.NON_EVENT_COLOR, plots.ACCENT_COLOR]

bars = ax_flow.barh(range(len(flow_values)), flow_values, color=flow_colors)
ax_flow.set_yticks(range(len(flow_labels)), labels=flow_labels)
ax_flow.invert_yaxis()
ax_flow.set_xlabel("Patients")
ax_flow.set_title("Cohort definition")
ax_flow.set_xlim(0, summary["n_total"] * 1.18)
for bar, value in zip(bars, flow_values):
    ax_flow.text(
        bar.get_width() + summary["n_total"] * 0.015, bar.get_y() + bar.get_height() / 2,
        f"{value}  ({value / summary['n_total'] * 100:.1f}%)", va="center", fontsize=9,
    )

class_labels = [
    "Non-event\n(alive at 2 years or\ndied > 730 d)",
    "Event\n(died <= 730 d after\nRT Start)",
]
class_values = [summary["n_nonevents"], summary["n_events"]]
class_bars = ax_class.bar(
    class_labels, class_values, color=[plots.NON_EVENT_COLOR, plots.EVENT_COLOR], width=0.6
)
ax_class.set_ylabel("Patients")
ax_class.set_title(f"Outcome in the eligible cohort (n = {summary['n_eligible']})")
ax_class.set_ylim(0, max(class_values) * 1.18)
for bar, value in zip(class_bars, class_values):
    ax_class.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_values) * 0.02,
        f"{value}\n({value / summary['n_eligible'] * 100:.1f}%)", ha="center", fontsize=9,
    )

fig.suptitle(
    "RADCURE two-year mortality: cohort flow and class balance",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "01_cohort_target_flow.png")

# Result:
#   2939 of 3346 patients (87.8%) are eligible; 555 events / 2384 non-events,
#   a prevalence of 18.88%. No disease site was excluded by a clinical rule.
# Decision:
#   The modelling cohort is the full RADCURE population restricted ONLY by
#   target observability.
# Next:
#   Motivate the evaluation split (Section 3.3).


# %%
# =============================================================================
# SECTION 3.3 — Split-strategy rationale
# =============================================================================
# Objective:
#   Document, with simple counts and rates only, why a naive chronological
#   holdout is rejected in favour of a stratified random 80/20 split.
#
# Rationale:
#   A temporal holdout would evaluate patients treated more recently -- but in
#   this file, more-recently-treated patients are disproportionately likely to
#   still be within their 730-day window at the apparent follow-up cutoff, so
#   they are disproportionately excluded from the eligible cohort. A naive
#   temporal test set is therefore not just smaller but differently censored
#   than its own training period, which confounds "temporal shift" with
#   "differential censoring selection". A stratified random split mixes all
#   treatment eras into both partitions and does not create that confound.
#
# Assumptions:
#   The chronological boundary is computed on the FULL file (n=3346), before
#   eligibility filtering -- computing it afterwards would itself be a
#   censoring-dependent, circular procedure.

section("SECTION 3.3 — Split-strategy rationale (temporal holdout rejected)")


def chronological_cutoff(dates: pd.Series, train_fraction: float) -> pd.Timestamp:
    """Return the smallest date D such that the INCLUSIVE cumulative count of
    `dates <= D` reaches at least `train_fraction` of all records, computed on
    the FULL cohort.

    The caller then splits with `historical = dates < D` (strict) and
    `later = dates >= D`, so every record sharing exactly date D stays
    together in "later" rather than being split across the boundary.
    """
    cumulative = dates.value_counts().sort_index().cumsum()
    target_n = int(np.ceil(train_fraction * len(dates)))
    return cumulative[cumulative >= target_n].index[0]


cutoff_date = chronological_cutoff(raw_df["RT Start"], train_fraction=1 - config.TEST_SIZE)
historical_mask = raw_df["RT Start"] < cutoff_date  # date D itself goes to "later"
temporal_test_mask = ~historical_mask

# `== 1` on the nullable Int64 target yields pandas' nullable boolean dtype,
# with <NA> for excluded rows; `.fillna(False)` makes it an ordinary mask
# without ever treating "unobservable" as "non-event".
is_event_full = (target_df[config.TARGET_COLUMN] == 1).fillna(False)

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

# %%
# --- FIGURE: why a naive temporal holdout was rejected -----------------------
# Exclusion rate for insufficient 730-day follow-up, by RT-Start year. The
# message is the trend, not any single bar.
observability_by_year = (
    pd.DataFrame(
        {
            "year": raw_df["RT Start"].dt.year,
            "excluded": ~eligible_mask.to_numpy(),
        }
    )
    .groupby("year")
    .agg(n=("excluded", "size"), n_excluded=("excluded", "sum"))
)
observability_by_year["excluded_pct"] = (
    observability_by_year["n_excluded"] / observability_by_year["n"] * 100
)
print("\nOutcome observability by RT-Start year:")
print(observability_by_year.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 4.8))
years = observability_by_year.index.astype(str)
bars = ax.bar(
    years, observability_by_year["excluded_pct"],
    color=[
        plots.EVENT_COLOR if pct > 10 else plots.NON_EVENT_COLOR
        for pct in observability_by_year["excluded_pct"]
    ],
)
ax.set_xlabel("Year of RT Start (offset dates)")
ax.set_ylabel("Patients excluded because the\ntwo-year outcome is unobservable (%)")
ax.set_title("Outcome observability depends strongly on treatment era")
ax.set_ylim(0, 112)
for bar, (_, row) in zip(bars, observability_by_year.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
        f"{row['excluded_pct']:.0f}%\nn={int(row['n'])}", ha="center", fontsize=8,
    )
ax.axvline(
    float(np.searchsorted(observability_by_year.index.to_numpy(), cutoff_date.year)) - 0.5,
    color=plots.ACCENT_COLOR, linestyle="--", linewidth=1.5,
    label=f"naive chronological 80/20 boundary ({cutoff_date.date()})",
)
ax.legend(loc="upper left")
fig.text(
    0.5, -0.06,
    "A naive chronological holdout would place the most heavily censored years in the test set, "
    "confounding calendar time with outcome observability.\nA stratified random split mixes all "
    "treatment eras into both partitions and avoids that confound.",
    ha="center", fontsize=9, style="italic",
)
fig.tight_layout()
plots.save_figure(fig, "01_temporal_observability.png")

# Result:
#   The exclusion rate stays around 1-5% through 2007, then rises to 11%
#   (2008), 22% (2009), 75% (2010) and 100% (2011): patients treated close to
#   the latest observed offset follow-up date simply have not had time to
#   accrue 730 days of follow-up. No survival-analysis method was used to
#   reach this conclusion -- only the counts above.
# Decision:
#   Use a stratified random 80/20 holdout as the evaluation split (created
#   once, in Section 5). A temporal holdout is not implemented.
# Next:
#   Run the feature leakage audit (Section 3.4) before any predictor is
#   admitted to X.


# %%
# =============================================================================
# SECTION 3.4 — Feature leakage audit
# =============================================================================
# Objective:
#   Classify every one of the 34 original columns exactly once against the
#   admission test -- "if a new patient is immediately about to receive their
#   first radiotherapy fraction, could this variable legitimately be known and
#   used at that moment?" -- and derive the candidate/excluded lists
#   PROGRAMMATICALLY, so the audit's conclusions and the code's behaviour
#   cannot silently diverge.
#
# Rationale:
#   The full table, its taxonomy and the reasoning behind every row live in
#   `src/radcure/leakage.py`. This section verifies coverage and prints the
#   resulting lists.

section("SECTION 3.4 — Feature leakage audit")

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
    "Derived candidate-predictor set no longer matches the confirmed audit outcome."
)
print("\n[OK] Candidate-predictor set matches the confirmed 13-variable outcome.")

# Stated explicitly rather than left as an omission: these three are excluded
# as PROXIES, not as direct target leakage.
for proxy_var, expected_category in [
    ("RT Start", "CALENDAR_ERA_PROXY"),
    ("RADCURE-challenge", "DATASET_MEMBERSHIP_PROXY"),
    ("ContrastEnhanced", "TECHNICAL_PROTOCOL_PROXY"),
]:
    row = leakage.LEAKAGE_AUDIT_TABLE.loc[leakage.LEAKAGE_AUDIT_TABLE["variable"] == proxy_var].iloc[0]
    assert row["category"] == expected_category and row["decision"] == "EXCLUDE"
    assert row["category"] != "DIRECT_TARGET_LEAKAGE"
print("[OK] RT Start / RADCURE-challenge / ContrastEnhanced are documented as proxies, not target leakage.")

# Result:
#   13 of 34 variables pass the temporal-availability audit; 21 are excluded
#   as an identifier, direct target leakage, follow-up information, a
#   post-landmark clinical event, post-landmark treatment information, mixed
#   temporal information, or one of the three proxies above.
# Decision:
#   Only the 13 CANDIDATE variables may be considered for X. Passing the
#   audit does not imply inclusion -- Section 3.5 narrows further, on
#   clinical rather than statistical grounds.


# %%
# =============================================================================
# SECTION 3.5 — Primary predictor set definition
# =============================================================================
# Objective:
#   Narrow the 13 leakage-safe candidates to the confirmed 8-variable PRIMARY
#   predictor set, and document, for each of the 5 candidates NOT admitted,
#   the specific non-leakage reason.
#
# Rationale:
#   - `M`       : near-zero variation -- in the eligible cohort (n=2939), 2927
#                 patients are M0, 10 are missing and 2 are MX, with no M1
#                 patient at all.
#   - `Stage`   : redundant given Ds Site + T + N together. Stage is NOT a
#                 deterministic function of TNM alone -- it is a SITE-SPECIFIC
#                 recombination (24 of 70 TNM combinations map to more than
#                 one Stage, resolving along AJCC-7 site-specific rules, e.g.
#                 nasopharyngeal T2N1M0 = Stage II vs. Stage III elsewhere).
#                 Using Stage alone would also lose substantial T/N
#                 granularity (Stage IVA alone spans ten T categories).
#   - `Subsite` : high cardinality (59 levels) with heterogeneous missingness:
#                 a STRUCTURAL absence for carcinoma of unknown primary (100%
#                 missing -- no identifiable primary subsite exists for that
#                 entity), versus substantial site-dependent missing data for
#                 nasopharynx (~52%).
#   - `Path`    : its dominant clinical distinction (SCC vs. nasopharyngeal
#                 carcinoma) is highly redundant with `Ds Site` -- 307 of 308
#                 NPC-histology records are nasopharyngeal and zero
#                 nasopharyngeal records are SCC-like. Not excluded for rarity.
#   - `HPV`     : availability at the prediction landmark cannot reliably be
#                 established for all historical patients (48.6% missing,
#                 site- and era-dependent documentation, with evidence some
#                 historical results were determined retrospectively).
#                 Reserved as a pre-specified sensitivity-only variable.
#
#   Selection was driven by temporal availability, leakage prevention,
#   information content, redundancy, data structure and clinical plausibility
#   -- not by screening variables on outcome performance.

section("SECTION 3.5 — Primary predictor set")

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

print("""
Predictor-selection funnel:
   34 raw columns
   -> 21 excluded by the study-design / leakage / temporality audit
   -> 13 temporally admissible clinical candidates
   ->  5 further predictor-design exclusions (M, Stage, Subsite, Path, HPV)
   ->  8 PRIMARY predictors (frozen)
""")

# Decision:
#   X contains exactly the 8 PRIMARY features. No automatic feature selection
#   is applied: the model already contains only eight pre-specified,
#   domain-informed predictors, and the encoded representation stays
#   manageable under regularisation and cross-validation.


# %%
# =============================================================================
# SECTION 4 — Deterministic semantic cleaning of the PRIMARY predictors
# =============================================================================
# Objective:
#   Apply the approved semantic cleaning rules to each PRIMARY predictor (plus
#   HPV, for the reserved sensitivity representation) and make the raw ->
#   cleaned effect visible: every raw level, every cleaned level, and the
#   resulting change in "not usable" counts.
#
# Rationale:
#   Every rule repairs how a value was WRITTEN -- case, an explicit
#   unknown/unassessable code, a non-numeric token -- based on the Data
#   Dictionary together with the findings of the separate EDA project. These
#   rules estimate no data-dependent parameter (no imputation value, no
#   scaling statistic, no frequency threshold), which is why they may run on
#   the full file, before the train/test split.
#
#   The exact mappings are fixed constants in `src/radcure/config.py`, so this
#   section applies and verifies already-approved rules rather than making new
#   cleaning decisions.

section("SECTION 4 — Deterministic semantic cleaning")

clinical_df = cohort.apply_primary_cleaning(raw_df)

# --- Age: no cleaning needed -------------------------------------------------
# Continuous, so a compact numeric summary is used rather than a value dump.
print("\nAge (n=3346, missing=0) — modelled as continuous, no binning:")
print(raw_df["Age"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_string())
n_top_coded_90 = int((raw_df["Age"] == 90.0).sum())
n_under_18 = int((raw_df["Age"] < 18).sum())
print(f"\n   {n_top_coded_90} patients have Age == 90.0, consistent with age top-coding / "
      f"de-identification; values are retained without clipping.")
print(f"   Patients under 18 retained without exclusion: {n_under_18}.")
assert raw_df["Age"].isna().sum() == 0

# --- Sex: no cleaning needed --------------------------------------------------
print_value_counts(raw_df["Sex"], "Sex")
assert set(raw_df["Sex"].dropna().unique()) == {"Male", "Female"}

# --- ECOG PS ------------------------------------------------------------------
print_value_counts(raw_df["ECOG PS"], "ECOG PS (RAW)")
print_value_counts(clinical_df["ECOG PS"], "ECOG PS (CLEANED)")

# --- Smoking PY ---------------------------------------------------------------
print_value_counts(raw_df["Smoking PY"], "Smoking PY (RAW, showing every non-numeric token)")
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
print_value_counts(clinical_df["Smoking Status"], "Smoking Status (CLEANED)")

# --- Ds Site --------------------------------------------------------------
# Three distinct level counts, kept apart so they are not conflated:
#   19 raw labels -> 17 after case/spelling harmonisation (the ONLY change
#   cleaning makes here) -> 15 observed in the ELIGIBLE cohort. The last step
#   is not cleaning: Orbit and Lacrimal gland (one raw patient each) fall
#   entirely within the 407 target-unobservable patients.
n_ds_site_raw = raw_df["Ds Site"].str.strip().nunique()
print_value_counts(raw_df["Ds Site"], "Ds Site (RAW, full cohort n=3346)")
print_value_counts(clinical_df["Ds Site"], "Ds Site (CLEANED — case duplicates merged only)")
assert clinical_df["Ds Site"].nunique() == n_ds_site_raw - 2, (
    "Expected exactly the two known case-duplicate pairs "
    "('nasal cavity'/'Nasal Cavity', 'esophagus'/'Esophagus') to merge; "
    f"raw levels={n_ds_site_raw}, cleaned levels={clinical_df['Ds Site'].nunique()}."
)

# --- T and N categories -------------------------------------------------------
print_value_counts(raw_df["T"], "T (RAW)")
print_value_counts(clinical_df["T"], "T (CLEANED)")
print_value_counts(raw_df["N"], "N (RAW)")
print_value_counts(clinical_df["N"], "N (CLEANED)")

# --- HPV (sensitivity-only representation; not part of X) --------------------
print_value_counts(raw_df["HPV"], "HPV (RAW)")
clinical_df["HPV"] = cleaning.clean_hpv(raw_df["HPV"])
print_value_counts(clinical_df["HPV"], "HPV (CLEANED — deliberately NOT labelled 'not tested')")

# --- Missingness before/after, on the ELIGIBLE cohort ------------------------
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

# Result:
#   Cleaning increases recorded "not usable" counts for ECOG PS, T, N and
#   Smoking PY, exactly as intended: it converts values that only LOOKED
#   present (`Unknown`, `TX`, `na`, a bound token) into honest missing values.
#   Every PRIMARY predictor stays under 1.6% missing. No clinically distinct
#   or rare-but-valid category was pooled or removed.
# Next:
#   Assemble the modelling cohort, create the frozen split, and persist both
#   (Section 5).


# %%
# =============================================================================
# SECTION 5 — Modelling cohort, frozen split, and the persisted artefact
# =============================================================================
# Objective:
#   Build the modelling frame from the eligible cohort, create the single
#   stratified 80/20 train/test split, and write the artefact that every later
#   analysis script consumes.
#
# Rationale:
#   Eligibility and predictor selection are both determined from information
#   available before the split (neither estimates a statistical parameter), so
#   both may safely precede it. The split is stratified on y to keep the
#   ~18.9% prevalence comparable across partitions.
#
#   `cohort.assign_frozen_split` is the ONLY `train_test_split` call site in
#   this project. Row order matters and is preserved: StratifiedKFold assigns
#   folds by POSITION, so a cohort reloaded in a different order would produce
#   different folds and different cross-validated numbers. The artefact
#   therefore stores rows in split order plus each row's original cohort
#   position, and the round-trip is verified below rather than assumed.

section("SECTION 5 — Modelling cohort, frozen split, persisted artefact")

cohort_df = cohort.build_modeling_cohort(raw_df)
print(f"Eligible modelling cohort: {cohort_df.shape[0]} patients x "
      f"{len(config.PRIMARY_FEATURES)} predictors + target.")
assert len(cohort_df) == config.EXPECTED_ELIGIBLE_N
assert list(cohort_df.columns) == [
    config.PATIENT_ID_COLUMN, *config.PRIMARY_FEATURES, config.TARGET_COLUMN
]

# --- Guards on the assembled frame -------------------------------------------
# The guard applies to the PREDICTORS only: `patient_id` is itself an audit
# EXCLUDED variable (an identifier) and is carried alongside the predictors for
# traceability and split-integrity checks, never admitted to X.
leakage.assert_no_excluded_column_in_frame(config.PRIMARY_FEATURES)
assert config.TARGET_COLUMN not in config.PRIMARY_FEATURES, "The target must never be a predictor."
assert config.PATIENT_ID_COLUMN not in config.PRIMARY_FEATURES
assert set(cohort_df["ECOG PS"].unique()) <= (config.ECOG_PS_VALID_LEVELS | {config.MISSING_LABEL})
assert set(cohort_df["T"].unique()) <= (config.T_VALID_LEVELS | {config.MISSING_LABEL})
assert set(cohort_df["N"].unique()) <= (config.N_VALID_LEVELS | {config.MISSING_LABEL})
assert cohort_df["Ds Site"].nunique() == config.EXPECTED_DS_SITE_LEVELS
assert pd.api.types.is_float_dtype(cohort_df["Smoking PY"]) and pd.api.types.is_float_dtype(cohort_df["Age"])
print("[OK] No leakage-excluded variable is present; the target is not among the predictors.")
print("[OK] Final category sets for ECOG PS / T / N / Ds Site match the approved cleaning specification.")
print("[OK] Age and Smoking PY are numeric with no residual non-numeric tokens.")

# --- The single stratified 80/20 split ---------------------------------------
split_df = cohort.assign_frozen_split(cohort_df)
cohort.assert_split_integrity(split_df)
print("\nFrozen split:")
print(cohort.split_summary(split_df).to_string())
print(f"\n[OK] Train/test sizes and event counts match the confirmed study design "
      f"({config.EXPECTED_TRAIN_N} / {config.EXPECTED_TEST_N}, "
      f"{config.EXPECTED_TRAIN_EVENTS} / {config.EXPECTED_TEST_EVENTS} events).")
print("[OK] Train and test patient identifiers are disjoint and together cover every "
      "eligible patient exactly once.")

# --- Persist, then verify the round-trip reproduces the split exactly --------
cohort_path = artifacts.save_modeling_cohort(split_df)
print(f"\n[artifact] {cohort_path.relative_to(PROJECT_ROOT)}")

X_train, y_train, id_train = cohort.partition(split_df, cohort.TRAIN_SPLIT)
X_train_reloaded, y_train_reloaded, id_train_reloaded = artifacts.load_training_partition()
assert X_train_reloaded.equals(X_train), "Reloaded training predictors differ from the in-memory split."
assert y_train_reloaded.equals(y_train), "Reloaded training target differs from the in-memory split."
assert id_train_reloaded.equals(id_train), "Reloaded training identifiers differ from the in-memory split."
print("[OK] Reloading the artefact reproduces the training partition exactly -- same values, "
      "same dtypes, same row order and same index, so downstream cross-validation folds are "
      "identical to the ones this split defines.")

# ------------------------------------------------------------------------------
# The held-out test partition is now frozen. Scripts 02-05 and 07 never load
# it; only analysis/06 does, for the single final evaluation.
# ------------------------------------------------------------------------------

# Decision:
#   The split is frozen and persisted. All learned preprocessing downstream is
#   fitted on the training partition, or inside CV folds drawn from it, only.


# %%
# =============================================================================
# SECTION 6 — PRIMARY preprocessing pipeline objects (construction only)
# =============================================================================
# Objective:
#   Build the ColumnTransformer that will later preprocess the PRIMARY
#   features, and verify -- by attempting to use it -- that it holds no fitted
#   state yet.
#
# Rationale:
#   Constructing an unfitted scikit-learn object estimates nothing and is
#   leakage-safe regardless of the split. Fitting it is a learned-preprocessing
#   operation and happens only inside a training pipeline, on the training
#   partition or on CV folds drawn from it.

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
    "\nNote (drop=None): no reference level is omitted from any categorical column, so a fitted "
    "coefficient must NOT be read as a simple 'category vs. one omitted reference category' "
    "effect -- every level, including the most frequent one, carries its own coefficient. "
    "analysis/07 returns to this when the final model is interpreted."
)

# Result:
#   The preprocessing pipeline is fully specified and demonstrably unfitted.
#   No rare-category pooling is included: all clinically valid categories are
#   retained, relying on `handle_unknown="ignore"` plus model regularisation.


# %%
# =============================================================================
# SECTION 7 — Foundation summary
# =============================================================================
section("SECTION 7 — Foundation summary")

print(f"""
Completed in this script:
  - Immutable raw data loaded and structurally verified (3346 x 34).
  - Prediction landmark and outcome clock fixed at RT Start; the
    diagnosis-anchored alternative and the fixed-vs-calendar horizon question
    both resolved with executable counts.
  - RT-Start-anchored 730-day target built; eligibility confirmed at
    {config.EXPECTED_ELIGIBLE_N} eligible / {config.EXPECTED_EVENTS_N} events /
    {config.EXPECTED_NONEVENTS_N} non-events / {config.EXPECTED_EXCLUDED_N} excluded.
  - Naive chronological holdout rejected, documented with censoring counts and
    rates only.
  - Leakage audit covering all 34 original columns exactly once; 13 candidates
    derived programmatically.
  - PRIMARY predictor set fixed at 8 variables; the 5 non-admitted candidates
    documented with their specific reasons.
  - Deterministic semantic cleaning applied and verified.
  - Single stratified 80/20 split created and FROZEN
    ({config.EXPECTED_TRAIN_N} / {config.EXPECTED_TEST_N} patients,
    {config.EXPECTED_TRAIN_EVENTS} / {config.EXPECTED_TEST_EVENTS} events).
  - Preprocessing pipeline constructed and confirmed UNFITTED.

Artefact produced:
  data/processed/{config.MODELING_COHORT_FILENAME}
      patient_id, the 8 semantically cleaned PRIMARY predictors, the target,
      and the frozen train/test assignment. Semantically cleaned but NOT
      learned-preprocessed: imputation, scaling and encoding remain inside the
      modelling pipeline, fitted training-only.

Figures produced:
  figures/01_cohort_target_flow.png
  figures/01_temporal_observability.png

Next: analysis/02_baseline_modeling.py — training-only baseline model-family
comparison, consuming the frozen training partition from this artefact.
""")
