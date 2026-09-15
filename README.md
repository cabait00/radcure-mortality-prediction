# RADCURE ML

Applied machine learning project using the RADCURE Clinical Dataset.

The project predicts **two-year all-cause mortality after the start of
radiotherapy** in **patients with head and neck tumours**, using clinical,
demographic and tumour-related characteristics available before treatment
initiation.

## Status

**Project Foundation complete.** The study design, feature leakage audit,
primary predictor set, and deterministic data-cleaning / preprocessing
specification have been defined, implemented and validated.

**Training-only baseline model-family comparison complete.** Five
default-hyperparameter candidates (Dummy, Logistic Regression, Decision
Tree, Random Forest, RBF SVC) were compared under a fixed 5-fold
stratified cross-validation design on the training split only. Logistic
Regression currently leads the default comparison; Logistic Regression,
Random Forest and RBF SVC are shortlisted for a controlled hyperparameter
tuning pass. The held-out test set has **not** been used for any
model-performance evaluation.

The primary predictor set: `Age`, `Sex`, `ECOG PS`, `Smoking PY`,
`Smoking Status`, `Ds Site`, `T`, `N`.

## TODO — final documentation: predictor-selection rationale

The final written documentation and presentation must explain the full
predictor-selection funnel explicitly, not just list its outcome:

- 34 raw dataset columns
- → 21 excluded during study-design / leakage / temporality audit
- → 13 temporally admissible clinical candidate predictors
- → 5 additional predictor-design exclusions
- → 8 prespecified primary predictors

The 8 primary predictors: `Age`, `Sex`, `ECOG PS`, `Smoking PY`,
`Smoking Status`, `Ds Site`, `T`, `N`.

The 5 admissible-but-not-primary candidates (temporally available, not
selected as primary): `Subsite`, `M`, `Stage`, `Path`, `HPV`.

Each exclusion must be explained individually in the final documentation.
It must also make explicit that:

> Predictor selection was driven primarily by temporal availability,
> leakage prevention, information content, redundancy, data structure and
> clinical-methodological plausibility — not by post-hoc maximization of
> cross-validation performance.

## Repository layout

- `analysis/01_project_foundation.py` — chronological, documented analysis
  script covering data loading, study design, the leakage audit, predictor
  selection, deterministic cleaning, and the frozen train/test split.
- `analysis/02_baseline_modeling.py` — chronological, documented analysis
  script covering the training-only baseline model-family comparison and
  its cross-validated results.
- `src/radcure/` — reusable, tested logic (cleaning, target construction,
  leakage audit, preprocessing pipeline objects, modelling utilities).
- `src/radcure/modeling.py` — the fixed cross-validation splitter, the
  baseline candidate-model registry, and the pipeline/evaluation helpers.
- `tests/` — validation tests for `src/radcure/`, including
  `tests/test_modeling.py`.
- `pytest.ini` — scopes test collection to `tests/` only.
- `data/raw/` — the immutable source workbook (not modified by this project).
- `reference/` — prior exploratory work this project builds on (not part
  of the tracked repository).

## Data source / citation

Clinical data are from the **RADCURE** collection on The Cancer Imaging
Archive (TCIA). Collection DOI: `10.7937/J47W-NM11`. The raw workbook is
not distributed with this repository.
