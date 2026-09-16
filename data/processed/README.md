# `data/processed/`

Derived data produced by the analysis pipeline. Not tracked in version
control — regenerate it by running `analysis/01_project_foundation.py`.

## `modeling_cohort.csv`

The project's single source of truth for the modelling cohort, written once by
`analysis/01` and loaded by every later stage.

One row per eligible patient (n = 2939), in **split order**: the training rows
first, in exactly the order `train_test_split` produced them, then the held-out
rows.

| column | meaning |
| --- | --- |
| `cohort_row` | the row's position in the pre-split eligible cohort, so the original pandas index can be restored exactly |
| `patient_id` | RADCURE patient identifier |
| `Age`, `Smoking PY` | numeric predictors, semantically cleaned |
| `Sex`, `ECOG PS`, `Smoking Status`, `Ds Site`, `T`, `N` | categorical predictors, semantically cleaned |
| `mortality_2y` | the target: 1 = died within 730 days of `RT Start`, 0 = non-event |
| `split` | `train` (n = 2351) or `test` (n = 588) |

### Two properties this file deliberately has

**It is semantically cleaned but NOT learned-preprocessed.** Imputation,
scaling and one-hot encoding are all fitted inside the modelling pipeline, on
training folds only. Persisting globally transformed features here would leak
held-out information into every downstream stage.

**Row order is load-bearing.** `StratifiedKFold(shuffle=True)` assigns folds by
position, so a cohort reloaded in a different order would produce different
folds and different cross-validated numbers. `radcure.cohort.partition` restores
both the order and the original index; `analysis/01` asserts the round-trip is
exact.
