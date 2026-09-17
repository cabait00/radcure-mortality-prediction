# `artifacts/`

Results passed between the analysis stages. Each file is written by exactly one
script and read by the ones after it, so no stage recomputes work an earlier
one already did.

Formats are deliberately plain: CSV for tables, JSON for configurations and
metrics. The one thing that genuinely needs a binary format — the fitted
pipeline — lives in `models/` instead.

| file | written by | contents |
| --- | --- | --- |
| `02_baseline_cv_results.csv` | `analysis/02` | per-fold CV scores for the five default-hyperparameter candidates |
| `02_shortlist.json` | `analysis/02` | which families go forward to tuning, and why |
| `03_tuned_results.json` | `analysis/03` | selected hyperparameters and CV metrics per family |
| `03_svc_grid_results.csv` | `analysis/03` | the full first-stage SVC `C` × `gamma` × `class_weight` surface |
| `04_oof_scores.csv` | `analysis/04` | one training-only out-of-fold score per patient per model — **not versioned** (patient-level) |
| `04_oof_metadata.json` | `analysis/04` | provenance of those scores: CV design, score method and frozen parameters per model |
| `04_model_comparison.csv` | `analysis/04` | every candidate considered, with its PRESPECIFIED / EXPLORATORY status |
| `04_final_model_specification.json` | `analysis/04` | the selected model family and why it was selected |
| `05_threshold_sweep.csv` | `analysis/05` | metrics at every achievable operating point |
| `05_threshold.json` | `analysis/05` | the frozen decision threshold and its selection rule |
| `06_heldout_predictions.csv` | `analysis/06` | per-patient held-out probabilities — **not versioned** (patient-level) |
| `06_heldout_metrics.json` | `analysis/06` | the final held-out metric set and confusion matrix |
| `07_coefficients.csv` | `analysis/07` | fitted coefficients, contrasts and category sample sizes |

## Two conventions worth knowing

**Patient-level artefacts are not versioned.** `04_oof_scores.csv` and
`06_heldout_predictions.csv` carry `patient_id` for one row per patient and are
listed in `.gitignore`. This is a repository-design choice, not a statement
that patient-level data cannot be redistributed under the source data's
license — the raw clinical workbook and the processed modelling cohort are,
in fact, both versioned in this repository (see
[`../data/README.md`](../data/README.md)). These two files are excluded
because they are regenerable patient-level intermediate model outputs, not
needed as part of the compact public review surface: their aggregate,
provenance-carrying counterparts (`04_oof_metadata.json`,
`06_heldout_metrics.json`) are versioned instead, and are the reproducible
record of every number quoted in the README.

**Floats round-trip exactly.** `radcure.artifacts.load_table` reads CSVs with
`float_precision="round_trip"`. `read_csv`'s default parser is fast but not
exactly correctly rounded, and it perturbs most values by about one unit in the
last place — negligible scientifically, but enough that a threshold re-derived
from a reloaded table would differ in its final digits from the value that was
written.
