# `data/` — source and provenance

This repository intentionally includes two data files: the immutable clinical
source workbook and the frozen derived modelling cohort generated from it.
This document is the authoritative record of their provenance and license.

## Source

```
RADCURE — Computed Tomography Images from Large Head and Neck Cohort
Version 4, updated 2024-12-19
The Cancer Imaging Archive (TCIA)
DOI: 10.7937/J47W-NM11
```

**Citation:**

> Welch, M. L., Kim, S., Hope, A., Huang, S. H., Lu, Z., Marsilla, J.,
> Kazmierski, M., Rey-McIntyre, K., Patel, T., O'Sullivan, B., Waldron, J.,
> Kwan, J., Su, J., Soltan Ghoraie, L., Chan, H. B., Yip, K., Giuliani, M.,
> Princess Margaret Head And Neck Site Group, Bratman, S., … Tadic, T. (2023).
> Computed Tomography Images from Large Head and Neck Cohort (RADCURE)
> (Version 4) [Dataset]. The Cancer Imaging Archive.
> DOI: [10.7937/J47W-NM11](https://doi.org/10.7937/J47W-NM11)

## License

The **RADCURE Version 4 clinical workbook** (`data/raw/`) is distributed by
TCIA under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. CC BY 4.0 permits sharing and adapting the material for any purpose,
provided that appropriate credit is given, a link to the license is provided,
and any changes made are indicated — which is what this document and
`data/processed/README.md` do for the derived cohort below.

This CC BY 4.0 statement applies to the **RADCURE clinical data only**. It
does not relicense the source code, documentation or any other content of
this repository, which retains its own project licensing.

The RADCURE **CT / RTSTRUCT imaging data** are governed by separate NIH
Controlled Data Access conditions and are **not included in this repository**
in any form.

Users of the clinical data in this repository must also follow the
[TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).
In particular:

- do not attempt to identify or contact research participants;
- redistribution of this data here does not imply endorsement of this
  project by TCIA, the National Cancer Institute, or the dataset authors.

## `data/raw/RADCURE_Clinical_v04_20241219.xlsx`

The official RADCURE Version 4 clinical source workbook, exactly as
distributed by TCIA (3346 rows × 34 columns). **Immutable within this
project**: every script in `analysis/` and every function in
`src/radcure/cleaning.py` only reads this file; nothing in the project ever
overwrites it.

## `data/processed/modeling_cohort.csv`

**Adapted material**, generated from the clinical source workbook above by
`analysis/01_project_foundation.py` (via `src/radcure/cohort.py`). CC BY 4.0
requires modifications to be identified; the changes made, relative to the
source workbook, are:

- deterministic semantic cleaning of the eight selected baseline predictor
  variables (`src/radcure/cleaning.py`) — harmonising spelling/case variants
  and converting explicit unknown/unassessable codes to an honest missing
  category; no value is invented or imputed at this stage;
- construction of the fixed 730-day mortality target, anchored at `RT Start`
  (`src/radcure/target.py`);
- exclusion of patients whose two-year outcome is not observable (alive with
  less than 730 days of documented follow-up) — 407 of the original 3346
  patients;
- restriction of the columns to the patient identifier, the eight frozen
  PRIMARY predictors, the constructed target, and split metadata — every
  other original column is dropped;
- addition of `cohort_row` (the row's position in the pre-split eligible
  cohort, so the exact original ordering can be restored);
- addition of the frozen `train` / `test` split assignment
  (`src/radcure/cohort.py::assign_frozen_split`).

**No globally learned preprocessing is included.** Imputation (the
`Smoking PY` median), scaling, and one-hot encoding remain fitted only inside
the training pipeline / cross-validation, strictly on the training partition
or on CV folds drawn from it — never on this persisted file as a whole. See
`data/processed/README.md` for the column-by-column layout and why row order
matters.

This file is generated, not manually maintained: it is intentionally
versioned here as a frozen reproducibility snapshot of what `analysis/01`
produces from the source workbook above, so the exact modelling cohort behind
every downstream result can be inspected and regenerated without re-deriving
it from the raw clinical data.
