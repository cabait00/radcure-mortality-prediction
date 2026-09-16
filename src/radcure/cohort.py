"""
Modelling-cohort assembly and the single frozen train/test split.

This module owns the deterministic path from the immutable raw workbook to
the frozen modelling cohort:

    raw workbook
      -> deterministic semantic cleaning of the PRIMARY predictors
      -> RT-Start-anchored 730-day target + observability eligibility
      -> the eligible modelling cohort (patient_id, 8 predictors, target)
      -> ONE stratified 80/20 train/test split

`assign_frozen_split` below is the ONLY place in this project that calls
`train_test_split`. `analysis/01_project_foundation.py` calls it once to
produce the persisted cohort artefact; `analysis/02`-`07` never split again,
they load the persisted assignment through `radcure.artifacts`.

Nothing here estimates a statistical parameter: semantic cleaning and target
construction are deterministic, so both may safely precede the split.
Learned preprocessing (imputation median, scaler statistics, encoder
vocabulary) lives in `preprocessing.py` and is fitted only inside the
training pipeline / cross-validation.

Row order is load-bearing, not cosmetic: `StratifiedKFold(shuffle=True)`
assigns folds by POSITION, so a cohort reloaded in a different row order
would produce different folds and different cross-validated numbers.
`assign_frozen_split` therefore returns the cohort already re-ordered into
split order (training rows in the order `train_test_split` produced them,
then test rows), and records each row's original cohort position in
`cohort_row` so the exact pandas index can be restored on load.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import cleaning, config, leakage, target

SPLIT_COLUMN = "split"
COHORT_ROW_COLUMN = "cohort_row"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

#: Column order of the persisted modelling-cohort artefact.
COHORT_ARTIFACT_COLUMNS = [
    COHORT_ROW_COLUMN,
    config.PATIENT_ID_COLUMN,
    *config.PRIMARY_FEATURES,
    config.TARGET_COLUMN,
    SPLIT_COLUMN,
]


def apply_primary_cleaning(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved deterministic semantic cleaning to every PRIMARY
    predictor that needs it, returning a copy of `raw_df`.

    `Age` and `Sex` need no cleaning (Age has no missing values and no
    non-numeric tokens; Sex is already exactly {Male, Female}). The six
    remaining predictors are harmonised by the rules in `cleaning.py`.
    """
    cleaned = raw_df.copy()
    cleaned["ECOG PS"] = cleaning.clean_ecog_ps(raw_df["ECOG PS"])
    cleaned["Smoking PY"] = cleaning.clean_smoking_py(raw_df["Smoking PY"])
    cleaned["Smoking Status"] = cleaning.clean_smoking_status(raw_df["Smoking Status"])
    cleaned["Ds Site"] = cleaning.clean_ds_site(raw_df["Ds Site"])
    cleaned["T"] = cleaning.clean_t_category(raw_df["T"])
    cleaned["N"] = cleaning.clean_n_category(raw_df["N"])
    return cleaned


def build_modeling_cohort(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the eligible modelling cohort from the raw workbook.

    Returns one row per ELIGIBLE patient (target observable within the
    fixed 730-day horizon), with `patient_id`, the 8 PRIMARY predictors
    and the integer target, indexed 0..n-1.

    Target observability is the ONLY restriction applied: no clinical,
    histological or frequency-based exclusion is performed.
    """
    cleaned_df = apply_primary_cleaning(raw_df)
    target_df = target.build_target(raw_df)
    eligible_mask = target_df[config.ELIGIBILITY_COLUMN]

    # Columns are selected explicitly rather than joined, so this function
    # behaves identically whether or not the caller's `raw_df` already
    # carries target-construction columns of its own.
    cohort_df = cleaned_df.loc[
        eligible_mask, [config.PATIENT_ID_COLUMN, *config.PRIMARY_FEATURES]
    ].copy()
    cohort_df[config.TARGET_COLUMN] = (
        target_df.loc[eligible_mask, config.TARGET_COLUMN].astype(int).to_numpy()
    )
    cohort_df = cohort_df.reset_index(drop=True)

    leakage.assert_no_excluded_column_in_frame(config.PRIMARY_FEATURES)
    assert cohort_df[config.PATIENT_ID_COLUMN].is_unique, (
        "patient_id is not unique within the eligible modelling cohort."
    )
    return cohort_df


def assign_frozen_split(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Create the project's single stratified 80/20 train/test split.

    This is the ONLY `train_test_split` call site in the project. It is
    invoked once, by `analysis/01`, to generate the persisted cohort
    artefact; every later stage loads that artefact instead of splitting
    again.

    The returned frame carries the full cohort in SPLIT ORDER -- training
    rows first, in exactly the order `train_test_split` produced them, then
    test rows -- plus:
      - `cohort_row` : the row's position in the input `cohort_df`, so the
        original pandas index can be restored exactly on load;
      - `split`      : "train" or "test".
    """
    X = cohort_df[config.PRIMARY_FEATURES]
    y = cohort_df[config.TARGET_COLUMN]

    X_train, X_test, _, _ = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    ordered_rows = list(X_train.index) + list(X_test.index)
    split_labels = [TRAIN_SPLIT] * len(X_train) + [TEST_SPLIT] * len(X_test)

    split_df = cohort_df.loc[ordered_rows].copy()
    split_df.insert(0, COHORT_ROW_COLUMN, list(split_df.index))
    split_df[SPLIT_COLUMN] = split_labels
    return split_df[COHORT_ARTIFACT_COLUMNS].reset_index(drop=True)


def partition(split_df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return `(X, y, patient_id)` for one side of the frozen split.

    File order is preserved and the original cohort index is restored from
    `cohort_row`, so the result is identical -- values, dtypes, order and
    index -- to what `train_test_split` returned in `analysis/01`.

    The six categorical predictors are normalised to `object` dtype: a CSV
    round-trip returns pandas' newer `str` dtype for them, which holds the
    same values but would make a direct `DataFrame.equals` comparison fail.
    """
    if split not in {TRAIN_SPLIT, TEST_SPLIT}:
        raise ValueError(f"Unknown split {split!r}; expected {TRAIN_SPLIT!r} or {TEST_SPLIT!r}.")

    rows = split_df.loc[split_df[SPLIT_COLUMN] == split]
    index = pd.Index(rows[COHORT_ROW_COLUMN].to_numpy(), name=None)

    X = rows[config.PRIMARY_FEATURES].copy()
    X[config.PRIMARY_CATEGORICAL_FEATURES] = X[config.PRIMARY_CATEGORICAL_FEATURES].astype(object)
    X.index = index

    y = pd.Series(rows[config.TARGET_COLUMN].astype(int).to_numpy(), index=index, name=config.TARGET_COLUMN)
    patient_id = pd.Series(
        rows[config.PATIENT_ID_COLUMN].to_numpy(),
        index=index,
        name=config.PATIENT_ID_COLUMN,
    )
    return X, y, patient_id


def split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-split n / events / non-events / prevalence table."""
    rows = []
    for split in (TRAIN_SPLIT, TEST_SPLIT):
        subset = split_df.loc[split_df[SPLIT_COLUMN] == split]
        n = len(subset)
        n_events = int((subset[config.TARGET_COLUMN] == 1).sum())
        rows.append(
            {
                "split": split,
                "n": n,
                "events": n_events,
                "non_events": n - n_events,
                "prevalence_pct": round(n_events / n * 100, 2) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def assert_split_integrity(split_df: pd.DataFrame) -> None:
    """Verify the frozen split artefact against the confirmed study design.

    Checks cohort size, per-split sizes and event counts, that train and
    test patient identifiers are disjoint, and that every eligible patient
    appears exactly once.
    """
    assert list(split_df.columns) == COHORT_ARTIFACT_COLUMNS, (
        f"Unexpected cohort-artefact columns: {list(split_df.columns)}"
    )
    assert len(split_df) == config.EXPECTED_ELIGIBLE_N, (
        f"Cohort has {len(split_df)} rows; expected {config.EXPECTED_ELIGIBLE_N}."
    )
    assert split_df[config.PATIENT_ID_COLUMN].is_unique, "A patient appears more than once in the split artefact."
    assert split_df[COHORT_ROW_COLUMN].is_unique, "Duplicate cohort_row values in the split artefact."
    assert set(split_df[SPLIT_COLUMN].unique()) == {TRAIN_SPLIT, TEST_SPLIT}

    train_rows = split_df.loc[split_df[SPLIT_COLUMN] == TRAIN_SPLIT]
    test_rows = split_df.loc[split_df[SPLIT_COLUMN] == TEST_SPLIT]

    assert len(train_rows) == config.EXPECTED_TRAIN_N
    assert int((train_rows[config.TARGET_COLUMN] == 1).sum()) == config.EXPECTED_TRAIN_EVENTS
    assert int((train_rows[config.TARGET_COLUMN] == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS
    assert len(test_rows) == config.EXPECTED_TEST_N
    assert int((test_rows[config.TARGET_COLUMN] == 1).sum()) == config.EXPECTED_TEST_EVENTS
    assert int((test_rows[config.TARGET_COLUMN] == 0).sum()) == config.EXPECTED_TEST_NONEVENTS

    train_ids = set(train_rows[config.PATIENT_ID_COLUMN])
    test_ids = set(test_rows[config.PATIENT_ID_COLUMN])
    assert not (train_ids & test_ids), "Train and test patient_id sets overlap."
    assert len(train_ids) + len(test_ids) == config.EXPECTED_ELIGIBLE_N, (
        "Train and test partitions do not cover every eligible patient exactly once."
    )
