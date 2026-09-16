"""
Deterministic semantic cleaning (Section 4) for the RADCURE clinical
variables used in this project.

Every function here implements a SEMANTIC correction only: harmonising a
spelling/case variant, converting an explicit unknown/unassessable code to
missing, or parsing a numeric string. No function estimates a statistical
parameter (a mean, a median, a frequency threshold, a scaler) from the data
it is given. These functions do not fit any data-dependent preprocessing
parameter, which is why they may be applied before the train/test split.

Learned preprocessing (imputation values, scaling parameters, encodings)
lives in ``preprocessing.py`` and is fitted only inside the training
pipeline / cross-validation, never here.

Functions with an explicit, closed valid-level contract (`clean_ecog_ps`,
`clean_t_category`, `clean_n_category`, `clean_hpv`) raise ``ValueError``
if they encounter a level they cannot classify as either "valid" or "map
to missing" -- a deliberate fail-loud guard: an unexpected raw value must
be inspected and the mapping dictionaries in ``config.py`` updated
explicitly, rather than silently passed through or discarded.
`clean_smoking_py` similarly raises ``ValueError`` for a string it cannot
parse as either numeric or a known missing token. `clean_ds_site` and
`clean_smoking_status` have no closed valid-level set to check against --
they perform case/spelling harmonisation only and intentionally pass any
other value through unchanged, so they do not raise.
"""

from __future__ import annotations

import pandas as pd

from . import config


# =============================================================================
# Loading (never writes to data/raw/)
# =============================================================================
def load_raw_clinical(
    path=config.RAW_DATA_PATH,
    sheet_name: str = config.CLINICAL_SHEET_NAME,
) -> pd.DataFrame:
    """Load the immutable raw RADCURE clinical sheet into memory.

    This function only reads the workbook; the raw file itself is never
    modified anywhere in this project.
    """
    return pd.read_excel(path, sheet_name=sheet_name)


def load_data_dictionary(
    path=config.RAW_DATA_PATH,
    sheet_name: str = config.DATA_DICTIONARY_SHEET_NAME,
) -> pd.DataFrame:
    """Load the workbook's own `Data Dictionary` sheet."""
    return pd.read_excel(path, sheet_name=sheet_name)


# =============================================================================
# Per-variable deterministic cleaning
# =============================================================================
def clean_ecog_ps(series: pd.Series) -> pd.Series:
    """Harmonise `ECOG PS` to the five valid ECOG grades plus an explicit
    `Missing` category (Section 4, Task 3).

    Rules:
      - `ECOG-Pt 0/1/2` -> `ECOG 0/1/2`   (exact semantic correction: the
        numeral is unambiguous, "-Pt" is a data-entry prefix)
      - `Unknown`        -> Missing        (explicit not-documented code)
      - `ECOG 0-1`       -> Missing        (ambiguous range; NOT arbitrarily
        assigned to either grade -- doing so would invent a clinical
        judgement that was never made)
      - true NaN         -> Missing

    ECOG 3 and ECOG 4 are deliberately kept separate (not pooled): each is
    a distinct clinical state, and the design matrix has no technical need
    for pooling (Section 4, Task 12).
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(config.ECOG_PS_EXACT_CORRECTIONS)
    mask = cleaned.isin(config.ECOG_PS_TO_MISSING)
    cleaned = cleaned.mask(mask)
    cleaned = cleaned.fillna(config.MISSING_LABEL)

    unexpected = set(cleaned.unique()) - config.ECOG_PS_VALID_LEVELS - {config.MISSING_LABEL}
    if unexpected:
        raise ValueError(f"Unexpected ECOG PS level(s) after cleaning: {unexpected}")
    return cleaned.astype(object)


def clean_ds_site(series: pd.Series) -> pd.Series:
    """Harmonise `Ds Site` case/spelling duplicates only (Section 4, Task 6).

    Only the two known case-duplicate pairs are merged. No clinically
    distinct or rare-but-valid site is pooled into an `Other` bucket --
    Section 3.2 established that rarity alone is not grounds to merge or
    remove patients.
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(config.DS_SITE_CASE_HARMONISATION)
    return cleaned.astype(object)


def clean_smoking_py(series: pd.Series) -> pd.Series:
    """Parse `Smoking PY` to numeric pack-years (Section 4, Task 4).

    `na` and every bound/range token (`<20`, `>50`, ...) become NaN -- a
    bound is not a measurement, and no numeric value is invented for it.
    The full numeric range is otherwise retained unmodified: no clipping,
    no winsorisation, no log transform. Imputation (a training-fold
    median) is applied later, inside the pipeline -- never here.
    """
    as_string = series.astype("string").str.strip()
    is_bound_token = as_string.isin(config.SMOKING_PY_MISSING_TOKENS)
    masked = as_string.mask(is_bound_token)
    numeric = pd.to_numeric(masked, errors="coerce")

    # Any remaining non-numeric, non-token string would indicate an
    # unanticipated raw value -- fail loudly rather than silently coercing.
    still_unparsed = as_string.notna() & ~is_bound_token & numeric.isna()
    if still_unparsed.any():
        bad_values = sorted(as_string[still_unparsed].unique())
        raise ValueError(f"Unparsable Smoking PY value(s): {bad_values}")

    return numeric.astype("float64")


def clean_smoking_status(series: pd.Series) -> pd.Series:
    """Normalise the casing of `Smoking Status` only (Section 4, Task 5).

    `Unknown` is retained as an explicit category and is NEVER imputed to
    another smoking status (Current / Ex-smoker / Non-smoker): doing so
    would assert a smoking history that was never documented.
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(config.SMOKING_STATUS_CASE_HARMONISATION)
    return cleaned.astype(object)


def clean_t_category(series: pd.Series) -> pd.Series:
    """Harmonise `T` to its valid AJCC categories plus an explicit
    `Missing` category (Section 4, Task 7).

    `TX` (explicit "cannot be assessed") and the four codes whose meaning
    is NOT documented (`T1 (2)`, `T2 (2)`, `T3 (2)`, `rT0`) become Missing.

    IMPORTANT: the `(2)` codes are NOT claimed to be duplicate encodings of
    a bare T value. Four of the five carry a `2nd Ca` flag dated shortly
    before `RT Start`, which is consistent with -- but does not prove --
    the code indexing a synchronous second primary tumour rather than the
    index tumour. Because the reference tumour cannot be established from
    this file, the value is treated as missing rather than guessed at.

    No T subcategory (T1a/T1b, T2a/T2b, T4a/T4b) is collapsed: each is
    site-specific and clinically distinct (Section 4, Task 7).
    """
    cleaned = series.astype("string").str.strip()
    mask = cleaned.isin(config.T_TO_MISSING_CODES)
    cleaned = cleaned.mask(mask)
    cleaned = cleaned.fillna(config.MISSING_LABEL)

    unexpected = set(cleaned.unique()) - config.T_VALID_LEVELS - {config.MISSING_LABEL}
    if unexpected:
        raise ValueError(f"Unexpected T level(s) after cleaning: {unexpected}")
    return cleaned.astype(object)


def clean_n_category(series: pd.Series) -> pd.Series:
    """Harmonise `N` to its valid AJCC categories plus an explicit
    `Missing` category (Section 4, Task 8).

    `NX` (explicit "cannot be assessed") becomes Missing. `N2` is retained
    as its own level and is NEVER assigned to `N2a`/`N2b`/`N2c`; no N
    subcategory is collapsed.
    """
    cleaned = series.astype("string").str.strip()
    mask = cleaned.isin(config.N_TO_MISSING_CODES)
    cleaned = cleaned.mask(mask)
    cleaned = cleaned.fillna(config.MISSING_LABEL)

    unexpected = set(cleaned.unique()) - config.N_VALID_LEVELS - {config.MISSING_LABEL}
    if unexpected:
        raise ValueError(f"Unexpected N level(s) after cleaning: {unexpected}")
    return cleaned.astype(object)


def clean_hpv(series: pd.Series) -> pd.Series:
    """Produce the deterministic HPV representation reserved for the
    (not yet implemented) PRIMARY + HPV sensitivity analysis
    (Section 4, Task 13).

    The Data Dictionary states only that a blank cell means "no data
    available" -- it does NOT state that the assay was never performed.
    The missing category is therefore labelled `Missing / not documented`,
    never `not tested`, and is never imputed to Positive/Negative.
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(config.HPV_VALUE_MAP)
    cleaned = cleaned.fillna(config.HPV_MISSING_LABEL)

    valid = {config.HPV_MISSING_LABEL, *config.HPV_VALUE_MAP.values()}
    unexpected = set(cleaned.unique()) - valid
    if unexpected:
        raise ValueError(f"Unexpected HPV level(s) after cleaning: {unexpected}")
    return cleaned.astype(object)


def missingness_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    missing_token_by_column: dict[str, str | None],
) -> pd.DataFrame:
    """Compact before/after "not usable" count table for the given columns.

    `missing_token_by_column` maps each column name to the value that
    represents "not usable" in the CLEANED column:
      - `None`   -> count actual NaN in the cleaned column (numeric
                    variables such as `Smoking PY`, which stay NaN and are
                    imputed later, inside the pipeline).
      - a string -> count occurrences of that explicit label (categorical
                    variables that were given an explicit "Missing" /
                    "Unknown" / "Missing / not documented" category by
                    deterministic cleaning).

    "before" always counts true blank cells in the RAW column
    (`raw_df[col].isna()`). Deterministic semantic cleaning can only ever
    convert a previously-valid-looking value (an explicit unknown/
    unassessable code, a non-numeric token) into "not usable" -- it never
    repairs a genuinely blank cell into a value -- so `after >= before` is
    asserted as an invariant.
    """
    n = len(raw_df)
    rows = []
    for col, token in missing_token_by_column.items():
        n_before = int(raw_df[col].isna().sum())
        if token is None:
            n_after = int(cleaned_df[col].isna().sum())
        else:
            n_after = int((cleaned_df[col] == token).sum())
        assert n_after >= n_before, (
            f"Deterministic cleaning DECREASED missingness for {col!r} "
            f"({n_before} -> {n_after}); semantic cleaning must never repair "
            "a blank cell into a value."
        )
        rows.append(
            {
                "variable": col,
                "missing_raw": n_before,
                "missing_after_cleaning": n_after,
                "pct_after_cleaning": round(n_after / n * 100, 2),
                "delta": n_after - n_before,
            }
        )
    return pd.DataFrame(rows)
