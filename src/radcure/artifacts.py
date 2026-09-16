"""
Persistence layer for the sequential `analysis/01` -> `analysis/07` pipeline.

Each analysis stage consumes the artefacts written by its predecessors and
writes its own, so no stage recomputes work an earlier one already did.
This module holds only the mechanics of that hand-off -- paths, save/load
helpers, and a clear error when an upstream artefact is missing. The
scientific reasoning stays in `analysis/`.

Formats are deliberately plain:
    CSV     for tables (inspectable in any editor or spreadsheet)
    JSON    for configurations, metrics and metadata
    joblib  only for the one thing that genuinely needs it -- the fitted
            scikit-learn pipeline

Directory roles:
    data/processed/  the frozen modelling cohort with its split assignment
    artifacts/       stage results (tables, configurations, metrics)
    models/          the fitted final pipeline
    figures/         generated figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import cohort, config

# =============================================================================
# Artefact names, grouped by the stage that PRODUCES them
# =============================================================================
BASELINE_CV_RESULTS = "02_baseline_cv_results.csv"
BASELINE_SHORTLIST = "02_shortlist.json"

TUNED_RESULTS = "03_tuned_results.json"
SVC_GRID_RESULTS = "03_svc_grid_results.csv"

OOF_SCORES = "04_oof_scores.csv"
OOF_METADATA = "04_oof_metadata.json"
MODEL_COMPARISON = "04_model_comparison.csv"
FINAL_MODEL_SPECIFICATION = "04_final_model_specification.json"

THRESHOLD_SWEEP = "05_threshold_sweep.csv"
THRESHOLD = "05_threshold.json"

HELDOUT_PREDICTIONS = "06_heldout_predictions.csv"
HELDOUT_METRICS = "06_heldout_metrics.json"

COEFFICIENTS = "07_coefficients.csv"

#: Which script to run when an artefact is missing.
_PRODUCED_BY = {
    BASELINE_CV_RESULTS: "analysis/02_baseline_modeling.py",
    BASELINE_SHORTLIST: "analysis/02_baseline_modeling.py",
    TUNED_RESULTS: "analysis/03_hyperparameter_tuning.py",
    SVC_GRID_RESULTS: "analysis/03_hyperparameter_tuning.py",
    OOF_SCORES: "analysis/04_model_complementarity_and_exploratory_models.py",
    OOF_METADATA: "analysis/04_model_complementarity_and_exploratory_models.py",
    MODEL_COMPARISON: "analysis/04_model_complementarity_and_exploratory_models.py",
    FINAL_MODEL_SPECIFICATION: "analysis/04_model_complementarity_and_exploratory_models.py",
    THRESHOLD_SWEEP: "analysis/05_threshold_selection_and_model_freeze.py",
    THRESHOLD: "analysis/05_threshold_selection_and_model_freeze.py",
    HELDOUT_PREDICTIONS: "analysis/06_final_heldout_test_evaluation.py",
    HELDOUT_METRICS: "analysis/06_final_heldout_test_evaluation.py",
    COEFFICIENTS: "analysis/07_final_model_interpretation.py",
    config.MODELING_COHORT_FILENAME: "analysis/01_project_foundation.py",
    config.FINAL_MODEL_FILENAME: "analysis/05_threshold_selection_and_model_freeze.py",
}


def _require(path: Path) -> Path:
    """Fail with an actionable message naming the upstream script."""
    if not path.exists():
        producer = _PRODUCED_BY.get(path.name, "an earlier analysis script")
        raise FileNotFoundError(
            f"Required artefact not found: {path}\n"
            f"Run {producer} first -- the analysis scripts form a sequential "
            f"pipeline and each stage consumes the artefacts of its predecessors."
        )
    return path


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Generic table / configuration helpers
# =============================================================================
def save_table(df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
    """Write one result table to `artifacts/` as CSV.

    `to_csv` already writes each float as the shortest decimal string that
    parses back to the identical float64, so no explicit `float_format` is
    needed here -- see `load_table` for the reading side, which is where
    precision would otherwise be lost.
    """
    path = _ensure_dir(config.ARTIFACTS_DIR / name)
    df.to_csv(path, index=index)
    return path


def load_table(name: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Read one result table back from `artifacts/`.

    `float_precision="round_trip"` is the default for a concrete reason:
    `read_csv`'s ordinary parser is fast but not exactly correctly rounded,
    and it perturbs a majority of these values by about one unit in the
    last place. That is scientifically negligible, but it would mean a
    quantity recomputed from a reloaded artefact is not bitwise identical
    to the one that was written -- so, for example, the decision threshold
    that `analysis/05` derives from the out-of-fold scores would differ in
    its final digits from the value `analysis/04` actually produced. The
    round-trip parser removes that discrepancy entirely.
    """
    read_csv_kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(_require(config.ARTIFACTS_DIR / name), **read_csv_kwargs)


def save_json(payload: dict, name: str) -> Path:
    """Write one configuration/metric artefact to `artifacts/` as JSON."""
    path = _ensure_dir(config.ARTIFACTS_DIR / name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_json(name: str) -> dict:
    """Read one configuration/metric artefact back from `artifacts/`."""
    return json.loads(_require(config.ARTIFACTS_DIR / name).read_text(encoding="utf-8"))


# =============================================================================
# The frozen modelling cohort (produced by analysis/01)
# =============================================================================
def save_modeling_cohort(split_df: pd.DataFrame) -> Path:
    """Persist the frozen modelling cohort with its split assignment.

    The frame must come from `cohort.assign_frozen_split`, i.e. already be
    in split order and carry `cohort_row` / `split`. Row order is preserved
    exactly, because cross-validation folds depend on it.
    """
    cohort.assert_split_integrity(split_df)
    path = _ensure_dir(config.PROCESSED_DATA_DIR / config.MODELING_COHORT_FILENAME)
    split_df.to_csv(path, index=False)
    return path


def load_modeling_cohort() -> pd.DataFrame:
    """Load the frozen modelling cohort, preserving file row order.

    Uses the same exactly-rounded float parser as `load_table`, so `Age` and
    `Smoking PY` are the identical float64 values `analysis/01` wrote.
    """
    path = _require(config.PROCESSED_DATA_DIR / config.MODELING_COHORT_FILENAME)
    return pd.read_csv(
        path,
        dtype={config.PATIENT_ID_COLUMN: "string"},
        float_precision="round_trip",
    )


def load_training_partition() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """`(X_train, y_train, id_train)` exactly as `analysis/01` produced them.

    The held-out partition is not read, not returned and not present in the
    caller's namespace -- during model development it is simply never
    loaded.
    """
    return cohort.partition(load_modeling_cohort(), cohort.TRAIN_SPLIT)


def load_heldout_partition() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """`(X_test, y_test, id_test)` exactly as `analysis/01` produced them.

    Reserved for the single final evaluation in `analysis/06`.
    """
    return cohort.partition(load_modeling_cohort(), cohort.TEST_SPLIT)


# =============================================================================
# The fitted final pipeline (produced by analysis/05)
# =============================================================================
def save_final_model(pipeline, name: str = config.FINAL_MODEL_FILENAME) -> Path:
    """Persist the fitted final pipeline with joblib."""
    path = _ensure_dir(config.MODELS_DIR / name)
    joblib.dump(pipeline, path)
    return path


def load_final_model(name: str = config.FINAL_MODEL_FILENAME):
    """Load the fitted final pipeline. It is used as-is and never refitted."""
    return joblib.load(_require(config.MODELS_DIR / name))
