"""
PRIMARY preprocessing pipeline OBJECTS (Section 4).

This module only CONSTRUCTS scikit-learn Pipeline / ColumnTransformer
objects. Nothing here is ever fitted -- every learned parameter (the
`Smoking PY` imputation median, the encoder's category list, the scaler's
mean/variance) must be estimated only inside the training pipeline / CV,
strictly after the train/test split. Calling `build_preprocessor()` and
inspecting its (unfitted) structure is therefore leakage-safe even before
a split exists; calling `.fit(...)` on it is not, and must never be done
on data that includes the held-out test set.

Design (Section 4, confirmed):
    numeric      = [Age, Smoking PY]
        SimpleImputer(strategy="median")   -- Age has 0 missing values, so
                                               this step is a no-op for Age;
                                               NO missing-indicator is added.
        StandardScaler()

    categorical  = [Sex, ECOG PS, Smoking Status, Ds Site, T, N]
        OneHotEncoder(handle_unknown="ignore", drop=None)
        -- these six columns already carry an explicit "Missing"/"Unknown"
           category from deterministic cleaning (cleaning.py), so no
           imputer is used on the categorical branch.

No frequency-based rare-category pooling is implemented at this stage
(Section 4, Task 12): all clinically valid categories are retained, and
`handle_unknown="ignore"` protects any transform-time level absent from a
given training fold.

IMPORTANT (drop=None): because no reference level is dropped, a fitted
logistic-regression coefficient on a one-hot column must NOT be described
as a simple "category vs. a single omitted reference category" effect --
every level (including the categorical variable's most frequent one) has
its own coefficient, and their sum is constrained only through the model's
intercept and any regularisation, not through an omitted baseline.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_numeric_pipeline() -> Pipeline:
    """Numeric branch: median imputation (no indicator) + standardisation.

    Scaling is required for Logistic Regression and SVC (both are
    scale-sensitive) and is harmless for tree-based models (invariant to
    monotone transforms) -- so one shared numeric pipeline is scientifically
    safe across every PRIMARY model family.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Categorical branch: one-hot encoding only.

    No imputer is needed here: `ECOG PS`, `T` and `N` already carry an
    explicit "Missing" level, and `Smoking Status` carries an explicit
    "Unknown" level, from deterministic cleaning -- these are ordinary
    category values by the time this pipeline sees them.
    """
    return Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]
    )


def build_preprocessor(
    numeric_features: list[str] = config.PRIMARY_NUMERIC_FEATURES,
    categorical_features: list[str] = config.PRIMARY_CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Construct (but do not fit) the full PRIMARY ColumnTransformer.

    Safe to call before any train/test split exists, since constructing an
    (unfitted) scikit-learn object estimates nothing. It must be fitted
    only via `.fit()` / `.fit_transform()` calls issued strictly inside a
    cross-validation fold or on the frozen training set -- never on data
    that includes the held-out test set.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), list(numeric_features)),
            ("categorical", build_categorical_pipeline(), list(categorical_features)),
        ]
    )
