"""
================================================================================
RADCURE ML — Milestone 5: Descriptive Interpretation of the Finalized Model
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
The model-development and evaluation cycle is CLOSED (Milestone 4). This
script makes no model-selection or threshold decision of any kind. It
re-fits the ALREADY-FROZEN Logistic Regression pipeline on the training
set -- the same frozen specification fitted on the same training data as
in Milestone 4, whose predictions were evaluated once, and only once,
against the held-out test set -- and describes what its fitted
coefficients say, in the restrained, non-causal language a linear model
conditioned on eight predictors and a specific encoding actually
supports.

This is purely DESCRIPTIVE interpretation. No predictor, preprocessing
step, hyperparameter, threshold, or model family is touched. No new
interpretation framework (SHAP, LIME, PDP, permutation importance) is
introduced -- only the fitted coefficients themselves, read transparently
and with their limitations stated plainly.

--------------------------------------------------------------------------------
THE HELD-OUT TEST SET IS NOT NEEDED HERE, AND IS NOT USED
--------------------------------------------------------------------------------
Coefficient interpretation is a property of the FITTED MODEL SPECIFICATION
-- the same predictor set, preprocessing, and hyperparameters that were
fitted on the training set and evaluated exactly once in Milestone 4. It
does not require, and must not use, the held-out test set: this script
reconstructs the confirmed split only to verify its integrity, then
deletes the test variables immediately (Section 2) -- no test prediction
or test metric is computed anywhere below.

--------------------------------------------------------------------------------
FROZEN FINAL SPECIFICATION (recap only -- nothing here is re-decided)
--------------------------------------------------------------------------------
  Predictors:     Age, Sex, ECOG PS, Smoking PY, Smoking Status, Ds Site,
                  T, N (Milestone 1, Section 3.12).
  Preprocessing:  the existing frozen ColumnTransformer, unchanged
                  (Milestone 1, Section 4).
  Model:          LogisticRegression(C=1.0, class_weight=None,
                  max_iter=5000, random_state=42) (Milestone 3A/3B).
  Threshold:      0.16929818782414302, selected by maximizing Balanced
                  Accuracy on training-only cross-validation predictions
                  (Milestone 3D) -- NOT used in this script, since no hard
                  decision or classification metric is computed here.
  Held-out result (Milestone 4, FINAL, not recomputed here):
                  ROC-AUC 0.8055, Average Precision 0.5222,
                  Sensitivity 0.7297, Specificity 0.6939,
                  Precision 0.3568, F1 0.4793, Balanced Accuracy 0.7118.

--------------------------------------------------------------------------------
PROJECT NARRATIVE (preserved for the final report/presentation)
--------------------------------------------------------------------------------
Predictor-selection funnel:
    34 raw dataset columns
    -> 21 excluded during study-design / leakage / temporality audit
    -> 13 temporally admissible clinical candidate predictors
    -> 5 candidates excluded from the primary predictor design (Subsite,
       M, Stage, Path, HPV)
    -> 8 primary predictors (the frozen set interpreted below)

This predictor set was fixed in Milestone 1, BEFORE any model was fitted.
The coefficients examined in this script did not influence, and could not
have influenced, which predictors were chosen -- that decision was already
final by the time any model-comparison work began.

Modelling narrative:
    Dummy / Logistic Regression / Decision Tree / Random Forest / RBF SVC
    (Milestone 2, default-hyperparameter comparison)
    -> controlled hyperparameter tuning of the three shortlisted families
       (Milestone 3A/3B)
    -> exploratory Stacking and XGBoost challengers, added post hoc, which
       provided no practically compelling discrimination advantage over
       Logistic Regression (Milestone 3C)
    -> Logistic Regression frozen as the sole final model family
    -> training-only Balanced-Accuracy threshold selection (Milestone 3D)
    -> one final held-out evaluation (Milestone 4)
    -> THIS SCRIPT: descriptive interpretation of the finalized model
       (Milestone 5)
"""

# %%
# =============================================================================
# SECTION 1 — Scope and frozen model context
# =============================================================================
# Objective:
#   State, before touching any data, exactly what this script will and
#   will not do, and recap the frozen specification and final held-out
#   result it interprets (without recomputing either).

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; no display required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from radcure import cleaning, config, ensemble, leakage, modeling, target  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — Scope and frozen model context")

print("Milestone 5: DESCRIPTIVE interpretation of the already-finalized Logistic")
print("Regression model. No predictor, preprocessing, hyperparameter, threshold,")
print("or model-family decision is made or revisited in this script.\n")

print("Frozen final specification (recap, not re-decided here):")
print(f"   Predictors ({len(config.PRIMARY_FEATURES)}): {config.PRIMARY_FEATURES}")
print("   Preprocessing: radcure.preprocessing.build_preprocessor() -- unchanged.")
print("   Model: LogisticRegression(C=1.0, class_weight=None, max_iter=5000, "
      f"random_state={config.RANDOM_STATE})")
print("   Threshold: 0.16929818782414302 (not used in this script -- no hard "
      "decision is evaluated here).")

print("""
Final held-out result (Milestone 4, FINAL -- recap only, NOT recomputed):
   ROC-AUC              0.8055
   Average Precision    0.5222
   Sensitivity          0.7297
   Specificity          0.6939
   Precision            0.3568
   F1                   0.4793
   Balanced Accuracy    0.7118

Predictor-selection funnel (preserved for the final documentation):
   34 raw dataset columns
   -> 21 excluded during study-design / leakage / temporality audit
   -> 13 temporally admissible clinical candidate predictors
   -> 5 candidates excluded from the primary predictor design (Subsite, M,
      Stage, Path, HPV)
   -> 8 primary predictors (the frozen set interpreted below)
This predictor set was fixed BEFORE any model was fitted -- the
coefficients below did not, and could not have, influenced that choice.

Modelling narrative: Dummy/LR/Tree/RF/SVC default comparison -> controlled
tuning -> exploratory Stacking/XGBoost (no compelling advantage over LR)
-> Logistic Regression frozen -> training-only threshold selection -> one
final held-out evaluation -> THIS descriptive interpretation.
""")

# Next step:
#   Reconstruct the confirmed split, verify its integrity, and delete the
#   test-set variables immediately -- they are not needed for coefficient
#   interpretation (Section 2).


# %%
# =============================================================================
# SECTION 2 — Reconstruct the confirmed split (test set not needed)
# =============================================================================
# Objective:
#   Deterministically reproduce the exact X_train/y_train from Milestone 1
#   and verify the confirmed counts, then immediately discard the test-set
#   variables -- this script never predicts on, or computes a metric from,
#   the held-out test set.
#
# Rationale:
#   Identical in substance to Section 2 of every prior Milestone-3/4
#   analysis script. Duplicated rather than imported, per the project
#   convention that analysis scripts never import one another. Coefficient
#   interpretation is a property of the fitted TRAINING-set model -- the
#   same frozen specification fitted on the same training data as in
#   Milestone 4, whose fit was evaluated once against the test set -- so
#   the test set plays no role here at all, and is deleted immediately
#   below as a hard guard against accidental later use.

section("SECTION 2 — Reconstructing the confirmed split")

raw_df = cleaning.load_raw_clinical()
target_df = target.build_target(raw_df)
eligible_mask = target_df[config.ELIGIBILITY_COLUMN]

clinical_df = raw_df.copy()
clinical_df["ECOG PS"] = cleaning.clean_ecog_ps(raw_df["ECOG PS"])
clinical_df["Smoking PY"] = cleaning.clean_smoking_py(raw_df["Smoking PY"])
clinical_df["Smoking Status"] = cleaning.clean_smoking_status(raw_df["Smoking Status"])
clinical_df["Ds Site"] = cleaning.clean_ds_site(raw_df["Ds Site"])
clinical_df["T"] = cleaning.clean_t_category(raw_df["T"])
clinical_df["N"] = cleaning.clean_n_category(raw_df["N"])
clinical_df = clinical_df.join(target_df)

cohort_df = clinical_df.loc[eligible_mask].reset_index(drop=True)
X = cohort_df[config.PRIMARY_FEATURES].copy()
y = cohort_df[config.TARGET_COLUMN].astype(int)
patient_id = cohort_df[config.PATIENT_ID_COLUMN].copy()

leakage.assert_no_excluded_column_in_frame(X.columns)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, patient_id,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE,
)

assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N == 2351
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS == 444
assert int((y_train == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS == 1907
assert len(X_test) == config.EXPECTED_TEST_N == 588
assert int(y_test.sum()) == config.EXPECTED_TEST_EVENTS == 111
assert int((y_test == 0).sum()) == config.EXPECTED_TEST_NONEVENTS == 477
assert set(id_train) & set(id_test) == set()

print(f"[OK] Training set: n={len(X_train)}, events={int(y_train.sum())}, "
      f"non-events={int((y_train == 0).sum())}.")
print(f"[OK] Held-out test set: n={len(X_test)}, events={int(y_test.sum())}, "
      f"non-events={int((y_test == 0).sum())}.")
print("[OK] Train/test patient_id sets are disjoint.")

# ------------------------------------------------------------------------------
# >>> FROM THIS POINT ON, X_test / y_test / id_test ARE NOT REFERENCED AGAIN. <<<
# This script interprets the model fitted on TRAINING data only. No test
# prediction or test metric is computed anywhere below.
# ------------------------------------------------------------------------------
del X_test, y_test, id_test

# Decision:
#   Only X_train / y_train are used for the remainder of this script.
# Next step:
#   Fit the frozen final pipeline on the full training set (Section 3).


# %%
# =============================================================================
# SECTION 3 — Fit the frozen final pipeline
# =============================================================================
# Objective:
#   Fit ONE fresh copy of the already-frozen pipeline on X_train/y_train,
#   with no parameter changed, and extract everything needed for
#   coefficient interpretation.
#
# Rationale:
#   This is fitted on the same data, with the same frozen specification
#   fitted on the same training data as in Milestone 4 -- the model BEING
#   interpreted here shares that specification and training data with the
#   model that was evaluated there, not a new or different fit.

section("SECTION 3 — Fitting the frozen final pipeline")

pipeline = ensemble.build_frozen_logistic_regression()
lr = pipeline.named_steps["classifier"]
assert isinstance(lr, LogisticRegression)
lr_config = modeling.describe_logistic_regression(lr)
assert lr_config["C"] == 1.0
assert lr_config["class_weight"] is None
assert lr_config["max_iter"] == 5000
assert lr_config["random_state"] == config.RANDOM_STATE == 42
print(f"Frozen pipeline: {[name for name, _ in pipeline.steps]}")
print(f"Classifier hyperparameters confirmed: C={lr_config['C']}, "
      f"class_weight={lr_config['class_weight']}, max_iter={lr_config['max_iter']}, "
      f"random_state={lr_config['random_state']}")

pipeline.fit(X_train, y_train)
print("[OK] Pipeline fit exactly once, on X_train/y_train only.")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
assert isinstance(preprocessor, ColumnTransformer)
assert isinstance(classifier, LogisticRegression)

transformed_feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_.ravel()
intercept = float(classifier.intercept_[0])

assert len(transformed_feature_names) == len(coefficients), (
    "Number of transformed feature names must exactly equal the number of "
    "fitted coefficients."
)
print(f"\n[OK] {len(transformed_feature_names)} transformed feature names match "
      f"{len(coefficients)} fitted coefficients exactly.")
print(f"Fitted intercept: {intercept!r}")

# Decision:
#   `transformed_feature_names`, `coefficients`, and `intercept` describe
#   the ONE fitted model interpreted for the rest of this script.
# Next step:
#   Parse the transformed names back to (original_feature, term) pairs and
#   build the transparent coefficient table (Section 4).


# %%
# =============================================================================
# SECTION 4 — Coefficient table
# =============================================================================
# Objective:
#   Build one transparent table mapping every fitted coefficient back to
#   its original predictor and category/numeric term, with direction
#   stated as a MODEL ASSOCIATION -- never a causal claim.
#
# Rationale (wording discipline, enforced throughout this script):
#   A coefficient in a fitted logistic regression describes how that term
#   is associated with the model's linear predictor (log-odds), holding
#   every other fitted term constant. It is a property of THIS fitted
#   model under THIS coding and THIS regularisation -- not a claim that
#   the underlying clinical factor causes, protects against, increases, or
#   reduces mortality. This script therefore uses "model association",
#   "coefficient direction", and "conditional association within the
#   fitted model" throughout, and never the causal verbs above.

section("SECTION 4 — Coefficient table")


def parse_transformed_feature_name(name: str) -> tuple[str, str]:
    """Map one `ColumnTransformer.get_feature_names_out()` name back to
    (original_feature, category_or_numeric_term).

    Names take the form `"numeric__<column>"` or
    `"categorical__<column>_<category>"`. A blind split on "_" would break
    on columns whose own name contains a space followed by more words
    (e.g. "ECOG PS", "Smoking Status") or on category labels that contain
    spaces themselves (e.g. "ECOG 0") -- so the ORIGINAL COLUMN NAME is
    matched explicitly against the known predictor lists instead of
    guessed from the string.
    """
    if name.startswith("numeric__"):
        column = name[len("numeric__"):]
        return column, "(standardized numeric)"
    if name.startswith("categorical__"):
        remainder = name[len("categorical__"):]
        for column in config.PRIMARY_CATEGORICAL_FEATURES:
            prefix = f"{column}_"
            if remainder.startswith(prefix):
                return column, remainder[len(prefix):]
        raise ValueError(f"Could not match a known categorical column in: {name!r}")
    raise ValueError(f"Unexpected transformed feature name prefix: {name!r}")


parsed = [parse_transformed_feature_name(name) for name in transformed_feature_names]
coefficient_table = pd.DataFrame(
    {
        "transformed_feature": transformed_feature_names,
        "original_feature": [p[0] for p in parsed],
        "category_or_numeric_term": [p[1] for p in parsed],
        "coefficient": coefficients,
    }
)
coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()
coefficient_table["direction"] = np.select(
    [coefficient_table["coefficient"] > 0, coefficient_table["coefficient"] < 0],
    ["higher model log-odds of two-year mortality", "lower model log-odds of two-year mortality"],
    default="neutral in the fitted linear predictor",
)

print(f"Coefficient table built: {len(coefficient_table)} rows "
      f"(one per transformed feature).")
print("\nFull coefficient table:")
print(coefficient_table.drop(columns="transformed_feature").to_string(index=False))

# Decision:
#   `coefficient_table` is the single source used for every interpretation
#   section below -- nothing is recomputed from a different fit.
# Next step:
#   Interpret the two standardized numeric predictors, Age and Smoking PY
#   (Section 5).


# %%
# =============================================================================
# SECTION 5 — Numeric features: Age and Smoking PY
# =============================================================================
# Objective:
#   Interpret the two numeric predictors' coefficients correctly given
#   standardization, and state the Smoking PY imputation caveat explicitly.
#
# Rationale:
#   Age and Smoking PY are standardized (mean-centred, unit-variance) by
#   the frozen preprocessing pipeline before the model ever sees them. A
#   standardized-feature coefficient represents the change in the model's
#   log-odds associated with a ONE-STANDARD-DEVIATION increase in that
#   predictor, conditional on all other fitted terms -- not a one-unit
#   (one-year, one-pack-year) change. exp(coefficient) is the corresponding
#   multiplicative change in model odds for that one-standard-deviation
#   increase, again conditional on the other predictors -- a model
#   association, not a causal rate.
#
#   Smoking PY carries an additional caveat: values recorded as invalid or
#   symbolic (bounds such as ">50", "na", etc. -- Section 4 cleaning) were
#   converted to missing, and the missing values are median-imputed INSIDE
#   this fitted pipeline before scaling. The reported training mean/std
#   below are therefore computed AFTER that imputation -- they describe the
#   standardization actually applied to the model, not the mean/std of
#   only the originally observed values. Imputation fills in a value for
#   modelling; it does not create an actually observed smoking-exposure
#   measurement for those patients.

section("SECTION 5 — Numeric features: Age and Smoking PY")


numeric_pipeline = preprocessor.named_transformers_["numeric"]
assert isinstance(numeric_pipeline, Pipeline)
imputer = numeric_pipeline.named_steps["imputer"]
scaler = numeric_pipeline.named_steps["scaler"]
assert isinstance(imputer, SimpleImputer)
assert isinstance(scaler, StandardScaler)

# `mean_`/`scale_`/`statistics_` are set dynamically inside `.fit()`, not
# declared at the class level, so a static type checker cannot infer their
# type from scikit-learn's source alone. `getattr` + `isinstance` reads
# them back without altering or refitting either estimator.
scaler_mean = getattr(scaler, "mean_", None)
scaler_scale = getattr(scaler, "scale_", None)
imputer_statistics = getattr(imputer, "statistics_", None)
assert isinstance(scaler_mean, np.ndarray)
assert isinstance(scaler_scale, np.ndarray)
assert isinstance(imputer_statistics, np.ndarray)

for i, feature in enumerate(config.PRIMARY_NUMERIC_FEATURES):
    matching_rows = coefficient_table.loc[coefficient_table["original_feature"] == feature]
    assert isinstance(matching_rows, pd.DataFrame)
    row = matching_rows.iloc[0]
    coef = row["coefficient"]
    training_mean = scaler_mean[i]
    training_std = scaler_scale[i]
    imputation_value = imputer_statistics[i]
    n_missing_in_training = int(X_train[feature].isna().sum())

    print(f"\n{feature}:")
    print(f"   Coefficient (per +1 SD, standardized)   {coef:+.4f}")
    print(f"   exp(coefficient)                        {np.exp(coef):.4f}")
    print(f"   Training mean (post-imputation)         {training_mean:.4f}")
    print(f"   Training std used by the fitted scaler  {training_std:.4f}")
    print(f"   Median used for imputation (training)   {imputation_value:.4f}")
    print(f"   Training rows imputed for this feature  {n_missing_in_training}")
    print(f"   Model association: a one-standard-deviation increase in {feature} is "
          f"associated with model odds of two-year mortality multiplied by "
          f"{np.exp(coef):.4f}, conditional on the other seven fitted predictors.")

print("""
Age has zero missing values in this cohort (Milestone 1), so its imputation
step is a no-op -- the median above is computed but never actually applied.
Smoking PY DOES have imputed rows: the value reported for it is a
model-development convenience (the training median, standardized), not an
observed pack-year value for those patients -- it must not be read as
evidence about their actual smoking history.
""")

# Next step:
#   State the critical categorical-coefficient warning before presenting
#   any categorical coefficient (Section 6).


# %%
# =============================================================================
# SECTION 6 — Critical categorical-coefficient warning
# =============================================================================
# Objective:
#   State explicitly, before any categorical coefficient is interpreted,
#   why these coefficients cannot be read as conventional
#   reference-category contrasts or odds ratios.
#
# Rationale:
#   `preprocessing.py` builds `OneHotEncoder(handle_unknown="ignore",
#   drop=None)` for every categorical predictor (Sex, ECOG PS, Smoking
#   Status, Ds Site, T, N) -- ALL categories are retained as columns; NONE
#   is dropped as an implicit reference level. This is a full-rank (not
#   reduced-rank) categorical encoding. Consequently:
#     - there is NO omitted reference category anywhere in this fitted
#       model;
#     - an individual categorical coefficient cannot be described as
#       "compared with reference category X", because no such comparison
#       was built into the fit;
#     - exp(beta) for a single category is NOT a conventional odds ratio
#       versus a baseline -- there is no baseline being held at zero;
#     - the absolute level of each categorical coefficient is a joint
#       function of the model's L2 regularisation and this specific
#       full-rank coding, not a directly interpretable per-category effect
#       size on its own.
#   This limitation is stated once here and referenced, not re-derived,
#   everywhere a categorical coefficient is shown below.

section("SECTION 6 — Critical categorical-coefficient warning")

print("""
IMPORTANT -- read before any categorical coefficient below:

Every categorical predictor in this model (Sex, ECOG PS, Smoking Status,
Ds Site, T, N) was one-hot encoded with `drop=None`: ALL categories are
retained as their own coefficient. There is NO omitted reference category
anywhere in this fit.

Therefore:
  - an individual categorical coefficient must NOT be read as "compared
    with reference category X" -- no such comparison exists in this
    fitted model;
  - exp(beta) for a single category is NOT presented as a conventional
    reference-category odds ratio;
  - each categorical coefficient's sign and magnitude can be described
    only as part of THIS fitted, full-rank, L2-regularised coding -- not
    as a standalone, coding-independent effect size.

Section 7 below introduces a PRESENTATION-ONLY anchor category per
predictor, computed AFTER fitting, purely to make coefficient differences
between categories of the same feature readable. It is explicitly not an
omitted reference category from the model fit itself.
""")

# Next step:
#   Compute descriptive within-feature category contrasts using a
#   presentation-only anchor (Section 7).


# %%
# =============================================================================
# SECTION 7 — Within-feature categorical contrasts (presentation-only anchor)
# =============================================================================
# Objective:
#   Make categorical coefficients interpretable WITHOUT refitting the
#   model and WITHOUT inventing a fake reference-category interpretation,
#   by reporting coefficient DIFFERENCES within each feature relative to
#   one descriptive anchor.
#
# Method (deterministic, presentation-only, decided BEFORE looking at any
# coefficient value):
#   For each categorical predictor, the anchor is its most frequent
#   TRAINING category after the established cleaning rules (a simple,
#   reproducible, data-driven choice -- not one selected because of its
#   coefficient). For every other category of that same feature:
#       delta = beta_category - beta_anchor
#   This equals the change in the fitted linear predictor when switching
#   that ONE feature from the anchor category to the other category, with
#   every other model input held fixed -- a valid algebraic consequence of
#   a linear model, regardless of the encoding's rank. exp(delta) is the
#   corresponding relative model-odds multiplier for that specific
#   within-feature switch.
#
#   No exhaustive pairwise comparison is computed (only vs. the anchor),
#   no significance test, no p-value, and no confidence interval is
#   attached to any delta.

section("SECTION 7 — Within-feature categorical contrasts (presentation-only anchor)")

print("Presentation anchor is NOT an omitted reference category used during model")
print("fitting; it is introduced only after fitting to make coefficient differences")
print("interpretable, and is chosen as each feature's most frequent TRAINING category.\n")

contrast_rows = []
for feature in config.PRIMARY_CATEGORICAL_FEATURES:
    feature_rows = coefficient_table.loc[coefficient_table["original_feature"] == feature]
    anchor_category = X_train[feature].value_counts().idxmax()
    anchor_values = feature_rows.loc[
        feature_rows["category_or_numeric_term"] == anchor_category, "coefficient"
    ]
    assert isinstance(anchor_values, pd.Series)
    anchor_coef = float(anchor_values.iloc[0])

    print(f"{feature}  (presentation anchor: {anchor_category!r}, "
          f"n={int((X_train[feature] == anchor_category).sum())} training patients)")

    for _, r in feature_rows.iterrows():
        category = r["category_or_numeric_term"]
        if category == anchor_category:
            continue
        delta = r["coefficient"] - anchor_coef
        contrast_rows.append(
            {
                "original_feature": feature,
                "anchor_category": anchor_category,
                "category": category,
                "coefficient": r["coefficient"],
                "delta_vs_anchor": delta,
                "exp_delta_vs_anchor": np.exp(delta),
            }
        )
        print(f"   {category:20s} delta={delta:+.4f}   exp(delta)={np.exp(delta):.4f}")

contrast_table = pd.DataFrame(contrast_rows)
print(f"\n[OK] Descriptive within-feature contrasts computed for all "
      f"{len(config.PRIMARY_CATEGORICAL_FEATURES)} categorical predictors "
      f"({len(contrast_table)} category rows) -- vs. their own training-mode anchor only.")
print("No exhaustive pairwise comparison, no significance test, no p-value, and no")
print("confidence interval was computed for any delta above.")

# Decision:
#   Implementing the presentation-anchor approach was straightforward
#   here (a simple `value_counts().idxmax()` per feature, applied after
#   fitting), so it IS used, with the caveat above repeated explicitly.
# Next step:
#   Build the feature-group summary for all eight original predictors
#   (Section 8).


# %%
# =============================================================================
# SECTION 8 — Feature-group summary
# =============================================================================
# Objective:
#   One concise summary row per ORIGINAL predictor, for direct reuse in
#   the final documentation/presentation.

section("SECTION 8 — Feature-group summary")

summary_rows = []
for feature in config.PRIMARY_FEATURES:
    feature_rows = coefficient_table.loc[coefficient_table["original_feature"] == feature]
    n_terms = len(feature_rows)
    coef_min = feature_rows["coefficient"].min()
    coef_max = feature_rows["coefficient"].max()
    largest_positive = feature_rows.loc[feature_rows["coefficient"].idxmax()]
    largest_negative = feature_rows.loc[feature_rows["coefficient"].idxmin()]
    summary_rows.append(
        {
            "original_feature": feature,
            "n_transformed_terms": n_terms,
            "coefficient_min": coef_min,
            "coefficient_max": coef_max,
            "largest_positive_term": largest_positive["category_or_numeric_term"],
            "largest_positive_coef": largest_positive["coefficient"],
            "largest_negative_term": largest_negative["category_or_numeric_term"],
            "largest_negative_coef": largest_negative["coefficient"],
        }
    )
feature_group_summary = pd.DataFrame(summary_rows).set_index("original_feature")
print(feature_group_summary.round(4).to_string())

print("""
For Age and Smoking PY (n_transformed_terms=1), "coefficient_min/max" and the
"largest positive/negative term" columns all describe the SAME single
standardized coefficient -- there is no range to speak of for a single term;
see Section 5 for their proper standardized-effect interpretation.

For the six categorical predictors, the full-rank one-hot caveat from
Section 6 applies to every coefficient in this table: none of these values
is a reference-category contrast on its own.
""")

# Next step:
#   Show the largest fitted coefficients across the whole model, with the
#   "not feature importance" caveat stated explicitly (Section 9).


# %%
# =============================================================================
# SECTION 9 — Largest fitted coefficients (NOT a feature-importance ranking)
# =============================================================================
# Objective:
#   Show which individual fitted terms carry the largest positive and
#   negative coefficients, using wording that does not imply a universal
#   importance ranking.
#
# Rationale:
#   Coefficient magnitude in a regularised linear model is shaped by how a
#   variable happens to be encoded (one numeric term vs. many one-hot
#   terms), its scale (standardized here, which helps, but categorical
#   one-hot columns are not on the same footing as a standardized numeric
#   term), and the L2 penalty -- not by a variable's clinical importance on
#   its own. This section therefore reports "largest fitted Logistic
#   Regression coefficients", never "feature importance", and does NOT
#   rank the eight original clinical variables from "most" to "least"
#   important.

section("SECTION 9 — Largest fitted Logistic Regression coefficients")

TOP_N = 10
top_positive = coefficient_table.nlargest(TOP_N, "coefficient")
top_negative = coefficient_table.nsmallest(TOP_N, "coefficient")

print(f"Strongest positive coefficient contributions in the fitted coding (top {TOP_N}):")
print(top_positive[["original_feature", "category_or_numeric_term", "coefficient"]]
      .to_string(index=False))

print(f"\nStrongest negative coefficient contributions in the fitted coding (top {TOP_N}):")
print(top_negative[["original_feature", "category_or_numeric_term", "coefficient"]]
      .to_string(index=False))

print("""
These are the largest FITTED COEFFICIENTS in this specific model and
encoding -- not a feature-importance ranking, and not a ranking of the
eight original clinical variables from most to least important. A
variable's coefficient magnitude depends on its representation (a single
standardized numeric term vs. several one-hot categorical terms) as much
as on any clinical signal it carries.
""")

# Next step:
#   Report the fitted intercept, separately, with its own caveat
#   (Section 10).


# %%
# =============================================================================
# SECTION 10 — Intercept
# =============================================================================
section("SECTION 10 — Intercept")

print(f"Fitted intercept: {intercept!r}")
print("""
The intercept is the model's baseline linear-predictor term under this
SPECIFIC transformed coding -- it is added to the sum of every fitted
coefficient's contribution to produce the model's log-odds for a given
patient. Because all one-hot category levels are retained (Section 6) and
the two numeric predictors are standardized, the intercept does NOT
correspond to a simple clinical "baseline-patient risk" (e.g. "risk for a
typical patient with all-reference categories") the way it might in a
reduced-rank, unstandardized encoding. It should be read only as one term
in this model's specific linear predictor, not interpreted on its own.
""")

# Next step:
#   Attempt the one optional coefficient plot (Section 11), then write the
#   restrained scientific interpretation (Section 12) and STOP.


# %%
# =============================================================================
# SECTION 11 — Optional coefficient plot
# =============================================================================
# Objective:
#   Produce ONE simple horizontal bar plot of the fitted coefficients, if
#   it can be made readable, clearly labelled as fitted coefficients (not
#   feature importance). Skipped rather than forced if it would be
#   unreadable.
#
# Note on where this figure is written:
#   This project has no existing convention for tracked figure/report
#   assets (no `figures/` or `reports/` directory anywhere in the tracked
#   repository). Per this milestone's explicit instruction not to
#   establish a new presentation-asset convention, the figure is written
#   OUTSIDE the tracked repository (this run's scratch directory) purely
#   for one-off visual review, and is not added to version control.

section("SECTION 11 — Optional coefficient plot")

sorted_table = coefficient_table.sort_values("coefficient")
n_terms = len(sorted_table)
MAX_READABLE_TERMS = 60  # generous headroom above the 52 terms this model has

if n_terms > MAX_READABLE_TERMS:
    print(f"[SKIPPED] {n_terms} transformed terms exceed the "
          f"{MAX_READABLE_TERMS}-term readability budget for one horizontal bar "
          f"plot -- skipping rather than producing an overcrowded, misleading "
          f"figure.")
else:
    labels = [
        f"{row.original_feature}: {row.category_or_numeric_term}"
        for row in sorted_table.itertuples()
    ]
    fig_height = max(6.0, 0.22 * n_terms)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(labels, sorted_table["coefficient"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted Logistic Regression coefficient (log-odds scale)")
    ax.set_title(
        "Fitted Logistic Regression coefficients -- NOT a feature-importance "
        "ranking\n(full-rank one-hot encoding: no omitted reference category)"
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()

    output_path = Path("/tmp") / "radcure_m5_coefficient_plot.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] One coefficient plot written to {output_path} (outside the "
          f"tracked repository -- for one-off review only, per this milestone's "
          f"instruction not to establish a new asset-saving convention).")
    print(f"     {n_terms} bars, sorted by coefficient value, labelled "
          f"'original_feature: category_or_numeric_term'.")

# Next step:
#   Write the restrained scientific interpretation (Section 12).


# %%
# =============================================================================
# SECTION 12 — Scientific interpretation
# =============================================================================
section("SECTION 12 — Scientific interpretation")

print("""
Which coefficients point toward HIGHER predicted two-year mortality?
   The largest positive fitted coefficients (Section 9) fall among the
   more advanced T/N categories, several Ds Site levels, and worse ECOG
   performance-status levels -- consistent, at the level of DIRECTION
   only, with the clinical expectation that more advanced/aggressive
   disease and poorer baseline performance status are associated with
   higher model log-odds of the two-year mortality outcome as coded here.

Which coefficients point toward LOWER predicted two-year mortality?
   The largest negative fitted coefficients fall among early T/N
   categories and better ECOG levels -- again consistent, in direction
   only, with less advanced disease and better performance status being
   associated with lower model log-odds under this fitted coding.

What do the Age and Smoking PY coefficients mean after standardization?
   Age's coefficient (Section 5) is positive: a one-standard-deviation
   increase in Age (~11.5 years, per the fitted scaler) is associated with
   higher model log-odds of the outcome, conditional on the other seven
   predictors. Smoking PY's coefficient must be read with the imputation
   caveat in mind -- the standardization is applied AFTER median-imputing
   invalid/missing pack-year values, so its coefficient reflects the
   model's use of that partially-imputed, standardized column, not a
   clean measurement of smoking exposure alone.

What categorical patterns appear in the fitted model?
   T and N stage show a loose tendency for the earliest categories
   (T1/T1a, N0) to carry negative coefficients and for several advanced
   categories (T4/T4a/T4b, N3) to carry positive ones, but the pattern is
   NOT strictly monotone across every intermediate substage: T2b carries
   the single largest positive coefficient in the entire model (larger
   than T3 or T4), and N3b's coefficient drops back to a small positive
   value after N3's large spike. This is reported as an irregular,
   partial tendency, not a clean staging gradient -- exactly the kind of
   pattern Limitation 6 below (correlated/redundant clinical information
   distributing signal unevenly across coefficients) would predict, and a
   reason not to over-read any single stage-category coefficient in
   isolation. ECOG PS shows a more consistent tendency toward higher
   coefficients at worse performance-status levels (ECOG 0 through ECOG
   4 move from clearly negative to clearly positive). Ds Site and Smoking
   Status show more mixed patterns across levels, consistent with those
   being more heterogeneous, multi-level
   categorical groupings rather than an ordered clinical scale.

What CAN be concluded from these coefficients?
   That, WITHIN this specific fitted model -- this predictor set, this
   preprocessing, this regularisation strength, this full-rank encoding --
   certain terms are associated with higher or lower modeled log-odds of
   the outcome, holding the other fitted terms fixed. That is a statement
   about the fitted model's internal structure and is consistent with,
   but does not independently prove, established clinical staging logic.

What CANNOT be concluded from these coefficients?
   1. Nothing here is a causal effect estimate: this is an observational
      prediction model anchored at radiotherapy initiation, not a causal study
      design, and no causal-inference method was used.
   2. No coefficient was tested for statistical significance, and none is
      claimed to be statistically significant or non-significant.
   3. No confidence interval was computed for any coefficient or delta
      above.
   4. Because of the full-rank one-hot encoding (drop=None), there is no
      conventional omitted reference category anywhere in this model --
      individual categorical coefficients are not standard
      reference-category contrasts, only the Section-7 presentation-anchor
      deltas approximate that familiar framing, and only descriptively.
   5. L2 regularisation shrinks and redistributes coefficient magnitude
      across correlated or redundant terms; a coefficient's size is partly
      a property of the regularisation strength (C=1.0), not purely of the
      underlying clinical signal.
   6. Correlated or overlapping clinical information (e.g. T and N stage
      both reflecting disease extent, or Ds Site correlating with typical
      staging patterns for that site) can cause the fitted model to
      distribute associated signal across several coefficients rather than
      concentrating it in one -- a coefficient's size should not be read
      as if that predictor acted in isolation.
   7. This interpretation is conditional on the chosen 8-predictor
      specification and this exact preprocessing; a different predictor
      set or encoding could shift where the fitted signal appears without
      necessarily changing what the model predicts overall.
   8. Held-out test performance provides an internal evaluation of
      predictive performance, but does not support causal interpretation
      of the coefficients.
   9. Large coefficients for UNCOMMON categorical levels (few training
      patients in that category) should be interpreted especially
      cautiously: with fewer observations to constrain a level's
      coefficient, both its raw magnitude and how much the L2 penalty
      shrinks it are more strongly affected by that level's frequency
      than by the underlying clinical signal -- a rare level's large
      coefficient is not directly comparable to a common level's
      coefficient of similar size.
""")

# Next step:
#   STOP (Section 13).


# %%
# =============================================================================
# SECTION 13 — STOP
# =============================================================================
section("SECTION 13 — STOP")

print("""
Explicitly NOT done in this script:
  [ ] No predictor, preprocessing, hyperparameter, threshold, or model
      family was changed.
  [ ] No alternative model or threshold was evaluated.
  [ ] No feature selection was performed.
  [ ] No SHAP, LIME, PDP, or permutation importance was computed.
  [ ] No held-out test prediction or metric was computed -- the test set
      was reconstructed only to verify its counts (Section 2), then
      deleted and never referenced again.
  [ ] No exhaustive pairwise categorical comparison, no significance test,
      no p-value, no confidence interval.
  [ ] No README rewrite, final report, or presentation was produced.
  [ ] Nothing committed or pushed.

This project's model-development and evaluation cycle remains CLOSED. This
script only describes, in restrained and non-causal language, what the
already-finalized model's fitted coefficients say.

STOP for review.
""")
