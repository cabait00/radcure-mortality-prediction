# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 07 Interpretation of the Finalized Model
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
One question: what did the frozen Logistic Regression learn, and how do
individual predictor values contribute to its score?

It loads the fitted pipeline that `analysis/05` persisted -- the same fitted
object `analysis/06` evaluated -- and describes its coefficients in the
restrained, non-causal language a regularised linear model on eight predictors
under a specific encoding actually supports. Nothing is refitted, no model
decision is made or revisited, and no new interpretation framework (SHAP, LIME,
PDP, permutation importance) is introduced.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  models/final_logistic_regression.joblib  fitted in analysis/05
          data/processed/modeling_cohort.csv       training partition, used
                                                   only for category sample
                                                   sizes
Produces  artifacts/07_coefficients.csv
          figures/07_coefficients_and_contrasts.png

The held-out partition is not loaded: coefficient interpretation is a property
of the fitted model, and held-out results play no part in it.

--------------------------------------------------------------------------------
FROZEN SPECIFICATION (recap only — nothing here is re-decided)
--------------------------------------------------------------------------------
  Predictors:     Age, Sex, ECOG PS, Smoking PY, Smoking Status, Ds Site, T, N
                  (analysis/01, Section 3.5), fixed BEFORE any model was
                  fitted -- so the coefficients below could not have influenced
                  which predictors were chosen.
  Preprocessing:  the frozen ColumnTransformer (analysis/01, Section 4).
  Model:          LogisticRegression(C=1.0, class_weight=None, max_iter=5000,
                  random_state=42) (analysis/03, selected in analysis/04).
  Threshold:      selected in analysis/05 -- not used here, since no hard
                  decision or classification metric is computed in this script.
"""

# %%
# =============================================================================
# SECTION 1 — Load the fitted model and the training category frequencies
# =============================================================================
# Objective:
#   Load the fitted pipeline and extract everything coefficient interpretation
#   needs, plus the training-set frequency of each categorical level.
#
# Rationale:
#   The model is LOADED, not refitted: `analysis/05` fitted it once on the
#   training partition, `analysis/06` evaluated that exact object, and this
#   script describes that same object. The training partition is read only to
#   count how many patients support each category -- no model is fitted on it
#   here.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from radcure import artifacts, config, modeling, plots  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 120)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — The fitted model being interpreted")

pipeline = artifacts.load_final_model()
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
assert isinstance(preprocessor, ColumnTransformer)
assert isinstance(classifier, LogisticRegression)

lr_config = modeling.describe_logistic_regression(classifier)
assert lr_config["C"] == 1.0
assert lr_config["class_weight"] is None
assert lr_config["max_iter"] == 5000
assert lr_config["random_state"] == config.RANDOM_STATE

print(f"Loaded models/{config.FINAL_MODEL_FILENAME} -- fitted in analysis/05, not refitted here.")
print(f"   LogisticRegression(C={lr_config['C']}, class_weight={lr_config['class_weight']}, "
      f"max_iter={lr_config['max_iter']}, random_state={lr_config['random_state']})")
print(f"   Predictors ({len(config.PRIMARY_FEATURES)}): {config.PRIMARY_FEATURES}")

transformed_feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_.ravel()
intercept = float(classifier.intercept_[0])
assert len(transformed_feature_names) == len(coefficients)
print(f"\n{len(transformed_feature_names)} transformed terms, {len(coefficients)} fitted "
      f"coefficients, intercept {intercept:+.7f}")

# Training partition -- used ONLY for category sample sizes and the
# imputation/scaling context reported in Section 2.
X_train, y_train, _ = artifacts.load_training_partition()
assert len(X_train) == config.EXPECTED_TRAIN_N
print(f"Training partition loaded for category counts only: n={len(X_train)}.")
print("The held-out partition is not loaded by this script.")


# %%
# =============================================================================
# SECTION 2 — The two numeric predictors
# =============================================================================
# Objective:
#   Interpret Age and Smoking PY correctly given standardisation, and state the
#   Smoking PY imputation caveat explicitly.
#
# Rationale:
#   Both are mean-centred and scaled to unit variance by the frozen pipeline
#   before the model sees them, so a coefficient describes the change in the
#   model's log-odds associated with a ONE-STANDARD-DEVIATION increase --
#   not a one-year or one-pack-year change -- conditional on the other fitted
#   terms. exp(coefficient) is the corresponding multiplicative change in model
#   odds: a model association, not a causal rate.
#
#   Smoking PY carries an extra caveat. Values recorded as bounds or symbolic
#   tokens (">50", "na", ...) were converted to missing by deterministic
#   cleaning, and the missing values are median-imputed INSIDE this fitted
#   pipeline before scaling. The mean/std reported below are therefore computed
#   AFTER that imputation: they describe the standardisation actually applied,
#   not the distribution of only the observed values. Imputation fills in a
#   value for modelling; it does not create an observed smoking-exposure
#   measurement.

section("SECTION 2 — Numeric predictors: Age and Smoking PY")

numeric_pipeline = preprocessor.named_transformers_["numeric"]
assert isinstance(numeric_pipeline, Pipeline)
imputer = numeric_pipeline.named_steps["imputer"]
scaler = numeric_pipeline.named_steps["scaler"]
assert isinstance(imputer, SimpleImputer)
assert isinstance(scaler, StandardScaler)

# `mean_`/`scale_`/`statistics_` are set dynamically inside `.fit()`, not
# declared at class level, so a static type checker cannot infer them from
# scikit-learn's source. `getattr` + `isinstance` reads them back without
# altering or refitting anything.
scaler_mean = getattr(scaler, "mean_", None)
scaler_scale = getattr(scaler, "scale_", None)
imputer_statistics = getattr(imputer, "statistics_", None)
assert isinstance(scaler_mean, np.ndarray)
assert isinstance(scaler_scale, np.ndarray)
assert isinstance(imputer_statistics, np.ndarray)

numeric_rows = []
for i, feature in enumerate(config.PRIMARY_NUMERIC_FEATURES):
    coef = float(coefficients[list(transformed_feature_names).index(f"numeric__{feature}")])
    n_imputed = int(X_train[feature].isna().sum())
    numeric_rows.append({
        "feature": feature,
        "coefficient": coef,
        "exp_coefficient": float(np.exp(coef)),
        "training_mean_post_imputation": float(scaler_mean[i]),
        "training_std": float(scaler_scale[i]),
        "imputation_median": float(imputer_statistics[i]),
        "n_imputed_in_training": n_imputed,
    })
    print(f"\n{feature}:")
    print(f"   Coefficient (per +1 SD)                 {coef:+.4f}")
    print(f"   exp(coefficient)                        {np.exp(coef):.4f}")
    print(f"   Training mean (post-imputation)         {scaler_mean[i]:.4f}")
    print(f"   Training std used by the fitted scaler  {scaler_scale[i]:.4f}")
    print(f"   Median used for imputation (training)   {imputer_statistics[i]:.4f}")
    print(f"   Training rows imputed for this feature  {n_imputed}")
    print(f"   -> a one-standard-deviation increase in {feature} "
          f"({scaler_scale[i]:.2f} units) is associated with model odds of two-year "
          f"mortality multiplied by {np.exp(coef):.4f}, conditional on the other seven "
          f"fitted predictors.")

print("""
Age has zero missing values in this cohort, so its imputation step is a no-op --
the median above is computed but never applied. Smoking PY does have imputed
rows: its coefficient reflects the model's use of a partially imputed,
standardised column, not a clean measurement of smoking exposure alone.
""")


# %%
# =============================================================================
# SECTION 3 — Why categorical coefficients are not reference-category effects
# =============================================================================
# Objective:
#   State the encoding consequence explicitly, before any categorical
#   coefficient is shown.
#
# Rationale:
#   The pipeline uses OneHotEncoder(handle_unknown="ignore", drop=None) for all
#   six categorical predictors: ALL categories are retained as columns and NONE
#   is dropped as an implicit reference level. This is a full-rank, not a
#   reduced-rank, encoding.

section("SECTION 3 — The full-rank encoding, and what it rules out")

print("""
Every categorical predictor (Sex, ECOG PS, Smoking Status, Ds Site, T, N) was
one-hot encoded with `drop=None`, so every category carries its own
coefficient and there is NO omitted reference category anywhere in this fit.

Therefore:
  - an individual categorical coefficient must NOT be read as "compared with
    reference category X" -- no such comparison exists in the fitted model;
  - exp(beta) for a single category is NOT a conventional reference-category
    odds ratio: there is no baseline being held at zero;
  - each coefficient's level is a joint function of this specific full-rank
    coding and the model's L2 regularisation, not a coding-independent effect
    size on its own.

Section 4 introduces a PRESENTATION anchor per predictor -- computed after
fitting, purely to make within-feature differences readable. It is explicitly
not an omitted reference category from the fit.
""")


# %%
# =============================================================================
# SECTION 4 — The coefficient and contrast table
# =============================================================================
# Objective:
#   Map every fitted coefficient back to its original predictor and category,
#   attach the training sample size that supports it, and compute a descriptive
#   within-feature contrast against one presentation anchor.
#
# Method (deterministic, presentation-only, decided BEFORE looking at any
# coefficient value):
#   For each categorical predictor the anchor is its most frequent TRAINING
#   category -- a simple, reproducible, data-driven choice, not one selected
#   because of its coefficient. For every other category of that feature:
#
#       delta = beta_category - beta_anchor
#
#   which is the change in the fitted linear predictor when that ONE feature
#   switches from the anchor category to the other, with every other model
#   input held fixed. That is a valid algebraic consequence of a linear model
#   regardless of the encoding's rank. exp(delta) is the corresponding relative
#   model-odds multiplier for that specific within-feature switch.
#
#   No exhaustive pairwise comparison, no significance test, no p-value and no
#   confidence interval is attached to any delta.

section("SECTION 4 — Coefficient and contrast table")


def parse_transformed_feature_name(name: str) -> tuple[str, str]:
    """Map one `ColumnTransformer.get_feature_names_out()` name back to
    (original_feature, category_or_numeric_term).

    Names take the form `"numeric__<column>"` or
    `"categorical__<column>_<category>"`. A blind split on "_" would break on
    columns whose own name contains a space ("ECOG PS", "Smoking Status") or on
    category labels containing spaces ("ECOG 0"), so the ORIGINAL COLUMN NAME
    is matched explicitly against the known predictor lists.
    """
    if name.startswith("numeric__"):
        return name[len("numeric__"):], "(standardized numeric)"
    if name.startswith("categorical__"):
        remainder = name[len("categorical__"):]
        for column in config.PRIMARY_CATEGORICAL_FEATURES:
            prefix = f"{column}_"
            if remainder.startswith(prefix):
                return column, remainder[len(prefix):]
        raise ValueError(f"Could not match a known categorical column in: {name!r}")
    raise ValueError(f"Unexpected transformed feature name prefix: {name!r}")


rows = []
for name, coef in zip(transformed_feature_names, coefficients):
    feature, term = parse_transformed_feature_name(name)
    if feature in config.PRIMARY_CATEGORICAL_FEATURES:
        in_category = X_train[feature] == term
        n_train = int(in_category.sum())
        n_events = int(y_train[in_category].sum())
    else:
        n_train = len(X_train)
        n_events = int(y_train.sum())
    rows.append({
        "original_feature": feature,
        "term": term,
        "coefficient": float(coef),
        "n_train": n_train,
        "n_events_train": n_events,
        "event_rate_train": n_events / n_train if n_train else float("nan"),
    })

coefficient_table = pd.DataFrame(rows)

# Presentation anchor per categorical feature, and the contrast against it.
anchors: dict[str, str] = {}
coefficient_table["is_presentation_anchor"] = False
coefficient_table["contrast_vs_anchor"] = np.nan
coefficient_table["exp_contrast_vs_anchor"] = np.nan

for feature in config.PRIMARY_CATEGORICAL_FEATURES:
    mask = coefficient_table["original_feature"] == feature
    anchor = str(X_train[feature].value_counts().idxmax())
    anchors[feature] = anchor
    anchor_coef = float(
        coefficient_table.loc[mask & (coefficient_table["term"] == anchor), "coefficient"].iloc[0]
    )
    coefficient_table.loc[mask, "contrast_vs_anchor"] = (
        coefficient_table.loc[mask, "coefficient"] - anchor_coef
    )
    coefficient_table.loc[mask, "exp_contrast_vs_anchor"] = np.exp(
        coefficient_table.loc[mask, "contrast_vs_anchor"]
    )
    coefficient_table.loc[mask & (coefficient_table["term"] == anchor), "is_presentation_anchor"] = True

print("Presentation anchor per categorical predictor (most frequent TRAINING category):")
for feature, anchor in anchors.items():
    n = int((X_train[feature] == anchor).sum())
    print(f"   {feature:16s} {anchor!r:28s} n={n}")

print(f"\nFull coefficient table ({len(coefficient_table)} terms):")
display_columns = [
    "original_feature", "term", "n_train", "coefficient",
    "contrast_vs_anchor", "exp_contrast_vs_anchor", "is_presentation_anchor",
]
for feature in config.PRIMARY_FEATURES:
    block = coefficient_table.loc[coefficient_table["original_feature"] == feature]
    print(f"\n--- {feature} ---")
    print(block[display_columns].drop(columns="original_feature")
          .sort_values("coefficient", ascending=False).round(4).to_string(index=False))

coefficients_path = artifacts.save_table(coefficient_table, artifacts.COEFFICIENTS)
print(f"\n[artifact] {coefficients_path.relative_to(PROJECT_ROOT)}")
print("The table also records each category's training event count and rate as descriptive")
print("context for its sample size. Those are cohort statistics, not model estimates.")

print("""
Category frequency and coefficient stability:
Large coefficients for rare categories should be interpreted cautiously,
because their estimates are supported by fewer observations. L2 regularisation
reduces, but does not eliminate, this instability. The concern is UNCERTAINTY:
a rare level's coefficient is less well determined and less comparable to a
common level's coefficient of similar magnitude. It is not a claim that low
frequency biases an estimate in any particular direction.
""")


# %%
# =============================================================================
# SECTION 5 — Interpretation figure
# =============================================================================
# Objective:
#   One readable horizontal figure showing the concrete terms -- not just
#   per-feature ranges -- with the training sample size behind each one.

section("SECTION 5 — Interpretation figure")

plot_table = coefficient_table.sort_values(
    ["original_feature", "coefficient"],
    key=lambda s: s.map({f: i for i, f in enumerate(config.PRIMARY_FEATURES)}) if s.name == "original_feature" else s,
    ascending=[True, True],
).reset_index(drop=True)

labels, values, colors, separators = [], [], [], []
previous_feature = None
for _, row in plot_table.iterrows():
    if previous_feature is not None and row["original_feature"] != previous_feature:
        separators.append(len(labels) - 0.5)
    previous_feature = row["original_feature"]
    if row["original_feature"] in config.PRIMARY_NUMERIC_FEATURES:
        labels.append(f"{row['original_feature']}  (per +1 SD)")
    else:
        anchor_mark = " ◆" if row["is_presentation_anchor"] else ""
        labels.append(f"{row['original_feature']}: {row['term']}  (n={row['n_train']}){anchor_mark}")
    values.append(row["coefficient"])
    colors.append(plots.EVENT_COLOR if row["coefficient"] > 0 else plots.NON_EVENT_COLOR)

fig, ax = plt.subplots(figsize=(11, max(9.0, 0.30 * len(labels))))
positions = np.arange(len(labels))
ax.barh(positions, values, color=colors, height=0.72)
ax.set_yticks(positions, labels=labels, fontsize=8)
ax.invert_yaxis()  # read top-to-bottom in the frozen predictor-set order
ax.axvline(0, color="black", linewidth=1.0)
for separator in separators:
    ax.axhline(separator, color=plots.NEUTRAL_COLOR, linewidth=0.7, linestyle=":")

ax.set_xlabel("Fitted Logistic Regression coefficient (log-odds scale)")
ax.set_ylabel("")
ax.set_ylim(len(labels) - 0.3, -0.7)
ax.set_title("What the frozen Logistic Regression learned", fontsize=13, pad=14)
ax.legend(
    handles=[
        Rectangle((0, 0), 1, 1, color=plots.EVENT_COLOR,
                  label="higher model log-odds of two-year mortality"),
        Rectangle((0, 0), 1, 1, color=plots.NON_EVENT_COLOR,
                  label="lower model log-odds"),
        Line2D([], [], linestyle="none", marker="D", color="black", markersize=5,
               label="this feature's presentation anchor"),
    ],
    loc="upper right", fontsize=8.5,
)

# Explicit margins rather than tight_layout: the footnote below the axes needs
# reserved space, which tight_layout would otherwise reclaim.
fig.subplots_adjust(left=0.30, right=0.97, top=0.965, bottom=0.085)
fig.text(
    0.5, 0.012,
    "Full-rank one-hot encoding (drop=None): there is NO omitted reference category, so a single "
    "categorical coefficient is not a reference-category effect.\n"
    "n = training patients in that category; coefficients for rare categories rest on fewer "
    "observations and are correspondingly less stable.\n"
    "Descriptive model structure only — not causal effects, not statistical significance, and "
    "not a feature-importance ranking.",
    ha="center", va="bottom", fontsize=8.5, style="italic",
)
plots.save_figure(fig, "07_coefficients_and_contrasts.png")


# %%
# =============================================================================
# SECTION 6 — Reading the fitted model
# =============================================================================

section("SECTION 6 — Reading the fitted model")

TOP_N = 8
top_positive = coefficient_table.nlargest(TOP_N, "coefficient")
top_negative = coefficient_table.nsmallest(TOP_N, "coefficient")
print(f"Largest positive fitted coefficients (top {TOP_N}):")
print(top_positive[["original_feature", "term", "n_train", "coefficient"]].round(4).to_string(index=False))
print(f"\nLargest negative fitted coefficients (top {TOP_N}):")
print(top_negative[["original_feature", "term", "n_train", "coefficient"]].round(4).to_string(index=False))

print(f"\nFitted intercept: {intercept:+.7f}")

print(f"""
These are the largest FITTED COEFFICIENTS in this specific model and encoding --
not a feature-importance ranking, and not a ranking of the eight clinical
variables from most to least important. A variable's coefficient magnitude
depends on its representation (one standardised numeric term vs. several
one-hot terms) as much as on any clinical signal it carries.

The intercept is one term in this model's linear predictor, added to the sum of
every coefficient's contribution. Because all one-hot levels are retained and
both numeric predictors are standardised, it does NOT correspond to a
"baseline-patient risk" the way it might in a reduced-rank, unstandardised
encoding, and should not be interpreted on its own.

WHAT THE COEFFICIENTS SHOW
--------------------------
Higher model log-odds: the largest positive coefficients fall among the more
advanced T and N categories, several Ds Site levels, and worse ECOG
performance-status levels. Lower model log-odds: early T/N categories and
better ECOG levels. Both directions are consistent with clinical staging logic
-- at the level of DIRECTION only, and as a property of this fitted model.

Age's coefficient is positive: a one-standard-deviation increase (~{scaler_scale[0]:.1f}
years, per the fitted scaler) is associated with higher model log-odds,
conditional on the other seven predictors. Smoking PY's must be read with the
imputation caveat from Section 2.

ECOG PS shows the most consistent tendency, moving from clearly negative at
ECOG 0 toward clearly positive at the worst grades. T and N show only a loose,
NON-monotone tendency: the earliest categories carry negative coefficients and
several advanced categories positive ones, but intermediate substages do not
line up in order. Ds Site and Smoking Status are more mixed, consistent with
their being heterogeneous multi-level groupings rather than ordered clinical
scales. That irregularity is what limitation 5 below would predict, and it is a
reason not to over-read any single stage-category coefficient in isolation.

WHAT CANNOT BE CONCLUDED
------------------------
  1. Nothing here is a causal effect estimate. This is an observational
     prediction model anchored at radiotherapy initiation, not a causal study
     design, and no causal-inference method was used.
  2. No coefficient was tested for statistical significance, and none is
     claimed to be significant or non-significant. No p-value and no confidence
     interval was computed for any coefficient or contrast.
  3. Because of the full-rank one-hot encoding there is no conventional omitted
     reference category: individual categorical coefficients are not standard
     reference-category contrasts, and only the Section-4 presentation-anchor
     deltas approximate that familiar framing -- descriptively.
  4. L2 regularisation shrinks and redistributes coefficient magnitude; a
     coefficient's size is partly a property of the regularisation strength
     (C=1.0), not purely of the underlying clinical signal.
  5. Correlated or overlapping clinical information -- T and N both reflecting
     disease extent, or Ds Site correlating with typical staging patterns for
     that site -- can cause the model to distribute associated signal across
     several coefficients rather than concentrating it in one.
  6. This interpretation is conditional on the chosen 8-predictor
     specification and this exact preprocessing. A different predictor set or
     encoding could shift where the fitted signal appears without necessarily
     changing what the model predicts overall.
  7. Large coefficients for uncommon categorical levels are less well
     determined than their magnitude alone suggests (Section 4).
""")


# %%
# =============================================================================
# SECTION 7 — Project summary
# =============================================================================
section("SECTION 7 — Project summary")

print(f"""
The pipeline, end to end:

  01  raw workbook -> deterministic cleaning -> RT-Start-anchored 730-day
      target -> {config.EXPECTED_ELIGIBLE_N} eligible patients -> one frozen
      stratified 80/20 split ({config.EXPECTED_TRAIN_N} / {config.EXPECTED_TEST_N})
  02  five default model families compared on training CV; three shortlisted
  03  one controlled grid search per shortlisted family
  04  out-of-fold complementarity, one stacking experiment, one exploratory
      XGBoost challenger -> an empirical performance plateau, and Logistic
      Regression selected on discrimination, simplicity and parsimony
  05  one decision threshold chosen from training-only out-of-fold predictions
      by maximising Balanced Accuracy (equivalently Youden's J); the model
      fitted on the full training partition and frozen
  06  held-out evaluation of that frozen model at that frozen threshold
  07  this description of what the frozen model learned

Every model-development decision used training data only. The held-out
partition is loaded only in analysis/06, for that dedicated evaluation stage,
and its results were not used to revise anything.

What this project does NOT establish: causal relationships; performance on an
independent population, institution or time period; or that a different
predictor set could not do better. External validation, and subgroup
validation, would be the relevant next questions before any clinical use --
both outside the scope defined here.
""")

# %%
