# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 02 Baseline Model-Family Comparison (training-only)
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
A training-only comparison of five default-hyperparameter classical model
families on the frozen PRIMARY predictor set, under a fixed 5-fold stratified
cross-validation design. It answers one question -- "which default model
configurations show consistent cross-validated discrimination before tuning?"
-- and shortlists the families worth a controlled tuning pass in
`analysis/03`.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  data/processed/modeling_cohort.csv (analysis/01), training partition
          only. The held-out partition is never loaded by this script.
Produces  artifacts/02_baseline_cv_results.csv  per-fold CV scores per model
          artifacts/02_shortlist.json           the families carried into 03
          figures/02_baseline_model_comparison.png

--------------------------------------------------------------------------------
DESIGN, FIXED BEFORE ANY RESULT IS SEEN
--------------------------------------------------------------------------------
  - Five DEFAULT-hyperparameter candidates: Dummy (prior), Logistic
    Regression, Decision Tree, Random Forest, RBF SVC. No tuning, no
    class_weight, no resampling, no probability calibration -- this is a
    default-family comparison, and mixing in a modelling choice such as
    class_weight would confound it. `class_weight` remains an explicit,
    controlled candidate for the tuning pass in `analysis/03`.
  - Three pre-specified CV metrics: roc_auc (PRIMARY, threshold-independent
    ranking quality), average_precision (SECONDARY, specifically informative
    at ~18.9% prevalence, where a no-skill PR curve sits near the prevalence
    rather than at 0.5), balanced_accuracy (SUPPLEMENTARY hard-prediction
    view). Ordinary accuracy is not used for model selection. Pre-specifying
    the metrics before any result exists prevents post-hoc metric shopping.
  - Every model is evaluated with the SAME fixed
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

The held-out test set is not loaded, fitted on, or predicted from anywhere in
this script.

Confirmed project environment: `ml` conda environment
(/home/c/miniconda3/envs/ml/bin/python) -- pandas 3.0.5, numpy 2.4.6,
scikit-learn 1.9.0, openpyxl 3.1.5, pytest 9.1.1.
"""

# %%
# =============================================================================
# SECTION 1 — Imports and scope
# =============================================================================

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import artifacts, config, modeling, plots  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


print("Imports OK. `radcure` package resolved from:", SRC_DIR)
print("Candidates:", list(modeling.build_model_registry().keys()))
print("CV metrics:", list(modeling.CV_SCORING.keys()), "(roc_auc = PRIMARY)")


# %%
# =============================================================================
# SECTION 2 — Load the frozen training partition
# =============================================================================
# Objective:
#   Load the training partition from the artefact `analysis/01` produced, and
#   confirm its counts.
#
# Rationale:
#   The split was created once, in `analysis/01`, and persisted at patient
#   level. Loading it here -- rather than re-deriving the cohort and splitting
#   again -- guarantees that every script in this pipeline trains on exactly
#   the same rows in exactly the same order, which matters because
#   StratifiedKFold assigns folds by position.

section("SECTION 2 — Loading the frozen training partition")

X_train, y_train, id_train = artifacts.load_training_partition()

assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
assert int((y_train == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS

print(f"Training partition: n={len(X_train)}  events={int(y_train.sum())}  "
      f"non-events={int((y_train == 0).sum())}  prevalence={y_train.mean() * 100:.2f}%")
print(f"Predictors ({len(config.PRIMARY_FEATURES)}): {config.PRIMARY_FEATURES}")
print("[OK] Training partition matches the frozen study design.")
print("The held-out partition is not loaded by this script.")


# %%
# =============================================================================
# SECTION 3 — Cross-validation and metric design
# =============================================================================
# Objective:
#   Fix the exact cross-validation splitter and the exact metrics every
#   candidate is judged on, in one place, before any model is touched.

section("SECTION 3 — Cross-validation and metric design")

cv = modeling.build_cv_splitter()
cv_attrs = modeling.describe_cv_splitter(cv)
print(f"CV splitter: StratifiedKFold(n_splits={cv.get_n_splits()}, shuffle={cv_attrs.shuffle}, "
      f"random_state={cv_attrs.random_state}) -- the SAME splitter for every model and metric.")
print(f"Metrics: {modeling.CV_SCORING}")

results: dict[str, dict[str, np.ndarray | float]] = {}


def run_and_report(name: str, pipeline) -> dict[str, np.ndarray | float]:
    """Run the pre-specified CV metrics for one candidate on TRAINING data
    only, print per-fold scores and mean +/- std, and surface any warning
    (e.g. a convergence warning) rather than silencing it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = modeling.evaluate_candidate(pipeline, X_train, y_train, cv)
    print(f"\n--- {name} ---")
    for metric_name in modeling.CV_SCORING:
        fold_scores = scores[f"{metric_name}_scores"]
        assert isinstance(fold_scores, np.ndarray)
        mean_ = float(scores[f"{metric_name}_mean"])
        std_ = float(scores[f"{metric_name}_std"])
        formatted = ", ".join(f"{s:.4f}" for s in fold_scores)
        print(f"   {metric_name:18s} folds=[{formatted}]  mean={mean_:.4f}  std={std_:.4f}")
    if caught:
        for w in caught:
            print(f"   [WARNING] {w.category.__name__}: {w.message}")
    else:
        print("   [OK] No warnings raised during cross-validation.")
    return scores


# %%
# =============================================================================
# SECTION 4 — Dummy baseline (no-signal reference)
# =============================================================================
# Objective:
#   Establish the no-signal reference empirically rather than by assertion.
#
# Rationale:
#   `DummyClassifier(strategy="prior")` predicts the training-fold class prior
#   for every patient, using no predictor information at all. With a constant
#   per-fold score, ROC-AUC is expected to be 0.5 (a constant score cannot
#   rank anything) and Average Precision to track the training-fold event
#   prevalence (~18.9%). Those are stated as expectations to check against the
#   computed numbers below, not as hard-coded results.

section("SECTION 4 — Dummy baseline (no-signal reference)")

pipelines = modeling.build_candidate_pipelines()
results["Dummy (prior)"] = run_and_report("Dummy (prior)", pipelines["Dummy (prior)"])


# %%
# =============================================================================
# SECTION 5 — Logistic Regression baseline
# =============================================================================
# Objective:
#   Establish the comparatively interpretable reference every nonlinear
#   candidate must improve on to justify its additional complexity.
#
# Rationale:
#   The shared preprocessor already standard-scales Age/Smoking PY and one-hot
#   encodes the six categorical predictors, so Logistic Regression receives a
#   well-conditioned design matrix without extra work. Coefficients are not
#   inspected here: interpretation belongs after model selection
#   (`analysis/07`), not in a comparison where the model has not yet been
#   chosen.

section("SECTION 5 — Logistic Regression baseline")

results["Logistic Regression"] = run_and_report("Logistic Regression", pipelines["Logistic Regression"])


# %%
# =============================================================================
# SECTION 6 — Nonlinear model families
# =============================================================================
# Objective:
#   Evaluate the remaining three default candidates under the identical fixed
#   CV design, so all five are judged on exactly the same folds and metrics.
#
# Rationale:
#   Decision Tree and Random Forest can capture non-linear / interaction
#   structure (e.g. between Ds Site, T and N) that a linear model cannot; the
#   RBF SVC can capture a different class of non-linear decision boundary.
#   None receives tuned hyperparameters here -- this section asks whether each
#   family's DEFAULT configuration shows real signal, not what its best
#   achievable performance is.

section("SECTION 6 — Nonlinear model families (Decision Tree, Random Forest, RBF SVC)")

for name in ["Decision Tree", "Random Forest", "RBF SVC"]:
    results[name] = run_and_report(name, pipelines[name])


# %%
# =============================================================================
# SECTION 7 — Comparison table, interpretation, and persisted results
# =============================================================================
# Objective:
#   Assemble one comparison table across all five candidates, interpret it
#   against the observed fold-to-fold spread, and persist it for `analysis/03`.
#
# Rationale:
#   Judgements below compare each model's mean against the Dummy reference and
#   against Logistic Regression, read alongside the OBSERVED fold-to-fold
#   standard deviations. No formal significance test is performed anywhere:
#   fold standard deviations describe spread across five overlapping training
#   folds -- they are not standard errors and not confidence intervals.

section("SECTION 7 — CV comparison table and interpretation")

table_rows = []
for name, scores in results.items():
    for metric_name in modeling.CV_SCORING:
        fold_scores = scores[f"{metric_name}_scores"]
        assert isinstance(fold_scores, np.ndarray)
        row: dict[str, object] = {"model": name, "metric": metric_name}
        row.update({f"fold_{i}": float(s) for i, s in enumerate(fold_scores)})
        row["mean"] = float(scores[f"{metric_name}_mean"])
        row["std"] = float(scores[f"{metric_name}_std"])
        table_rows.append(row)

baseline_results = pd.DataFrame(table_rows)

# Wide view for reading and for the figure below.
comparison_table = baseline_results.pivot(index="model", columns="metric", values=["mean", "std"])
comparison_table.columns = [f"{metric}_{stat}" for stat, metric in comparison_table.columns]
comparison_table = comparison_table[
    [f"{m}_{s}" for m in modeling.CV_SCORING for s in ("mean", "std")]
].round(4)
print(comparison_table.to_string())

# --- Empirical check of the Dummy baseline's expected behaviour ------------
dummy_auc_mean = float(cast(float, comparison_table.at["Dummy (prior)", "roc_auc_mean"]))
dummy_ap_mean = float(
    cast(float, comparison_table.at["Dummy (prior)", "average_precision_mean"])
)
train_prevalence = float(y_train.mean())
print(f"\nDummy ROC-AUC mean = {dummy_auc_mean:.4f} (expected ~0.50 for a constant-score classifier).")
print(f"Dummy Average Precision mean = {dummy_ap_mean:.4f} vs. training prevalence = "
      f"{train_prevalence:.4f} (expected to track prevalence for a no-signal classifier).")

logreg_auc_mean = float(
    cast(float, comparison_table.at["Logistic Regression", "roc_auc_mean"])
)
logreg_auc_std = float(
    cast(float, comparison_table.at["Logistic Regression", "roc_auc_std"])
)
print(f"\nLogistic Regression ROC-AUC = {logreg_auc_mean:.4f} (std={logreg_auc_std:.4f}) "
      f"vs. Dummy {dummy_auc_mean:.4f}: difference {logreg_auc_mean - dummy_auc_mean:+.4f}.")

print("\nNonlinear candidates vs. Logistic Regression (ROC-AUC):")
for name in ["Decision Tree", "Random Forest", "RBF SVC"]:
    m = float(cast(float, comparison_table.at[name, "roc_auc_mean"]))
    s = float(cast(float, comparison_table.at[name, "roc_auc_std"]))
    print(f"   {name:16s} mean={m:.4f} (std={s:.4f})  difference vs. LogReg={m - logreg_auc_mean:+.4f}")

ranking_auc = comparison_table["roc_auc_mean"].sort_values(ascending=False).index.tolist()
ranking_ap = comparison_table["average_precision_mean"].sort_values(ascending=False).index.tolist()
print(f"\nRanked by ROC-AUC:           {ranking_auc}")
print(f"Ranked by Average Precision: {ranking_ap}")
print(f"Same ranking? {'YES' if ranking_auc == ranking_ap else 'NO -- see interpretation below'}")

# Result / Interpretation (read against the table printed above):
#
#   - Dummy behaves exactly as a no-signal classifier should: ROC-AUC 0.5000
#     with zero std, and Average Precision matching the training prevalence to
#     four decimals. That confirms the reference empirically. Every real model
#     sits well above it.
#
#   - Logistic Regression leads on all three metrics (ROC-AUC 0.7897, AP
#     0.4404, Balanced Accuracy 0.5879). It is the strongest default
#     candidate here, not a fallback kept despite weaker performance. Random
#     Forest's ROC-AUC is 0.034 below it, RBF SVC's 0.061 below; both gaps
#     should be read alongside the fold-to-fold standard deviations
#     (~0.02-0.03 for every non-Dummy candidate).
#
#   - Decision Tree vs. Random Forest: the tree is substantially worse on
#     ROC-AUC (0.5850 vs. 0.7554) and AP (0.2358 vs. 0.3815). Its Balanced
#     Accuracy is in fact slightly HIGHER (0.5850 vs. 0.5783) -- Random Forest
#     does not dominate on every metric -- but that does not compensate for
#     much weaker ranking performance on the two metrics that matter most
#     here.
#
#   - ROC-AUC and AP agree only PARTIALLY. Random Forest and RBF SVC swap
#     order: Random Forest has the higher ROC-AUC (0.7554 vs. 0.7285) while
#     RBF SVC has the higher AP (0.4135 vs. 0.3815). The SVC shows a better
#     precision-recall trade-off across thresholds despite weaker overall ROC
#     ranking -- a genuine metric-dependent disagreement worth recording.
#
#   - RBF SVC's Balanced Accuracy (0.5324, std 0.0066) is markedly lower than
#     Logistic Regression's or Random Forest's. That reflects its DEFAULT hard
#     decision rule, not its ranking quality. Both `C` and `gamma` govern an
#     RBF SVC's flexibility and are left at scikit-learn defaults here, so
#     this result justifies one controlled tuning pass rather than a verdict
#     on the family.
#
#   - The unpruned Decision Tree is the candidate most likely to show a
#     training-CV vs. generalisation gap later, given its tendency to overfit
#     without a depth constraint. Noted as a risk.

# %%
# --- FIGURE: baseline model comparison ---------------------------------------
# Derived entirely from `baseline_results`, the table persisted below.
figure_metrics = [
    ("roc_auc", "ROC-AUC (PRIMARY)", 0.5, "no-skill (0.50)"),
    ("average_precision", "Average Precision (SECONDARY)", train_prevalence,
     f"no-skill = prevalence ({train_prevalence:.3f})"),
]
model_order = comparison_table["roc_auc_mean"].sort_values(ascending=False).index.tolist()
# Colour carries one meaning only: the Dummy row is the no-signal reference,
# every real candidate shares a single neutral colour so the bar LENGTHS, not
# the palette, do the comparing.
bar_colors = [
    plots.NEUTRAL_COLOR if m == "Dummy (prior)" else plots.NON_EVENT_COLOR
    for m in model_order
]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
for ax, (metric, title, reference, reference_label) in zip(axes, figure_metrics):
    means = np.asarray(
        [
            float(cast(float, comparison_table.at[m, f"{metric}_mean"]))
            for m in model_order
        ],
        dtype=float,
    )
    stds = np.asarray(
        [
            float(cast(float, comparison_table.at[m, f"{metric}_std"]))
            for m in model_order
        ],
        dtype=float,
    )
    positions = np.arange(len(model_order))
    axis_scale = float(means.max())
    ax.barh(
        positions, means, xerr=stds, color=bar_colors,
        error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.2},
    )
    ax.set_yticks(positions, labels=model_order)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_xlim(0, axis_scale * 1.35)
    ax.axvline(reference, color=plots.NEUTRAL_COLOR, linestyle="--", linewidth=1.2, label=reference_label)
    ax.legend(loc="lower right", fontsize=8)
    for pos, mean_, std_ in zip(positions, means, stds):
        text_x = float(mean_ + std_ + axis_scale * 0.03)
        ax.text(text_x, pos, f"{mean_:.4f}", va="center", fontsize=9)

fig.suptitle(
    "Default-hyperparameter baseline comparison — 5-fold CV on the training partition\n"
    "(bars show the mean across folds, whiskers the fold-to-fold standard deviation)",
    fontsize=12.5, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "02_baseline_model_comparison.png")


# %%
# =============================================================================
# SECTION 8 — Shortlist for tuning, and persisted artefacts
# =============================================================================
# Objective:
#   Record which families go forward to `analysis/03`, with the reason for
#   each, and write both artefacts this stage owns.

section("SECTION 8 — Shortlist for analysis/03")

shortlist = {
    "shortlisted": ["Logistic Regression", "Random Forest", "RBF SVC"],
    "not_shortlisted": ["Decision Tree", "Dummy (prior)"],
    "selection_basis": "training-only 5-fold CV; no test-set information used",
    "rationale": {
        "Logistic Regression": (
            "Highest mean on all three metrics of any default candidate "
            f"(ROC-AUC {logreg_auc_mean:.4f}), and the most transparent model in the "
            "comparison."
        ),
        "Random Forest": (
            "Second-strongest on the primary metric and a nonlinear tree-ensemble "
            "family; depth / leaf-size / feature-subsampling and class weighting are "
            "all still at defaults, so it warrants one controlled tuning pass."
        ),
        "RBF SVC": (
            "Lower ROC-AUC than Random Forest but HIGHER Average Precision -- a genuine "
            "metric-dependent disagreement. Its two governing hyperparameters, C and "
            "gamma, were left at scikit-learn defaults and strongly determine an RBF "
            "SVC's flexibility, so the default result is not a verdict on the family."
        ),
        "Decision Tree": (
            "Not shortlisted: substantially worse than Random Forest on ROC-AUC "
            "(0.5850 vs. 0.7554) and Average Precision (0.2358 vs. 0.3815). Its "
            "slightly higher Balanced Accuracy does not compensate for much weaker "
            "ranking performance."
        ),
        "Dummy (prior)": (
            "Retained only as the permanent no-signal reference, never as a modelling "
            "candidate."
        ),
    },
}

print("Shortlisted for tuning:", shortlist["shortlisted"])
for name in shortlist["shortlisted"] + shortlist["not_shortlisted"]:
    print(f"\n   {name}\n      {shortlist['rationale'][name]}")

print("\nRanked by ROC-AUC:")
print(comparison_table.sort_values("roc_auc_mean", ascending=False).to_string())

results_path = artifacts.save_table(baseline_results, artifacts.BASELINE_CV_RESULTS)
shortlist_path = artifacts.save_json(shortlist, artifacts.BASELINE_SHORTLIST)
print(f"\n[artifact] {results_path.relative_to(PROJECT_ROOT)}")
print(f"[artifact] {shortlist_path.relative_to(PROJECT_ROOT)}")

print("""
Next: analysis/03_hyperparameter_tuning.py — one controlled GridSearchCV per
shortlisted family, under the identical CV design, comparing each tuned result
against the default results persisted above. The held-out partition remains
untouched throughout model development.
""")
