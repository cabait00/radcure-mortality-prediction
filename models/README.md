# `models/`

Fitted scikit-learn objects. Not tracked in version control (binary, and
regenerable) — recreate by running
`analysis/05_threshold_selection_and_model_freeze.py`.

## `final_logistic_regression.joblib`

The frozen final pipeline, fitted once on the complete training partition
(n = 2351) in `analysis/05`:

```
Pipeline(
    preprocessor = ColumnTransformer(
        numeric     = [Age, Smoking PY] -> SimpleImputer(median) -> StandardScaler
        categorical = [Sex, ECOG PS, Smoking Status, Ds Site, T, N]
                      -> OneHotEncoder(handle_unknown="ignore", drop=None)
    ),
    classifier   = LogisticRegression(C=1.0, class_weight=None,
                                      max_iter=5000, random_state=42),
)
```

Every learned preprocessing parameter — the `Smoking PY` imputation median, the
scaler's mean and variance, the encoder's category vocabulary — was estimated
inside that single `fit` call, on training data only.

`analysis/06` loads this object and evaluates it once on the held-out
partition; `analysis/07` loads the same object and describes its coefficients.
Neither refits it.

Its frozen decision threshold lives separately, in
`artifacts/05_threshold.json`.
