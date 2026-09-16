"""
radcure
=======

Reusable, tested logic for the RADCURE two-year mortality prediction
project. The chronological scientific narrative lives in the ``analysis/``
scripts; this package holds only the deterministic, reproducible building
blocks that the narrative calls and displays evidence for.

Modules, roughly in the order the pipeline uses them:

    config        static configuration: paths, study-design parameters, the
                  deterministic cleaning maps, the frozen predictor set and
                  the frozen expected results
    cleaning      raw-data loading and deterministic semantic cleaning
    target        the RT-Start-anchored 730-day target and its eligibility rule
    leakage       the machine-checkable 34-variable feature leakage audit
    cohort        modelling-cohort assembly and the single frozen train/test
                  split (the only ``train_test_split`` call site)
    preprocessing the learned-preprocessing pipeline objects (never fitted here)
    modeling      the fixed CV splitter, the baseline candidate registry and
                  the CV evaluation helper
    tuning        the fixed hyperparameter search spaces and search builders
    ensemble      the frozen tuned pipelines, out-of-fold scores and stacking
    evaluation    classification metrics and the threshold search
    artifacts     persistence for the 01 -> 07 artefact pipeline
    plots         shared figure style and helpers

Nothing in this package fits a statistical parameter at import time, and
nothing in it writes to ``data/raw/``.
"""

__version__ = "0.1.0"
