"""
radcure
=======

Reusable, tested logic for the RADCURE two-year mortality prediction
project. The chronological scientific narrative lives in the top-level
``analysis/`` script(s); this package holds only the deterministic and
reproducible building blocks that the narrative calls and displays evidence
for (raw-data loading, semantic cleaning, target construction, the leakage
audit, and the preprocessing pipeline objects).

Nothing in this package fits a statistical parameter at import time and
nothing in it writes to ``data/raw/``.
"""

__version__ = "0.1.0"
