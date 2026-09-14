"""
RT-Start-anchored two-year mortality target construction (Sections 3.8-3.9).

Confirmed design:
    - prediction landmark  = `RT Start` (immediately before the first
      radiotherapy fraction)
    - outcome time origin  = `RT Start` (same clock as the landmark), so
      every patient is given the identical 730-day prospective horizon
      measured from the moment the prediction is actually made -- unlike
      a diagnosis-anchored alternative, which was evaluated and rejected
      (see `build_target_diagnosis_anchored` and the Section 3.8 cell of
      the analysis script) because it produces patient-specific REMAINING
      prediction horizons after the actual landmark, not a uniform one
    - horizon              = a FIXED 730 days (not a calendar 2-year
      offset -- the two definitions were shown to reclassify zero patients
      in this cohort, and the fixed-day rule avoids leap-year ambiguity)

Boundary rule:
    event      : deceased and (Date of Death - RT Start).days <= 730
    non-event  : deceased and (Date of Death - RT Start).days >  730
                 OR alive and (Last FU - RT Start).days       >= 730
    excluded   : alive and (Last FU - RT Start).days           < 730
                 (target is genuinely unobservable -- these patients are
                 NEVER assigned to class 0)

No survival-analysis methodology (Kaplan-Meier, log-rank, Cox) is used
anywhere in this module or this project; only simple event/censoring
counts and rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_days_from_rt_start(df: pd.DataFrame) -> pd.Series:
    """Days from `RT Start` to the event (death) for deceased patients, or
    to last documented contact for living patients.

    Uses only observed date columns (`RT Start`, `Date of Death`,
    `Last FU`); no derived/reconstructed date is involved.
    """
    is_dead = df["Status"].eq("Dead")
    days_to_death = (df["Date of Death"] - df["RT Start"]).dt.days
    days_to_last_fu = (df["Last FU"] - df["RT Start"]).dt.days
    days = np.where(is_dead, days_to_death, days_to_last_fu).astype(float)
    return pd.Series(days, index=df.index, name=config.TIME_FROM_LANDMARK_COLUMN)


def assert_no_deaths_before_landmark(df: pd.DataFrame) -> None:
    """Verify that no patient's recorded date of death precedes the
    prediction landmark (`RT Start`). This is a structural sanity check on
    the cohort (Section 3.8/3.9), not a target-construction step.
    """
    is_dead = df["Status"].eq("Dead")
    days_to_death = (df["Date of Death"] - df["RT Start"]).dt.days
    n_before = int(((days_to_death < 0) & is_dead).sum())
    assert n_before == 0, (
        f"{n_before} deceased patient(s) have Date of Death before RT Start -- "
        "the landmark/outcome-clock design assumption is violated."
    )


def build_target(
    df: pd.DataFrame,
    horizon_days: int = config.OUTCOME_HORIZON_DAYS,
) -> pd.DataFrame:
    """Construct the RT-Start-anchored binary two-year mortality target.

    Returns a DataFrame (same index as `df`) with three columns:
      - `days_from_rt_start` : float, see `compute_days_from_rt_start`
      - `target_eligible`    : bool, True iff the 2-year outcome is observable
      - `mortality_2y`       : pandas nullable Int64, 1 (event) / 0
                                (non-event) / <NA> (excluded -- outcome
                                unobservable). <NA> rows must never be
                                silently treated as 0.
    """
    is_dead = df["Status"].eq("Dead")
    t = compute_days_from_rt_start(df)

    is_event = is_dead & (t <= horizon_days)
    is_nonevent_died_later = is_dead & (t > horizon_days)
    is_nonevent_alive_long_enough = (~is_dead) & (t >= horizon_days)
    is_excluded = (~is_dead) & (t < horizon_days)

    # Every row must fall into exactly one of the four groups.
    n_total = len(df)
    n_covered = int(is_event.sum() + is_nonevent_died_later.sum()
                     + is_nonevent_alive_long_enough.sum() + is_excluded.sum())
    assert n_covered == n_total, (
        f"Target-group partition is not exhaustive/disjoint: {n_covered} of {n_total} rows covered."
    )

    mortality_2y = pd.Series(pd.array([pd.NA] * n_total, dtype="Int64"), index=df.index)
    mortality_2y[is_event] = 1
    mortality_2y[is_nonevent_died_later | is_nonevent_alive_long_enough] = 0

    eligible = is_event | is_nonevent_died_later | is_nonevent_alive_long_enough

    out = pd.DataFrame(
        {
            config.TIME_FROM_LANDMARK_COLUMN: t,
            config.ELIGIBILITY_COLUMN: eligible,
            config.TARGET_COLUMN: mortality_2y,
        },
        index=df.index,
    )
    return out


def reconstruct_diagnosis_date(df: pd.DataFrame) -> pd.Series:
    """Reconstruct each patient's diagnosis date from documented follow-up
    duration, as `Last FU - round(Length FU * 365)` days, using the
    dataset's apparent 365-day encoding of `Length FU` (empirically,
    `Length FU * 365` is integral to within floating-point tolerance for
    every record in this file).

    IMPORTANT: this is a DERIVED / RECONSTRUCTED quantity, not a directly
    observed date. It is used ONLY for the diagnosis-to-RT-Start diagnostic
    and the RADCURE-challenge temporal-separation comparison in the main
    analysis script (Section 3.8) -- it is never used as a predictor, and
    it is never used to construct the confirmed outcome (whose clock is
    `RT Start`, not diagnosis). Reproducing the challenge's observed
    training/test temporal separation with this reconstruction strongly
    corroborates the reconstruction, but does not make the diagnosis date
    directly observed or prove it exact.
    """
    days_from_diagnosis = (df["Length FU"].astype(float) * 365).round()
    return df["Last FU"] - pd.to_timedelta(days_from_diagnosis, unit="D")


def classify_two_year_outcome(is_dead: pd.Series, t: pd.Series, horizon) -> dict[str, pd.Series]:
    """Shared 4-way boundary-rule classification, factored out so the
    Section 3.8 design-comparison diagnostics (diagnosis-anchored vs.
    RT-Start-anchored; fixed-730-day vs. calendar-two-year) can reuse the
    exact same rule without duplicating it. NOT used by `build_target`
    itself, which is left untouched.

    `horizon` may be a scalar (the fixed-730-day rule) or a per-row
    array/Series of the same length as `t` (e.g. a calendar-derived,
    patient-specific horizon in days).

    Returns a dict of four mutually exclusive, jointly exhaustive boolean
    masks: "event", "nonevent_died_later", "nonevent_alive", "excluded".
    """
    return {
        "event": is_dead & (t <= horizon),
        "nonevent_died_later": is_dead & (t > horizon),
        "nonevent_alive": (~is_dead) & (t >= horizon),
        "excluded": (~is_dead) & (t < horizon),
    }


def build_target_diagnosis_anchored(
    df: pd.DataFrame,
    horizon_days: int = config.OUTCOME_HORIZON_DAYS,
) -> pd.DataFrame:
    """Construct a DIAGNOSIS-anchored binary two-year mortality target,
    for the Section 3.8 ONE-OFF DESIGN COMPARISON ONLY.

    This is NOT the project's outcome definition (see `build_target`,
    anchored at `RT Start`), and its output must NEVER be used to
    construct X, y, or any modelling artefact. Its sole purpose is to
    demonstrate, in executable code, why the RT-Start anchor was chosen:
    the diagnosis-anchored endpoint produces patient-specific remaining
    prediction horizons AFTER the actual prediction landmark (`RT Start`)
    and therefore does not align with the intended prospective prediction
    question, whereas every patient anchored at `RT Start` is given the
    identical 730-day prospective horizon.

    Time is `round(Length FU * 365)` -- `Length FU` is documented as
    "duration of follow-up from diagnosis to last contact date, in
    years", so this is the direct, dictionary-defined diagnosis-to-outcome
    time; it does not require the separately reconstructed diagnosis DATE.

    Returns a DataFrame (same index as `df`) with three columns:
      - `days_from_diagnosis`             : float
      - `diagnosis_anchored_eligible`     : bool
      - `diagnosis_anchored_mortality_2y` : pandas nullable Int64, 1/0/<NA>
    """
    is_dead = df["Status"].eq("Dead")
    t = (df["Length FU"].astype(float) * 365).round()
    t.name = "days_from_diagnosis"

    groups = classify_two_year_outcome(is_dead, t, horizon_days)

    n_total = len(df)
    n_covered = int(sum(int(mask.sum()) for mask in groups.values()))
    assert n_covered == n_total, (
        f"Diagnosis-anchored target partition is not exhaustive/disjoint: {n_covered} of {n_total} rows covered."
    )

    mortality_2y = pd.Series(pd.array([pd.NA] * n_total, dtype="Int64"), index=df.index)
    mortality_2y[groups["event"]] = 1
    mortality_2y[groups["nonevent_died_later"] | groups["nonevent_alive"]] = 0

    eligible = groups["event"] | groups["nonevent_died_later"] | groups["nonevent_alive"]

    return pd.DataFrame(
        {
            "days_from_diagnosis": t,
            "diagnosis_anchored_eligible": eligible,
            "diagnosis_anchored_mortality_2y": mortality_2y,
        },
        index=df.index,
    )


def target_summary(
    target_df: pd.DataFrame,
    eligibility_col: str = config.ELIGIBILITY_COLUMN,
    target_col: str = config.TARGET_COLUMN,
) -> dict:
    """Compact eligibility/event/prevalence summary of a target-construction
    result, for printing and for assertion against the expected values in
    `config.py`.

    Column-name parameters default to the confirmed RT-Start-anchored
    target's columns (`build_target`'s output), so existing call sites are
    unaffected; pass the diagnosis-anchored column names (see
    `build_target_diagnosis_anchored`) to summarise that comparison-only
    target instead.
    """
    n_total = len(target_df)
    n_excluded = int((~target_df[eligibility_col]).sum())
    n_eligible = n_total - n_excluded
    n_events = int((target_df[target_col] == 1).sum())
    n_nonevents = n_eligible - n_events
    return {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_excluded": n_excluded,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "prevalence_pct": round(n_events / n_eligible * 100, 2) if n_eligible else float("nan"),
    }
