"""
Machine-checkable feature leakage audit (Section 3.4).

Every one of the 34 original RADCURE columns is listed here EXACTLY ONCE
with an explicit predictor decision and reason. The candidate/excluded
predictor lists used downstream are DERIVED from this table -- they are
never retyped by hand elsewhere -- so that the audit's conclusions and the
code's behaviour cannot silently drift apart.

Admission test applied to every variable (Section 3.4, principle 1):
    "If a new patient is immediately about to receive their first
    radiotherapy fraction, could this variable legitimately be known and
    used at that moment?"

Category taxonomy (a variable's PRIMARY concern; a secondary concern, where
one exists, is noted in the `secondary_category` field and in `reason`):

    BASELINE_AVAILABLE               -- available at the landmark, no proxy risk
    IDENTIFIER                       -- carries no clinical information
    DIRECT_TARGET_LEAKAGE            -- is the outcome, or defines it
    FOLLOWUP_INFORMATION             -- recorded only during/because of follow-up
    POST_LANDMARK_EVENT              -- a clinical event observed after the landmark
    POST_LANDMARK_TREATMENT_INFORMATION -- describes treatment as DELIVERED, not planned
    MIXED_TEMPORAL_INFORMATION       -- some values pre-date the landmark, most do not
    DATASET_MEMBERSHIP_PROXY         -- encodes which partition of the raw file a row is in
    CALENDAR_ERA_PROXY               -- available at the landmark, but a proxy for treatment era
    TECHNICAL_PROTOCOL_PROXY         -- available at the landmark, but describes acquisition
                                          protocol rather than the patient
    TEMPORALLY_AMBIGUOUS             -- baseline in content, but the *recorded* value may have
                                          been generated/updated after the landmark

IMPORTANT: `RT Start`, `RADCURE-challenge` and `ContrastEnhanced` are each
excluded as a PROXY (CALENDAR_ERA_PROXY / DATASET_MEMBERSHIP_PROXY /
TECHNICAL_PROTOCOL_PROXY), never as DIRECT_TARGET_LEAKAGE -- none of the
three is derived from the outcome, and each is knowable at the landmark.

Passing this audit (`decision == "CANDIDATE"`) means a variable is
TEMPORALLY ADMISSIBLE. It does NOT mean the variable belongs in a
parsimonious model -- that further, non-leakage decision is made in
Section 3.5 (see `config.PRIMARY_FEATURES`).
"""

from __future__ import annotations

import pandas as pd

from . import config

# =============================================================================
# The audit table: one row per original column, in raw file order.
# =============================================================================
# NOTE: the raw workbook's `M` column name carries a trailing space ("M ").
# It is kept verbatim here so that this table's `variable` column matches
# `raw_df.columns` exactly, which is what the coverage assertions below
# rely on.

_LEAKAGE_AUDIT_RECORDS: list[dict[str, str]] = [
    dict(
        variable="patient_id",
        meaning="Patient ID randomly assigned prior to anonymising the DICOM PHI tag.",
        temporal_availability="Assigned at data curation; carries no clinical content at any time.",
        category="IDENTIFIER",
        secondary_category="",
        decision="EXCLUDE",
        reason="Random identifier with zero clinical information; retained only for traceability and split-integrity checks, never admitted to X.",
        uncertainty="None.",
    ),
    dict(
        variable="Age",
        meaning="Patient age, years.",
        temporal_availability="Known at diagnosis and at the landmark.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Core demographic predictor, complete, no leakage path.",
        uncertainty="Dictionary does not state age at diagnosis vs. at RT Start (median gap ~43 days); immaterial.",
    ),
    dict(
        variable="Sex",
        meaning="Patient sex, male or female.",
        temporal_availability="Known at diagnosis and at the landmark.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Standard baseline demographic predictor, complete.",
        uncertainty="None.",
    ),
    dict(
        variable="ECOG PS",
        meaning="ECOG performance-status grade (0=fully active ... 4=disabled, 5=Dead).",
        temporal_availability="Clinical assessment, presumed at consultation prior to treatment.",
        category="BASELINE_AVAILABLE",
        secondary_category="TEMPORALLY_AMBIGUOUS",
        decision="CANDIDATE",
        reason="Principal non-tumour clinical predictor. The scale's own grade 5 (=Dead) would be direct leakage if present; verified zero records carry ECOG 5, so this path is closed.",
        uncertainty="Dictionary does not explicitly state the assessment time point (unlike Smoking Status).",
    ),
    dict(
        variable="Smoking PY",
        meaning="Number of packs smoked per year (pack-years).",
        temporal_availability="Fixed exposure history, known at consultation.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Cumulative exposure measure, fixed at baseline.",
        uncertainty="Stored as text with non-numeric tokens; a cleaning issue, not a leakage issue.",
    ),
    dict(
        variable="Smoking Status",
        meaning="Smoking status AT FIRST CONSULTATION: current/ex/non-smoker.",
        temporal_availability="Explicitly stated by the Dictionary to be assessed at first consultation.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="The only variable whose Dictionary entry names its own observation time point.",
        uncertainty="None.",
    ),
    dict(
        variable="Ds Site",
        meaning="Primary cancer site.",
        temporal_availability="Established at diagnosis, before treatment planning.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Primary tumour site, fixed at diagnosis.",
        uncertainty="None.",
    ),
    dict(
        variable="Subsite",
        meaning="Primary cancer subsite.",
        temporal_availability="Established at diagnosis, before treatment planning.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Temporally admissible; excluded from the PRIMARY model at the Section 3.5 predictor-set stage for cardinality and structural missingness, NOT for a leakage reason.",
        uncertainty="No uncertainty markers found in any of the 63 raw values.",
    ),
    dict(
        variable="T",
        meaning="AJCC 7th-edition T (primary tumour) category.",
        temporal_availability="Underlying tumour extent assessed at baseline; the AJCC-7 CODE was retrospectively harmonised for most of the cohort (AJCC 7 entered clinical use in 2010; 90.8% of patients started RT earlier).",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="BASELINE_AVAILABLE",
        decision="CANDIDATE",
        reason="The underlying clinical information (tumour extent) was available at the landmark; only the coding vocabulary was applied retrospectively. Not outcome-derived.",
        uncertainty="Cannot verify from the file whether re-coding ever drew on post-treatment findings (though AJCC is a pre-treatment staging system).",
    ),
    dict(
        variable="N",
        meaning="AJCC 7th-edition N (regional node) category.",
        temporal_availability="Same as T.",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="BASELINE_AVAILABLE",
        decision="CANDIDATE",
        reason="Same as T.",
        uncertainty="Same as T.",
    ),
    dict(
        variable="M ",
        meaning="AJCC 7th-edition M (distant metastasis) category at staging.",
        temporal_availability="Same as T. NOTE: distinct from the follow-up variable `Distant`.",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="BASELINE_AVAILABLE",
        decision="CANDIDATE",
        reason="Same as T; admissible, but see Section 3.5 for its exclusion from the PRIMARY set on near-zero-variance grounds (not leakage).",
        uncertainty="Same as T.",
    ),
    dict(
        variable="Stage",
        meaning="AJCC 7th-edition composite stage group.",
        temporal_availability="Same as T; a site-specific recombination of T, N, M.",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="BASELINE_AVAILABLE",
        decision="CANDIDATE",
        reason="Admissible; excluded from the PRIMARY set (Section 3.5) as redundant given Ds Site + T + N together, not because it is a simple deterministic function of TNM alone (it is not -- confirmed empirically).",
        uncertainty="53 records use codes outside the AJCC-7 head-and-neck stage-group set.",
    ),
    dict(
        variable="Path",
        meaning="Pathologic diagnosis / histology type.",
        temporal_availability="Established on pre-treatment diagnostic biopsy.",
        category="BASELINE_AVAILABLE",
        secondary_category="",
        decision="CANDIDATE",
        reason="Baseline histology; admissible. Excluded from the PRIMARY set (Section 3.5) because its dominant clinical distinction is highly redundant with Ds Site, not because of rarity.",
        uncertainty="3 patients (0.09%) had `Postop RT alone`, where histology could in principle reflect a post-surgical specimen; surgery still preceded the RT landmark.",
    ),
    dict(
        variable="HPV",
        meaning="Tumour HPV status by p16 IHC +/- HPV DNA PCR. Blank = no data available.",
        temporal_availability="A fixed biological property of the tumour, present at the landmark for every patient -- but the RECORDED RESULT may not have been available at the landmark for a substantial, unquantifiable fraction of the cohort.",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="BASELINE_AVAILABLE",
        decision="CANDIDATE",
        reason="Passes as a candidate (biological property, not outcome-derived); documentation rate is implausibly high in 1999-2002 and DECLINES after 2008, consistent with some historical results being determined retrospectively on archived tissue. Reserved for a pre-specified sensitivity analysis only (Section 3.5), not the PRIMARY model.",
        uncertainty="The fraction of results determined retrospectively cannot be quantified from this file. Missingness is site- and era-dependent, so its meaning is not stable across the cohort.",
    ),
    dict(
        variable="Tx Modality",
        meaning="How surgery, radiation and chemotherapy are combined (mapping terminology referenced but not included in the workbook).",
        temporal_availability="Planned-versus-realised status CANNOT be resolved from the workbook (the referenced mapping terminology is absent), so landmark admissibility is uncertain rather than affirmatively post-landmark.",
        category="TEMPORALLY_AMBIGUOUS",
        secondary_category="POST_LANDMARK_TREATMENT_INFORMATION",
        decision="EXCLUDE",
        reason="Excluded CONSERVATIVELY from PRIMARY: admissibility cannot be determined one way or the other (missing mapping terminology), and the variable is near-redundant with Chemo. Not asserted to be post-landmark as a fact -- only that its admissibility is unresolved.",
        uncertainty="Whether any recorded value is the planned or the delivered modality cannot be established from this file.",
    ),
    dict(
        variable="Chemo",
        meaning="Yes = RECEIVED concurrent chemoradiotherapy; none = did not receive.",
        temporal_availability="Post-landmark: describes what was received, not what was planned.",
        category="POST_LANDMARK_TREATMENT_INFORMATION",
        secondary_category="",
        decision="EXCLUDE",
        reason="Dictionary wording ('received') is explicit; this is a statement about what happened during/after the landmark.",
        uncertainty="None.",
    ),
    dict(
        variable="RT Start",
        meaning="Date of first radiotherapy fraction, with date offset applied.",
        temporal_availability="IS the prediction landmark -- known at the landmark by definition.",
        category="CALENDAR_ERA_PROXY",
        secondary_category="",
        decision="EXCLUDE",
        reason="NOT direct target leakage (nothing about it is derived from the outcome) -- excluded because, as a FEATURE, the calendar date is a proxy for treatment era, which drives censoring, case mix, HPV-testing availability and imaging protocol. Outside the clinical/demographic/tumour scope of the research question. Retained for landmark/outcome-clock definition and the temporal-holdout comparison only.",
        uncertainty="None.",
    ),
    dict(
        variable="Dose",
        meaning="Total RT dose DELIVERED during radiotherapy.",
        temporal_availability="Known only at completion of the RT course, ~7 weeks after the landmark.",
        category="POST_LANDMARK_TREATMENT_INFORMATION",
        secondary_category="",
        decision="EXCLUDE",
        reason="Data Dictionary wording ('delivered') is explicit: this is the realised total RT "
               "dose, known only once the course is completed. At the prediction landmark, "
               "immediately before RT initiation, the realised total delivered dose is not yet "
               "known.",
        uncertainty="Whether any record was ever back-filled with the planned prescription instead of the delivered dose.",
    ),
    dict(
        variable="Fx",
        meaning="Number of RT fractions DELIVERED.",
        temporal_availability="Same as Dose.",
        category="POST_LANDMARK_TREATMENT_INFORMATION",
        secondary_category="",
        decision="EXCLUDE",
        reason="Same as Dose.",
        uncertainty="Same as Dose.",
    ),
    dict(
        variable="Last FU",
        meaning="Most recent date of contact at which outcomes were updated, OR date of death.",
        temporal_availability="Follow-up; used to construct the target for living patients.",
        category="DIRECT_TARGET_LEAKAGE",
        secondary_category="",
        decision="EXCLUDE",
        reason="Target-construction input: for living patients, (Last FU - RT Start) determines both the class-0 label and cohort eligibility.",
        uncertainty="None.",
    ),
    dict(
        variable="Status",
        meaning="Binary indicator of vital status at last contact date.",
        temporal_availability="Follow-up; IS the outcome.",
        category="DIRECT_TARGET_LEAKAGE",
        secondary_category="",
        decision="EXCLUDE",
        reason="This is the event indicator the target is built from.",
        uncertainty="None.",
    ),
    dict(
        variable="Length FU",
        meaning="Duration of follow-up from DIAGNOSIS to last contact date, in years.",
        temporal_availability="Follow-up; encodes both survival time and censoring time from a different time origin (diagnosis, not RT Start).",
        category="DIRECT_TARGET_LEAKAGE",
        secondary_category="FOLLOWUP_INFORMATION",
        decision="EXCLUDE",
        reason="A near-affine transform of the outcome time even though the confirmed clock is RT Start; strongly correlated with vital status.",
        uncertainty="None.",
    ),
    dict(
        variable="Date of Death",
        meaning="Date of death, with date offset applied.",
        temporal_availability="Follow-up; used to construct the target for deceased patients.",
        category="DIRECT_TARGET_LEAKAGE",
        secondary_category="",
        decision="EXCLUDE",
        reason="Target-construction input: (Date of Death - RT Start) <= 730 defines the positive class.",
        uncertainty="None.",
    ),
    dict(
        variable="Cause of Death",
        meaning="Cause of death, if applicable.",
        temporal_availability="Follow-up; a recorded value occurs only among deceased patients.",
        category="FOLLOWUP_INFORMATION",
        secondary_category="DIRECT_TARGET_LEAKAGE",
        decision="EXCLUDE",
        reason="Observed 1052 of 1058 deceased patients (99.4%) have a recorded cause, 6 do not, "
               "and 0 living patients have one -- an observed cause therefore directly reveals "
               "death status, so even an is-recorded indicator built from this column would be "
               "informative about vital status and the variable is not admissible as a baseline "
               "predictor.",
        uncertainty="None.",
    ),
    dict(
        variable="Local",
        meaning="Binary indicator of local relapse.",
        temporal_availability="Post-landmark follow-up event; on-landmark-dated cases are almost entirely coded `Persistent` (57/57), a determination only possible after treatment.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Represents a post-treatment outcome, not baseline disease status.",
        uncertainty="None.",
    ),
    dict(
        variable="Date Local",
        meaning="Date of local relapse, if applicable.",
        temporal_availability="Post-landmark follow-up.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Same as Local.",
        uncertainty="None.",
    ),
    dict(
        variable="Regional",
        meaning="Binary indicator of regional relapse.",
        temporal_availability="Post-landmark follow-up event; 37/38 on-landmark-dated cases are `Persistent`.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Same reasoning as Local.",
        uncertainty="1 record dated 57 days before RT Start is unexplained.",
    ),
    dict(
        variable="Date Regional",
        meaning="Date of regional relapse, if applicable.",
        temporal_availability="Post-landmark follow-up.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Same as Regional.",
        uncertainty="None.",
    ),
    dict(
        variable="Distant",
        meaning="Binary indicator of distant metastasis (a FOLLOW-UP event, distinct from baseline `M`).",
        temporal_availability="Post-landmark follow-up event.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Post-treatment event; baseline metastatic status is separately carried by `M `.",
        uncertainty="2 records dated 19 days before RT Start are unexplained.",
    ),
    dict(
        variable="Date Distant",
        meaning="Date of distant metastasis, if applicable.",
        temporal_availability="Post-landmark follow-up.",
        category="POST_LANDMARK_EVENT",
        secondary_category="",
        decision="EXCLUDE",
        reason="Same as Distant.",
        uncertainty="None.",
    ),
    dict(
        variable="2nd Ca",
        meaning="Binary indicator of second cancer diagnosis.",
        temporal_availability="MIXED: of 439 dated events, 98 precede RT Start, 2 are on it, 339 follow it. The raw flag does not distinguish prior history from a follow-up event.",
        category="MIXED_TEMPORAL_INFORMATION",
        secondary_category="",
        decision="EXCLUDE",
        reason="77% of dated events are post-landmark; the raw flag cannot be split into prior/subsequent components without the date, and the date is predominantly follow-up-derived.",
        uncertainty="A theoretically constructible baseline-only 'prior second cancer' indicator (flagging ~3% of the eligible cohort) was considered and deliberately NOT implemented for simplicity (Section 3.4, Task 7).",
    ),
    dict(
        variable="Date 2nd Ca",
        meaning="Date of second cancer diagnosis, if applicable.",
        temporal_availability="Same as 2nd Ca.",
        category="MIXED_TEMPORAL_INFORMATION",
        secondary_category="",
        decision="EXCLUDE",
        reason="Predominantly follow-up-derived.",
        uncertainty="Same as 2nd Ca.",
    ),
    dict(
        variable="RADCURE-challenge",
        meaning="training/test/0 membership in the 2020 UHN RADCURE challenge cohort.",
        temporal_availability="Assigned post-hoc by the challenge organisers on diagnosis date; carries no clinical content.",
        category="DATASET_MEMBERSHIP_PROXY",
        secondary_category="CALENDAR_ERA_PROXY",
        decision="EXCLUDE",
        reason="NOT direct target leakage -- but it is a coarse era indicator (assigned by diagnosis date) whose groups differ sharply in follow-up availability (censoring 0.3% / 23.6% / 28.3%) and in undocumented selection (group '0'). Not observable for a future patient and not a property of the patient.",
        uncertainty="Group '0' membership criterion remains undocumented (Section 3.2).",
    ),
    dict(
        variable="ContrastEnhanced",
        meaning="Imaging enhanced by contrast agent (1=enhanced, 0=regular).",
        temporal_availability="Known at the landmark (staging imaging precedes RT) -- but describes imaging ACQUISITION PROTOCOL, not patient biology.",
        category="TECHNICAL_PROTOCOL_PROXY",
        secondary_category="CALENDAR_ERA_PROXY",
        decision="EXCLUDE",
        reason="NOT direct target leakage -- excluded because it is a near-perfect proxy for treatment era (0.0% enhanced in 1999-2001 rising to ~90% by 2004) and describes scanner protocol, outside the clinical/demographic/tumour scope of the research question.",
        uncertainty="None.",
    ),
]

LEAKAGE_AUDIT_TABLE: pd.DataFrame = pd.DataFrame(_LEAKAGE_AUDIT_RECORDS)

# =============================================================================
# Derived, single-source-of-truth predictor lists
# =============================================================================
CANDIDATE_PREDICTORS: tuple[str, ...] = tuple(
    LEAKAGE_AUDIT_TABLE.loc[LEAKAGE_AUDIT_TABLE["decision"] == "CANDIDATE", "variable"]
)
EXCLUDED_VARIABLES: tuple[str, ...] = tuple(
    LEAKAGE_AUDIT_TABLE.loc[LEAKAGE_AUDIT_TABLE["decision"] == "EXCLUDE", "variable"]
)


def assert_full_coverage(raw_columns) -> None:
    """Verify that the audit table covers every raw column exactly once.

    Raises AssertionError (with the offending set(s) named) if:
      - any raw column is missing from the audit table,
      - the audit table names a variable absent from the raw file, or
      - any variable is audited more than once.
    """
    audited = list(LEAKAGE_AUDIT_TABLE["variable"])
    audited_set = set(audited)
    raw_set = set(raw_columns)

    duplicated = {v for v in audited if audited.count(v) > 1}
    assert not duplicated, f"Variable(s) audited more than once: {duplicated}"

    missing_from_audit = raw_set - audited_set
    assert not missing_from_audit, f"Raw column(s) not covered by the audit: {missing_from_audit}"

    extra_in_audit = audited_set - raw_set
    assert not extra_in_audit, f"Audited variable(s) absent from the raw file: {extra_in_audit}"

    assert len(LEAKAGE_AUDIT_TABLE) == len(raw_columns) == config.EXPECTED_RAW_COLUMNS, (
        f"Expected exactly {config.EXPECTED_RAW_COLUMNS} variables audited and in the raw file; "
        f"got audit={len(LEAKAGE_AUDIT_TABLE)}, raw={len(raw_columns)}."
    )


def assert_no_excluded_column_in_frame(columns) -> None:
    """Guard: none of the leakage-audit EXCLUDED variables may appear in a
    predictor frame (X). Called just before modelling in the main script.
    """
    offending = set(columns) & set(EXCLUDED_VARIABLES)
    assert not offending, f"Excluded (leakage/proxy) variable(s) present in X: {offending}"
