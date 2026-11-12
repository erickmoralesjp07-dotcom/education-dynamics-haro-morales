from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_ID = "SCR-0009"

SCREENED_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_01.csv"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

PENDING_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
    / "batch_01_pending_full_text_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSESSMENT_FILE = (
    OUTPUT_DIR
    / "scr_0009_full_text_assessment.csv"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "scr_0009_full_text_assessment_diagnostic.csv"
)

TEXT_COLUMNS = [
    "FULL_TEXT_REQUIRED",
    "FULL_TEXT_REQUESTED",
    "FULL_TEXT_OBTAINED",
    "FULL_TEXT_LOCAL_FILE",
    "CONSENSUS_DECISION",
    "CONSENSUS_REASON",
    "CONSENSUS_DATE",
    "SCREENING_STATUS",
]

DECISION = "Include_final_candidate"

REASON = (
    "Methodologically adequate multi-cycle "
    "PISA inequality study"
)

RISK_OF_BIAS = (
    "Moderate risk of bias"
)

NOTES = (
    "The study uses PISA 2012, 2015, and 2018, "
    "two-level mixed-effects models, country fixed "
    "effects, sampling weights, replicate weights, "
    "plausible values, and robustness checks using "
    "reading outcomes, larger schools, and alternative "
    "inequality measures. It reconstructs the 2012 "
    "wealth index using the later two-parameter IRT "
    "approach for cross-cycle consistency. Important "
    "limitations remain: the analysis is observational "
    "and does not identify causality; treatment of all "
    "plausible values and variance pooling should be "
    "confirmed in the published version and annexes; "
    "missing-data handling is not fully documented; "
    "and no code repository or replication package was "
    "identified. The Research Square preprint was used "
    "for assessment, but the final citation must use "
    "the 2025 Review of Income and Wealth article, "
    "DOI 10.1111/roiw.70040."
)


def prepare_columns(data):
    for column in TEXT_COLUMNS:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    return data


def update_screening_file(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    data = prepare_columns(data)

    mask = data[
        "SCREENING_ID"
    ].eq(SCREENING_ID)

    match_count = int(
        mask.sum()
    )

    if match_count != 1:
        raise ValueError(
            f"{SCREENING_ID} aparece "
            f"{match_count} veces en {path}."
        )

    assessment_datetime = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    data.loc[
        mask,
        "FULL_TEXT_REQUIRED",
    ] = "Yes"

    data.loc[
        mask,
        "FULL_TEXT_REQUESTED",
    ] = "Yes"

    data.loc[
        mask,
        "FULL_TEXT_OBTAINED",
    ] = "Yes"

    data.loc[
        mask,
        "FULL_TEXT_LOCAL_FILE",
    ] = (
        "literature_review/full_text/"
        "screening_batch_01/"
        "SCR-0009_full_text.pdf"
    )

    data.loc[
        mask,
        "CONSENSUS_DECISION",
    ] = DECISION

    data.loc[
        mask,
        "CONSENSUS_REASON",
    ] = REASON

    data.loc[
        mask,
        "CONSENSUS_DATE",
    ] = assessment_datetime

    data.loc[
        mask,
        "SCREENING_STATUS",
    ] = "Full_text_assessed"

    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return data


def main():
    screened = update_screening_file(
        SCREENED_FILE
    )

    queue = update_screening_file(
        QUEUE_FILE
    )

    queue_row = queue.loc[
        queue[
            "SCREENING_ID"
        ].eq(SCREENING_ID)
    ].iloc[0]

    assessment = pd.DataFrame(
        [
            {
                "SCREENING_ID": (
                    SCREENING_ID
                ),
                "TITLE": (
                    queue_row["TITLE"]
                ),
                "AUTHORS": (
                    queue_row["AUTHORS"]
                ),
                "YEAR": (
                    queue_row["YEAR"]
                ),
                "JOURNAL_OR_SOURCE": (
                    queue_row[
                        "JOURNAL_OR_SOURCE"
                    ]
                ),
                "DOI_NORMALIZED": (
                    queue_row[
                        "DOI_NORMALIZED"
                    ]
                ),
                "ORIGINAL_PREPRINT_DOI": (
                    queue_row.get(
                        "ORIGINAL_PREPRINT_DOI",
                        "",
                    )
                ),
                "VERSION_ASSESSED": (
                    "Research Square "
                    "preprint v1"
                ),
                "CITATION_VERSION": (
                    "Published journal "
                    "article"
                ),
                "FULL_TEXT_DECISION": (
                    DECISION
                ),
                "FULL_TEXT_REASON": (
                    REASON
                ),
                "RISK_OF_BIAS": (
                    RISK_OF_BIAS
                ),
                "SAMPLE_DESIGN_ADEQUATE": (
                    True
                ),
                "PISA_SAMPLING_WEIGHTS_USED": (
                    True
                ),
                "PISA_REPLICATE_WEIGHTS_USED": (
                    True
                ),
                "PLAUSIBLE_VALUES_IDENTIFIED": (
                    True
                ),
                "MULTILEVEL_STRUCTURE_USED": (
                    True
                ),
                "COUNTRY_FIXED_EFFECTS_USED": (
                    True
                ),
                "ROBUSTNESS_CHECKS_REPORTED": (
                    True
                ),
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    False
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    False
                ),
                "CODE_AVAILABLE": (
                    False
                ),
                "ASSESSMENT_NOTES": (
                    NOTES
                ),
                "ASSESSMENT_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            }
        ]
    )

    assessment.to_csv(
        ASSESSMENT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    screened_selected = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].eq(SCREENING_ID),
            comparison_columns,
        ]
        .reset_index(
            drop=True
        )
        .fillna("")
    )

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].eq(SCREENING_ID),
            comparison_columns,
        ]
        .reset_index(
            drop=True
        )
        .fillna("")
    )

    files_match = (
        screened_selected.equals(
            queue_selected
        )
    )

    pending_removed = False
    pending_count_after = None

    if PENDING_FILE.exists():
        pending = pd.read_csv(
            PENDING_FILE
        )

        if (
            "SCREENING_ID"
            not in pending.columns
        ):
            raise ValueError(
                "El archivo pendiente no "
                "contiene SCREENING_ID."
            )

        original_matches = int(
            pending[
                "SCREENING_ID"
            ].eq(SCREENING_ID).sum()
        )

        pending = pending.loc[
            ~pending[
                "SCREENING_ID"
            ].eq(SCREENING_ID)
        ].copy()

        pending_count_after = len(
            pending
        )

        pending_removed = (
            original_matches == 1
            and pending_count_after == 11
        )

        pending.to_csv(
            PENDING_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    valid = (
        len(assessment) == 1
        and assessment.loc[
            0,
            "FULL_TEXT_DECISION",
        ] == DECISION
        and assessment.loc[
            0,
            "DOI_NORMALIZED",
        ] == "10.1111/roiw.70040"
        and files_match
        and pending_removed
        and pending_count_after == 11
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "SCREENING_ID": (
                    SCREENING_ID
                ),
                "ASSESSED_RECORDS": 1,
                "FINAL_CANDIDATES": 1,
                "FILES_MATCH": (
                    files_match
                ),
                "REMOVED_FROM_PENDING": (
                    pending_removed
                ),
                "PENDING_RECORDS_AFTER": (
                    pending_count_after
                ),
                "CITATION_DOI": (
                    "10.1111/roiw.70040"
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if valid
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== EVALUACION SCR-0009 ====="
    )

    print(
        assessment[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_REASON",
                "RISK_OF_BIAS",
                "PISA_SAMPLING_WEIGHTS_USED",
                "PISA_REPLICATE_WEIGHTS_USED",
                "PLAUSIBLE_VALUES_IDENTIFIED",
                "MULTILEVEL_STRUCTURE_USED",
                "COUNTRY_FIXED_EFFECTS_USED",
                "ROBUSTNESS_CHECKS_REPORTED",
                "CAUSAL_CLAIM_SUPPORTED",
                "CODE_AVAILABLE",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== DIAGNOSTICO ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "Archivo creado:"
    )

    print(
        ASSESSMENT_FILE
    )

    print()
    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "La evaluacion de "
            "SCR-0009 no coincide "
            "con el resultado esperado."
        )


if __name__ == "__main__":
    main()