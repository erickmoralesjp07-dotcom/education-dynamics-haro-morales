from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_ID = "SCR-0002"

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

ASSESSMENT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
    / "scr_0002_full_text_assessment.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "scr_0002_full_text_assessment_diagnostic.csv"
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DECISION = "Include_with_major_limitations"

REASON = (
    "Non-nested feature selection and "
    "optimistically biased cross-validation"
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

    match_count = int(mask.sum())

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
        "SCR-0002_full_text.pdf"
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
    required_files = [
        SCREENED_FILE,
        QUEUE_FILE,
        PENDING_FILE,
        ASSESSMENT_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    assessment = pd.read_csv(
        ASSESSMENT_FILE
    )

    required_assessment_columns = [
        "SCREENING_ID",
        "FULL_TEXT_DECISION",
        "FULL_TEXT_REASON",
        "RISK_OF_BIAS",
    ]

    missing_columns = [
        column
        for column in required_assessment_columns
        if column not in assessment.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en la evaluacion "
            f"de SCR-0002: {missing_columns}"
        )

    assessment_row = assessment.loc[
        assessment[
            "SCREENING_ID"
        ].eq(SCREENING_ID)
    ]

    if len(assessment_row) != 1:
        raise ValueError(
            "La evaluacion de SCR-0002 "
            "debe contener exactamente una fila."
        )

    saved_decision = str(
        assessment_row[
            "FULL_TEXT_DECISION"
        ].iloc[0]
    ).strip()

    saved_reason = str(
        assessment_row[
            "FULL_TEXT_REASON"
        ].iloc[0]
    ).strip()

    if saved_decision != DECISION:
        raise ValueError(
            "La decision guardada no coincide: "
            f"{saved_decision}"
        )

    if saved_reason != REASON:
        raise ValueError(
            "El motivo guardado no coincide: "
            f"{saved_reason}"
        )

    screened = update_screening_file(
        SCREENED_FILE
    )

    queue = update_screening_file(
        QUEUE_FILE
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
        .reset_index(drop=True)
        .fillna("")
    )

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].eq(SCREENING_ID),
            comparison_columns,
        ]
        .reset_index(drop=True)
        .fillna("")
    )

    files_match = (
        screened_selected.equals(
            queue_selected
        )
    )

    pending = pd.read_csv(
        PENDING_FILE
    )

    if "SCREENING_ID" not in pending.columns:
        raise ValueError(
            "El archivo de pendientes no "
            "contiene SCREENING_ID."
        )

    matches_before = int(
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

    pending.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    removed_from_pending = (
        matches_before == 1
        and pending_count_after == 10
    )

    valid = (
        saved_decision == DECISION
        and saved_reason == REASON
        and files_match
        and removed_from_pending
        and pending_count_after == 10
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "SCREENING_ID": SCREENING_ID,
                "ASSESSED_RECORDS": 1,
                "INCLUDE_WITH_MAJOR_LIMITATIONS": 1,
                "FILES_MATCH": files_match,
                "REMOVED_FROM_PENDING": (
                    removed_from_pending
                ),
                "PENDING_RECORDS_AFTER": (
                    pending_count_after
                ),
                "SELECTION_LEAKAGE_FLAG": True,
                "NESTED_FEATURE_SELECTION": False,
                "REPORTED_OOS_R2": 0.8936,
                "CITATION_DOI": (
                    "10.1016/j.ijedudev."
                    "2022.102561"
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
        "===== REGISTRO SCR-0002 ====="
    )

    print(
        "DECISION:",
        saved_decision,
    )

    print(
        "RISK_OF_BIAS:",
        assessment_row[
            "RISK_OF_BIAS"
        ].iloc[0],
    )

    print(
        "FILES_MATCH:",
        files_match,
    )

    print(
        "REMOVED_FROM_PENDING:",
        removed_from_pending,
    )

    print(
        "PENDING_RECORDS_AFTER:",
        pending_count_after,
    )

    print(
        "SELECTION_LEAKAGE_FLAG:",
        True,
    )

    print(
        "NESTED_FEATURE_SELECTION:",
        False,
    )

    print(
        "REPORTED_OOS_R2:",
        0.8936,
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "El registro de SCR-0002 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()