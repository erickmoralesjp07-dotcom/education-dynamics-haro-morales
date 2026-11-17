from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
)

CURRENT_CONSOLIDATED_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_consolidated.csv"
)

SCR_0002_FILE = (
    ASSESSMENT_DIR
    / "scr_0002_full_text_assessment.csv"
)

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

OUTPUT_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_consolidated.csv"
)

SUMMARY_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_summary.csv"
)

PENDING_FILE = (
    ASSESSMENT_DIR
    / "batch_01_pending_full_text_candidates.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "batch_01_full_text_assessment_consolidation_diagnostic.csv"
)

STANDARD_COLUMNS = [
    "SCREENING_ID",
    "TITLE",
    "AUTHORS",
    "YEAR",
    "JOURNAL_OR_SOURCE",
    "DOI_NORMALIZED",
    "THEMATIC_BLOCK",
    "VERIFICATION_FLAGS",
    "FULL_TEXT_DECISION",
    "FULL_TEXT_REASON",
    "RISK_OF_BIAS",
    "SAMPLE_DESIGN_ADEQUATE",
    "MISSING_DATA_TREATMENT_ADEQUATE",
    "VALIDATION_ADEQUATE",
    "CAUSAL_CLAIM_SUPPORTED",
    "CODE_AVAILABLE",
    "ASSESSMENT_NOTES",
    "ASSESSMENT_DATETIME",
]


def to_boolean(value):
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "si",
    }


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def main():
    required_files = [
        CURRENT_CONSOLIDATED_FILE,
        SCR_0002_FILE,
        SCREENED_FILE,
        QUEUE_FILE,
    ]

    for path in required_files:
        require_file(path)

    consolidated = pd.read_csv(
        CURRENT_CONSOLIDATED_FILE
    )

    scr_0002 = pd.read_csv(
        SCR_0002_FILE
    )

    screened = pd.read_csv(
        SCREENED_FILE
    )

    queue = pd.read_csv(
        QUEUE_FILE
    )

    if len(scr_0002) != 1:
        raise ValueError(
            "SCR-0002 debe tener exactamente "
            "una evaluacion."
        )

    scr_row = scr_0002.iloc[0]

    if (
        str(
            scr_row["SCREENING_ID"]
        ).strip()
        != "SCR-0002"
    ):
        raise ValueError(
            "La evaluacion no corresponde "
            "a SCR-0002."
        )

    expected_decision = (
        "Include_with_major_limitations"
    )

    expected_reason = (
        "Non-nested feature selection and "
        "optimistically biased cross-validation"
    )

    if (
        str(
            scr_row[
                "FULL_TEXT_DECISION"
            ]
        ).strip()
        != expected_decision
    ):
        raise ValueError(
            "La decision de SCR-0002 "
            "no coincide."
        )

    if (
        str(
            scr_row[
                "FULL_TEXT_REASON"
            ]
        ).strip()
        != expected_reason
    ):
        raise ValueError(
            "El motivo de SCR-0002 "
            "no coincide."
        )

    queue_match = queue.loc[
        queue[
            "SCREENING_ID"
        ].eq("SCR-0002")
    ]

    if len(queue_match) != 1:
        raise ValueError(
            "SCR-0002 debe aparecer una vez "
            "en la cola."
        )

    queue_row = queue_match.iloc[0]

    scr_0002_standard = pd.DataFrame(
        [
            {
                "SCREENING_ID": "SCR-0002",
                "TITLE": (
                    scr_row["TITLE"]
                ),
                "AUTHORS": (
                    scr_row["AUTHORS"]
                ),
                "YEAR": (
                    scr_row["YEAR"]
                ),
                "JOURNAL_OR_SOURCE": (
                    scr_row[
                        "JOURNAL_OR_SOURCE"
                    ]
                ),
                "DOI_NORMALIZED": (
                    scr_row[
                        "DOI_NORMALIZED"
                    ]
                ),
                "THEMATIC_BLOCK": (
                    queue_row[
                        "THEMATIC_BLOCK"
                    ]
                ),
                "VERIFICATION_FLAGS": (
                    queue_row[
                        "VERIFICATION_FLAGS"
                    ]
                ),
                "FULL_TEXT_DECISION": (
                    scr_row[
                        "FULL_TEXT_DECISION"
                    ]
                ),
                "FULL_TEXT_REASON": (
                    scr_row[
                        "FULL_TEXT_REASON"
                    ]
                ),
                "RISK_OF_BIAS": (
                    scr_row[
                        "RISK_OF_BIAS"
                    ]
                ),
                "SAMPLE_DESIGN_ADEQUATE": (
                    to_boolean(
                        scr_row.get(
                            "SAMPLE_DESIGN_ADEQUATE",
                            False,
                        )
                    )
                ),
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    to_boolean(
                        scr_row.get(
                            "MISSING_DATA_TREATMENT_ADEQUATE",
                            False,
                        )
                    )
                ),
                "VALIDATION_ADEQUATE": (
                    to_boolean(
                        scr_row.get(
                            "VALIDATION_ADEQUATE",
                            False,
                        )
                    )
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    to_boolean(
                        scr_row.get(
                            "CAUSAL_CLAIM_SUPPORTED",
                            False,
                        )
                    )
                ),
                "CODE_AVAILABLE": (
                    to_boolean(
                        scr_row.get(
                            "CODE_AVAILABLE",
                            False,
                        )
                    )
                ),
                "ASSESSMENT_NOTES": (
                    scr_row[
                        "ASSESSMENT_NOTES"
                    ]
                ),
                "ASSESSMENT_DATETIME": (
                    scr_row[
                        "ASSESSMENT_DATETIME"
                    ]
                ),
            }
        ]
    )

    missing_consolidated_columns = [
        column
        for column in STANDARD_COLUMNS
        if column
        not in consolidated.columns
    ]

    if missing_consolidated_columns:
        raise ValueError(
            "Faltan columnas en el consolidado: "
            f"{missing_consolidated_columns}"
        )

    consolidated = consolidated.loc[
        ~consolidated[
            "SCREENING_ID"
        ].eq("SCR-0002")
    ].copy()

    consolidated = pd.concat(
        [
            consolidated[
                STANDARD_COLUMNS
            ],
            scr_0002_standard[
                STANDARD_COLUMNS
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_count = int(
        consolidated[
            "SCREENING_ID"
        ]
        .duplicated()
        .sum()
    )

    consolidated = (
        consolidated
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(
            drop=True
        )
    )

    consolidated.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        consolidated
        .groupby(
            [
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ],
            as_index=False,
        )
        .agg(
            RECORD_COUNT=(
                "SCREENING_ID",
                "size",
            )
        )
        .sort_values(
            [
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    full_text_candidate_ids = set(
        screened.loc[
            screened[
                "REVIEWER_1_DECISION"
            ].isin(
                [
                    "Include",
                    "Unclear",
                ]
            ),
            "SCREENING_ID",
        ]
    )

    assessed_ids = set(
        consolidated[
            "SCREENING_ID"
        ]
    )

    pending_ids = (
        full_text_candidate_ids
        - assessed_ids
    )

    pending = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].isin(
                pending_ids
            )
        ]
        .sort_values(
            [
                "REVIEWER_1_DECISION",
                "RELEVANCE_SCORE",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    pending.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_final_candidate"
        ).sum()
    )

    limited_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_with_major_limitations"
        ).sum()
    )

    excluded_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    scr_0002_present = (
        "SCR-0002"
        in assessed_ids
    )

    scr_0002_not_pending = (
        "SCR-0002"
        not in pending_ids
    )

    comparison_columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(
                assessed_ids
            ),
            comparison_columns,
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(
            drop=True
        )
        .fillna("")
    )

    screened_selected = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].isin(
                assessed_ids
            ),
            comparison_columns,
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(
            drop=True
        )
        .fillna("")
    )

    files_match = (
        queue_selected.equals(
            screened_selected
        )
    )

    valid = (
        len(full_text_candidate_ids)
        == 18
        and len(consolidated)
        == 8
        and duplicate_count
        == 0
        and final_count
        == 2
        and limited_count
        == 3
        and excluded_count
        == 3
        and len(pending)
        == 10
        and scr_0002_present
        and scr_0002_not_pending
        and files_match
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "FULL_TEXT_CANDIDATES": (
                    len(
                        full_text_candidate_ids
                    )
                ),
                "ASSESSED_FULL_TEXTS": (
                    len(consolidated)
                ),
                "FINAL_CANDIDATES": (
                    final_count
                ),
                "INCLUDE_WITH_MAJOR_LIMITATIONS": (
                    limited_count
                ),
                "FULL_TEXT_EXCLUSIONS": (
                    excluded_count
                ),
                "PENDING_FULL_TEXTS": (
                    len(pending)
                ),
                "DUPLICATE_ASSESSMENTS": (
                    duplicate_count
                ),
                "SCR_0002_INCLUDED": (
                    scr_0002_present
                ),
                "SCR_0002_REMOVED_FROM_PENDING": (
                    scr_0002_not_pending
                ),
                "FILES_MATCH": (
                    files_match
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
        "===== CONSOLIDACION CON "
        "SCR-0002 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== EVALUACIONES ====="
    )

    print(
        consolidated[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_REASON",
                "RISK_OF_BIAS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESUMEN ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PENDIENTES ====="
    )

    pending_columns = [
        "SCREENING_ID",
        "REVIEWER_1_DECISION",
        "TITLE",
        "DOI_NORMALIZED",
        "VERIFICATION_FLAGS",
    ]

    print(
        pending[
            pending_columns
        ].to_string(
            index=False
        )
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
            "La consolidacion con SCR-0002 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()