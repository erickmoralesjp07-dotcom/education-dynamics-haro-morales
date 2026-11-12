from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
)

ML_FILE = (
    ASSESSMENT_DIR
    / "batch_01_ml_full_text_assessment.csv"
)

REMAINING_FILE = (
    ASSESSMENT_DIR
    / "batch_01_remaining_full_text_assessment.csv"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

SCREENED_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_01.csv"
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


def main():
    required_files = [
        ML_FILE,
        REMAINING_FILE,
        QUEUE_FILE,
        SCREENED_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    ml = pd.read_csv(ML_FILE)
    remaining = pd.read_csv(
        REMAINING_FILE
    )

    queue = pd.read_csv(
        QUEUE_FILE
    )

    screened = pd.read_csv(
        SCREENED_FILE
    )

    common_columns = [
        "SCREENING_ID",
        "TITLE",
        "FULL_TEXT_DECISION",
        "RISK_OF_BIAS",
        "ASSESSMENT_NOTES",
        "ASSESSMENT_DATETIME",
    ]

    ml_normalized = ml[
        common_columns
    ].copy()

    ml_normalized[
        "FULL_TEXT_REASON"
    ] = ml[
        "FULL_TEXT_EXCLUSION_REASON"
    ]

    ml_normalized[
        "SAMPLE_DESIGN_ADEQUATE"
    ] = False

    ml_normalized[
        "MISSING_DATA_TREATMENT_ADEQUATE"
    ] = False

    ml_normalized[
        "VALIDATION_ADEQUATE"
    ] = ml[
        "EXTERNAL_VALIDATION"
    ]

    ml_normalized[
        "CAUSAL_CLAIM_SUPPORTED"
    ] = False

    ml_normalized[
        "CODE_AVAILABLE"
    ] = ml[
        "CODE_AVAILABLE"
    ]

    remaining_columns = [
        "SCREENING_ID",
        "TITLE",
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

    consolidated = pd.concat(
        [
            ml_normalized[
                remaining_columns
            ],
            remaining[
                remaining_columns
            ],
        ],
        ignore_index=True,
    )

    duplicate_count = int(
        consolidated[
            "SCREENING_ID"
        ].duplicated().sum()
    )

    consolidated = (
        consolidated.sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
    )

    queue_metadata_columns = [
        "SCREENING_ID",
        "YEAR",
        "AUTHORS",
        "JOURNAL_OR_SOURCE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "VERIFICATION_FLAGS",
    ]

    consolidated = consolidated.merge(
        queue[
            queue_metadata_columns
        ],
        on="SCREENING_ID",
        how="left",
        validate="one_to_one",
    )

    ordered_columns = [
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

    consolidated = consolidated[
        ordered_columns
    ]

    consolidated.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        consolidated.groupby(
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
        .reset_index(drop=True)
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

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(
                assessed_ids
            ),
            [
                "SCREENING_ID",
                "CONSENSUS_DECISION",
                "CONSENSUS_REASON",
                "SCREENING_STATUS",
            ],
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
        .fillna("")
    )

    screened_selected = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].isin(
                assessed_ids
            ),
            [
                "SCREENING_ID",
                "CONSENSUS_DECISION",
                "CONSENSUS_REASON",
                "SCREENING_STATUS",
            ],
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
        .fillna("")
    )

    files_match = (
        queue_selected.equals(
            screened_selected
        )
    )

    valid = (
        len(consolidated) == 6
        and duplicate_count == 0
        and final_count == 1
        and limited_count == 2
        and excluded_count == 3
        and len(pending) == 12
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
        "===== CONSOLIDACION DE TEXTO "
        "COMPLETO LOTE 01 ====="
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

    print(
        pending[
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
                "TITLE",
                "DOI_NORMALIZED",
                "VERIFICATION_FLAGS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Archivos creados:"
    )

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(PENDING_FILE)
    print(DIAGNOSTIC_FILE)

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
            "La consolidacion no coincide "
            "con los resultados esperados."
        )


if __name__ == "__main__":
    main()