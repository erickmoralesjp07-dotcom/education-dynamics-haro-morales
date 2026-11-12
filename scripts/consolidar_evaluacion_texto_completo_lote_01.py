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

SCR_0009_FILE = (
    ASSESSMENT_DIR
    / "scr_0009_full_text_assessment.csv"
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

STANDARD_COLUMNS = [
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


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido: {path}"
        )


def to_boolean(series):
    if series.dtype == bool:
        return series

    normalized = (
        series
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "si",
        ]
    )


def normalize_ml_assessment(data):
    required_columns = [
        "SCREENING_ID",
        "TITLE",
        "FULL_TEXT_DECISION",
        "FULL_TEXT_EXCLUSION_REASON",
        "RISK_OF_BIAS",
        "EXTERNAL_VALIDATION",
        "CODE_AVAILABLE",
        "ASSESSMENT_NOTES",
        "ASSESSMENT_DATETIME",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en la evaluacion ML: "
            f"{missing_columns}"
        )

    normalized = pd.DataFrame(
        {
            "SCREENING_ID": (
                data["SCREENING_ID"]
            ),
            "TITLE": (
                data["TITLE"]
            ),
            "FULL_TEXT_DECISION": (
                data["FULL_TEXT_DECISION"]
            ),
            "FULL_TEXT_REASON": (
                data[
                    "FULL_TEXT_EXCLUSION_REASON"
                ]
            ),
            "RISK_OF_BIAS": (
                data["RISK_OF_BIAS"]
            ),
            "SAMPLE_DESIGN_ADEQUATE": (
                False
            ),
            "MISSING_DATA_TREATMENT_ADEQUATE": (
                False
            ),
            "VALIDATION_ADEQUATE": (
                to_boolean(
                    data[
                        "EXTERNAL_VALIDATION"
                    ]
                )
            ),
            "CAUSAL_CLAIM_SUPPORTED": (
                False
            ),
            "CODE_AVAILABLE": (
                to_boolean(
                    data[
                        "CODE_AVAILABLE"
                    ]
                )
            ),
            "ASSESSMENT_NOTES": (
                data["ASSESSMENT_NOTES"]
            ),
            "ASSESSMENT_DATETIME": (
                data["ASSESSMENT_DATETIME"]
            ),
        }
    )

    return normalized[
        STANDARD_COLUMNS
    ]


def normalize_remaining_assessment(data):
    missing_columns = [
        column
        for column in STANDARD_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en la evaluacion "
            f"de los cuatro estudios: {missing_columns}"
        )

    normalized = data[
        STANDARD_COLUMNS
    ].copy()

    boolean_columns = [
        "SAMPLE_DESIGN_ADEQUATE",
        "MISSING_DATA_TREATMENT_ADEQUATE",
        "VALIDATION_ADEQUATE",
        "CAUSAL_CLAIM_SUPPORTED",
        "CODE_AVAILABLE",
    ]

    for column in boolean_columns:
        normalized[column] = to_boolean(
            normalized[column]
        )

    return normalized


def normalize_scr_0009_assessment(data):
    required_columns = [
        "SCREENING_ID",
        "TITLE",
        "FULL_TEXT_DECISION",
        "FULL_TEXT_REASON",
        "RISK_OF_BIAS",
        "SAMPLE_DESIGN_ADEQUATE",
        "MISSING_DATA_TREATMENT_ADEQUATE",
        "PISA_SAMPLING_WEIGHTS_USED",
        "PISA_REPLICATE_WEIGHTS_USED",
        "PLAUSIBLE_VALUES_IDENTIFIED",
        "MULTILEVEL_STRUCTURE_USED",
        "COUNTRY_FIXED_EFFECTS_USED",
        "ROBUSTNESS_CHECKS_REPORTED",
        "CAUSAL_CLAIM_SUPPORTED",
        "CODE_AVAILABLE",
        "ASSESSMENT_NOTES",
        "ASSESSMENT_DATETIME",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en la evaluacion "
            f"SCR-0009: {missing_columns}"
        )

    validation_columns = [
        "PISA_SAMPLING_WEIGHTS_USED",
        "PISA_REPLICATE_WEIGHTS_USED",
        "PLAUSIBLE_VALUES_IDENTIFIED",
        "MULTILEVEL_STRUCTURE_USED",
        "COUNTRY_FIXED_EFFECTS_USED",
        "ROBUSTNESS_CHECKS_REPORTED",
    ]

    validation_matrix = pd.DataFrame(
        {
            column: to_boolean(
                data[column]
            )
            for column
            in validation_columns
        }
    )

    normalized = pd.DataFrame(
        {
            "SCREENING_ID": (
                data["SCREENING_ID"]
            ),
            "TITLE": (
                data["TITLE"]
            ),
            "FULL_TEXT_DECISION": (
                data["FULL_TEXT_DECISION"]
            ),
            "FULL_TEXT_REASON": (
                data["FULL_TEXT_REASON"]
            ),
            "RISK_OF_BIAS": (
                data["RISK_OF_BIAS"]
            ),
            "SAMPLE_DESIGN_ADEQUATE": (
                to_boolean(
                    data[
                        "SAMPLE_DESIGN_ADEQUATE"
                    ]
                )
            ),
            "MISSING_DATA_TREATMENT_ADEQUATE": (
                to_boolean(
                    data[
                        "MISSING_DATA_TREATMENT_ADEQUATE"
                    ]
                )
            ),
            "VALIDATION_ADEQUATE": (
                validation_matrix.all(
                    axis=1
                )
            ),
            "CAUSAL_CLAIM_SUPPORTED": (
                to_boolean(
                    data[
                        "CAUSAL_CLAIM_SUPPORTED"
                    ]
                )
            ),
            "CODE_AVAILABLE": (
                to_boolean(
                    data[
                        "CODE_AVAILABLE"
                    ]
                )
            ),
            "ASSESSMENT_NOTES": (
                data["ASSESSMENT_NOTES"]
            ),
            "ASSESSMENT_DATETIME": (
                data["ASSESSMENT_DATETIME"]
            ),
        }
    )

    return normalized[
        STANDARD_COLUMNS
    ]


def main():
    required_files = [
        ML_FILE,
        REMAINING_FILE,
        SCR_0009_FILE,
        QUEUE_FILE,
        SCREENED_FILE,
    ]

    for path in required_files:
        require_file(path)

    ml_raw = pd.read_csv(
        ML_FILE
    )

    remaining_raw = pd.read_csv(
        REMAINING_FILE
    )

    scr_0009_raw = pd.read_csv(
        SCR_0009_FILE
    )

    queue = pd.read_csv(
        QUEUE_FILE
    )

    screened = pd.read_csv(
        SCREENED_FILE
    )

    ml = normalize_ml_assessment(
        ml_raw
    )

    remaining = (
        normalize_remaining_assessment(
            remaining_raw
        )
    )

    scr_0009 = (
        normalize_scr_0009_assessment(
            scr_0009_raw
        )
    )

    consolidated = pd.concat(
        [
            ml,
            remaining,
            scr_0009,
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

    metadata_columns = [
        "SCREENING_ID",
        "YEAR",
        "AUTHORS",
        "JOURNAL_OR_SOURCE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "VERIFICATION_FLAGS",
    ]

    missing_metadata_columns = [
        column
        for column in metadata_columns
        if column not in queue.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "Faltan columnas en la cola: "
            f"{missing_metadata_columns}"
        )

    consolidated = consolidated.merge(
        queue[
            metadata_columns
        ],
        on="SCREENING_ID",
        how="left",
        validate="one_to_one",
    )

    output_columns = [
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
        output_columns
    ]

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

    comparison_columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    missing_queue_comparison = [
        column
        for column in comparison_columns
        if column not in queue.columns
    ]

    missing_screened_comparison = [
        column
        for column in comparison_columns
        if column not in screened.columns
    ]

    if missing_queue_comparison:
        raise ValueError(
            "Faltan columnas de comparacion "
            f"en la cola: {missing_queue_comparison}"
        )

    if missing_screened_comparison:
        raise ValueError(
            "Faltan columnas de comparacion "
            "en el lote cribado: "
            f"{missing_screened_comparison}"
        )

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

    scr_0009_present = (
        "SCR-0009"
        in assessed_ids
    )

    scr_0009_not_pending = (
        "SCR-0009"
        not in pending_ids
    )

    valid = (
        len(full_text_candidate_ids)
        == 18
        and len(consolidated)
        == 7
        and duplicate_count
        == 0
        and final_count
        == 2
        and limited_count
        == 2
        and excluded_count
        == 3
        and len(pending)
        == 11
        and files_match
        and scr_0009_present
        and scr_0009_not_pending
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
                "SCR_0009_INCLUDED": (
                    scr_0009_present
                ),
                "SCR_0009_REMOVED_FROM_PENDING": (
                    scr_0009_not_pending
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
        "===== CONSOLIDACION ACTUALIZADA "
        "DE TEXTO COMPLETO LOTE 01 ====="
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
            "La consolidacion actualizada "
            "no coincide con los resultados "
            "esperados."
        )


if __name__ == "__main__":
    main()