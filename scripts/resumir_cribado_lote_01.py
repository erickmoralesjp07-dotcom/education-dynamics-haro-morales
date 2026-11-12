from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_batches"
    / "screening_batch_01.csv"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
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


def main():
    if not BATCH_FILE.exists():
        raise FileNotFoundError(BATCH_FILE)

    if not QUEUE_FILE.exists():
        raise FileNotFoundError(QUEUE_FILE)

    batch = pd.read_csv(BATCH_FILE)
    queue = pd.read_csv(QUEUE_FILE)

    if len(batch) != 25:
        raise ValueError(
            "El lote 01 debe tener 25 registros."
        )

    expected_decisions = {
        "Include",
        "Exclude",
        "Unclear",
    }

    decisions = set(
        batch[
            "REVIEWER_1_DECISION"
        ]
        .fillna("")
        .str.strip()
        .unique()
    )

    invalid_decisions = (
        decisions
        - expected_decisions
    )

    if invalid_decisions:
        raise ValueError(
            "Existen decisiones invalidas o vacias: "
            f"{sorted(invalid_decisions)}"
        )

    batch_ids = set(
        batch["SCREENING_ID"]
    )

    queue_batch = queue.loc[
        queue["SCREENING_ID"].isin(
            batch_ids
        )
    ].copy()

    comparison_columns = [
        "SCREENING_ID",
        "REVIEWER_1_DECISION",
        "REVIEWER_1_REASON",
        "REVIEWER_1_NOTES",
        "SCREENING_STATUS",
    ]

    batch_comparison = (
        batch[comparison_columns]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
        .fillna("")
    )

    queue_comparison = (
        queue_batch[comparison_columns]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
        .fillna("")
    )

    records_match = (
        batch_comparison.equals(
            queue_comparison
        )
    )

    include_count = int(
        batch[
            "REVIEWER_1_DECISION"
        ].eq("Include").sum()
    )

    exclude_count = int(
        batch[
            "REVIEWER_1_DECISION"
        ].eq("Exclude").sum()
    )

    unclear_count = int(
        batch[
            "REVIEWER_1_DECISION"
        ].eq("Unclear").sum()
    )

    completed_count = int(
        batch[
            "SCREENING_STATUS"
        ].eq("Reviewer_1_completed")
        .sum()
    )

    full_text_candidates = batch.loc[
        batch[
            "REVIEWER_1_DECISION"
        ].isin(
            [
                "Include",
                "Unclear",
            ]
        )
    ].copy()

    full_text_candidates[
        "FULL_TEXT_PRIORITY"
    ] = full_text_candidates[
        "REVIEWER_1_DECISION"
    ].map(
        {
            "Include": 1,
            "Unclear": 2,
        }
    )

    full_text_candidates = (
        full_text_candidates.sort_values(
            [
                "FULL_TEXT_PRIORITY",
                "RELEVANCE_SCORE",
                "CITED_BY_COUNT",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    exclusion_summary = (
        batch.loc[
            batch[
                "REVIEWER_1_DECISION"
            ].eq("Exclude")
        ]
        .groupby(
            "REVIEWER_1_REASON",
            as_index=False,
        )
        .agg(
            RECORD_COUNT=(
                "SCREENING_ID",
                "size",
            )
        )
        .sort_values(
            "RECORD_COUNT",
            ascending=False,
        )
    )

    decision_summary = (
        batch.groupby(
            "REVIEWER_1_DECISION",
            as_index=False,
        )
        .agg(
            RECORD_COUNT=(
                "SCREENING_ID",
                "size",
            ),
            WITH_DOI=(
                "DOI_NORMALIZED",
                lambda values: (
                    values.fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
            ),
            WITH_ABSTRACT=(
                "ABSTRACT",
                lambda values: (
                    values.fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
            ),
        )
    )

    screened_file = (
        OUTPUT_DIR
        / "screened_batch_01.csv"
    )

    full_text_file = (
        OUTPUT_DIR
        / "batch_01_full_text_candidates.csv"
    )

    exclusion_file = (
        OUTPUT_DIR
        / "batch_01_exclusion_summary.csv"
    )

    decision_file = (
        OUTPUT_DIR
        / "batch_01_decision_summary.csv"
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "batch_01_screening_diagnostic.csv"
    )

    batch.to_csv(
        screened_file,
        index=False,
        encoding="utf-8-sig",
    )

    full_text_candidates.to_csv(
        full_text_file,
        index=False,
        encoding="utf-8-sig",
    )

    exclusion_summary.to_csv(
        exclusion_file,
        index=False,
        encoding="utf-8-sig",
    )

    decision_summary.to_csv(
        decision_file,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(batch) == 25
        and include_count == 15
        and exclude_count == 7
        and unclear_count == 3
        and completed_count == 25
        and records_match
        and len(full_text_candidates) == 18
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "BATCH_ID": "BATCH-01",
                "TOTAL_RECORDS": len(batch),
                "INCLUDE_RECORDS": include_count,
                "EXCLUDE_RECORDS": exclude_count,
                "UNCLEAR_RECORDS": unclear_count,
                "COMPLETED_RECORDS": (
                    completed_count
                ),
                "FULL_TEXT_CANDIDATES": (
                    len(full_text_candidates)
                ),
                "BATCH_QUEUE_MATCH": (
                    records_match
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
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== RESUMEN CRIBADO LOTE 01 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== DECISIONES ====="
    )

    print(
        decision_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== MOTIVOS DE EXCLUSION ====="
    )

    print(
        exclusion_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== CANDIDATOS A TEXTO COMPLETO ====="
    )

    display_columns = [
        "SCREENING_ID",
        "REVIEWER_1_DECISION",
        "YEAR",
        "TITLE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "VERIFICATION_FLAGS",
    ]

    print(
        full_text_candidates[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(screened_file)
    print(full_text_file)
    print(exclusion_file)
    print(decision_file)
    print(diagnostic_file)

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
            "El resumen del lote 01 no coincide "
            "con las decisiones esperadas."
        )


if __name__ == "__main__":
    main()