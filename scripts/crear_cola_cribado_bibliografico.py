from pathlib import Path
from datetime import datetime
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
)

BATCH_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_batches"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

BATCH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 25

INPUT_FILES = [
    (
        "PRIORITY",
        RESULTS_DIR
        / "refined_priority_screening.csv",
    ),
    (
        "VERIFY_NO_ABSTRACT",
        RESULTS_DIR
        / "refined_verify_without_abstract.csv",
    ),
    (
        "SECONDARY",
        RESULTS_DIR
        / "refined_secondary_screening.csv",
    ),
]


def safe_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def load_tier(tier, path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    data["SCREENING_TIER"] = tier
    data["SOURCE_FILE"] = str(
        path.relative_to(
            PROJECT_ROOT
        )
    )

    return data


def main():
    frames = []

    for tier, path in INPUT_FILES:
        frame = load_tier(
            tier,
            path,
        )

        frames.append(frame)

    queue = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    duplicate_keys = [
        "DUPLICATE_KEY",
    ]

    duplicate_count = int(
        queue.duplicated(
            duplicate_keys,
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "La cola contiene registros duplicados: "
            f"{duplicate_count}"
        )

    queue.insert(
        0,
        "SCREENING_ID",
        [
            f"SCR-{number:04d}"
            for number in range(
                1,
                len(queue) + 1,
            )
        ],
    )

    queue["BATCH_NUMBER"] = (
        queue.index // BATCH_SIZE
    ) + 1

    queue["BATCH_ID"] = queue[
        "BATCH_NUMBER"
    ].apply(
        lambda value: (
            f"BATCH-{int(value):02d}"
        )
    )

    queue["REVIEWER_1"] = (
        "Erick Morales"
    )

    queue["REVIEWER_1_DECISION"] = ""
    queue["REVIEWER_1_REASON"] = ""
    queue["REVIEWER_1_NOTES"] = ""
    queue["REVIEWER_1_DATE"] = ""

    queue["REVIEWER_2"] = (
        "Second reviewer pending"
    )

    queue["REVIEWER_2_DECISION"] = ""
    queue["REVIEWER_2_REASON"] = ""
    queue["REVIEWER_2_NOTES"] = ""
    queue["REVIEWER_2_DATE"] = ""

    queue["CONFLICT_FLAG"] = False
    queue["CONSENSUS_DECISION"] = ""
    queue["CONSENSUS_REASON"] = ""
    queue["CONSENSUS_DATE"] = ""

    queue["FULL_TEXT_REQUIRED"] = ""
    queue["FULL_TEXT_REQUESTED"] = ""
    queue["FULL_TEXT_OBTAINED"] = ""
    queue["FULL_TEXT_LOCAL_FILE"] = ""

    queue["SCREENING_STATUS"] = "Pending"

    selected_columns = [
        "SCREENING_ID",
        "BATCH_ID",
        "BATCH_NUMBER",
        "SCREENING_TIER",
        "DATABASE",
        "QUERY_ID",
        "THEMATIC_BLOCK",
        "TITLE",
        "AUTHORS",
        "YEAR",
        "JOURNAL_OR_SOURCE",
        "DOCUMENT_TYPE",
        "DOI_NORMALIZED",
        "URL",
        "ABSTRACT",
        "RELEVANCE_SCORE",
        "CITED_BY_COUNT",
        "OPEN_ACCESS",
        "PREPRINT_OR_REPOSITORY",
        "FUTURE_RECORD",
        "VERIFICATION_FLAGS",
        "REFINED_REASON",
        "REVIEWER_1",
        "REVIEWER_1_DECISION",
        "REVIEWER_1_REASON",
        "REVIEWER_1_NOTES",
        "REVIEWER_1_DATE",
        "REVIEWER_2",
        "REVIEWER_2_DECISION",
        "REVIEWER_2_REASON",
        "REVIEWER_2_NOTES",
        "REVIEWER_2_DATE",
        "CONFLICT_FLAG",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "CONSENSUS_DATE",
        "FULL_TEXT_REQUIRED",
        "FULL_TEXT_REQUESTED",
        "FULL_TEXT_OBTAINED",
        "FULL_TEXT_LOCAL_FILE",
        "SCREENING_STATUS",
        "DUPLICATE_KEY",
        "SOURCE_FILE",
    ]

    missing_columns = [
        column
        for column in selected_columns
        if column not in queue.columns
    ]

    for column in missing_columns:
        queue[column] = ""

    queue = queue[
        selected_columns
    ].copy()

    queue_file = (
        RESULTS_DIR
        / "title_abstract_screening_queue.csv"
    )

    queue.to_csv(
        queue_file,
        index=False,
        encoding="utf-8-sig",
    )

    existing_batches = list(
        BATCH_DIR.glob(
            "screening_batch_*.csv"
        )
    )

    for path in existing_batches:
        path.unlink()

    for batch_number, batch in queue.groupby(
        "BATCH_NUMBER",
        sort=True,
    ):
        batch_file = (
            BATCH_DIR
            / (
                f"screening_batch_"
                f"{int(batch_number):02d}.csv"
            )
        )

        batch.to_csv(
            batch_file,
            index=False,
            encoding="utf-8-sig",
        )

    batch_summary = (
        queue.groupby(
            [
                "BATCH_ID",
                "BATCH_NUMBER",
                "SCREENING_TIER",
            ],
            as_index=False,
        )
        .agg(
            RECORD_COUNT=(
                "SCREENING_ID",
                "size",
            ),
            WITH_ABSTRACT=(
                "ABSTRACT",
                lambda values: (
                    values
                    .fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
            ),
            WITH_DOI=(
                "DOI_NORMALIZED",
                lambda values: (
                    values
                    .fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
            ),
            PREPRINT_FLAGS=(
                "PREPRINT_OR_REPOSITORY",
                "sum",
            ),
            FUTURE_FLAGS=(
                "FUTURE_RECORD",
                "sum",
            ),
        )
    )

    batch_summary_file = (
        RESULTS_DIR
        / "screening_batch_summary.csv"
    )

    batch_summary.to_csv(
        batch_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    expected_batches = math.ceil(
        len(queue) / BATCH_SIZE
    )

    actual_batch_files = len(
        list(
            BATCH_DIR.glob(
                "screening_batch_*.csv"
            )
        )
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "QUEUE_RECORDS": len(queue),
                "PRIORITY_RECORDS": int(
                    queue[
                        "SCREENING_TIER"
                    ].eq("PRIORITY").sum()
                ),
                "VERIFY_NO_ABSTRACT_RECORDS": int(
                    queue[
                        "SCREENING_TIER"
                    ].eq(
                        "VERIFY_NO_ABSTRACT"
                    ).sum()
                ),
                "SECONDARY_RECORDS": int(
                    queue[
                        "SCREENING_TIER"
                    ].eq("SECONDARY").sum()
                ),
                "BATCH_SIZE": BATCH_SIZE,
                "EXPECTED_BATCHES": (
                    expected_batches
                ),
                "ACTUAL_BATCH_FILES": (
                    actual_batch_files
                ),
                "DUPLICATE_RECORDS": (
                    duplicate_count
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(queue) == 268
                    and duplicate_count == 0
                    and expected_batches
                    == actual_batch_files
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "title_abstract_screening_queue_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== COLA DE CRIBADO ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESUMEN DE LOTES ====="
    )

    print(
        batch_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PRIMER LOTE ====="
    )

    display_columns = [
        "SCREENING_ID",
        "SCREENING_TIER",
        "YEAR",
        "TITLE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "VERIFICATION_FLAGS",
    ]

    first_batch = queue.loc[
        queue["BATCH_NUMBER"].eq(1)
    ]

    print(
        first_batch[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(queue_file)
    print(batch_summary_file)
    print(BATCH_DIR)
    print(diagnostic_file)

    print()
    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )


if __name__ == "__main__":
    main()