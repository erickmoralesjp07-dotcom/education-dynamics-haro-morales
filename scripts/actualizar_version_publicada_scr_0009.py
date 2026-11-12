from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUEUE_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

BATCH_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_batches"
    / "screening_batch_01.csv"
)

SCREENED_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_01.csv"
)

FULL_TEXT_CANDIDATES_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "batch_01_full_text_candidates.csv"
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
    / "full_text_verification"
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

SCREENING_ID = "SCR-0009"

ORIGINAL_PREPRINT_DOI = (
    "10.21203/rs.3.rs-3709698/v1"
)

PUBLISHED_DOI = (
    "10.1111/roiw.70040"
)

PUBLISHED_TITLE = (
    "Within-School Wealth Inequality and "
    "Learning Achievement: Evidence From PISA"
)

PUBLISHED_SOURCE = (
    "Review of Income and Wealth"
)

PUBLISHED_YEAR = 2025

PUBLISHED_DOCUMENT_TYPE = (
    "journal-article"
)

PUBLISHED_URL = (
    "https://doi.org/10.1111/roiw.70040"
)

AUTHORS = (
    "Lucas Néstor Sempé; Lucio Esposito"
)

FILES_TO_UPDATE = [
    QUEUE_FILE,
    BATCH_FILE,
    SCREENED_FILE,
    FULL_TEXT_CANDIDATES_FILE,
    PENDING_FILE,
]


def prepare_text_column(
    data,
    column,
):
    if column not in data.columns:
        data[column] = ""

    data[column] = (
        data[column]
        .astype("string")
        .fillna("")
    )

    return data


def update_file(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    if "SCREENING_ID" not in data.columns:
        raise ValueError(
            f"Falta SCREENING_ID en {path}"
        )

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

    text_columns = [
        "TITLE",
        "AUTHORS",
        "JOURNAL_OR_SOURCE",
        "DOCUMENT_TYPE",
        "DOI_NORMALIZED",
        "URL",
        "VERIFICATION_FLAGS",
        "REVIEWER_1_NOTES",
    ]

    for column in text_columns:
        data = prepare_text_column(
            data,
            column,
        )

    if (
        "ORIGINAL_PREPRINT_DOI"
        not in data.columns
    ):
        data[
            "ORIGINAL_PREPRINT_DOI"
        ] = ""

    data[
        "ORIGINAL_PREPRINT_DOI"
    ] = (
        data[
            "ORIGINAL_PREPRINT_DOI"
        ]
        .astype("string")
        .fillna("")
    )

    if (
        "PUBLISHED_VERSION_STATUS"
        not in data.columns
    ):
        data[
            "PUBLISHED_VERSION_STATUS"
        ] = ""

    data[
        "PUBLISHED_VERSION_STATUS"
    ] = (
        data[
            "PUBLISHED_VERSION_STATUS"
        ]
        .astype("string")
        .fillna("")
    )

    if "VERSION_UPDATE_DATE" not in data.columns:
        data[
            "VERSION_UPDATE_DATE"
        ] = ""

    data[
        "VERSION_UPDATE_DATE"
    ] = (
        data[
            "VERSION_UPDATE_DATE"
        ]
        .astype("string")
        .fillna("")
    )

    previous_notes = str(
        data.loc[
            mask,
            "REVIEWER_1_NOTES",
        ].iloc[0]
    ).strip()

    version_note = (
        "Published version identified: "
        "Sempé and Esposito, 2025, "
        "Review of Income and Wealth, "
        "DOI 10.1111/roiw.70040. "
        "The 2023 Research Square record "
        "is retained only as version history."
    )

    combined_notes = (
        previous_notes
        + (
            " "
            if previous_notes
            else ""
        )
        + version_note
    )

    data.loc[
        mask,
        "ORIGINAL_PREPRINT_DOI",
    ] = ORIGINAL_PREPRINT_DOI

    data.loc[
        mask,
        "TITLE",
    ] = PUBLISHED_TITLE

    data.loc[
        mask,
        "AUTHORS",
    ] = AUTHORS

    if "YEAR" not in data.columns:
        data["YEAR"] = pd.NA

    data.loc[
        mask,
        "YEAR",
    ] = PUBLISHED_YEAR

    data.loc[
        mask,
        "JOURNAL_OR_SOURCE",
    ] = PUBLISHED_SOURCE

    data.loc[
        mask,
        "DOCUMENT_TYPE",
    ] = PUBLISHED_DOCUMENT_TYPE

    data.loc[
        mask,
        "DOI_NORMALIZED",
    ] = PUBLISHED_DOI

    data.loc[
        mask,
        "URL",
    ] = PUBLISHED_URL

    data.loc[
        mask,
        "VERIFICATION_FLAGS",
    ] = ""

    data.loc[
        mask,
        "PUBLISHED_VERSION_STATUS",
    ] = "Published_version_verified"

    data.loc[
        mask,
        "VERSION_UPDATE_DATE",
    ] = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    data.loc[
        mask,
        "REVIEWER_1_NOTES",
    ] = combined_notes

    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return data


def main():
    updated_rows = []

    for path in FILES_TO_UPDATE:
        data = update_file(path)

        row = data.loc[
            data[
                "SCREENING_ID"
            ].eq(SCREENING_ID)
        ].iloc[0]

        updated_rows.append(
            {
                "FILE": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "SCREENING_ID": (
                    row["SCREENING_ID"]
                ),
                "TITLE": row["TITLE"],
                "YEAR": row["YEAR"],
                "JOURNAL_OR_SOURCE": (
                    row[
                        "JOURNAL_OR_SOURCE"
                    ]
                ),
                "DOI_NORMALIZED": (
                    row[
                        "DOI_NORMALIZED"
                    ]
                ),
                "ORIGINAL_PREPRINT_DOI": (
                    row[
                        "ORIGINAL_PREPRINT_DOI"
                    ]
                ),
                "PUBLISHED_VERSION_STATUS": (
                    row[
                        "PUBLISHED_VERSION_STATUS"
                    ]
                ),
            }
        )

    audit = pd.DataFrame(
        updated_rows
    )

    audit_file = (
        OUTPUT_DIR
        / "scr_0009_published_version_update.csv"
    )

    audit.to_csv(
        audit_file,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(audit) == 5
        and audit[
            "DOI_NORMALIZED"
        ].eq(PUBLISHED_DOI).all()
        and audit[
            "ORIGINAL_PREPRINT_DOI"
        ].eq(
            ORIGINAL_PREPRINT_DOI
        ).all()
        and audit[
            "PUBLISHED_VERSION_STATUS"
        ].eq(
            "Published_version_verified"
        ).all()
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
                "FILES_UPDATED": (
                    len(audit)
                ),
                "PUBLISHED_DOI": (
                    PUBLISHED_DOI
                ),
                "ORIGINAL_PREPRINT_DOI": (
                    ORIGINAL_PREPRINT_DOI
                ),
                "PUBLISHED_YEAR": (
                    PUBLISHED_YEAR
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if valid
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "scr_0009_published_version_update_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== ACTUALIZACION SCR-0009 ====="
    )

    print(
        audit.to_string(
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
    print("Archivos creados:")
    print(audit_file)
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
            "La actualizacion publicada "
            "no coincide entre archivos."
        )


if __name__ == "__main__":
    main()