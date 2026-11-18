from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_DIR = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
)

SCREENED_FILE = (
    ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_01.csv"
)

QUEUE_FILE = (
    ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

CONSOLIDATED_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_consolidated.csv"
)

PENDING_FILE = (
    ASSESSMENT_DIR
    / "batch_01_pending_full_text_candidates.csv"
)

NOT_RETRIEVED_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_reports_not_retrieved.csv"
)

UNVERIFIED_FILE = (
    ASSESSMENT_DIR
    / "batch_01_unverified_records_excluded.csv"
)

CLOSURE_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_closure_summary.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "batch_01_full_text_closure_diagnostic.csv"
)

SCRIPT_TIME = (
    datetime.now()
    .astimezone()
    .isoformat()
)

NOT_RETRIEVED = {
    "SCR-0005": {
        "reason": (
            "Published Wiley article identified, but the "
            "full text could not be retrieved. The direct "
            "Wiley request returned HTTP 403 and the SSRN "
            "version was unavailable."
        ),
        "doi": "10.1111/1467-9361.00162",
        "access_status": (
            "REPORT_NOT_RETRIEVED"
        ),
    },
    "SCR-0007": {
        "reason": (
            "Published Emerald article identified, but no "
            "open full text was located and the publisher "
            "page required restricted or paid access."
        ),
        "doi": "10.1108/jes-05-2023-0246",
        "access_status": (
            "REPORT_NOT_RETRIEVED_PAYWALL"
        ),
    },
    "SCR-0013": {
        "reason": (
            "SSRN record identified, but both attempted "
            "delivery URLs returned HTTP 403 and no "
            "alternative legal full-text copy was located."
        ),
        "doi": "10.2139/ssrn.2179525",
        "access_status": (
            "REPORT_NOT_RETRIEVED"
        ),
    },
}

UNVERIFIED = {
    "SCR-0024": {
        "decision": (
            "Exclude_unverified_record"
        ),
        "reason": (
            "Unverified future-dated 2026 preprint record "
            "with no retrievable full text. The record "
            "cannot be treated as an assessed published "
            "study."
        ),
        "doi": "10.35542/osf.io/fdytm_v1",
        "status": (
            "Excluded_unverified_record"
        ),
    },
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def prepare_text_columns(data):
    columns = [
        "FULL_TEXT_REQUIRED",
        "FULL_TEXT_REQUESTED",
        "FULL_TEXT_OBTAINED",
        "FULL_TEXT_LOCAL_FILE",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "CONSENSUS_DATE",
        "SCREENING_STATUS",
    ]

    for column in columns:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    return data


def update_registries(data):
    data = prepare_text_columns(data)

    for screening_id, record in (
        NOT_RETRIEVED.items()
    ):
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        if int(mask.sum()) != 1:
            raise ValueError(
                f"{screening_id} no aparece "
                "exactamente una vez."
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
        ] = "No"

        data.loc[
            mask,
            "FULL_TEXT_LOCAL_FILE",
        ] = ""

        data.loc[
            mask,
            "CONSENSUS_DECISION",
        ] = "Report_not_retrieved"

        data.loc[
            mask,
            "CONSENSUS_REASON",
        ] = record["reason"]

        data.loc[
            mask,
            "CONSENSUS_DATE",
        ] = SCRIPT_TIME

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = record["access_status"]

    for screening_id, record in (
        UNVERIFIED.items()
    ):
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        if int(mask.sum()) != 1:
            raise ValueError(
                f"{screening_id} no aparece "
                "exactamente una vez."
            )

        data.loc[
            mask,
            "FULL_TEXT_REQUIRED",
        ] = "No"

        data.loc[
            mask,
            "FULL_TEXT_REQUESTED",
        ] = "No"

        data.loc[
            mask,
            "FULL_TEXT_OBTAINED",
        ] = "No"

        data.loc[
            mask,
            "FULL_TEXT_LOCAL_FILE",
        ] = ""

        data.loc[
            mask,
            "CONSENSUS_DECISION",
        ] = record["decision"]

        data.loc[
            mask,
            "CONSENSUS_REASON",
        ] = record["reason"]

        data.loc[
            mask,
            "CONSENSUS_DATE",
        ] = SCRIPT_TIME

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = record["status"]

    return data


def main():
    required_files = [
        SCREENED_FILE,
        QUEUE_FILE,
        CONSOLIDATED_FILE,
        PENDING_FILE,
    ]

    for path in required_files:
        require_file(path)

    screened = pd.read_csv(
        SCREENED_FILE
    )

    queue = pd.read_csv(
        QUEUE_FILE
    )

    consolidated = pd.read_csv(
        CONSOLIDATED_FILE
    )

    pending_before = pd.read_csv(
        PENDING_FILE
    )

    screened = update_registries(
        screened
    )

    queue = update_registries(
        queue
    )

    screened.to_csv(
        SCREENED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    queue.to_csv(
        QUEUE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    not_retrieved_rows = []

    for screening_id, record in (
        NOT_RETRIEVED.items()
    ):
        row = queue.loc[
            queue[
                "SCREENING_ID"
            ].eq(screening_id)
        ].iloc[0]

        not_retrieved_rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": row.get(
                    "TITLE",
                    "",
                ),
                "AUTHORS": row.get(
                    "AUTHORS",
                    "",
                ),
                "YEAR": row.get(
                    "YEAR",
                    "",
                ),
                "JOURNAL_OR_SOURCE": row.get(
                    "JOURNAL_OR_SOURCE",
                    "",
                ),
                "DOI_NORMALIZED": (
                    record["doi"]
                ),
                "RETRIEVAL_STATUS": (
                    record["access_status"]
                ),
                "RETRIEVAL_REASON": (
                    record["reason"]
                ),
                "FULL_TEXT_REQUESTED": (
                    "Yes"
                ),
                "FULL_TEXT_OBTAINED": (
                    "No"
                ),
                "RECORD_DATETIME": (
                    SCRIPT_TIME
                ),
            }
        )

    not_retrieved = pd.DataFrame(
        not_retrieved_rows
    )

    not_retrieved.to_csv(
        NOT_RETRIEVED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    unverified_rows = []

    for screening_id, record in (
        UNVERIFIED.items()
    ):
        row = queue.loc[
            queue[
                "SCREENING_ID"
            ].eq(screening_id)
        ].iloc[0]

        unverified_rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": row.get(
                    "TITLE",
                    "",
                ),
                "AUTHORS": row.get(
                    "AUTHORS",
                    "",
                ),
                "YEAR": row.get(
                    "YEAR",
                    "",
                ),
                "DOI_NORMALIZED": (
                    record["doi"]
                ),
                "DECISION": (
                    record["decision"]
                ),
                "EXCLUSION_REASON": (
                    record["reason"]
                ),
                "VERIFICATION_FLAGS": (
                    row.get(
                        "VERIFICATION_FLAGS",
                        "",
                    )
                ),
                "RECORD_DATETIME": (
                    SCRIPT_TIME
                ),
            }
        )

    unverified = pd.DataFrame(
        unverified_rows
    )

    unverified.to_csv(
        UNVERIFIED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    open_ids = (
        set(NOT_RETRIEVED.keys())
        | set(UNVERIFIED.keys())
    )

    pending_after = pending_before.loc[
        ~pending_before[
            "SCREENING_ID"
        ].isin(open_ids)
    ].copy()

    pending_after.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    assessed_count = len(
        consolidated
    )

    reports_not_retrieved = len(
        not_retrieved
    )

    unverified_excluded = len(
        unverified
    )

    final_candidates = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_final_candidate"
        ).sum()
    )

    limited_includes = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_with_major_limitations"
        ).sum()
    )

    full_text_exclusions = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    closure = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    SCRIPT_TIME
                ),
                "FULL_TEXT_CANDIDATES": 18,
                "REPORTS_RETRIEVED": 14,
                "REPORTS_NOT_RETRIEVED": (
                    reports_not_retrieved
                ),
                "UNVERIFIED_RECORDS_EXCLUDED": (
                    unverified_excluded
                ),
                "FULL_TEXTS_ASSESSED": (
                    assessed_count
                ),
                "FINAL_CANDIDATES": (
                    final_candidates
                ),
                "INCLUDE_WITH_MAJOR_LIMITATIONS": (
                    limited_includes
                ),
                "FULL_TEXT_EXCLUSIONS": (
                    full_text_exclusions
                ),
                "OPEN_PENDING_RECORDS": (
                    len(pending_after)
                ),
            }
        ]
    )

    closure.to_csv(
        CLOSURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    affected_ids = sorted(
        open_ids
    )

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(affected_ids),
            comparison_columns,
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
            ].isin(affected_ids),
            comparison_columns,
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
        assessed_count == 14
        and reports_not_retrieved == 3
        and unverified_excluded == 1
        and final_candidates == 5
        and limited_includes == 6
        and full_text_exclusions == 3
        and len(pending_before) == 4
        and len(pending_after) == 0
        and files_match
    )

    diagnostic = closure.copy()

    diagnostic[
        "FILES_MATCH"
    ] = files_match

    diagnostic[
        "VALIDATION_STATUS"
    ] = (
        "OK"
        if valid
        else "REVIEW"
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== CIERRE DE TEXTOS "
        "PENDIENTES LOTE 01 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== INFORMES NO "
        "RECUPERADOS ====="
    )

    print(
        not_retrieved[
            [
                "SCREENING_ID",
                "DOI_NORMALIZED",
                "RETRIEVAL_STATUS",
                "RETRIEVAL_REASON",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== REGISTRO NO "
        "VERIFICADO ====="
    )

    print(
        unverified[
            [
                "SCREENING_ID",
                "DOI_NORMALIZED",
                "DECISION",
                "EXCLUSION_REASON",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Archivos creados:"
    )

    print(NOT_RETRIEVED_FILE)
    print(UNVERIFIED_FILE)
    print(CLOSURE_FILE)
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
            "El cierre del lote 01 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()