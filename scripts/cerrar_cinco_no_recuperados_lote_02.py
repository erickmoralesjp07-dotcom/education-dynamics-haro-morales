from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PENDING_FILE = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "batch_02_pending_full_text_candidates.csv"
)

OUTPUT_FILE = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "batch_02_unretrieved_full_text_closure.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "batch_02_unretrieved_full_text_closure_diagnostic.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()

REASONS = {
    "SCR-0033": (
        "Full text not retrieved after six attempts. "
        "Publisher and DOI routes returned HTTP 403, "
        "and one alternative PDF route returned HTTP 404."
    ),
    "SCR-0034": (
        "Full text not retrieved after three attempts. "
        "DOI and Taylor & Francis routes returned HTTP 403."
    ),
    "SCR-0036": (
        "Full text not retrieved after two attempts. "
        "Repository landing page was not a PDF and the "
        "direct Bryant University file route returned HTTP 403."
    ),
    "SCR-0037": (
        "Full text not retrieved after six attempts. "
        "DOI, Mannheim and SAGE routes returned HTTP 403; "
        "the handle route did not return a valid PDF."
    ),
    "SCR-0050": (
        "Full text not retrieved. Nottingham returned HTTP 404 "
        "and the legacy CORE identifier reports that no repository "
        "ID exists for 33565871."
    ),
}


def main():
    if not PENDING_FILE.exists():
        raise FileNotFoundError(
            PENDING_FILE
        )

    pending = pd.read_csv(
        PENDING_FILE,
        encoding="utf-8-sig",
    )

    expected_ids = set(
        REASONS
    )

    actual_ids = set(
        pending[
            "SCREENING_ID"
        ].astype(str)
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Los pendientes actuales no coinciden con "
            "los cinco registros esperados."
        )

    rows = []

    for screening_id in sorted(expected_ids):
        source = pending.loc[
            pending[
                "SCREENING_ID"
            ].eq(screening_id)
        ].iloc[0]

        rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": source.get(
                    "TITLE",
                    "",
                ),
                "DOI_NORMALIZED": source.get(
                    "DOI_NORMALIZED",
                    "",
                ),
                "FULL_TEXT_STATUS": (
                    "NOT_RETRIEVED_AFTER_EXHAUSTIVE_SEARCH"
                ),
                "CLOSURE_REASON": REASONS[
                    screening_id
                ],
                "ASSESSMENT_STATUS": (
                    "NOT_ASSESSED_NO_FULL_TEXT"
                ),
                "CLOSURE_DATETIME": NOW,
            }
        )

    closure = pd.DataFrame(
        rows
    )

    closure.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(closure) == 5
        and set(
            closure[
                "SCREENING_ID"
            ]
        ) == expected_ids
        and closure[
            "FULL_TEXT_STATUS"
        ].eq(
            "NOT_RETRIEVED_AFTER_EXHAUSTIVE_SEARCH"
        ).all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "BATCH_ID": (
                    "screening_batch_02"
                ),
                "UNRETRIEVED_FULL_TEXTS": (
                    len(closure)
                ),
                "ASSESSED_FULL_TEXTS": 13,
                "FINAL_CANDIDATES": 7,
                "INCLUDE_WITH_MAJOR_LIMITATIONS": 4,
                "FULL_TEXT_EXCLUSIONS": 2,
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
        "===== CIERRE DE NO RECUPERADOS "
        "LOTE 02 ====="
    )

    print(
        closure[
            [
                "SCREENING_ID",
                "FULL_TEXT_STATUS",
                "ASSESSMENT_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "NO RECUPERADOS:",
        len(closure),
    )

    print(
        "EVALUADOS:",
        13,
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    print()
    print(
        "ARCHIVO:",
        OUTPUT_FILE,
    )

    if not valid:
        raise ValueError(
            "El cierre de no recuperados "
            "requiere revision."
        )


if __name__ == "__main__":
    main()