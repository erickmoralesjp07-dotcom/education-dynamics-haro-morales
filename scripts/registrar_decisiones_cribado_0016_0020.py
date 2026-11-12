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

TEXT_COLUMNS = [
    "REVIEWER_1_DECISION",
    "REVIEWER_1_REASON",
    "REVIEWER_1_NOTES",
    "REVIEWER_1_DATE",
    "SCREENING_STATUS",
]

DECISIONS = {
    "SCR-0016": {
        "decision": "Exclude",
        "reason": (
            "Economic outcome without educational outcome"
        ),
        "notes": (
            "The study examines educational expenditure "
            "primarily in relation to economic growth and "
            "regional income. It does not directly model "
            "educational access, adult attainment, "
            "literacy, learning, or educational inequality."
        ),
    },
    "SCR-0017": {
        "decision": "Include",
        "reason": (
            "Central methodological study of measurement "
            "error in cross-country education data"
        ),
        "notes": (
            "Highly relevant to the use of changes in "
            "educational attainment across countries. "
            "Discusses measurement error in first-"
            "differenced cross-country education data, "
            "linearity assumptions, coefficient "
            "heterogeneity, and human-capital measurement."
        ),
    },
    "SCR-0018": {
        "decision": "Include",
        "reason": (
            "Relevant multi-country PISA study with "
            "educational performance outcomes"
        ),
        "notes": (
            "Uses PISA microdata for 22 countries and "
            "education-specific institutional measures. "
            "Relevant for international educational "
            "comparisons, student performance, PISA, and "
            "cross-country institutional heterogeneity."
        ),
    },
    "SCR-0019": {
        "decision": "Exclude",
        "reason": (
            "No international or cross-country component"
        ),
        "notes": (
            "The study uses longitudinal individual-level "
            "data from India to estimate the educational "
            "effects of natural disasters. Despite its "
            "valid panel design, it does not address "
            "cross-country educational dynamics or "
            "international indicator comparability."
        ),
    },
    "SCR-0020": {
        "decision": "Include",
        "reason": (
            "Relevant time-series study of education "
            "financing and access"
        ),
        "notes": (
            "Directly examines education expenditure and "
            "access through primary, secondary, and "
            "tertiary enrolment over 1990-2015. Relevant "
            "to the financing-access relationship and "
            "UNESCO policy context. Full text must verify "
            "the UNESCO 26 percent claim, econometric "
            "specification, and journal quality."
        ),
    },
}


def prepare_text_columns(data):
    for column in TEXT_COLUMNS:
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

    required_columns = [
        "SCREENING_ID",
        *TEXT_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {path}: "
            f"{missing_columns}"
        )

    data = prepare_text_columns(data)

    current_datetime = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    for screening_id, decision in DECISIONS.items():
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        match_count = int(mask.sum())

        if match_count != 1:
            raise ValueError(
                f"{screening_id} aparece "
                f"{match_count} veces en {path}."
            )

        existing_decision = str(
            data.loc[
                mask,
                "REVIEWER_1_DECISION",
            ].iloc[0]
        ).strip()

        if existing_decision not in {
            "",
            decision["decision"],
        }:
            raise ValueError(
                f"{screening_id} ya tiene una decision "
                f"diferente: {existing_decision}"
            )

        data.loc[
            mask,
            "REVIEWER_1_DECISION",
        ] = decision["decision"]

        data.loc[
            mask,
            "REVIEWER_1_REASON",
        ] = decision["reason"]

        data.loc[
            mask,
            "REVIEWER_1_NOTES",
        ] = decision["notes"]

        data.loc[
            mask,
            "REVIEWER_1_DATE",
        ] = current_datetime

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = "Reviewer_1_completed"

    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return data


def main():
    batch = update_file(BATCH_FILE)
    queue = update_file(QUEUE_FILE)

    columns = [
        "SCREENING_ID",
        "REVIEWER_1_DECISION",
        "REVIEWER_1_REASON",
        "SCREENING_STATUS",
    ]

    queue_selected = (
        queue.loc[
            queue["SCREENING_ID"].isin(
                DECISIONS.keys()
            ),
            columns,
        ]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
    )

    batch_selected = (
        batch.loc[
            batch["SCREENING_ID"].isin(
                DECISIONS.keys()
            ),
            columns,
        ]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
    )

    files_match = batch_selected.equals(
        queue_selected
    )

    include_count = int(
        queue_selected[
            "REVIEWER_1_DECISION"
        ].eq("Include").sum()
    )

    exclude_count = int(
        queue_selected[
            "REVIEWER_1_DECISION"
        ].eq("Exclude").sum()
    )

    unclear_count = int(
        queue_selected[
            "REVIEWER_1_DECISION"
        ].eq("Unclear").sum()
    )

    completed_count = int(
        queue_selected[
            "SCREENING_STATUS"
        ].eq(
            "Reviewer_1_completed"
        ).sum()
    )

    valid = (
        len(queue_selected) == 5
        and include_count == 3
        and exclude_count == 2
        and unclear_count == 0
        and completed_count == 5
        and files_match
    )

    print(
        "===== DECISIONES SCR-0016 "
        "A SCR-0020 ====="
    )

    print(
        queue_selected.to_string(
            index=False
        )
    )

    print()
    print("INCLUDE:", include_count)
    print("EXCLUDE:", exclude_count)
    print("UNCLEAR:", unclear_count)
    print("COMPLETADOS:", completed_count)
    print(
        "LOTE_Y_COLA_COINCIDEN:",
        files_match,
    )

    print(
        "VALIDACION_STATUS:",
        "OK" if valid else "REVIEW",
    )

    if not valid:
        raise ValueError(
            "Las decisiones registradas no coinciden "
            "con el resultado esperado."
        )


if __name__ == "__main__":
    main()