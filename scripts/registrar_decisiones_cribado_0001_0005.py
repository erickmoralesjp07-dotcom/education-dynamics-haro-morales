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
    "SCR-0001": {
        "decision": "Include",
        "reason": (
            "Relevant cross-country longitudinal "
            "study of educational attainment"
        ),
        "notes": (
            "Include provisionally as grey literature. "
            "Uses 85-country panel data and country/year "
            "fixed effects. Verify document type, "
            "peer-review status, and whether a published "
            "version exists."
        ),
    },
    "SCR-0002": {
        "decision": "Include",
        "reason": (
            "Directly relevant predictive cross-country "
            "educational indicator study"
        ),
        "notes": (
            "Central methodological study. Uses 165 "
            "countries, LASSO variable selection, linear "
            "regression, and leave-one-out validation. "
            "Verify publication year because metadata "
            "reports 2020 while DOI indicates 2022."
        ),
    },
    "SCR-0003": {
        "decision": "Exclude",
        "reason": (
            "Economic outcome without educational outcome"
        ),
        "notes": (
            "Education expenditure is a predictor, but "
            "the primary outcome is GDP per capita rather "
            "than an educational outcome. The 2026 "
            "publication status also requires verification."
        ),
    },
    "SCR-0004": {
        "decision": "Exclude",
        "reason": "Duplicate version",
        "notes": (
            "SSRN version of the same study represented "
            "by SCR-0001. Retain one canonical record only "
            "and later verify whether a peer-reviewed "
            "version exists."
        ),
    },
    "SCR-0005": {
        "decision": "Include",
        "reason": (
            "Central cross-country study of educational-"
            "attainment inequality and convergence"
        ),
        "notes": (
            "High-priority study for the theoretical "
            "framework and convergence analysis. Examines "
            "beta and sigma convergence, Gini decomposition, "
            "development gaps, and gender gaps from "
            "1960 to 1990."
        ),
    },
}


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

    # Las columnas vacias pueden ser inferidas como float64.
    # Las convertimos explicitamente a texto antes de asignar.
    for column in TEXT_COLUMNS:
        data[column] = (
            data[column]
            .fillna("")
            .astype("string")
        )

    current_datetime = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    for screening_id, decision in DECISIONS.items():
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        match_count = int(
            mask.sum()
        )

        if match_count != 1:
            raise ValueError(
                f"{screening_id} aparece "
                f"{match_count} veces en {path}."
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
    batch = update_file(
        BATCH_FILE
    )

    queue = update_file(
        QUEUE_FILE
    )

    selected = (
        queue.loc[
            queue["SCREENING_ID"].isin(
                DECISIONS.keys()
            ),
            [
                "SCREENING_ID",
                "TITLE",
                "REVIEWER_1_DECISION",
                "REVIEWER_1_REASON",
                "SCREENING_STATUS",
            ],
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
    )

    batch_selected = (
        batch.loc[
            batch["SCREENING_ID"].isin(
                DECISIONS.keys()
            ),
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
                "REVIEWER_1_REASON",
                "SCREENING_STATUS",
            ],
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
    )

    queue_check = selected[
        [
            "SCREENING_ID",
            "REVIEWER_1_DECISION",
            "REVIEWER_1_REASON",
            "SCREENING_STATUS",
        ]
    ].reset_index(drop=True)

    files_match = (
        batch_selected.equals(
            queue_check
        )
    )

    print(
        "===== DECISIONES SCR-0001 "
        "A SCR-0005 ====="
    )

    print(
        selected.to_string(
            index=False
        )
    )

    include_count = int(
        selected[
            "REVIEWER_1_DECISION"
        ].eq("Include").sum()
    )

    exclude_count = int(
        selected[
            "REVIEWER_1_DECISION"
        ].eq("Exclude").sum()
    )

    completed_count = int(
        selected[
            "SCREENING_STATUS"
        ].eq(
            "Reviewer_1_completed"
        ).sum()
    )

    valid = (
        len(selected) == 5
        and include_count == 3
        and exclude_count == 2
        and completed_count == 5
        and files_match
    )

    print()
    print(
        "INCLUDE:",
        include_count,
    )

    print(
        "EXCLUDE:",
        exclude_count,
    )

    print(
        "COMPLETADOS:",
        completed_count,
    )

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
            "Las decisiones no coinciden "
            "con el resultado esperado."
        )


if __name__ == "__main__":
    main()