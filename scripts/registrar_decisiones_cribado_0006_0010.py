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
    "SCR-0006": {
        "decision": "Include",
        "reason": (
            "Relevant cross-country comparison using "
            "international educational assessments"
        ),
        "notes": (
            "Compares immigrant educational achievement "
            "across ten countries using PISA, TIMSS, and "
            "PIRLS. Relevant for cross-country "
            "heterogeneity and international assessment "
            "comparability. Verify whether a peer-reviewed "
            "version replaced the SSRN record."
        ),
    },
    "SCR-0007": {
        "decision": "Include",
        "reason": (
            "Relevant cross-country panel study of "
            "intergenerational educational persistence"
        ),
        "notes": (
            "Uses data for 93 countries, panel regression, "
            "country clustering, educational inequality, "
            "privilege, governance, and gender-specific "
            "educational mobility. Relevant to inequality "
            "and regime heterogeneity."
        ),
    },
    "SCR-0008": {
        "decision": "Include",
        "reason": (
            "Relevant longitudinal UNESCO-based dynamic "
            "and inequality analysis"
        ),
        "notes": (
            "Uses UNESCO indicators from 1998 to 2024, "
            "hierarchical clustering, Gini and Theil "
            "indices, autoregressive dynamics, a COVID-19 "
            "structural break, and long-run equilibria. "
            "Obtain full text and assess journal quality."
        ),
    },
    "SCR-0009": {
        "decision": "Include",
        "reason": (
            "Relevant multi-cycle PISA study of "
            "school-level inequality and achievement"
        ),
        "notes": (
            "Uses PISA 2012, 2015, and 2018 across "
            "countries and evaluates wealth inequality, "
            "achievement, interactions, and robustness. "
            "Include provisionally and search for a "
            "peer-reviewed version replacing the "
            "Research Square preprint."
        ),
    },
    "SCR-0010": {
        "decision": "Include",
        "reason": (
            "Relevant cross-country longitudinal link "
            "between PISA performance and later attainment"
        ),
        "notes": (
            "Links PISA performance at age 15 with "
            "educational attainment at ages 25 to 27 in "
            "Australia, Denmark, and Switzerland. Relevant "
            "for connecting learning measures to later "
            "adult educational outcomes."
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

    completed_count = int(
        queue_selected[
            "SCREENING_STATUS"
        ].eq("Reviewer_1_completed").sum()
    )

    valid = (
        len(queue_selected) == 5
        and include_count == 5
        and exclude_count == 0
        and completed_count == 5
        and files_match
    )

    print(
        "===== DECISIONES SCR-0006 "
        "A SCR-0010 ====="
    )

    print(
        queue_selected.to_string(
            index=False
        )
    )

    print()
    print("INCLUDE:", include_count)
    print("EXCLUDE:", exclude_count)
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