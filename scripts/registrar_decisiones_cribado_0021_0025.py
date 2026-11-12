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
    "SCR-0021": {
        "decision": "Include",
        "reason": (
            "Central methodological study of PISA "
            "sample eligibility and comparability"
        ),
        "notes": (
            "Highly relevant to international and "
            "intertemporal PISA comparisons. Demonstrates "
            "how school participation, grade delay, and "
            "sample eligibility affect measured progress "
            "and inequality. Combines PISA with demographic "
            "survey data and should support the PISA "
            "methodology and limitations sections."
        ),
    },
    "SCR-0022": {
        "decision": "Include",
        "reason": (
            "Relevant multi-country study distinguishing "
            "literacy and educational attainment"
        ),
        "notes": (
            "Uses the International Adult Literacy Survey "
            "to distinguish functional literacy from "
            "formal schooling across countries. Relevant "
            "to the conceptual distinction between youth "
            "literacy, adult attainment, and their measured "
            "associations. Verify whether a journal version "
            "replaced the repository report."
        ),
    },
    "SCR-0023": {
        "decision": "Exclude",
        "reason": (
            "Economic outcome without educational outcome"
        ),
        "notes": (
            "The paper includes education expenditure, "
            "enrolment, and attainment, but the main "
            "dependent outcome and interpretation concern "
            "GDP per capita and economic development. It "
            "does not primarily model educational dynamics."
        ),
    },
    "SCR-0024": {
        "decision": "Unclear",
        "reason": (
            "Relevant cross-country panel preprint "
            "requiring publication and methodological "
            "verification"
        ),
        "notes": (
            "Uses 131 countries and an educational outcome, "
            "but it is an OSF preprint dated 2026 with only "
            "four years of observations. Verify publication "
            "status, country and time effects, temporal "
            "dependence, endogeneity, robustness, and "
            "whether a peer-reviewed version exists."
        ),
    },
    "SCR-0025": {
        "decision": "Unclear",
        "reason": (
            "Potentially relevant machine-learning study "
            "requiring leakage and validation assessment"
        ),
        "notes": (
            "The predictive topic is relevant, but the "
            "reported R2 of 0.9881 requires careful review. "
            "Verify the geographic unit, period, target "
            "definition, predictor overlap, possible target "
            "leakage, train-test independence, grouped or "
            "temporal validation, journal quality, and "
            "reproducibility before inclusion."
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
        and include_count == 2
        and exclude_count == 1
        and unclear_count == 2
        and completed_count == 5
        and files_match
    )

    print(
        "===== DECISIONES SCR-0021 "
        "A SCR-0025 ====="
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