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
    "SCR-0011": {
        "decision": "Exclude",
        "reason": (
            "Economic outcome without educational outcome"
        ),
        "notes": (
            "The model uses schooling and education "
            "expenditure as predictors, but the primary "
            "outcome is regional productivity. It does "
            "not directly model educational access, "
            "attainment, learning, literacy, or inequality."
        ),
    },
    "SCR-0012": {
        "decision": "Include",
        "reason": (
            "Relevant PISA methodology using plausible "
            "values, complex sampling, and multilevel models"
        ),
        "notes": (
            "Relevant methodological evidence for the "
            "future PISA component. Examines mathematics, "
            "reading, and science performance using PISA "
            "2018, plausible values, complex sampling, "
            "multilevel regression, and individual, school, "
            "and regional factors. Although it studies one "
            "country, it supports correct PISA methodology."
        ),
    },
    "SCR-0013": {
        "decision": "Include",
        "reason": (
            "Relevant cross-country PISA study of "
            "educational opportunity inequality"
        ),
        "notes": (
            "Uses PISA 2009 data for 72 countries and "
            "examines inequality of educational opportunity "
            "in mathematics, science, and reading. Relevant "
            "to cross-country inequality and achievement. "
            "Include provisionally and search for a "
            "peer-reviewed version replacing the SSRN record."
        ),
    },
    "SCR-0014": {
        "decision": "Exclude",
        "reason": (
            "No international or cross-country component"
        ),
        "notes": (
            "The study uses longitudinal Swedish "
            "administrative data and flexible machine "
            "learning within a causal framework, but focuses "
            "on individual neighbourhood trajectories and "
            "high-school graduation in one country. It is "
            "outside the central multinational macro-level "
            "scope of the review."
        ),
    },
    "SCR-0015": {
        "decision": "Unclear",
        "reason": (
            "Potentially relevant predictive study requiring "
            "full-text methodological verification"
        ),
        "notes": (
            "The comparison of XGBoost and Random Forest "
            "using international education indicators is "
            "directly relevant, but the abstract does not "
            "report the country count, period, grouping "
            "strategy, or data source details. Verify journal "
            "quality, possible target leakage from engineered "
            "ratios, independence of the 80-20 split, "
            "cross-validation design, and reproducibility "
            "before inclusion."
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

        match_count = int(
            mask.sum()
        )

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
    batch = update_file(
        BATCH_FILE
    )

    queue = update_file(
        QUEUE_FILE
    )

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

    files_match = (
        batch_selected.equals(
            queue_selected
        )
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
        and exclude_count == 2
        and unclear_count == 1
        and completed_count == 5
        and files_match
    )

    print(
        "===== DECISIONES SCR-0011 "
        "A SCR-0015 ====="
    )

    print(
        queue_selected.to_string(
            index=False
        )
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
        "UNCLEAR:",
        unclear_count,
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
            "Las decisiones registradas no coinciden "
            "con el resultado esperado."
        )


if __name__ == "__main__":
    main()