from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENED_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_01.csv"
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
    / "full_text_assessment"
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

ASSESSMENTS = {
    "SCR-0015": {
        "decision": "Exclude_full_text",
        "reason": (
            "Target leakage and inadequate "
            "validation design"
        ),
        "notes": (
            "Enrollment Rate is the target, but the "
            "predictors include Literacy-to-Enrollment "
            "Ratio and Enrollment-literacy gap, both of "
            "which mathematically contain the target. "
            "The study also uses a random 80-20 split "
            "without grouping by country or year, reports "
            "inconsistent R2 values between table and "
            "figure, provides no simple baseline, and "
            "does not provide code or sufficient dataset "
            "coverage information."
        ),
        "quality": "Critical risk of bias",
        "leakage": True,
        "external_validation": False,
        "baseline": False,
        "code_available": False,
    },
    "SCR-0025": {
        "decision": "Exclude_full_text",
        "reason": (
            "Outcome-predictor circularity and "
            "inadequate external validation"
        ),
        "notes": (
            "The regional education score is a composite "
            "of attainment and post-secondary measures, "
            "while dominant predictors include Level 3 "
            "achievement at age 18, attainment variables, "
            "and higher-education participation. This "
            "creates potential construct overlap and "
            "target leakage. The random 80-20 split does "
            "not separate regions or years, no simple "
            "baseline is reported, and the claimed R2 of "
            "0.9881 is not supported by external, spatial, "
            "or temporal validation."
        ),
        "quality": "Critical risk of bias",
        "leakage": True,
        "external_validation": False,
        "baseline": False,
        "code_available": False,
    },
}


def prepare_columns(data):
    text_columns = [
        "FULL_TEXT_REQUIRED",
        "FULL_TEXT_REQUESTED",
        "FULL_TEXT_OBTAINED",
        "FULL_TEXT_LOCAL_FILE",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "CONSENSUS_DATE",
        "SCREENING_STATUS",
    ]

    for column in text_columns:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    return data


def update_screening_file(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    data = prepare_columns(data)

    assessment_datetime = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    for screening_id, assessment in (
        ASSESSMENTS.items()
    ):
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        if int(mask.sum()) != 1:
            raise ValueError(
                f"{screening_id} aparece "
                f"{int(mask.sum())} veces en {path}."
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
        ] = "Yes"

        data.loc[
            mask,
            "FULL_TEXT_LOCAL_FILE",
        ] = (
            "literature_review/full_text/"
            "screening_batch_01/"
            f"{screening_id}_full_text.pdf"
        )

        data.loc[
            mask,
            "CONSENSUS_DECISION",
        ] = assessment["decision"]

        data.loc[
            mask,
            "CONSENSUS_REASON",
        ] = assessment["reason"]

        data.loc[
            mask,
            "CONSENSUS_DATE",
        ] = assessment_datetime

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = "Full_text_excluded"

    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return data


def main():
    screened = update_screening_file(
        SCREENED_FILE
    )

    queue = update_screening_file(
        QUEUE_FILE
    )

    rows = []

    for screening_id, assessment in (
        ASSESSMENTS.items()
    ):
        source_row = queue.loc[
            queue["SCREENING_ID"].eq(
                screening_id
            )
        ].iloc[0]

        rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": source_row["TITLE"],
                "FULL_TEXT_DECISION": (
                    assessment["decision"]
                ),
                "FULL_TEXT_EXCLUSION_REASON": (
                    assessment["reason"]
                ),
                "RISK_OF_BIAS": (
                    assessment["quality"]
                ),
                "TARGET_LEAKAGE_RISK": (
                    assessment["leakage"]
                ),
                "EXTERNAL_VALIDATION": (
                    assessment[
                        "external_validation"
                    ]
                ),
                "BASELINE_REPORTED": (
                    assessment["baseline"]
                ),
                "CODE_AVAILABLE": (
                    assessment[
                        "code_available"
                    ]
                ),
                "ASSESSMENT_NOTES": (
                    assessment["notes"]
                ),
                "ASSESSMENT_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            }
        )

    assessment_table = pd.DataFrame(
        rows
    )

    output_file = (
        OUTPUT_DIR
        / "batch_01_ml_full_text_assessment.csv"
    )

    assessment_table.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    screened_selected = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].isin(
                ASSESSMENTS.keys()
            ),
            columns,
        ]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
    )

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(
                ASSESSMENTS.keys()
            ),
            columns,
        ]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
    )

    files_match = (
        screened_selected.equals(
            queue_selected
        )
    )

    excluded_count = int(
        queue_selected[
            "CONSENSUS_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    valid = (
        len(assessment_table) == 2
        and excluded_count == 2
        and assessment_table[
            "TARGET_LEAKAGE_RISK"
        ].all()
        and files_match
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "ASSESSED_RECORDS": (
                    len(assessment_table)
                ),
                "FULL_TEXT_EXCLUSIONS": (
                    excluded_count
                ),
                "CRITICAL_RISK_RECORDS": int(
                    assessment_table[
                        "RISK_OF_BIAS"
                    ].eq(
                        "Critical risk of bias"
                    ).sum()
                ),
                "TARGET_LEAKAGE_FLAGS": int(
                    assessment_table[
                        "TARGET_LEAKAGE_RISK"
                    ].sum()
                ),
                "FILES_MATCH": files_match,
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
        / "batch_01_ml_full_text_assessment_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== EVALUACION DE TEXTO "
        "COMPLETO ML ====="
    )

    print(
        assessment_table[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_EXCLUSION_REASON",
                "RISK_OF_BIAS",
                "TARGET_LEAKAGE_RISK",
                "EXTERNAL_VALIDATION",
                "BASELINE_REPORTED",
                "CODE_AVAILABLE",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("===== DIAGNOSTICO =====")

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print("Archivo creado:")
    print(output_file)

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
            "La evaluacion no coincide "
            "con el resultado esperado."
        )


if __name__ == "__main__":
    main()