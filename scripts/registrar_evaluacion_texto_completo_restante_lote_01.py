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

TEXT_COLUMNS = [
    "FULL_TEXT_REQUIRED",
    "FULL_TEXT_REQUESTED",
    "FULL_TEXT_OBTAINED",
    "FULL_TEXT_LOCAL_FILE",
    "CONSENSUS_DECISION",
    "CONSENSUS_REASON",
    "CONSENSUS_DATE",
    "SCREENING_STATUS",
]

ASSESSMENTS = {
    "SCR-0008": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Severe complete-case selection and "
            "aggregated dynamic design"
        ),
        "risk": "High risk of bias",
        "notes": (
            "The study is relevant because it uses UNESCO "
            "UIS indicators, hierarchical clustering, Gini "
            "and Theil indices, and an AR(1) model with a "
            "COVID-19 structural-break dummy. However, only "
            "16 of 210 countries or territories were retained "
            "because of data completeness, creating severe "
            "selection risk. Global trends are based on this "
            "restricted sample, the AR(1) is estimated on "
            "aggregated averages rather than a country panel, "
            "and the steady state is an algebraic property "
            "rather than evidence of structural educational "
            "equilibrium. The study may support methods and "
            "limitations but not global inference."
        ),
        "sample_design_adequate": False,
        "missing_data_adequate": False,
        "validation_adequate": False,
        "causal_claim_supported": False,
        "code_available": False,
    },
    "SCR-0010": {
        "decision": "Include_final_candidate",
        "reason": (
            "Methodologically adequate longitudinal "
            "multi-country study"
        ),
        "risk": "Moderate risk of bias",
        "notes": (
            "The study links PISA 2000 or 2003 participants "
            "to longitudinal outcomes at ages 25 to 27 in "
            "Australia, Denmark, and Switzerland. It applies "
            "PISA sampling and replication weights, combines "
            "them with longitudinal follow-up weights, "
            "clusters standard errors by school, models item "
            "and booklet effects, and reports robustness "
            "checks. Important limitations include attrition "
            "of 36 percent response in Australia and 44 "
            "percent in Denmark, limited country diversity, "
            "different PISA major domains, and an "
            "associational rather than causal design."
        ),
        "sample_design_adequate": True,
        "missing_data_adequate": True,
        "validation_adequate": True,
        "causal_claim_supported": False,
        "code_available": False,
    },
    "SCR-0018": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Incomplete PISA design treatment and "
            "uncertain instrument validity"
        ),
        "risk": "High risk of bias",
        "notes": (
            "The study uses four PISA waves for 22 countries, "
            "country and year fixed effects, multilevel "
            "models, and instrumental-variable estimators. "
            "It recognizes the five plausible values and "
            "tests endogeneity, weak identification, "
            "underidentification, and overidentifying "
            "restrictions. However, the reviewed text does "
            "not establish correct use of final sampling "
            "weights, replicate weights, or repeated "
            "estimation and pooling across plausible values. "
            "The proposed instruments may directly affect "
            "educational outcomes through urbanization, "
            "resources, or population structure. Results "
            "should not be treated as definitive causal "
            "effects."
        ),
        "sample_design_adequate": False,
        "missing_data_adequate": False,
        "validation_adequate": True,
        "causal_claim_supported": False,
        "code_available": False,
    },
    "SCR-0020": {
        "decision": "Exclude_full_text",
        "reason": (
            "Unsupported causal claims and "
            "inadequate time-series design"
        ),
        "risk": "Critical risk of bias",
        "notes": (
            "The analysis uses only 26 annual observations "
            "and 24 observations in Granger tests. It does "
            "not document unit-root, stationarity, "
            "cointegration, or error-correction analysis. "
            "The projected education-expenditure variable "
            "is deterministically constructed as 26 percent "
            "of total expenditure and is not an observed "
            "policy treatment. Nominal trends, limited "
            "controls, and unsupported causal "
            "interpretations create major risk of spurious "
            "results. The claimed UNESCO 26 percent "
            "benchmark is supported only through secondary "
            "references and must not be treated as an "
            "official UNESCO recommendation without a "
            "primary source."
        ),
        "sample_design_adequate": False,
        "missing_data_adequate": False,
        "validation_adequate": False,
        "causal_claim_supported": False,
        "code_available": False,
    },
}


def prepare_columns(data):
    for column in TEXT_COLUMNS:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    return data


def screening_status(decision):
    if decision == "Exclude_full_text":
        return "Full_text_excluded"

    return "Full_text_assessed"


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

    for screening_id, assessment in ASSESSMENTS.items():
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
        ] = screening_status(
            assessment["decision"]
        )

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

    assessment_rows = []

    for screening_id, assessment in ASSESSMENTS.items():
        source_row = queue.loc[
            queue["SCREENING_ID"].eq(
                screening_id
            )
        ].iloc[0]

        assessment_rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": source_row["TITLE"],
                "FULL_TEXT_DECISION": (
                    assessment["decision"]
                ),
                "FULL_TEXT_REASON": (
                    assessment["reason"]
                ),
                "RISK_OF_BIAS": (
                    assessment["risk"]
                ),
                "SAMPLE_DESIGN_ADEQUATE": (
                    assessment[
                        "sample_design_adequate"
                    ]
                ),
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    assessment[
                        "missing_data_adequate"
                    ]
                ),
                "VALIDATION_ADEQUATE": (
                    assessment[
                        "validation_adequate"
                    ]
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    assessment[
                        "causal_claim_supported"
                    ]
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
        assessment_rows
    )

    assessment_file = (
        OUTPUT_DIR
        / "batch_01_remaining_full_text_assessment.csv"
    )

    assessment_table.to_csv(
        assessment_file,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_columns = [
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
            comparison_columns,
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(
            drop=True
        )
    )

    queue_selected = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(
                ASSESSMENTS.keys()
            ),
            comparison_columns,
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(
            drop=True
        )
    )

    files_match = (
        screened_selected.equals(
            queue_selected
        )
    )

    final_candidate_count = int(
        assessment_table[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_final_candidate"
        ).sum()
    )

    limited_include_count = int(
        assessment_table[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_with_major_limitations"
        ).sum()
    )

    exclusion_count = int(
        assessment_table[
            "FULL_TEXT_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    valid = (
        len(assessment_table) == 4
        and final_candidate_count == 1
        and limited_include_count == 2
        and exclusion_count == 1
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
                "FINAL_CANDIDATES": (
                    final_candidate_count
                ),
                "INCLUDE_WITH_MAJOR_LIMITATIONS": (
                    limited_include_count
                ),
                "FULL_TEXT_EXCLUSIONS": (
                    exclusion_count
                ),
                "FILES_MATCH": (
                    files_match
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
        / "batch_01_remaining_full_text_assessment_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== EVALUACION DE LOS CUATRO "
        "TEXTOS COMPLETOS ====="
    )

    print(
        assessment_table[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_REASON",
                "RISK_OF_BIAS",
                "SAMPLE_DESIGN_ADEQUATE",
                "MISSING_DATA_TREATMENT_ADEQUATE",
                "VALIDATION_ADEQUATE",
                "CAUSAL_CLAIM_SUPPORTED",
                "CODE_AVAILABLE",
            ]
        ].to_string(
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
    print(
        "Archivo creado:"
    )

    print(
        assessment_file
    )

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