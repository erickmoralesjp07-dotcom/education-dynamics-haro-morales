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

SUMMARY_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_summary.csv"
)

PENDING_FILE = (
    ASSESSMENT_DIR
    / "batch_01_pending_full_text_candidates.csv"
)

SIX_ASSESSMENTS_FILE = (
    ASSESSMENT_DIR
    / "batch_01_six_additional_full_text_assessments.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "batch_01_six_additional_full_text_assessments_diagnostic.csv"
)

ASSESSMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

STANDARD_COLUMNS = [
    "SCREENING_ID",
    "TITLE",
    "AUTHORS",
    "YEAR",
    "JOURNAL_OR_SOURCE",
    "DOI_NORMALIZED",
    "THEMATIC_BLOCK",
    "VERIFICATION_FLAGS",
    "FULL_TEXT_DECISION",
    "FULL_TEXT_REASON",
    "RISK_OF_BIAS",
    "SAMPLE_DESIGN_ADEQUATE",
    "MISSING_DATA_TREATMENT_ADEQUATE",
    "VALIDATION_ADEQUATE",
    "CAUSAL_CLAIM_SUPPORTED",
    "CODE_AVAILABLE",
    "ASSESSMENT_NOTES",
    "ASSESSMENT_DATETIME",
]

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
    "SCR-0001": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Endogeneity concerns and unverified "
            "exogeneity of disaster exposure"
        ),
        "risk": "High risk of bias",
        "sample": True,
        "missing": False,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The study uses a panel of 85 countries from "
            "1960 to 1990, Barro-Lee educational-attainment "
            "data, country and year fixed effects, and robust "
            "standard errors. It nevertheless assumes that "
            "natural disasters are exogenous and acknowledges "
            "that economic-damage measures create endogeneity. "
            "The macro-country design cannot fully isolate "
            "causal effects and may be affected by measurement "
            "error, omitted variables and heterogeneous data "
            "quality. It is useful for discussing shocks and "
            "long-run educational dynamics, but causal claims "
            "require caution."
        ),
    },
    "SCR-0006": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Descriptive cross-survey comparison with "
            "selection and comparability limitations"
        ),
        "risk": "High risk of bias",
        "sample": False,
        "missing": False,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The study compares immigrant and native student "
            "achievement using PISA, TIMSS and PIRLS. Its "
            "cross-survey comparison is informative and tests "
            "whether patterns are robust across assessments. "
            "However, survey target populations, subjects, "
            "ages and sampling designs differ; response-rate "
            "problems are noted for some countries; immigrant "
            "selection is not causally resolved; and the main "
            "regression analyses are descriptive OLS models. "
            "The study is suitable as contextual evidence but "
            "not as a causal estimate of immigrant status."
        ),
    },
    "SCR-0012": {
        "decision": "Include_final_candidate",
        "reason": (
            "Adequate multilevel PISA analysis using "
            "plausible values and complex sample design"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": False,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The study evaluates mathematics, reading and "
            "science achievement among Kazakhstani students "
            "using PISA 2018. It explicitly reports multilevel "
            "regression analysis, plausible values and complex "
            "sample-design procedures, and separates individual, "
            "school and regional factors. It remains an "
            "observational single-country analysis and does not "
            "identify causal effects. Missing-data treatment and "
            "replication materials are not sufficiently clear, "
            "but the methodological alignment with PISA makes "
            "it a final candidate."
        ),
    },
    "SCR-0017": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong methodological analysis of measurement "
            "error and heterogeneity in education-growth models"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The study compares Barro-Lee, Kyriacou and World "
            "Values Survey education measures and demonstrates "
            "severe attenuation from measurement error in "
            "first-differenced country data. It evaluates "
            "alternative time windows, instrumental variables, "
            "capital controls, non-linearity and heterogeneous "
            "country coefficients. It explicitly warns that "
            "reverse causality and omitted policies prevent "
            "simple causal interpretation. The careful treatment "
            "of measurement error and specification sensitivity "
            "makes it a strong methodological reference."
        ),
    },
    "SCR-0021": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong analysis of PISA eligibility, coverage "
            "and changing sample composition"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The OECD study examines Turkey across PISA 2003 "
            "to 2012 and shows that eligibility rose from "
            "approximately 45 percent to 80 percent. It combines "
            "PISA and Demographic and Health Survey data, uses "
            "survey weighting, five plausible values, composite "
            "standard errors and multiple robustness checks. "
            "The study directly demonstrates how changing sample "
            "coverage can bias trend interpretation. It is "
            "limited to one national case and does not estimate "
            "a causal intervention effect, but it is essential "
            "for interpreting international assessment trends."
        ),
    },
    "SCR-0022": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Inadequate plausible-value treatment and "
            "potential earnings-sample selection"
        ),
        "risk": "High risk of bias",
        "sample": True,
        "missing": False,
        "validation": True,
        "causal": False,
        "code": False,
        "notes": (
            "The study uses International Adult Literacy Survey "
            "data, population weights, interval-regression "
            "methods for banded earnings and robust standard "
            "errors. It finds that literacy is associated with "
            "earnings across countries. However, it averages "
            "plausible values to construct literacy scores rather "
            "than estimating and pooling across plausible values. "
            "The working sample excludes individuals without "
            "usable earnings observations, creating selection "
            "risk, and causal identification is absent. It is "
            "useful with major limitations."
        ),
    },
}


def prepare_text_columns(data):
    for column in TEXT_COLUMNS:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    return data


def update_registry(data):
    data = prepare_text_columns(data)

    now = (
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
        ] = now

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = "Full_text_assessed"

    return data


def main():
    required_files = [
        SCREENED_FILE,
        QUEUE_FILE,
        CONSOLIDATED_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    screened = pd.read_csv(
        SCREENED_FILE
    )

    queue = pd.read_csv(
        QUEUE_FILE
    )

    consolidated = pd.read_csv(
        CONSOLIDATED_FILE
    )

    screened = update_registry(
        screened
    )

    queue = update_registry(
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

    rows = []

    for screening_id, assessment in (
        ASSESSMENTS.items()
    ):
        queue_row = queue.loc[
            queue[
                "SCREENING_ID"
            ].eq(screening_id)
        ].iloc[0]

        rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": queue_row.get(
                    "TITLE",
                    "",
                ),
                "AUTHORS": queue_row.get(
                    "AUTHORS",
                    "",
                ),
                "YEAR": queue_row.get(
                    "YEAR",
                    "",
                ),
                "JOURNAL_OR_SOURCE": (
                    queue_row.get(
                        "JOURNAL_OR_SOURCE",
                        "",
                    )
                ),
                "DOI_NORMALIZED": (
                    queue_row.get(
                        "DOI_NORMALIZED",
                        "",
                    )
                ),
                "THEMATIC_BLOCK": (
                    queue_row.get(
                        "THEMATIC_BLOCK",
                        "",
                    )
                ),
                "VERIFICATION_FLAGS": (
                    queue_row.get(
                        "VERIFICATION_FLAGS",
                        "",
                    )
                ),
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
                    assessment["sample"]
                ),
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    assessment["missing"]
                ),
                "VALIDATION_ADEQUATE": (
                    assessment["validation"]
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    assessment["causal"]
                ),
                "CODE_AVAILABLE": (
                    assessment["code"]
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

    six_assessments = pd.DataFrame(
        rows
    )

    six_assessments.to_csv(
        SIX_ASSESSMENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    missing_columns = [
        column
        for column in STANDARD_COLUMNS
        if column not in consolidated.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el consolidado: "
            f"{missing_columns}"
        )

    ids_to_replace = set(
        ASSESSMENTS.keys()
    )

    consolidated = consolidated.loc[
        ~consolidated[
            "SCREENING_ID"
        ].isin(ids_to_replace)
    ].copy()

    consolidated = pd.concat(
        [
            consolidated[
                STANDARD_COLUMNS
            ],
            six_assessments[
                STANDARD_COLUMNS
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    consolidated = (
        consolidated
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
    )

    duplicate_count = int(
        consolidated[
            "SCREENING_ID"
        ]
        .duplicated()
        .sum()
    )

    consolidated.to_csv(
        CONSOLIDATED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        consolidated
        .groupby(
            [
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ],
            as_index=False,
        )
        .agg(
            RECORD_COUNT=(
                "SCREENING_ID",
                "size",
            )
        )
        .sort_values(
            [
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ]
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_ids = set(
        screened.loc[
            screened[
                "REVIEWER_1_DECISION"
            ].isin(
                [
                    "Include",
                    "Unclear",
                ]
            ),
            "SCREENING_ID",
        ]
    )

    assessed_ids = set(
        consolidated[
            "SCREENING_ID"
        ]
    )

    pending_ids = (
        candidate_ids
        - assessed_ids
    )

    pending = (
        screened.loc[
            screened[
                "SCREENING_ID"
            ].isin(pending_ids)
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
    )

    pending.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_final_candidate"
        ).sum()
    )

    limited_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_with_major_limitations"
        ).sum()
    )

    excluded_count = int(
        consolidated[
            "FULL_TEXT_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    files_match = (
        queue.loc[
            queue[
                "SCREENING_ID"
            ].isin(assessed_ids),
            [
                "SCREENING_ID",
                "CONSENSUS_DECISION",
                "CONSENSUS_REASON",
                "SCREENING_STATUS",
            ],
        ]
        .sort_values(
            "SCREENING_ID"
        )
        .reset_index(drop=True)
        .fillna("")
        .equals(
            screened.loc[
                screened[
                    "SCREENING_ID"
                ].isin(assessed_ids),
                [
                    "SCREENING_ID",
                    "CONSENSUS_DECISION",
                    "CONSENSUS_REASON",
                    "SCREENING_STATUS",
                ],
            ]
            .sort_values(
                "SCREENING_ID"
            )
            .reset_index(drop=True)
            .fillna("")
        )
    )

    expected_pending_ids = {
        "SCR-0005",
        "SCR-0007",
        "SCR-0013",
        "SCR-0024",
    }

    valid = (
        len(consolidated) == 14
        and duplicate_count == 0
        and final_count == 5
        and limited_count == 6
        and excluded_count == 3
        and len(pending) == 4
        and pending_ids
        == expected_pending_ids
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
                "FULL_TEXT_CANDIDATES": (
                    len(candidate_ids)
                ),
                "ASSESSED_FULL_TEXTS": (
                    len(consolidated)
                ),
                "FINAL_CANDIDATES": (
                    final_count
                ),
                "INCLUDE_WITH_MAJOR_LIMITATIONS": (
                    limited_count
                ),
                "FULL_TEXT_EXCLUSIONS": (
                    excluded_count
                ),
                "PENDING_FULL_TEXTS": (
                    len(pending)
                ),
                "DUPLICATE_ASSESSMENTS": (
                    duplicate_count
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

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== REGISTRO DE SEIS "
        "EVALUACIONES ====="
    )

    print(
        six_assessments[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_REASON",
                "RISK_OF_BIAS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== CONSOLIDADO ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PENDIENTES ====="
    )

    print(
        pending[
            [
                "SCREENING_ID",
                "TITLE",
                "DOI_NORMALIZED",
                "VERIFICATION_FLAGS",
            ]
        ].to_string(
            index=False
        )
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
            "El registro conjunto de seis "
            "evaluaciones requiere revision."
        )


if __name__ == "__main__":
    main()