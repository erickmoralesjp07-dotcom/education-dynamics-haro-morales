from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BATCH_FILE = (
    ROOT
    / "literature_review"
    / "screening_batches"
    / "screening_batch_02.csv"
)

QUEUE_FILE = (
    ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

OUTPUT_DIR = (
    ROOT
    / "literature_review"
    / "screening_results"
)

DIAGNOSTIC_DIR = (
    ROOT
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

SCREENED_FILE = (
    OUTPUT_DIR
    / "screened_batch_02.csv"
)

CANDIDATES_FILE = (
    OUTPUT_DIR
    / "batch_02_full_text_candidates.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "batch_02_screening_summary.csv"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "batch_02_title_abstract_screening_diagnostic.csv"
)

SCREENING_TIME = (
    datetime.now()
    .astimezone()
    .isoformat()
)

DECISIONS = {
    "SCR-0026": {
        "decision": "Unclear",
        "reason": "",
        "notes": (
            "Highly relevant PISA inequality topic, but the "
            "2026 record reports PISA 2025 results and requires "
            "verification of publication status, data release, "
            "source integrity and methodological documentation."
        ),
    },
    "SCR-0027": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Cross-country and regional analysis of tertiary "
            "educational inequality and convergence using "
            "three-level hierarchical models across 25 European "
            "countries and 248 regions."
        ),
    },
    "SCR-0028": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Long-run international educational-attainment "
            "database covering 74 countries from 1870 to 2010, "
            "with measurement-error correction and convergence "
            "analysis."
        ),
    },
    "SCR-0029": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Methodologically relevant PISA study examining "
            "construct validity of opportunity-to-learn measures "
            "and socioeconomic inequality in mathematics."
        ),
    },
    "SCR-0030": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "PISA 2006 analysis of regional divergence in Spain "
            "using efficiency estimation and a two-stage model. "
            "Text required to evaluate plausible values, weights "
            "and two-stage inference."
        ),
    },
    "SCR-0031": {
        "decision": "Exclude",
        "reason": (
            "Single-country disability study outside the "
            "international comparative scope"
        ),
        "notes": (
            "Longitudinal Scottish study of disability and "
            "attainment. Relevant to inequality generally, but "
            "outside the cross-country, international-assessment "
            "and educational-dynamics scope of this review."
        ),
    },
    "SCR-0032": {
        "decision": "Exclude",
        "reason": (
            "Health outcome rather than educational outcome"
        ),
        "notes": (
            "Educational attainment is an explanatory variable, "
            "while the primary outcome is physical functioning "
            "among older adults. It does not analyze educational "
            "attainment, achievement or education-system dynamics "
            "as the outcome."
        ),
    },
    "SCR-0033": {
        "decision": "Unclear",
        "reason": "",
        "notes": (
            "Relevant panel analysis of education expenditure, "
            "human capital and growth in Central America. The "
            "2026 bibliographic status and complete methods must "
            "be verified before inclusion."
        ),
    },
    "SCR-0034": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Long-run fixed-effects panel covering 104 countries "
            "from 1870 to 2010, directly relevant to technology, "
            "human capital, gender differences and convergence."
        ),
    },
    "SCR-0035": {
        "decision": "Unclear",
        "reason": "",
        "notes": (
            "Relevant multi-country panel topic, but the abstract "
            "does not report sample size, period, estimator, "
            "diagnostics or validation. The 2026 record and source "
            "must be verified from the full text."
        ),
    },
    "SCR-0036": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Panel of 50 US states with state and time fixed "
            "effects. Relevant to education expenditure and "
            "inequality, although the outcome is income "
            "inequality rather than educational inequality."
        ),
    },
    "SCR-0037": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Strong comparative design using 153 surveys from "
            "40 Sub-Saharan African countries to analyze trends "
            "in intergenerational educational inequality."
        ),
    },
    "SCR-0038": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Directly relevant methodological study using 1,680 "
            "survey and census datasets and out-of-sample "
            "validation to improve educational-attainment "
            "estimation."
        ),
    },
    "SCR-0039": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Longitudinal analysis of educational continuation "
            "by migrant country of origin in Australia. Relevant "
            "to attainment heterogeneity and migration settings."
        ),
    },
    "SCR-0040": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Panel analysis of education expenditure and "
            "economic growth across 19 OECD countries. Full text "
            "needed to inspect stationarity, dependence, "
            "endogeneity and estimator selection."
        ),
    },
    "SCR-0041": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Cross-country PISA 2018 multilevel and meta-analytic "
            "study treating ICC as educational inequality across "
            "79 countries. Preprint status must be retained."
        ),
    },
    "SCR-0042": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Cross-country analysis linking pre-primary structural "
            "quality indicators to later PISA science performance "
            "between 2000 and 2015."
        ),
    },
    "SCR-0043": {
        "decision": "Exclude",
        "reason": (
            "Duplicate or later record of study already retained"
        ),
        "notes": (
            "The title and topic correspond to the natural-"
            "disaster study already retained and evaluated as "
            "SCR-0001. The 2026 SSRN record should be stored only "
            "as version history, not as a separate study."
        ),
    },
    "SCR-0044": {
        "decision": "Unclear",
        "reason": "",
        "notes": (
            "Highly relevant PISA 2022 comparative analysis using "
            "plausible values and sampling weights. The 2026 "
            "publication record, journal status and full methods "
            "must be verified."
        ),
    },
    "SCR-0045": {
        "decision": "Exclude",
        "reason": (
            "Healthcare and composite human-development outcome "
            "outside the educational-outcome scope"
        ),
        "notes": (
            "The primary focus is healthcare accessibility and "
            "the dependent variable is IHDI. Education expenditure "
            "is only one predictor and the study does not directly "
            "model educational attainment or achievement."
        ),
    },
    "SCR-0046": {
        "decision": "Exclude",
        "reason": (
            "Narrow autism-support simulation outside the "
            "comparative educational-attainment scope"
        ),
        "notes": (
            "System-dynamics simulation of family and teacher "
            "support for autism inclusion. The parameters come "
            "from prior literature and the study does not analyze "
            "international attainment, inequality or comparable "
            "education-system trajectories."
        ),
    },
    "SCR-0047": {
        "decision": "Exclude",
        "reason": (
            "Local subnational HDI study outside the comparative "
            "educational-outcome scope"
        ),
        "notes": (
            "Analysis is limited to five districts or "
            "municipalities in Yogyakarta and uses HDI rather "
            "than educational attainment or learning achievement "
            "as the dependent variable."
        ),
    },
    "SCR-0048": {
        "decision": "Exclude",
        "reason": (
            "Single-country disability-policy study outside the "
            "international comparative scope"
        ),
        "notes": (
            "Retrospective analysis of school settings and "
            "attainment among people with disabilities in France. "
            "Important policy evidence, but not aligned with the "
            "international comparative focus of this review."
        ),
    },
    "SCR-0049": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Cross-country analysis linking corruption measures "
            "to PISA scores, dropout, enrollment and schooling "
            "years. Full text needed to evaluate fixed effects, "
            "sample size and complex PISA design treatment."
        ),
    },
    "SCR-0050": {
        "decision": "Include",
        "reason": "",
        "notes": (
            "Comparative TIMSS study of eight MENA countries using "
            "plausible values, jackknife standard errors, mean and "
            "quantile analysis, and gender decomposition."
        ),
    },
}


def prepare_columns(data):
    text_columns = [
        "REVIEWER_1_DECISION",
        "REVIEWER_1_EXCLUSION_REASON",
        "REVIEWER_1_NOTES",
        "SCREENING_STATUS",
        "SCREENING_DATETIME",
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


def apply_decisions(data):
    data = prepare_columns(data)

    for screening_id, record in (
        DECISIONS.items()
    ):
        mask = data[
            "SCREENING_ID"
        ].eq(screening_id)

        count = int(mask.sum())

        if count != 1:
            raise ValueError(
                f"{screening_id} aparece "
                f"{count} veces."
            )

        data.loc[
            mask,
            "REVIEWER_1_DECISION",
        ] = record["decision"]

        data.loc[
            mask,
            "REVIEWER_1_EXCLUSION_REASON",
        ] = record["reason"]

        data.loc[
            mask,
            "REVIEWER_1_NOTES",
        ] = record["notes"]

        data.loc[
            mask,
            "SCREENING_STATUS",
        ] = "Title_abstract_screened"

        data.loc[
            mask,
            "SCREENING_DATETIME",
        ] = SCREENING_TIME

    return data


def main():
    if not BATCH_FILE.exists():
        raise FileNotFoundError(
            BATCH_FILE
        )

    batch = pd.read_csv(
        BATCH_FILE,
        encoding="utf-8-sig",
    )

    if len(batch) != 25:
        raise ValueError(
            "El lote 02 debe contener "
            f"25 registros y contiene {len(batch)}."
        )

    expected_ids = set(
        DECISIONS.keys()
    )

    batch_ids = set(
        batch[
            "SCREENING_ID"
        ].astype(str)
    )

    if batch_ids != expected_ids:
        missing = sorted(
            expected_ids - batch_ids
        )

        unexpected = sorted(
            batch_ids - expected_ids
        )

        raise ValueError(
            f"IDs faltantes: {missing}; "
            f"IDs inesperados: {unexpected}"
        )

    screened = apply_decisions(
        batch.copy()
    )

    screened.to_csv(
        SCREENED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    batch_updated = apply_decisions(
        batch.copy()
    )

    batch_updated.to_csv(
        BATCH_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    queue_updated = False

    if QUEUE_FILE.exists():
        queue = pd.read_csv(
            QUEUE_FILE,
            encoding="utf-8-sig",
        )

        if (
            "SCREENING_ID"
            in queue.columns
        ):
            queue = prepare_columns(
                queue
            )

            for screening_id, record in (
                DECISIONS.items()
            ):
                mask = queue[
                    "SCREENING_ID"
                ].eq(screening_id)

                if int(mask.sum()) != 1:
                    raise ValueError(
                        f"{screening_id} no aparece "
                        "exactamente una vez en "
                        "la cola general."
                    )

                queue.loc[
                    mask,
                    "REVIEWER_1_DECISION",
                ] = record["decision"]

                queue.loc[
                    mask,
                    "REVIEWER_1_EXCLUSION_REASON",
                ] = record["reason"]

                queue.loc[
                    mask,
                    "REVIEWER_1_NOTES",
                ] = record["notes"]

                queue.loc[
                    mask,
                    "SCREENING_STATUS",
                ] = "Title_abstract_screened"

                queue.loc[
                    mask,
                    "SCREENING_DATETIME",
                ] = SCREENING_TIME

            queue.to_csv(
                QUEUE_FILE,
                index=False,
                encoding="utf-8-sig",
            )

            queue_updated = True

    candidate_mask = screened[
        "REVIEWER_1_DECISION"
    ].isin(
        [
            "Include",
            "Unclear",
        ]
    )

    candidates = (
        screened.loc[
            candidate_mask
        ]
        .sort_values(
            [
                "REVIEWER_1_DECISION",
                "SCREENING_ID",
            ]
        )
        .reset_index(drop=True)
    )

    candidates.to_csv(
        CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    include_count = int(
        screened[
            "REVIEWER_1_DECISION"
        ].eq("Include").sum()
    )

    exclude_count = int(
        screened[
            "REVIEWER_1_DECISION"
        ].eq("Exclude").sum()
    )

    unclear_count = int(
        screened[
            "REVIEWER_1_DECISION"
        ].eq("Unclear").sum()
    )

    missing_decisions = int(
        screened[
            "REVIEWER_1_DECISION"
        ].astype("string")
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    duplicate_ids = int(
        screened[
            "SCREENING_ID"
        ].duplicated().sum()
    )

    summary = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    SCREENING_TIME
                ),
                "BATCH_ID": (
                    "screening_batch_02"
                ),
                "TOTAL_RECORDS": (
                    len(screened)
                ),
                "INCLUDE": (
                    include_count
                ),
                "EXCLUDE": (
                    exclude_count
                ),
                "UNCLEAR": (
                    unclear_count
                ),
                "FULL_TEXT_CANDIDATES": (
                    len(candidates)
                ),
                "MISSING_DECISIONS": (
                    missing_decisions
                ),
                "DUPLICATE_SCREENING_IDS": (
                    duplicate_ids
                ),
                "QUEUE_UPDATED": (
                    queue_updated
                ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(screened) == 25
        and include_count == 14
        and exclude_count == 7
        and unclear_count == 4
        and len(candidates) == 18
        and missing_decisions == 0
        and duplicate_ids == 0
        and queue_updated
    )

    diagnostic = summary.copy()

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
        "===== CRIBADO DEL LOTE 02 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== DECISIONES ====="
    )

    print(
        screened[
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
                "REVIEWER_1_EXCLUSION_REASON",
                "TITLE",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== CANDIDATOS A TEXTO "
        "COMPLETO ====="
    )

    print(
        candidates[
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
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
        "Archivos creados:"
    )

    print(SCREENED_FILE)
    print(CANDIDATES_FILE)
    print(SUMMARY_FILE)
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
            "El cribado del lote 02 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()