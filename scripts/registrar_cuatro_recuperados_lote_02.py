from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_DIR = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
)

CONSOLIDATED_FILE = (
    ASSESSMENT_DIR
    / "batch_02_full_text_assessment_consolidated.csv"
)

SUMMARY_FILE = (
    ASSESSMENT_DIR
    / "batch_02_full_text_assessment_summary.csv"
)

PENDING_FILE = (
    ASSESSMENT_DIR
    / "batch_02_pending_full_text_candidates.csv"
)

CANDIDATES_FILE = (
    ROOT
    / "literature_review"
    / "screening_results"
    / "batch_02_full_text_candidates.csv"
)

SCREENED_FILE = (
    ROOT
    / "literature_review"
    / "screening_results"
    / "screened_batch_02.csv"
)

QUEUE_FILE = (
    ROOT
    / "literature_review"
    / "search_results"
    / "title_abstract_screening_queue.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "batch_02_thirteen_full_text_assessments_diagnostic.csv"
)

ASSESSMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()

DECISIONS = {
    "SCR-0027": {
        "decision": "Include_final_candidate",
        "reason": (
            "Large harmonized European Social Survey analysis "
            "with multilevel regional and country comparisons"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": False,
        "validation": True,
        "causal": False,
        "notes": (
            "The study combines European Social Survey waves 1 to 10 "
            "and analyzes 285,506 individuals in 298 regions from "
            "25 European countries. It uses multilevel linear "
            "probability models, country-specific regressions, "
            "intraclass correlations and residual diagnostics. "
            "The large harmonized sample and explicit regional "
            "structure support inclusion. Limitations include "
            "observational repeated cross-sectional data, possible "
            "differences in regional coding across survey rounds, "
            "and limited support for causal interpretation."
        ),
    },
    "SCR-0029": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong methodological assessment of construct validity "
            "in PISA opportunity-to-learn measurement"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": False,
        "validation": True,
        "causal": False,
        "notes": (
            "The study reexamines PISA 2012 data and evaluates the "
            "construct validity of opportunity-to-learn indicators. "
            "It identifies construct-irrelevant variance and separates "
            "two latent dimensions that challenge earlier mediation "
            "claims. The cross-country probability samples and explicit "
            "measurement critique make it methodologically relevant. "
            "Because the design is observational and cross-sectional, "
            "the reported relationships should not be interpreted "
            "causally."
        ),
    },
    "SCR-0039": {
        "decision": "Include_final_candidate",
        "reason": (
            "National longitudinal cohort analysis with sampling and "
            "attrition weights, probit models and robustness checks"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "notes": (
            "The study uses the nationally representative 2003 "
            "Longitudinal Survey of Australian Youth cohort derived "
            "from PISA, including 10,370 participants. It estimates "
            "probit models, reports average marginal effects, applies "
            "weights for initial sampling and later attrition, uses "
            "robust standard errors and conducts subgroup robustness "
            "analyses. Limitations include observational identification, "
            "incomplete country-of-origin indicators and possible "
            "residual attrition or omitted-variable bias."
        ),
    },
    "SCR-0049": {
        "decision": "Exclude_full_text",
        "reason": (
            "Conference poster with insufficient methodological detail "
            "and unresolved endogeneity and comparability problems"
        ),
        "risk": "Critical risk of bias",
        "sample": False,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The available document is a brief conference poster of "
            "approximately 790 extracted words. It reports panel GLS "
            "regressions with country and time fixed effects and "
            "associations between governance and PISA outcomes, but "
            "does not provide enough information about the country "
            "sample, missing data, weighting, uncertainty propagation "
            "or complete model results. The poster itself recommends "
            "2SLS, improved specifications and more comparable data, "
            "confirming unresolved endogeneity and specification risks."
        ),
    },
}

REQUIRED_COLUMNS = [
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

REGISTRY_COLUMNS = [
    "FULL_TEXT_REQUIRED",
    "FULL_TEXT_REQUESTED",
    "FULL_TEXT_OBTAINED",
    "FULL_TEXT_LOCAL_FILE",
    "CONSENSUS_DECISION",
    "CONSENSUS_REASON",
    "CONSENSUS_DATE",
    "SCREENING_STATUS",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def update_registry(data):
    for column in REGISTRY_COLUMNS:
        if column not in data.columns:
            data[column] = ""

        data[column] = (
            data[column]
            .astype("string")
            .fillna("")
        )

    for screening_id, result in DECISIONS.items():
        mask = data["SCREENING_ID"].eq(screening_id)

        if int(mask.sum()) != 1:
            raise ValueError(
                f"{screening_id} no aparece exactamente una vez "
                "en el registro."
            )

        data.loc[mask, "FULL_TEXT_REQUIRED"] = "Yes"
        data.loc[mask, "FULL_TEXT_REQUESTED"] = "Yes"
        data.loc[mask, "FULL_TEXT_OBTAINED"] = "Yes"

        data.loc[mask, "FULL_TEXT_LOCAL_FILE"] = (
            "literature_review/full_text/"
            "screening_batch_02/"
            f"{screening_id}_full_text.pdf"
        )

        data.loc[mask, "CONSENSUS_DECISION"] = (
            result["decision"]
        )

        data.loc[mask, "CONSENSUS_REASON"] = (
            result["reason"]
        )

        data.loc[mask, "CONSENSUS_DATE"] = NOW

        data.loc[mask, "SCREENING_STATUS"] = (
            "Full_text_assessed"
        )

    return data


def main():
    required_files = [
        CONSOLIDATED_FILE,
        CANDIDATES_FILE,
        SCREENED_FILE,
        QUEUE_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    existing = pd.read_csv(
        CONSOLIDATED_FILE,
        encoding="utf-8-sig",
    )

    candidates = pd.read_csv(
        CANDIDATES_FILE,
        encoding="utf-8-sig",
    )

    screened = pd.read_csv(
        SCREENED_FILE,
        encoding="utf-8-sig",
    )

    queue = pd.read_csv(
        QUEUE_FILE,
        encoding="utf-8-sig",
    )

    candidate_lookup = (
        candidates
        .set_index("SCREENING_ID")
        .to_dict(orient="index")
    )

    new_rows = []

    for screening_id, result in DECISIONS.items():
        if screening_id not in candidate_lookup:
            raise ValueError(
                f"{screening_id} no aparece en candidatos."
            )

        source = candidate_lookup[screening_id]

        new_rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": clean_text(
                    source.get("TITLE", "")
                ),
                "AUTHORS": clean_text(
                    source.get("AUTHORS", "")
                ),
                "YEAR": clean_text(
                    source.get("YEAR", "")
                ),
                "JOURNAL_OR_SOURCE": clean_text(
                    source.get("JOURNAL_OR_SOURCE", "")
                ),
                "DOI_NORMALIZED": clean_text(
                    source.get("DOI_NORMALIZED", "")
                ),
                "THEMATIC_BLOCK": clean_text(
                    source.get("THEMATIC_BLOCK", "")
                ),
                "VERIFICATION_FLAGS": clean_text(
                    source.get("VERIFICATION_FLAGS", "")
                ),
                "FULL_TEXT_DECISION": result["decision"],
                "FULL_TEXT_REASON": result["reason"],
                "RISK_OF_BIAS": result["risk"],
                "SAMPLE_DESIGN_ADEQUATE": result["sample"],
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    result["missing"]
                ),
                "VALIDATION_ADEQUATE": (
                    result["validation"]
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    result["causal"]
                ),
                "CODE_AVAILABLE": False,
                "ASSESSMENT_NOTES": result["notes"],
                "ASSESSMENT_DATETIME": NOW,
            }
        )

    additions = pd.DataFrame(
        new_rows,
        columns=REQUIRED_COLUMNS,
    )

    existing = existing.loc[
        ~existing["SCREENING_ID"].isin(
            DECISIONS.keys()
        )
    ].copy()

    consolidated = pd.concat(
        [
            existing,
            additions,
        ],
        ignore_index=True,
    )

    consolidated = (
        consolidated
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
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

    assessed_ids = set(
        consolidated["SCREENING_ID"]
        .astype(str)
    )

    pending = candidates.loc[
        ~candidates["SCREENING_ID"].isin(
            assessed_ids
        )
    ].copy()

    pending.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    screened = update_registry(screened)
    queue = update_registry(queue)

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

    duplicate_count = int(
        consolidated[
            "SCREENING_ID"
        ].duplicated().sum()
    )

    expected_pending = {
        "SCR-0033",
        "SCR-0034",
        "SCR-0036",
        "SCR-0037",
        "SCR-0050",
    }

    actual_pending = set(
        pending["SCREENING_ID"]
        .astype(str)
    )

    valid = (
        len(consolidated) == 13
        and final_count == 7
        and limited_count == 4
        and excluded_count == 2
        and len(pending) == 5
        and actual_pending == expected_pending
        and duplicate_count == 0
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "BATCH_ID": "screening_batch_02",
                "FULL_TEXTS_ASSESSED": len(
                    consolidated
                ),
                "FINAL_CANDIDATES": final_count,
                "INCLUDE_WITH_MAJOR_LIMITATIONS": (
                    limited_count
                ),
                "FULL_TEXT_EXCLUSIONS": excluded_count,
                "PENDING_FULL_TEXTS": len(pending),
                "DUPLICATE_ASSESSMENTS": duplicate_count,
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
        "===== EVALUACION DE CUATRO "
        "RECUPERADOS LOTE 02 ====="
    )

    print(
        additions[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "===== RESULTADO CONSOLIDADO ====="
    )

    print(
        consolidated[
            "FULL_TEXT_DECISION"
        ].value_counts().to_string()
    )

    print()
    print(
        "EVALUADOS:",
        len(consolidated),
    )

    print(
        "PENDIENTES:",
        len(pending),
    )

    print(
        "IDS PENDIENTES:",
        ", ".join(
            pending[
                "SCREENING_ID"
            ].astype(str)
        ),
    )

    print(
        "DUPLICADOS:",
        duplicate_count,
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "La consolidacion de las cuatro "
            "evaluaciones requiere revision."
        )


if __name__ == "__main__":
    main()