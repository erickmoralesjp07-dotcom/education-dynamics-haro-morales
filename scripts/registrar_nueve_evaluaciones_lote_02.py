from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCREENED_FILE = (
    ROOT / "literature_review" / "screening_results"
    / "screened_batch_02.csv"
)

QUEUE_FILE = (
    ROOT / "literature_review" / "search_results"
    / "title_abstract_screening_queue.csv"
)

CANDIDATES_FILE = (
    ROOT / "literature_review" / "screening_results"
    / "batch_02_full_text_candidates.csv"
)

ASSESSMENT_DIR = (
    ROOT / "literature_review" / "full_text_assessment"
)

OUTPUT_FILE = (
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

DIAGNOSTIC_FILE = (
    ROOT / "results" / "diagnostics"
    / "batch_02_nine_full_text_assessments_diagnostic.csv"
)

ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_FILE.parent.mkdir(parents=True, exist_ok=True)

NOW = datetime.now().astimezone().isoformat()

DECISIONS = {
    "SCR-0026": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Descriptive secondary analysis without "
            "student-level uncertainty estimation"
        ),
        "risk": "High risk of bias",
        "sample": False,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The study compares Philippine PISA science results "
            "for 2018, 2022 and 2025 using public databases, "
            "technical reports and policy documents. It recognizes "
            "that PISA is observational and cross-sectional. "
            "Although it mentions sampling weights, replicate "
            "weights and standard errors, it primarily reproduces "
            "published estimates and performs descriptive and "
            "documentary comparison rather than a reproducible "
            "student-level analysis. It is useful as recent context "
            "but should not support causal or model-performance claims."
        ),
    },
    "SCR-0028": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong long-run attainment database with explicit "
            "measurement-error and robustness analysis"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "notes": (
            "The study constructs a historical educational-attainment "
            "database for 74 countries from 1870 to 2010. It documents "
            "missing enrollment data, imputation, measurement error, "
            "differential mortality bias and multiple robustness "
            "experiments. Some early observations rely on neighboring "
            "countries and constructed historical values, so uncertainty "
            "is substantial before 1960. Its transparent treatment of "
            "measurement limitations and convergence makes it a strong "
            "methodological and substantive reference."
        ),
    },
    "SCR-0030": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Two-stage DEA design with inadequate complex-survey "
            "and plausible-value inference"
        ),
        "risk": "High risk of bias",
        "sample": False,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The study uses PISA 2006 data for Spanish regions and "
            "calculates DEA efficiency followed by second-stage "
            "regressions. It acknowledges five plausible values and "
            "runs separate calculations, but appears to average results "
            "rather than pool uncertainty using PISA procedures. "
            "Replicate weights and complex sampling variance are not "
            "clearly incorporated. Generated DEA scores in a second-stage "
            "Tobit model also create inference concerns."
        ),
    },
    "SCR-0035": {
        "decision": "Exclude_full_text",
        "reason": (
            "Insufficiently specified and non-reproducible "
            "empirical panel analysis"
        ),
        "risk": "Critical risk of bias",
        "sample": False,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The full text describes a multi-country panel framework "
            "and repeatedly claims that panel methods address "
            "heterogeneity and endogeneity. However, it does not clearly "
            "report the country sample, observation period, complete "
            "model specification, estimator implementation, diagnostics, "
            "missing-data procedure or reproducible results required to "
            "evaluate the empirical claims. Causal language is stronger "
            "than the documented design supports."
        ),
    },
    "SCR-0038": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong out-of-sample validation of educational-"
            "attainment measurement methods"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "notes": (
            "Using 1,680 survey and census datasets, the study compares "
            "the standard-duration method with new models for splitting "
            "binned schooling data. It uses out-of-sample predictive "
            "validation and reports materially lower error for the "
            "space-time approach. The PDF uses a custom character "
            "encoding that affected text extraction, but its identity "
            "and results were manually confirmed. It is directly relevant "
            "to measurement quality and predictive validation."
        ),
    },
    "SCR-0040": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Small OECD panel with unresolved dependence, "
            "endogeneity and causal-identification concerns"
        ),
        "risk": "High risk of bias",
        "sample": True,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The study analyzes 19 OECD countries using panel unit-root "
            "tests, cointegration procedures, the Hausman test and a "
            "fixed-effects regression. The design is relevant but the "
            "small cross-sectional dimension, possible cross-sectional "
            "dependence, reverse causality between spending and growth, "
            "and limited endogeneity treatment weaken interpretation. "
            "The results should be treated as associations."
        ),
    },
    "SCR-0041": {
        "decision": "Include_final_candidate",
        "reason": (
            "Strong multilevel PISA analysis with plausible values, "
            "weights, missing-data treatment and meta-analysis"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "notes": (
            "The preprint analyzes 79 PISA 2018 systems using ten "
            "plausible values, student and school weights, multilevel "
            "models, FIML for missing data and meta-analysis of ICCs. "
            "Results from plausible values are combined using multiple-"
            "imputation procedures. The main limitations are its "
            "observational design, sensitivity of ICC interpretation "
            "and preprint status. Methodologically it is a strong "
            "candidate for educational-inequality analysis."
        ),
    },
    "SCR-0042": {
        "decision": "Include_with_major_limitations",
        "reason": (
            "Aggregated country-level PISA analysis with questionable "
            "temporal alignment and uncertainty treatment"
        ),
        "risk": "High risk of bias",
        "sample": False,
        "missing": False,
        "validation": False,
        "causal": False,
        "notes": (
            "The study relates pre-primary structural indicators to "
            "aggregated PISA science scores for 33 countries across "
            "cycles. It uses country-level regression and reports an "
            "R-squared of 0.65. The approach does not clearly propagate "
            "PISA plausible-value and replicate-weight uncertainty, "
            "uses retrospective temporal matching and risks ecological "
            "bias, serial dependence and policy overinterpretation. "
            "It remains useful as exploratory comparative evidence."
        ),
    },
    "SCR-0044": {
        "decision": "Include_final_candidate",
        "reason": (
            "Technically adequate PISA 2022 comparative analysis "
            "with plausible values and replicate weights"
        ),
        "risk": "Moderate risk of bias",
        "sample": True,
        "missing": True,
        "validation": True,
        "causal": False,
        "notes": (
            "The study analyzes 52,392 students from ten Asian "
            "economies using ten reading plausible values, final "
            "student weights and 80 replicate weights. It estimates "
            "each plausible value and combines results, explicitly "
            "avoids causal interpretation and checks the effect of "
            "complete-case exclusion. Remaining limitations include "
            "self-reported predictors, observational associations and "
            "possible residual selection from missing data."
        ),
    },
}

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


def prepare_registry(data):
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
    data = prepare_registry(data)

    for screening_id, result in DECISIONS.items():
        mask = data["SCREENING_ID"].eq(screening_id)

        if int(mask.sum()) != 1:
            raise ValueError(
                f"{screening_id} no aparece exactamente una vez."
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
    for path in [
        SCREENED_FILE,
        QUEUE_FILE,
        CANDIDATES_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    screened = pd.read_csv(
        SCREENED_FILE,
        encoding="utf-8-sig",
    )

    queue = pd.read_csv(
        QUEUE_FILE,
        encoding="utf-8-sig",
    )

    candidates = pd.read_csv(
        CANDIDATES_FILE,
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

    lookup = candidates.set_index(
        "SCREENING_ID"
    ).to_dict(orient="index")

    rows = []

    for screening_id, result in DECISIONS.items():
        source = lookup[screening_id]

        rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": source.get("TITLE", ""),
                "AUTHORS": source.get("AUTHORS", ""),
                "YEAR": source.get("YEAR", ""),
                "JOURNAL_OR_SOURCE": source.get(
                    "JOURNAL_OR_SOURCE",
                    "",
                ),
                "DOI_NORMALIZED": source.get(
                    "DOI_NORMALIZED",
                    "",
                ),
                "THEMATIC_BLOCK": source.get(
                    "THEMATIC_BLOCK",
                    "",
                ),
                "VERIFICATION_FLAGS": source.get(
                    "VERIFICATION_FLAGS",
                    "",
                ),
                "FULL_TEXT_DECISION": result["decision"],
                "FULL_TEXT_REASON": result["reason"],
                "RISK_OF_BIAS": result["risk"],
                "SAMPLE_DESIGN_ADEQUATE": result["sample"],
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    result["missing"]
                ),
                "VALIDATION_ADEQUATE": result["validation"],
                "CAUSAL_CLAIM_SUPPORTED": result["causal"],
                "CODE_AVAILABLE": False,
                "ASSESSMENT_NOTES": result["notes"],
                "ASSESSMENT_DATETIME": NOW,
            }
        )

    assessments = pd.DataFrame(rows).sort_values(
        "SCREENING_ID"
    )

    assessments.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        assessments.groupby(
            [
                "FULL_TEXT_DECISION",
                "RISK_OF_BIAS",
            ],
            as_index=False,
        )
        .agg(RECORD_COUNT=("SCREENING_ID", "size"))
        .sort_values("FULL_TEXT_DECISION")
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    assessed_ids = set(DECISIONS)

    pending = candidates.loc[
        ~candidates["SCREENING_ID"].isin(assessed_ids)
    ].copy()

    pending.to_csv(
        PENDING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_count = int(
        assessments["FULL_TEXT_DECISION"]
        .eq("Include_final_candidate")
        .sum()
    )

    limited_count = int(
        assessments["FULL_TEXT_DECISION"]
        .eq("Include_with_major_limitations")
        .sum()
    )

    excluded_count = int(
        assessments["FULL_TEXT_DECISION"]
        .eq("Exclude_full_text")
        .sum()
    )

    duplicate_count = int(
        assessments["SCREENING_ID"].duplicated().sum()
    )

    comparison_columns = [
        "SCREENING_ID",
        "CONSENSUS_DECISION",
        "CONSENSUS_REASON",
        "SCREENING_STATUS",
    ]

    files_match = (
        screened.loc[
            screened["SCREENING_ID"].isin(assessed_ids),
            comparison_columns,
        ]
        .sort_values("SCREENING_ID")
        .reset_index(drop=True)
        .fillna("")
        .equals(
            queue.loc[
                queue["SCREENING_ID"].isin(assessed_ids),
                comparison_columns,
            ]
            .sort_values("SCREENING_ID")
            .reset_index(drop=True)
            .fillna("")
        )
    )

    valid = (
        len(assessments) == 9
        and final_count == 4
        and limited_count == 4
        and excluded_count == 1
        and len(pending) == 9
        and duplicate_count == 0
        and files_match
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "BATCH_ID": "screening_batch_02",
                "FULL_TEXTS_ASSESSED": len(assessments),
                "FINAL_CANDIDATES": final_count,
                "INCLUDE_WITH_MAJOR_LIMITATIONS": limited_count,
                "FULL_TEXT_EXCLUSIONS": excluded_count,
                "PENDING_FULL_TEXTS": len(pending),
                "DUPLICATE_ASSESSMENTS": duplicate_count,
                "FILES_MATCH": files_match,
                "VALIDATION_STATUS": (
                    "OK" if valid else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("===== EVALUACION DE NUEVE TEXTOS LOTE 02 =====")

    print(
        assessments[
            [
                "SCREENING_ID",
                "FULL_TEXT_DECISION",
                "FULL_TEXT_REASON",
                "RISK_OF_BIAS",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== DIAGNOSTICO =====")
    print(diagnostic.to_string(index=False))

    print()
    print("===== PENDIENTES DE TEXTO COMPLETO =====")

    print(
        pending[
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
                "TITLE",
                "DOI_NORMALIZED",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[0, "VALIDATION_STATUS"],
    )

    if not valid:
        raise ValueError(
            "La evaluación del lote 02 requiere revisión."
        )


if __name__ == "__main__":
    main()