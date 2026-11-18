from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "all_batches_final_candidates.csv"
)

OUTPUT_DIR = (
    ROOT
    / "literature_review"
    / "evidence_synthesis"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_candidates_master_evidence_table.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_candidates_master_evidence_summary.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "final_candidates_master_evidence_diagnostic.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def classify_design(row):
    notes = clean_value(
        row.get(
            "ASSESSMENT_NOTES",
            "",
        )
    ).lower()

    title = clean_value(
        row.get(
            "TITLE",
            "",
        )
    ).lower()

    text = f"{title} {notes}"

    if (
        "longitudinal" in text
        or "cohort" in text
    ):
        return "Longitudinal/cohort"

    if (
        "multilevel" in text
        or "intraclass correlation" in text
    ):
        return "Multilevel/comparative"

    if (
        "panel" in text
        or "fixed effects" in text
    ):
        return "Panel/fixed-effects"

    if (
        "measurement" in text
        or "construct validity" in text
        or "validation" in text
    ):
        return "Measurement/validation"

    if (
        "pisa" in text
        or "timss" in text
    ):
        return "International assessment analysis"

    return "Comparative observational"


def classify_dataset(row):
    text = " ".join(
        [
            clean_value(
                row.get(
                    "TITLE",
                    "",
                )
            ),
            clean_value(
                row.get(
                    "ASSESSMENT_NOTES",
                    "",
                )
            ),
        ]
    ).lower()

    datasets = []

    if "pisa" in text:
        datasets.append(
            "PISA"
        )

    if "timss" in text:
        datasets.append(
            "TIMSS"
        )

    if (
        "european social survey" in text
        or " ess " in f" {text} "
    ):
        datasets.append(
            "European Social Survey"
        )

    if (
        "longitudinal survey of australian youth"
        in text
        or "lsay" in text
    ):
        datasets.append(
            "LSAY"
        )

    if (
        "survey" in text
        and not datasets
    ):
        datasets.append(
            "Survey data"
        )

    if not datasets:
        datasets.append(
            "Other/constructed dataset"
        )

    return "; ".join(
        datasets
    )


def extract_scope(row):
    notes = clean_value(
        row.get(
            "ASSESSMENT_NOTES",
            "",
        )
    )

    if not notes:
        return ""

    sentences = [
        sentence.strip()
        for sentence in notes.replace(
            "\n",
            " ",
        ).split(".")
        if sentence.strip()
    ]

    if not sentences:
        return ""

    return (
        sentences[0]
        + "."
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    candidates = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    if len(candidates) != 12:
        raise ValueError(
            "Se esperaban exactamente "
            "12 candidatos finales."
        )

    if candidates[
        "SCREENING_ID"
    ].duplicated().any():
        raise ValueError(
            "Existen SCREENING_ID duplicados."
        )

    rows = []

    for _, source in (
        candidates.iterrows()
    ):
        rows.append(
            {
                "SCREENING_ID": clean_value(
                    source.get(
                        "SCREENING_ID",
                        "",
                    )
                ),
                "BATCH_ID": clean_value(
                    source.get(
                        "BATCH_ID",
                        "",
                    )
                ),
                "TITLE": clean_value(
                    source.get(
                        "TITLE",
                        "",
                    )
                ),
                "AUTHORS": clean_value(
                    source.get(
                        "AUTHORS",
                        "",
                    )
                ),
                "YEAR": clean_value(
                    source.get(
                        "YEAR",
                        "",
                    )
                ),
                "JOURNAL_OR_SOURCE": clean_value(
                    source.get(
                        "JOURNAL_OR_SOURCE",
                        "",
                    )
                ),
                "DOI_NORMALIZED": clean_value(
                    source.get(
                        "DOI_NORMALIZED",
                        "",
                    )
                ),
                "THEMATIC_BLOCK": clean_value(
                    source.get(
                        "THEMATIC_BLOCK",
                        "",
                    )
                ),
                "STUDY_DESIGN_CATEGORY": (
                    classify_design(source)
                ),
                "DATASET_CATEGORY": (
                    classify_dataset(source)
                ),
                "SCOPE_OR_SAMPLE_EVIDENCE": (
                    extract_scope(source)
                ),
                "FULL_TEXT_REASON": clean_value(
                    source.get(
                        "FULL_TEXT_REASON",
                        "",
                    )
                ),
                "RISK_OF_BIAS": clean_value(
                    source.get(
                        "RISK_OF_BIAS",
                        "",
                    )
                ),
                "SAMPLE_DESIGN_ADEQUATE": (
                    source.get(
                        "SAMPLE_DESIGN_ADEQUATE",
                        False,
                    )
                ),
                "MISSING_DATA_TREATMENT_ADEQUATE": (
                    source.get(
                        "MISSING_DATA_TREATMENT_ADEQUATE",
                        False,
                    )
                ),
                "VALIDATION_ADEQUATE": (
                    source.get(
                        "VALIDATION_ADEQUATE",
                        False,
                    )
                ),
                "CAUSAL_CLAIM_SUPPORTED": (
                    source.get(
                        "CAUSAL_CLAIM_SUPPORTED",
                        False,
                    )
                ),
                "CODE_AVAILABLE": source.get(
                    "CODE_AVAILABLE",
                    False,
                ),
                "ASSESSMENT_NOTES": clean_value(
                    source.get(
                        "ASSESSMENT_NOTES",
                        "",
                    )
                ),
                "SYNTHESIS_STATUS": (
                    "Ready_for_evidence_synthesis"
                ),
                "SYNTHESIS_DATETIME": NOW,
            }
        )

    master = pd.DataFrame(
        rows
    ).sort_values(
        "SCREENING_ID"
    ).reset_index(
        drop=True
    )

    master.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        master.groupby(
            [
                "STUDY_DESIGN_CATEGORY",
                "DATASET_CATEGORY",
                "RISK_OF_BIAS",
            ],
            as_index=False,
        )
        .agg(
            STUDY_COUNT=(
                "SCREENING_ID",
                "size",
            )
        )
        .sort_values(
            [
                "STUDY_DESIGN_CATEGORY",
                "DATASET_CATEGORY",
                "RISK_OF_BIAS",
            ]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(master) == 12
        and master[
            "SCREENING_ID"
        ].nunique() == 12
        and master[
            "SYNTHESIS_STATUS"
        ].eq(
            "Ready_for_evidence_synthesis"
        ).all()
        and master[
            "TITLE"
        ].ne("").all()
        and master[
            "FULL_TEXT_REASON"
        ].ne("").all()
        and master[
            "ASSESSMENT_NOTES"
        ].ne("").all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "FINAL_CANDIDATES_EXPECTED": 12,
                "MASTER_TABLE_RECORDS": len(
                    master
                ),
                "UNIQUE_SCREENING_IDS": int(
                    master[
                        "SCREENING_ID"
                    ].nunique()
                ),
                "STUDY_DESIGN_CATEGORIES": int(
                    master[
                        "STUDY_DESIGN_CATEGORY"
                    ].nunique()
                ),
                "DATASET_CATEGORIES": int(
                    master[
                        "DATASET_CATEGORY"
                    ].nunique()
                ),
                "READY_FOR_SYNTHESIS": int(
                    master[
                        "SYNTHESIS_STATUS"
                    ].eq(
                        "Ready_for_evidence_synthesis"
                    ).sum()
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
        "===== TABLA MAESTRA DE "
        "EVIDENCIA ====="
    )

    print()
    print(
        master[
            [
                "SCREENING_ID",
                "STUDY_DESIGN_CATEGORY",
                "DATASET_CATEGORY",
                "RISK_OF_BIAS",
                "SYNTHESIS_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "REGISTROS:",
        len(master),
    )

    print(
        "IDS UNICOS:",
        master[
            "SCREENING_ID"
        ].nunique(),
    )

    print(
        "CATEGORIAS DE DISEÑO:",
        master[
            "STUDY_DESIGN_CATEGORY"
        ].nunique(),
    )

    print(
        "CATEGORIAS DE DATOS:",
        master[
            "DATASET_CATEGORY"
        ].nunique(),
    )

    print(
        "LISTOS PARA SINTESIS:",
        master[
            "SYNTHESIS_STATUS"
        ].eq(
            "Ready_for_evidence_synthesis"
        ).sum(),
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    print()
    print(
        "ARCHIVO MAESTRO:",
        OUTPUT_FILE,
    )

    print(
        "ARCHIVO RESUMEN:",
        SUMMARY_FILE,
    )

    if not valid:
        raise ValueError(
            "La tabla maestra requiere revisión."
        )


if __name__ == "__main__":
    main()