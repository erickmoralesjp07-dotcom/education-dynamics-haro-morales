from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSESSMENT_DIR = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
)

BATCH_01_FILE = (
    ASSESSMENT_DIR
    / "batch_01_full_text_assessment_consolidated.csv"
)

BATCH_02_FILE = (
    ASSESSMENT_DIR
    / "batch_02_full_text_assessment_consolidated.csv"
)

OUTPUT_FILE = (
    ASSESSMENT_DIR
    / "all_batches_full_text_assessment_consolidated.csv"
)

SUMMARY_FILE = (
    ASSESSMENT_DIR
    / "all_batches_full_text_assessment_summary.csv"
)

FINAL_CANDIDATES_FILE = (
    ASSESSMENT_DIR
    / "all_batches_final_candidates.csv"
)

LIMITED_CANDIDATES_FILE = (
    ASSESSMENT_DIR
    / "all_batches_candidates_with_major_limitations.csv"
)

EXCLUSIONS_FILE = (
    ASSESSMENT_DIR
    / "all_batches_full_text_exclusions.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "all_batches_full_text_assessment_diagnostic.csv"
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


def load_batch(path, batch_id):
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError(
            f"{path.name} está vacío."
        )

    required_columns = {
        "SCREENING_ID",
        "TITLE",
        "FULL_TEXT_DECISION",
        "FULL_TEXT_REASON",
        "RISK_OF_BIAS",
        "ASSESSMENT_NOTES",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} no contiene: "
            f"{sorted(missing_columns)}"
        )

    data = data.copy()

    data.insert(
        0,
        "BATCH_ID",
        batch_id,
    )

    return data


def main():
    batch_01 = load_batch(
        BATCH_01_FILE,
        "screening_batch_01",
    )

    batch_02 = load_batch(
        BATCH_02_FILE,
        "screening_batch_02",
    )

    source_columns_match = (
        list(
            batch_01.columns
        )
        == list(
            batch_02.columns
        )
    )

    if not source_columns_match:
        raise ValueError(
            "Las columnas de los dos lotes "
            "no coinciden exactamente."
        )

    consolidated = pd.concat(
        [
            batch_01,
            batch_02,
        ],
        ignore_index=True,
    )

    consolidated = (
        consolidated
        .sort_values(
            [
                "SCREENING_ID",
                "BATCH_ID",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_ids = (
        consolidated.loc[
            consolidated[
                "SCREENING_ID"
            ].duplicated(
                keep=False
            ),
            "SCREENING_ID",
        ]
        .astype(str)
        .tolist()
    )

    duplicate_count = len(
        set(duplicate_ids)
    )

    final_candidates = (
        consolidated.loc[
            consolidated[
                "FULL_TEXT_DECISION"
            ].eq(
                "Include_final_candidate"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    limited_candidates = (
        consolidated.loc[
            consolidated[
                "FULL_TEXT_DECISION"
            ].eq(
                "Include_with_major_limitations"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    exclusions = (
        consolidated.loc[
            consolidated[
                "FULL_TEXT_DECISION"
            ].eq(
                "Exclude_full_text"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    consolidated.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_candidates.to_csv(
        FINAL_CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    limited_candidates.to_csv(
        LIMITED_CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    exclusions.to_csv(
        EXCLUSIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        consolidated
        .groupby(
            [
                "BATCH_ID",
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
                "BATCH_ID",
                "FULL_TEXT_DECISION",
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
        len(batch_01) == 14
        and len(batch_02) == 13
        and len(consolidated) == 27
        and len(final_candidates) == 12
        and len(limited_candidates) == 10
        and len(exclusions) == 5
        and duplicate_count == 0
        and source_columns_match
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "BATCHES_CONSOLIDATED": 2,
                "BATCH_01_ASSESSED": len(
                    batch_01
                ),
                "BATCH_02_ASSESSED": len(
                    batch_02
                ),
                "TOTAL_FULL_TEXTS_ASSESSED": len(
                    consolidated
                ),
                "FINAL_CANDIDATES": len(
                    final_candidates
                ),
                "INCLUDE_WITH_MAJOR_LIMITATIONS": len(
                    limited_candidates
                ),
                "FULL_TEXT_EXCLUSIONS": len(
                    exclusions
                ),
                "DUPLICATE_SCREENING_IDS": (
                    duplicate_count
                ),
                "SOURCE_COLUMNS_MATCH": (
                    source_columns_match
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
        "===== CONSOLIDACION GENERAL "
        "LOTES 01 Y 02 ====="
    )

    print()
    print(
        "LOTE 01 EVALUADOS:",
        len(batch_01),
    )

    print(
        "LOTE 02 EVALUADOS:",
        len(batch_02),
    )

    print(
        "TOTAL EVALUADOS:",
        len(consolidated),
    )

    print()
    print(
        "CANDIDATOS FINALES:",
        len(final_candidates),
    )

    print(
        "INCLUIDOS CON LIMITACIONES:",
        len(limited_candidates),
    )

    print(
        "EXCLUIDOS:",
        len(exclusions),
    )

    print(
        "DUPLICADOS:",
        duplicate_count,
    )

    print()
    print(
        "===== DECISIONES POR LOTE ====="
    )

    print(
        consolidated.groupby(
            [
                "BATCH_ID",
                "FULL_TEXT_DECISION",
            ]
        ).size().to_string()
    )

    print()
    print(
        "===== CANDIDATOS FINALES ====="
    )

    print(
        final_candidates[
            [
                "SCREENING_ID",
                "BATCH_ID",
                "TITLE",
                "RISK_OF_BIAS",
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

    print()
    print(
        "ARCHIVO CONSOLIDADO:",
        OUTPUT_FILE,
    )

    print(
        "ARCHIVO DE CANDIDATOS:",
        FINAL_CANDIDATES_FILE,
    )

    print(
        "ARCHIVO DE RESUMEN:",
        SUMMARY_FILE,
    )

    if not valid:
        raise ValueError(
            "La consolidación general "
            "requiere revisión."
        )


if __name__ == "__main__":
    main()