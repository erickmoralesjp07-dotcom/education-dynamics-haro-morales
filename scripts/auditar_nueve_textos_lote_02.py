from pathlib import Path
from datetime import datetime
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = (
    ROOT
    / "literature_review"
    / "extracted_text"
    / "screening_batch_02"
)

CANDIDATES_FILE = (
    ROOT
    / "literature_review"
    / "screening_results"
    / "batch_02_full_text_candidates.csv"
)

OUTPUT_DIR = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "batch_02_methodological_audit"
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

AUDIT_FILE = (
    OUTPUT_DIR
    / "batch_02_nine_texts_methodological_audit.csv"
)

EVIDENCE_FILE = (
    OUTPUT_DIR
    / "batch_02_nine_texts_critical_evidence.txt"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "batch_02_nine_texts_methodological_audit_diagnostic.csv"
)

STUDY_IDS = [
    "SCR-0026",
    "SCR-0028",
    "SCR-0030",
    "SCR-0035",
    "SCR-0038",
    "SCR-0040",
    "SCR-0041",
    "SCR-0042",
    "SCR-0044",
]

TERM_GROUPS = {
    "sample": [
        "sample",
        "countries",
        "country",
        "students",
        "observations",
        "participants",
        "surveys",
        "census",
    ],
    "panel_time": [
        "panel",
        "longitudinal",
        "time series",
        "fixed effect",
        "random effect",
        "year effect",
        "trend",
        "convergence",
    ],
    "pisa_design": [
        "pisa",
        "plausible value",
        "sampling weight",
        "student weight",
        "replicate weight",
        "balanced repeated replication",
        "brr",
        "complex sampling",
        "jackknife",
    ],
    "validation": [
        "cross-validation",
        "out-of-sample",
        "training",
        "test set",
        "validation",
        "rmse",
        "mae",
        "sensitivity analysis",
        "robustness",
    ],
    "causality": [
        "causal",
        "causality",
        "endogeneity",
        "instrumental variable",
        "reverse causality",
        "omitted variable",
        "association",
        "correlation",
    ],
    "missing_data": [
        "missing data",
        "missing values",
        "imputation",
        "complete case",
        "listwise deletion",
        "multiple imputation",
    ],
    "uncertainty": [
        "standard error",
        "confidence interval",
        "bootstrap",
        "clustered",
        "driscoll",
        "heteroskedastic",
    ],
    "reproducibility": [
        "data availability",
        "code availability",
        "github",
        "supplementary material",
        "repository",
        "open data",
    ],
    "limitations": [
        "limitation",
        "limitations",
        "caution",
        "cannot infer",
        "should be interpreted",
        "future research",
    ],
}

PRIORITY_TERMS = [
    "we use",
    "we used",
    "we estimate",
    "we estimated",
    "method",
    "methodology",
    "data",
    "sample",
    "model",
    "regression",
    "fixed effect",
    "multilevel",
    "plausible value",
    "sampling weight",
    "cross-validation",
    "out-of-sample",
    "robustness",
    "sensitivity",
    "endogeneity",
    "missing",
    "limitation",
    "conclusion",
]


def clean_line(line):
    line = line.replace(
        "\u00ad",
        "",
    )

    return re.sub(
        r"\s+",
        " ",
        line,
    ).strip()


def count_words(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def score_line(line):
    lower = line.lower()
    matched_terms = []

    for group_terms in (
        TERM_GROUPS.values()
    ):
        for term in group_terms:
            if term in lower:
                matched_terms.append(
                    term
                )

    priority_matches = [
        term
        for term in PRIORITY_TERMS
        if term in lower
    ]

    score = (
        len(set(matched_terms))
        + 2 * len(
            set(priority_matches)
        )
    )

    if re.search(
        r"\b\d{2,}\b",
        line,
    ):
        score += 1

    return (
        score,
        sorted(
            set(
                matched_terms
                + priority_matches
            )
        ),
    )


def select_evidence(
    lines,
    maximum=35,
):
    candidates = []

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        line = clean_line(
            raw_line
        )

        if len(line) < 35:
            continue

        score, terms = score_line(
            line
        )

        if score <= 0:
            continue

        candidates.append(
            {
                "line_number": (
                    line_number
                ),
                "text": line,
                "score": score,
                "terms": terms,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["line_number"],
        )
    )

    selected = []
    selected_lines = []

    for item in candidates:
        if any(
            abs(
                item["line_number"]
                - existing
            ) <= 1
            for existing
            in selected_lines
        ):
            continue

        selected.append(
            item
        )

        selected_lines.append(
            item["line_number"]
        )

        if len(selected) >= maximum:
            break

    selected.sort(
        key=lambda item: (
            item["line_number"]
        )
    )

    return selected


def main():
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            CANDIDATES_FILE
        )

    candidates = pd.read_csv(
        CANDIDATES_FILE,
        encoding="utf-8-sig",
    )

    candidate_lookup = (
        candidates
        .set_index(
            "SCREENING_ID"
        )
        .to_dict(
            orient="index"
        )
    )

    audit_rows = []
    evidence_blocks = []

    print(
        "===== AUDITORIA METODOLOGICA "
        "DE NUEVE TEXTOS LOTE 02 ====="
    )

    for screening_id in STUDY_IDS:
        text_file = (
            TEXT_DIR
            / f"{screening_id}_full_text.txt"
        )

        if not text_file.exists():
            raise FileNotFoundError(
                text_file
            )

        text = text_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lower_text = text.lower()
        lines = text.splitlines()

        evidence = select_evidence(
            lines,
            maximum=35,
        )

        metadata = candidate_lookup[
            screening_id
        ]

        row = {
            "SCREENING_ID": (
                screening_id
            ),
            "TITLE": metadata.get(
                "TITLE",
                "",
            ),
            "YEAR": metadata.get(
                "YEAR",
                "",
            ),
            "DOI_NORMALIZED": (
                metadata.get(
                    "DOI_NORMALIZED",
                    "",
                )
            ),
            "REVIEWER_1_DECISION": (
                metadata.get(
                    "REVIEWER_1_DECISION",
                    "",
                )
            ),
            "WORD_COUNT": (
                count_words(text)
            ),
            "SELECTED_EVIDENCE_LINES": (
                len(evidence)
            ),
        }

        for group_name, terms in (
            TERM_GROUPS.items()
        ):
            row[
                f"{group_name.upper()}_TERM_COUNT"
            ] = sum(
                lower_text.count(
                    term
                )
                for term in terms
            )

        audit_rows.append(
            row
        )

        evidence_blocks.append(
            "\n".join(
                [
                    "=" * 80,
                    f"SCREENING_ID: {screening_id}",
                    (
                        "TITLE: "
                        f"{metadata.get('TITLE', '')}"
                    ),
                    (
                        "DECISION_AT_SCREENING: "
                        f"{metadata.get('REVIEWER_1_DECISION', '')}"
                    ),
                    (
                        "WORD_COUNT: "
                        f"{count_words(text)}"
                    ),
                    (
                        "SELECTED_EVIDENCE_LINES: "
                        f"{len(evidence)}"
                    ),
                    "=" * 80,
                    "",
                    *[
                        (
                            f"{item['line_number']}: "
                            f"{item['text']}"
                        )
                        for item in evidence
                    ],
                    "",
                ]
            )
        )

        print(
            screening_id,
            "WORDS=",
            count_words(text),
            "EVIDENCE=",
            len(evidence),
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    EVIDENCE_FILE.write_text(
        "\n".join(
            evidence_blocks
        ),
        encoding="utf-8",
    )

    valid = (
        len(audit) == 9
        and audit[
            "WORD_COUNT"
        ].gt(500).all()
        and audit[
            "SELECTED_EVIDENCE_LINES"
        ].ge(15).all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "BATCH_ID": (
                    "screening_batch_02"
                ),
                "TEXTS_AUDITED": (
                    len(audit)
                ),
                "TEXTS_OVER_500_WORDS": int(
                    audit[
                        "WORD_COUNT"
                    ].gt(500).sum()
                ),
                "TEXTS_WITH_AT_LEAST_15_EVIDENCE_LINES": int(
                    audit[
                        "SELECTED_EVIDENCE_LINES"
                    ].ge(15).sum()
                ),
                "TOTAL_SELECTED_EVIDENCE_LINES": int(
                    audit[
                        "SELECTED_EVIDENCE_LINES"
                    ].sum()
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

    print()
    print(
        "===== RESUMEN DE AUDITORIA ====="
    )

    print(
        audit[
            [
                "SCREENING_ID",
                "WORD_COUNT",
                "SELECTED_EVIDENCE_LINES",
                "SAMPLE_TERM_COUNT",
                "PANEL_TIME_TERM_COUNT",
                "PISA_DESIGN_TERM_COUNT",
                "VALIDATION_TERM_COUNT",
                "CAUSALITY_TERM_COUNT",
                "MISSING_DATA_TERM_COUNT",
                "LIMITATIONS_TERM_COUNT",
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
        "Archivos creados:"
    )

    print(AUDIT_FILE)
    print(EVIDENCE_FILE)
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
            "La auditoria de los nueve "
            "textos requiere revision."
        )


if __name__ == "__main__":
    main()