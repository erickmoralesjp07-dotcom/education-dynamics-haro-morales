from pathlib import Path
from datetime import datetime
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "extracted_text"
    / "screening_batch_01"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_assessment"
    / "methodological_audit"
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "batch_01_six_pending_methodological_summary.txt"
)

COUNTS_FILE = (
    OUTPUT_DIR
    / "batch_01_six_pending_critical_term_counts.csv"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "batch_01_six_pending_methodological_summary_diagnostic.csv"
)

CONFIGURATION = {
    "SCR-0001": {
        "sections": [
            "data",
            "method",
            "empirical",
            "results",
            "conclusion",
            "limitation",
        ],
        "terms": [
            "cross-country",
            "natural disaster",
            "educational attainment",
            "barro and lee",
            "panel",
            "fixed effect",
            "random effect",
            "instrument",
            "endogeneity",
            "robustness",
            "causal",
            "missing",
            "sample",
        ],
    },
    "SCR-0006": {
        "sections": [
            "data",
            "method",
            "empirical",
            "results",
            "conclusion",
            "limitation",
        ],
        "terms": [
            "pisa",
            "timss",
            "pirls",
            "immigrant",
            "native",
            "sample weight",
            "sampling weight",
            "replicate weight",
            "plausible value",
            "multilevel",
            "country fixed",
            "decomposition",
            "selection",
            "robustness",
            "causal",
        ],
    },
    "SCR-0012": {
        "sections": [
            "method",
            "methodology",
            "results",
            "conclusion",
            "discussion",
            "limitation",
        ],
        "terms": [
            "pisa",
            "kazakhstan",
            "plausible value",
            "sampling weight",
            "replicate weight",
            "student weight",
            "multilevel",
            "hierarchical",
            "regression",
            "sample",
            "mathematics",
            "reading",
            "science",
            "robustness",
        ],
    },
    "SCR-0017": {
        "sections": [
            "data",
            "method",
            "empirical",
            "results",
            "conclusion",
            "measurement",
        ],
        "terms": [
            "economic growth",
            "schooling",
            "education quality",
            "education quantity",
            "measurement error",
            "barro and lee",
            "international test",
            "country",
            "regression",
            "instrument",
            "endogeneity",
            "robustness",
            "causal",
            "sample",
        ],
    },
    "SCR-0021": {
        "sections": [
            "data",
            "method",
            "results",
            "conclusion",
            "discussion",
            "limitation",
        ],
        "terms": [
            "pisa 2003",
            "pisa 2006",
            "pisa 2009",
            "pisa 2012",
            "turkey",
            "eligibility",
            "coverage",
            "target population",
            "enrolment",
            "enrollment",
            "sample weight",
            "plausible value",
            "selection bias",
            "trend",
            "robustness",
        ],
    },
    "SCR-0022": {
        "sections": [
            "data",
            "method",
            "results",
            "conclusion",
            "discussion",
            "limitation",
        ],
        "terms": [
            "international adult literacy survey",
            "ials",
            "functional literacy",
            "educational attainment",
            "earnings",
            "sample weight",
            "plausible value",
            "measurement error",
            "regression",
            "instrument",
            "selection",
            "country",
            "robustness",
            "causal",
        ],
    },
}

GLOBAL_TERMS = [
    "missing data",
    "imputation",
    "data availability",
    "code availability",
    "github",
    "bootstrap",
    "confidence interval",
    "standard error",
    "baseline",
    "cross-validation",
    "out-of-sample",
]


def clean_line(line):
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def select_passages(lines, terms, limit=45):
    scored = []

    for index, line in enumerate(lines):
        cleaned = clean_line(line)

        if len(cleaned) < 25:
            continue

        lower = cleaned.lower()

        matched = [
            term
            for term in terms
            if term.lower() in lower
        ]

        if not matched:
            continue

        score = len(matched)

        if any(
            keyword in lower
            for keyword in [
                "we use",
                "we estimate",
                "we employ",
                "the sample",
                "our data",
                "results show",
                "limitation",
                "however",
                "weight",
                "plausible value",
                "fixed effect",
                "instrument",
                "robust",
            ]
        ):
            score += 2

        scored.append(
            {
                "index": index,
                "line_number": index + 1,
                "text": cleaned,
                "matched": matched,
                "score": score,
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            item["line_number"],
        )
    )

    selected = []
    used_indexes = set()

    for item in scored:
        index = item["index"]

        if any(
            abs(index - used) <= 1
            for used in used_indexes
        ):
            continue

        selected.append(item)
        used_indexes.add(index)

        if len(selected) >= limit:
            break

    selected.sort(
        key=lambda item: item["line_number"]
    )

    return selected


def main():
    blocks = []
    count_rows = []

    print(
        "===== RESUMEN METODOLOGICO "
        "DE SEIS ESTUDIOS ====="
    )

    for screening_id, config in (
        CONFIGURATION.items()
    ):
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

        lines = text.splitlines()
        lower_text = text.lower()

        terms = list(
            dict.fromkeys(
                config["terms"]
                + GLOBAL_TERMS
            )
        )

        selected = select_passages(
            lines,
            terms,
            limit=45,
        )

        blocks.append(
            "\n".join(
                [
                    "=" * 78,
                    f"SCREENING_ID: {screening_id}",
                    (
                        "SELECTED_PASSAGES: "
                        f"{len(selected)}"
                    ),
                    "=" * 78,
                    "",
                    *[
                        (
                            f"{item['line_number']}: "
                            f"{item['text']}"
                        )
                        for item in selected
                    ],
                    "",
                ]
            )
        )

        row = {
            "SCREENING_ID": screening_id,
            "SELECTED_PASSAGES": len(
                selected
            ),
        }

        for term in terms:
            column = (
                "TERM_"
                + re.sub(
                    r"[^A-Z0-9]+",
                    "_",
                    term.upper(),
                ).strip("_")
            )

            row[column] = lower_text.count(
                term.lower()
            )

        count_rows.append(row)

        print(
            screening_id,
            "PASAJES=",
            len(selected),
        )

    OUTPUT_FILE.write_text(
        "\n".join(blocks),
        encoding="utf-8",
    )

    counts = pd.DataFrame(
        count_rows
    )

    counts.to_csv(
        COUNTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        len(counts) == 6
        and counts[
            "SELECTED_PASSAGES"
        ].ge(10).all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "STUDIES_SUMMARIZED": (
                    len(counts)
                ),
                "STUDIES_WITH_AT_LEAST_10_PASSAGES": int(
                    counts[
                        "SELECTED_PASSAGES"
                    ].ge(10).sum()
                ),
                "TOTAL_SELECTED_PASSAGES": int(
                    counts[
                        "SELECTED_PASSAGES"
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

    print(OUTPUT_FILE)
    print(COUNTS_FILE)
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
            "El resumen metodologico "
            "requiere revision."
        )


if __name__ == "__main__":
    main()