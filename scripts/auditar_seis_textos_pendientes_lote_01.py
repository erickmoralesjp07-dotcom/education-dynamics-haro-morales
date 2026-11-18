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

SUMMARY_FILE = (
    OUTPUT_DIR
    / "batch_01_six_pending_methodological_audit.csv"
)

PASSAGES_FILE = (
    OUTPUT_DIR
    / "batch_01_six_pending_methodological_passages.txt"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "batch_01_six_pending_methodological_audit_diagnostic.csv"
)

STUDIES = {
    "SCR-0001": {
        "expected_title_terms": [
            "natural disasters",
            "educational attainment",
        ],
        "method_terms": [
            "natural disaster",
            "educational attainment",
            "cross-country",
            "country",
            "countries",
            "panel",
            "1960",
            "1990",
            "data",
            "sample",
            "barro",
            "lee",
            "regression",
            "fixed effect",
            "random effect",
            "instrument",
            "endogeneity",
            "causal",
            "robustness",
            "sensitivity",
            "missing",
            "limitation",
        ],
    },
    "SCR-0006": {
        "expected_title_terms": [
            "how different are immigrants",
            "educational achievement",
        ],
        "method_terms": [
            "immigrant",
            "immigrants",
            "educational achievement",
            "pisa",
            "timss",
            "pirls",
            "country",
            "countries",
            "survey",
            "sample",
            "weight",
            "plausible value",
            "replicate weight",
            "regression",
            "multilevel",
            "fixed effect",
            "decomposition",
            "selection",
            "robustness",
            "causal",
            "limitation",
        ],
    },
    "SCR-0012": {
        "expected_title_terms": [
            "pisa-2018",
            "kazakhstan",
        ],
        "method_terms": [
            "pisa",
            "2018",
            "kazakhstan",
            "sample",
            "student",
            "school",
            "plausible value",
            "weight",
            "replicate weight",
            "regression",
            "multilevel",
            "hierarchical",
            "factor",
            "achievement",
            "mathematics",
            "reading",
            "science",
            "robustness",
            "limitation",
        ],
    },
    "SCR-0017": {
        "expected_title_terms": [
            "education for growth",
            "why and for whom",
        ],
        "method_terms": [
            "education",
            "growth",
            "economic growth",
            "schooling",
            "quality",
            "quantity",
            "measurement error",
            "country",
            "countries",
            "sample",
            "regression",
            "instrument",
            "endogeneity",
            "fixed effect",
            "causal",
            "robustness",
            "sensitivity",
            "barro",
            "lee",
            "limitation",
        ],
    },
    "SCR-0021": {
        "expected_title_terms": [
            "who makes it into pisa",
            "turkey",
        ],
        "method_terms": [
            "pisa",
            "turkey",
            "eligibility",
            "sample",
            "coverage",
            "enrolment",
            "enrollment",
            "exclusion",
            "selection",
            "representative",
            "weight",
            "plausible value",
            "2003",
            "2012",
            "trend",
            "bias",
            "sensitivity",
            "robustness",
            "limitation",
        ],
    },
    "SCR-0022": {
        "expected_title_terms": [
            "functional literacy",
            "educational attainment",
            "earnings",
        ],
        "method_terms": [
            "functional literacy",
            "educational attainment",
            "earnings",
            "country",
            "countries",
            "international adult literacy survey",
            "ials",
            "sample",
            "weight",
            "plausible value",
            "regression",
            "instrument",
            "measurement error",
            "selection",
            "causal",
            "robustness",
            "limitation",
        ],
    },
}

GLOBAL_TERMS = [
    "data availability",
    "code availability",
    "github",
    "open data",
    "supplementary",
    "appendix",
    "missing data",
    "imputation",
    "standard error",
    "confidence interval",
    "bootstrap",
    "cross-validation",
    "out-of-sample",
    "baseline",
]


def normalize_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def count_words(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def find_passages(
    lines,
    terms,
    context_before=1,
    context_after=2,
):
    matching_indexes = []

    for index, line in enumerate(lines):
        lower_line = line.lower()

        if any(
            term.lower() in lower_line
            for term in terms
        ):
            matching_indexes.append(index)

    indexes_to_keep = set()

    for index in matching_indexes:
        beginning = max(
            0,
            index - context_before,
        )

        ending = min(
            len(lines),
            index + context_after + 1,
        )

        for keep_index in range(
            beginning,
            ending,
        ):
            indexes_to_keep.add(
                keep_index
            )

    return [
        (
            index + 1,
            lines[index],
        )
        for index
        in sorted(indexes_to_keep)
    ]


def main():
    summary_rows = []
    passages_blocks = []

    print(
        "===== AUDITORIA DE SEIS "
        "TEXTOS PENDIENTES ====="
    )

    for screening_id, configuration in (
        STUDIES.items()
    ):
        text_file = (
            TEXT_DIR
            / f"{screening_id}_full_text.txt"
        )

        if not text_file.exists():
            raise FileNotFoundError(
                text_file
            )

        raw_text = text_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lines = raw_text.splitlines()
        lower_text = raw_text.lower()

        file_size = (
            text_file.stat().st_size
        )

        character_count = len(
            raw_text
        )

        word_count = count_words(
            raw_text
        )

        expected_title_matches = {
            term: (
                term.lower()
                in lower_text
            )
            for term
            in configuration[
                "expected_title_terms"
            ]
        }

        title_terms_found = sum(
            expected_title_matches.values()
        )

        title_terms_expected = len(
            expected_title_matches
        )

        identity_status = (
            "IDENTITY_PROBABLE"
            if title_terms_found
            == title_terms_expected
            else "IDENTITY_REVIEW"
        )

        all_terms = (
            configuration[
                "method_terms"
            ]
            + GLOBAL_TERMS
        )

        term_counts = {
            term: lower_text.count(
                term.lower()
            )
            for term in all_terms
        }

        passages = find_passages(
            lines,
            all_terms,
        )

        nonempty_passages = [
            (
                line_number,
                normalize_text(line),
            )
            for line_number, line
            in passages
            if normalize_text(line)
        ]

        passages_blocks.append(
            "\n".join(
                [
                    "=" * 78,
                    f"SCREENING_ID: {screening_id}",
                    (
                        "IDENTITY_STATUS: "
                        f"{identity_status}"
                    ),
                    (
                        "TEXT_FILE: "
                        f"{text_file.relative_to(PROJECT_ROOT)}"
                    ),
                    (
                        "WORD_COUNT: "
                        f"{word_count}"
                    ),
                    "=" * 78,
                    "",
                    *[
                        (
                            f"{line_number}: "
                            f"{line}"
                        )
                        for line_number, line
                        in nonempty_passages
                    ],
                    "",
                ]
            )
        )

        summary_row = {
            "SCREENING_ID": (
                screening_id
            ),
            "TEXT_FILE": str(
                text_file.relative_to(
                    PROJECT_ROOT
                )
            ),
            "FILE_SIZE_BYTES": (
                file_size
            ),
            "CHARACTER_COUNT": (
                character_count
            ),
            "WORD_COUNT": (
                word_count
            ),
            "IDENTITY_STATUS": (
                identity_status
            ),
            "TITLE_TERMS_FOUND": (
                title_terms_found
            ),
            "TITLE_TERMS_EXPECTED": (
                title_terms_expected
            ),
            "METHODOLOGICAL_PASSAGE_COUNT": (
                len(
                    nonempty_passages
                )
            ),
        }

        for term, count in (
            term_counts.items()
        ):
            column_name = (
                "TERM_"
                + re.sub(
                    r"[^A-Z0-9]+",
                    "_",
                    term.upper(),
                ).strip("_")
            )

            summary_row[
                column_name
            ] = count

        summary_rows.append(
            summary_row
        )

        print()
        print(
            screening_id,
            "IDENTITY_STATUS=",
            identity_status,
            "WORDS=",
            word_count,
            "PASSAGES=",
            len(nonempty_passages),
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    PASSAGES_FILE.write_text(
        "\n".join(
            passages_blocks
        ),
        encoding="utf-8",
    )

    assessed_count = len(
        summary
    )

    identity_probable_count = int(
        summary[
            "IDENTITY_STATUS"
        ].eq(
            "IDENTITY_PROBABLE"
        ).sum()
    )

    valid = (
        assessed_count == 6
        and summary[
            "FILE_SIZE_BYTES"
        ].gt(1000).all()
        and summary[
            "WORD_COUNT"
        ].gt(1000).all()
        and summary[
            "METHODOLOGICAL_PASSAGE_COUNT"
        ].gt(0).all()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "REQUESTED_STUDIES": 6,
                "TEXTS_FOUND": (
                    assessed_count
                ),
                "TEXTS_OVER_1000_WORDS": int(
                    summary[
                        "WORD_COUNT"
                    ].gt(1000).sum()
                ),
                "IDENTITIES_PROBABLE": (
                    identity_probable_count
                ),
                "METHODOLOGICAL_AUDITS_CREATED": int(
                    summary[
                        "METHODOLOGICAL_PASSAGE_COUNT"
                    ].gt(0).sum()
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
        "===== RESUMEN ====="
    )

    print(
        summary[
            [
                "SCREENING_ID",
                "WORD_COUNT",
                "IDENTITY_STATUS",
                "TITLE_TERMS_FOUND",
                "TITLE_TERMS_EXPECTED",
                "METHODOLOGICAL_PASSAGE_COUNT",
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

    print(SUMMARY_FILE)
    print(PASSAGES_FILE)
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
            "La auditoria metodologica "
            "requiere revision."
        )


if __name__ == "__main__":
    main()