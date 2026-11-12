from pathlib import Path
from datetime import datetime
import html
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_batches"
    / "screening_batch_01.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_forms"
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
    / "screening_form_batch_01.md"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "screening_form_batch_01_diagnostic.csv"
)


def clean_text(value):
    if pd.isna(value):
        return ""

    text = html.unescape(
        str(value)
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def format_value(value):
    text = clean_text(value)
    return text if text else "Not available"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    if len(data) != 25:
        raise ValueError(
            "El lote 01 debe contener "
            f"25 registros y contiene {len(data)}."
        )

    required = [
        "SCREENING_ID",
        "TITLE",
        "AUTHORS",
        "YEAR",
        "JOURNAL_OR_SOURCE",
        "DOCUMENT_TYPE",
        "DOI_NORMALIZED",
        "URL",
        "ABSTRACT",
        "THEMATIC_BLOCK",
        "VERIFICATION_FLAGS",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    lines = []

    lines.append(
        "# Title and Abstract Screening Form"
    )

    lines.append("")
    lines.append(
        "## Batch 01"
    )

    lines.append("")
    lines.append(
        f"Generated: "
        f"{datetime.now().astimezone().isoformat()}"
    )

    lines.append("")
    lines.append(
        "Allowed decisions: "
        "`Include`, `Exclude`, or `Unclear`."
    )

    lines.append("")
    lines.append(
        "Decisions must be based on title and "
        "abstract relevance to the project, not "
        "only on the automatic score."
    )

    lines.append("")
    lines.append(
        "Suggested exclusion reasons:"
    )

    lines.append("")
    lines.append(
        "- Wrong population or context"
    )
    lines.append(
        "- Wrong educational outcome"
    )
    lines.append(
        "- Wrong method or study design"
    )
    lines.append(
        "- No international or longitudinal component"
    )
    lines.append(
        "- Economic outcome without educational outcome"
    )
    lines.append(
        "- Non-scientific document"
    )
    lines.append(
        "- Preprint replaced by published article"
    )
    lines.append(
        "- Future or unverifiable publication"
    )
    lines.append(
        "- Language cannot be evaluated"
    )

    lines.append("")
    lines.append("---")

    for position, row in data.iterrows():
        record_number = position + 1

        screening_id = format_value(
            row["SCREENING_ID"]
        )

        title = format_value(
            row["TITLE"]
        )

        lines.append("")
        lines.append(
            f"## {record_number}. "
            f"{screening_id}"
        )

        lines.append("")
        lines.append(
            f"### {title}"
        )

        lines.append("")
        lines.append(
            f"- **Authors:** "
            f"{format_value(row['AUTHORS'])}"
        )

        lines.append(
            f"- **Year:** "
            f"{format_value(row['YEAR'])}"
        )

        lines.append(
            f"- **Source:** "
            f"{format_value(row['JOURNAL_OR_SOURCE'])}"
        )

        lines.append(
            f"- **Document type:** "
            f"{format_value(row['DOCUMENT_TYPE'])}"
        )

        lines.append(
            f"- **DOI:** "
            f"{format_value(row['DOI_NORMALIZED'])}"
        )

        lines.append(
            f"- **URL:** "
            f"{format_value(row['URL'])}"
        )

        lines.append(
            f"- **Thematic block:** "
            f"{format_value(row['THEMATIC_BLOCK'])}"
        )

        lines.append(
            f"- **Verification flags:** "
            f"{format_value(row['VERIFICATION_FLAGS'])}"
        )

        lines.append("")
        lines.append(
            "**Abstract**"
        )

        lines.append("")
        lines.append(
            format_value(
                row["ABSTRACT"]
            )
        )

        lines.append("")
        lines.append(
            "**Reviewer 1 decision:** "
            "`PENDING`"
        )

        lines.append("")
        lines.append(
            "**Reviewer 1 exclusion reason:** "
            "`PENDING`"
        )

        lines.append("")
        lines.append(
            "**Reviewer 1 notes:** "
            "`PENDING`"
        )

        lines.append("")
        lines.append("---")

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "BATCH_ID": "BATCH-01",
                "INPUT_RECORDS": len(data),
                "OUTPUT_FILE": str(
                    OUTPUT_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "RECORDS_WITH_ABSTRACT": int(
                    data[
                        "ABSTRACT"
                    ]
                    .fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
                "PREPRINT_FLAGS": int(
                    data[
                        "PREPRINT_OR_REPOSITORY"
                    ].fillna(False).sum()
                ),
                "FUTURE_FLAGS": int(
                    data[
                        "FUTURE_RECORD"
                    ].fillna(False).sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(data) == 25
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
        "===== FICHA DE CRIBADO LOTE 01 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print("Archivo creado:")
    print(OUTPUT_FILE)

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()