from pathlib import Path
from datetime import datetime
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "openalex_crossref_unique_records.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


EDUCATION_TERMS = [
    "education",
    "educational",
    "school",
    "student achievement",
    "learning achievement",
    "literacy",
    "attainment",
    "out-of-school",
    "pisa",
    "unesco",
    "sdg 4",
]

CORE_TERMS = [
    "system dynamics",
    "dynamic system",
    "differential equation",
    "state space",
    "panel data",
    "longitudinal",
    "cross-country",
    "educational attainment",
    "education expenditure",
    "education financing",
    "plausible values",
    "convergence",
    "cross-validation",
    "machine learning",
]

EXCLUSION_TERMS = [
    "dementia",
    "accelerated aging",
    "polygenic",
    "physical activity",
    "sports psychology",
    "medical education",
    "nursing education",
    "earth systems science education",
    "business economics education",
    "vibration lab",
    "differential analyzer",
    "chinese language education",
    "teacher digital literacy",
    "university teachers",
    "household wealth",
    "life expectancy",
]


def clean_text(value):
    value = "" if pd.isna(value) else str(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def contains_any(text, terms):
    return any(term in text for term in terms)


def classify_record(row):
    text = " ".join(
        [
            clean_text(row.get("TITLE")),
            clean_text(row.get("ABSTRACT")),
            clean_text(row.get("JOURNAL_OR_SOURCE")),
        ]
    )

    education_match = contains_any(
        text,
        EDUCATION_TERMS,
    )

    core_match = contains_any(
        text,
        CORE_TERMS,
    )

    exclusion_match = contains_any(
        text,
        EXCLUSION_TERMS,
    )

    abstract_available = bool(
        clean_text(row.get("ABSTRACT"))
    )

    score = float(
        row.get("RELEVANCE_SCORE", 0)
    )

    if exclusion_match:
        preliminary_status = "Exclude_preliminary"
        reason = "Clearly unrelated domain"

    elif education_match and core_match and score >= 7:
        preliminary_status = "Priority_screening"
        reason = "Education and core-method match"

    elif education_match and score >= 5:
        preliminary_status = "Secondary_screening"
        reason = "Education match; weaker methodological match"

    else:
        preliminary_status = "Low_priority"
        reason = "Insufficient combined relevance"

    return pd.Series(
        {
            "EDUCATION_MATCH": education_match,
            "CORE_METHOD_MATCH": core_match,
            "EXCLUSION_TERM_MATCH": exclusion_match,
            "ABSTRACT_AVAILABLE": abstract_available,
            "PRELIMINARY_STATUS": preliminary_status,
            "PRELIMINARY_REASON": reason,
        }
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    data = pd.read_csv(INPUT_FILE)

    classifications = data.apply(
        classify_record,
        axis=1,
    )

    data = pd.concat(
        [
            data,
            classifications,
        ],
        axis=1,
    )

    priority_order = {
        "Priority_screening": 1,
        "Secondary_screening": 2,
        "Low_priority": 3,
        "Exclude_preliminary": 4,
    }

    data["PRIORITY_ORDER"] = (
        data["PRELIMINARY_STATUS"]
        .map(priority_order)
    )

    data = data.sort_values(
        [
            "PRIORITY_ORDER",
            "ABSTRACT_AVAILABLE",
            "RELEVANCE_SCORE",
            "CITED_BY_COUNT",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    priority = data.loc[
        data["PRELIMINARY_STATUS"].eq(
            "Priority_screening"
        )
    ].copy()

    secondary = data.loc[
        data["PRELIMINARY_STATUS"].eq(
            "Secondary_screening"
        )
    ].copy()

    excluded = data.loc[
        data["PRELIMINARY_STATUS"].eq(
            "Exclude_preliminary"
        )
    ].copy()

    all_output = (
        OUTPUT_DIR
        / "bibliographic_prioritization_all.csv"
    )

    priority_output = (
        OUTPUT_DIR
        / "priority_title_abstract_screening.csv"
    )

    secondary_output = (
        OUTPUT_DIR
        / "secondary_title_abstract_screening.csv"
    )

    excluded_output = (
        OUTPUT_DIR
        / "preliminary_domain_exclusions.csv"
    )

    diagnostic_output = (
        DIAGNOSTIC_DIR
        / "bibliographic_prioritization_diagnostic.csv"
    )

    data.to_csv(
        all_output,
        index=False,
        encoding="utf-8-sig",
    )

    priority.to_csv(
        priority_output,
        index=False,
        encoding="utf-8-sig",
    )

    secondary.to_csv(
        secondary_output,
        index=False,
        encoding="utf-8-sig",
    )

    excluded.to_csv(
        excluded_output,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "INPUT_RECORDS": len(data),
                "PRIORITY_RECORDS": len(priority),
                "SECONDARY_RECORDS": len(secondary),
                "LOW_PRIORITY_RECORDS": int(
                    data[
                        "PRELIMINARY_STATUS"
                    ].eq("Low_priority").sum()
                ),
                "PRELIMINARY_EXCLUSIONS": len(excluded),
                "PRIORITY_WITH_ABSTRACT": int(
                    priority[
                        "ABSTRACT_AVAILABLE"
                    ].sum()
                ),
                "PRIORITY_WITH_DOI": int(
                    priority[
                        "DOI_NORMALIZED"
                    ].fillna("")
                    .ne("")
                    .sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(data) > 0
                    and len(priority) > 0
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        diagnostic_output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== PRIORIZACION BIBLIOGRAFICA ====="
    )

    print(
        diagnostic.to_string(index=False)
    )

    print()
    print(
        "===== PRIMEROS 50 REGISTROS "
        "PRIORITARIOS ====="
    )

    columns = [
        "RELEVANCE_SCORE",
        "YEAR",
        "TITLE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "DATABASE",
        "ABSTRACT_AVAILABLE",
    ]

    print(
        priority[columns]
        .head(50)
        .to_string(index=False)
    )

    print()
    print("Archivos creados:")
    print(all_output)
    print(priority_output)
    print(secondary_output)
    print(excluded_output)
    print(diagnostic_output)

    print()
    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )


if __name__ == "__main__":
    main()