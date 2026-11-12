from pathlib import Path
from datetime import datetime
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_results"
    / "bibliographic_prioritization_all.csv"
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TITLE_EDUCATION_TERMS = [
    "education",
    "educational",
    "school",
    "student achievement",
    "learning achievement",
    "literacy",
    "attainment",
    "out-of-school",
    "pisa",
    "sdg 4",
    "human capital",
]

METHOD_TERMS = [
    "system dynamics",
    "dynamic system",
    "differential equation",
    "state space",
    "panel data",
    "longitudinal",
    "cross-country",
    "cross-national",
    "multi-country",
    "convergence",
    "forecast",
    "prediction",
    "predictive",
    "cross-validation",
    "machine learning",
    "random forest",
    "ridge",
    "elastic net",
]

DIMENSION_TERMS = [
    "educational attainment",
    "education expenditure",
    "education spending",
    "education financing",
    "school enrollment",
    "school enrolment",
    "out-of-school",
    "literacy",
    "learning achievement",
    "student achievement",
    "educational inequality",
    "inequality of educational opportunity",
    "plausible values",
    "pisa",
    "unesco",
]

EXCLUSION_TERMS = [
    "dementia",
    "accelerated aging",
    "polygenic",
    "genetic prediction",
    "physical activity",
    "sports psychology",
    "medical education",
    "nursing education",
    "earth systems science education",
    "business economics education",
    "vibration lab",
    "differential analyzer",
    "chinese language education",
    "university teachers",
    "digital literacy influences",
    "cohabiting unions",
    "partnering market",
    "maternal phenotype",
    "life expectancy",
    "household wealth",
    "entrepreneurial exit",
    "functional trajectories",
    "nonlinear fractional differential equations",
]

SOFTWARE_TERMS = [
    "cran package",
    "software package",
    "r package",
    "python package",
]

PREPRINT_MARKERS = [
    "ssrn",
    "osf.io",
    "research square",
    "rs.3.rs-",
    "preprint",
]


def clean_text(value):
    value = (
        ""
        if pd.isna(value)
        else str(value)
    )

    value = re.sub(
        r"<[^>]+>",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def contains_any(text, terms):
    return any(
        term in text
        for term in terms
    )


def evaluate_record(row):
    title = clean_text(
        row.get("TITLE")
    )

    abstract = clean_text(
        row.get("ABSTRACT")
    )

    source = clean_text(
        row.get("JOURNAL_OR_SOURCE")
    )

    doi = clean_text(
        row.get("DOI_NORMALIZED")
    )

    url = clean_text(
        row.get("URL")
    )

    combined = " ".join(
        [
            title,
            abstract,
            source,
        ]
    )

    title_education = contains_any(
        title,
        TITLE_EDUCATION_TERMS,
    )

    title_method = contains_any(
        title,
        METHOD_TERMS,
    )

    title_dimension = contains_any(
        title,
        DIMENSION_TERMS,
    )

    combined_method = contains_any(
        combined,
        METHOD_TERMS,
    )

    combined_dimension = contains_any(
        combined,
        DIMENSION_TERMS,
    )

    exclusion_match = contains_any(
        combined,
        EXCLUSION_TERMS,
    )

    software_match = contains_any(
        title,
        SOFTWARE_TERMS,
    )

    preprint = contains_any(
        " ".join(
            [
                doi,
                url,
                source,
            ]
        ),
        PREPRINT_MARKERS,
    )

    abstract_available = bool(
        abstract
    )

    try:
        year = int(float(row.get("YEAR")))
    except (TypeError, ValueError):
        year = None

    future_record = (
        year is not None
        and year >= 2026
    )

    strict_match = (
        title_education
        and (
            title_method
            or title_dimension
            or (
                combined_method
                and combined_dimension
            )
        )
        and not exclusion_match
        and not software_match
    )

    if software_match:
        status = "Exclude"
        reason = "Software or package"

    elif exclusion_match:
        status = "Exclude"
        reason = "Clearly unrelated domain"

    elif not title_education:
        status = "Exclude"
        reason = (
            "Education relevance absent "
            "from title"
        )

    elif strict_match and abstract_available:
        status = "Priority_manual_screening"
        reason = (
            "Education title plus method "
            "or dimension match"
        )

    elif strict_match:
        status = "Verify_without_abstract"
        reason = (
            "Relevant title but abstract "
            "is unavailable"
        )

    else:
        status = "Secondary_manual_screening"
        reason = (
            "Educational title but weak "
            "combined methodological match"
        )

    verification_flags = []

    if preprint:
        verification_flags.append(
            "PREPRINT_OR_REPOSITORY"
        )

    if future_record:
        verification_flags.append(
            "YEAR_2026_VERIFY"
        )

    if not abstract_available:
        verification_flags.append(
            "NO_ABSTRACT"
        )

    if not doi:
        verification_flags.append(
            "NO_DOI"
        )

    return pd.Series(
        {
            "TITLE_EDUCATION_MATCH": (
                title_education
            ),
            "TITLE_METHOD_MATCH": (
                title_method
            ),
            "TITLE_DIMENSION_MATCH": (
                title_dimension
            ),
            "COMBINED_METHOD_MATCH": (
                combined_method
            ),
            "COMBINED_DIMENSION_MATCH": (
                combined_dimension
            ),
            "STRICT_RELEVANCE_MATCH": (
                strict_match
            ),
            "PREPRINT_OR_REPOSITORY": (
                preprint
            ),
            "FUTURE_RECORD": future_record,
            "REFINED_STATUS": status,
            "REFINED_REASON": reason,
            "VERIFICATION_FLAGS": "|".join(
                verification_flags
            ),
        }
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    classification = data.apply(
        evaluate_record,
        axis=1,
    )

    data = pd.concat(
        [
            data,
            classification,
        ],
        axis=1,
    )

    status_order = {
        "Priority_manual_screening": 1,
        "Verify_without_abstract": 2,
        "Secondary_manual_screening": 3,
        "Exclude": 4,
    }

    data["REFINED_ORDER"] = (
        data["REFINED_STATUS"]
        .map(status_order)
    )

    data = data.sort_values(
        [
            "REFINED_ORDER",
            "RELEVANCE_SCORE",
            "CITED_BY_COUNT",
            "YEAR",
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
        data["REFINED_STATUS"].eq(
            "Priority_manual_screening"
        )
    ].copy()

    verify = data.loc[
        data["REFINED_STATUS"].eq(
            "Verify_without_abstract"
        )
    ].copy()

    secondary = data.loc[
        data["REFINED_STATUS"].eq(
            "Secondary_manual_screening"
        )
    ].copy()

    excluded = data.loc[
        data["REFINED_STATUS"].eq(
            "Exclude"
        )
    ].copy()

    all_file = (
        OUTPUT_DIR
        / "refined_bibliographic_screening_all.csv"
    )

    priority_file = (
        OUTPUT_DIR
        / "refined_priority_screening.csv"
    )

    verify_file = (
        OUTPUT_DIR
        / "refined_verify_without_abstract.csv"
    )

    secondary_file = (
        OUTPUT_DIR
        / "refined_secondary_screening.csv"
    )

    excluded_file = (
        OUTPUT_DIR
        / "refined_preliminary_exclusions.csv"
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "refined_bibliographic_screening_diagnostic.csv"
    )

    data.to_csv(
        all_file,
        index=False,
        encoding="utf-8-sig",
    )

    priority.to_csv(
        priority_file,
        index=False,
        encoding="utf-8-sig",
    )

    verify.to_csv(
        verify_file,
        index=False,
        encoding="utf-8-sig",
    )

    secondary.to_csv(
        secondary_file,
        index=False,
        encoding="utf-8-sig",
    )

    excluded.to_csv(
        excluded_file,
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
                "PRIORITY_MANUAL_SCREENING": (
                    len(priority)
                ),
                "VERIFY_WITHOUT_ABSTRACT": (
                    len(verify)
                ),
                "SECONDARY_MANUAL_SCREENING": (
                    len(secondary)
                ),
                "PRELIMINARY_EXCLUSIONS": (
                    len(excluded)
                ),
                "PRIORITY_WITH_DOI": int(
                    priority[
                        "DOI_NORMALIZED"
                    ]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "PRIORITY_PREPRINT_FLAGS": int(
                    priority[
                        "PREPRINT_OR_REPOSITORY"
                    ].sum()
                ),
                "PRIORITY_FUTURE_FLAGS": int(
                    priority[
                        "FUTURE_RECORD"
                    ].sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(priority) > 0
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== CRIBADO BIBLIOGRAFICO "
        "REFINADO ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PRIMEROS 60 CANDIDATOS ====="
    )

    display_columns = [
        "RELEVANCE_SCORE",
        "YEAR",
        "TITLE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "DATABASE",
        "VERIFICATION_FLAGS",
    ]

    print(
        priority[
            display_columns
        ]
        .head(60)
        .to_string(index=False)
    )

    print()
    print("Archivos creados:")
    print(all_file)
    print(priority_file)
    print(verify_file)
    print(secondary_file)
    print(excluded_file)
    print(diagnostic_file)

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