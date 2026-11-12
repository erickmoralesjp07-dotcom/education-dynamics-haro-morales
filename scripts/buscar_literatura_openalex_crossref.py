from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import time
import unicodedata

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_DIR = PROJECT_ROOT / "literature_review"
RAW_DIR = LITERATURE_DIR / "search_raw"
OUTPUT_DIR = LITERATURE_DIR / "search_results"
DIAGNOSTIC_DIR = PROJECT_ROOT / "results" / "diagnostics"

for directory in [
    RAW_DIR,
    OUTPUT_DIR,
    DIAGNOSTIC_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


YEAR_FROM = 2000
YEAR_TO = 2026
RESULTS_PER_QUERY = 40

USER_AGENT = (
    "EducationDynamicsHaroMorales/0.1 "
    "(academic literature review)"
)

SEARCHES = [
    {
        "query_id": "LIT-Q01",
        "block": "dynamic_systems_education",
        "query": (
            '"education system" '
            '"system dynamics" '
            'longitudinal'
        ),
    },
    {
        "query_id": "LIT-Q02",
        "block": "differential_equations_education",
        "query": (
            'education '
            '"differential equations" '
            '"dynamic model"'
        ),
    },
    {
        "query_id": "LIT-Q03",
        "block": "international_education_panels",
        "query": (
            '"educational indicators" '
            '"panel data" '
            '"cross-country"'
        ),
    },
    {
        "query_id": "LIT-Q04",
        "block": "adult_attainment",
        "query": (
            '"educational attainment" '
            'longitudinal '
            '"cross-country"'
        ),
    },
    {
        "query_id": "LIT-Q05",
        "block": "education_finance",
        "query": (
            '"education expenditure" '
            'attainment '
            '"panel data"'
        ),
    },
    {
        "query_id": "LIT-Q06",
        "block": "predictive_validation",
        "query": (
            '"education indicators" '
            'prediction '
            '"cross-validation"'
        ),
    },
    {
        "query_id": "LIT-Q07",
        "block": "state_space_education",
        "query": (
            'education '
            '"state space model" '
            'longitudinal'
        ),
    },
    {
        "query_id": "LIT-Q08",
        "block": "convergence_regimes",
        "query": (
            '"educational attainment" '
            'convergence '
            '"cross-country"'
        ),
    },
    {
        "query_id": "LIT-Q09",
        "block": "pisa_inequality",
        "query": (
            'PISA '
            '"plausible values" '
            'inequality'
        ),
    },
    {
        "query_id": "LIT-Q10",
        "block": "machine_learning_education",
        "query": (
            '"educational indicators" '
            '"machine learning" '
            'prediction'
        ),
    },
]


def normalize_doi(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"^https?://(dx\.)?doi\.org/",
        "",
        value,
    )

    value = re.sub(
        r"^doi:\s*",
        "",
        value,
    )

    return value.strip()


def normalize_title(value):
    if value is None:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def invert_abstract(index):
    if not isinstance(index, dict):
        return ""

    positions = []

    for word, word_positions in index.items():
        for position in word_positions:
            positions.append(
                (
                    int(position),
                    str(word),
                )
            )

    positions.sort(
        key=lambda item: item[0]
    )

    return " ".join(
        word
        for _, word in positions
    )


def authors_openalex(work):
    authorships = work.get(
        "authorships"
    ) or []

    names = []

    for authorship in authorships:
        author = (
            authorship.get("author")
            or {}
        )

        name = author.get(
            "display_name"
        )

        if name:
            names.append(str(name))

    return "; ".join(names)


def authors_crossref(item):
    authors = item.get("author") or []

    names = []

    for author in authors:
        given = str(
            author.get("given") or ""
        ).strip()

        family = str(
            author.get("family") or ""
        ).strip()

        name = " ".join(
            part
            for part in [
                given,
                family,
            ]
            if part
        )

        if name:
            names.append(name)

    return "; ".join(names)


def year_crossref(item):
    for field in [
        "published-print",
        "published-online",
        "published",
        "issued",
    ]:
        date = item.get(field) or {}

        parts = date.get(
            "date-parts"
        ) or []

        if (
            parts
            and parts[0]
            and parts[0][0]
        ):
            return int(parts[0][0])

    return None


def save_json(data, path):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def search_openalex(session, search):
    parameters = {
        "search": search["query"],
        "filter": (
            f"from_publication_date:"
            f"{YEAR_FROM}-01-01,"
            f"to_publication_date:"
            f"{YEAR_TO}-12-31"
        ),
        "sort": "relevance_score:desc",
        "per-page": RESULTS_PER_QUERY,
    }

    response = None

    for attempt in range(1, 6):
        response = session.get(
            "https://api.openalex.org/works",
            params=parameters,
            timeout=90,
        )

        if response.status_code != 429:
            break

        retry_after = response.headers.get(
            "Retry-After"
        )

        wait_seconds = (
            float(retry_after)
            if retry_after
            else float(5 * attempt)
        )

        print(
            f"  OpenAlex limite 429. "
            f"Reintento {attempt}/5 despues de "
            f"{wait_seconds:.1f} segundos."
        )

        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(
            "OpenAlex no produjo ninguna respuesta."
        )

    response.raise_for_status()

    data = response.json()

    raw_file = (
        RAW_DIR
        / f"{search['query_id']}_openalex.json"
    )

    save_json(
        data,
        raw_file,
    )

    rows = []

    for rank, work in enumerate(
        data.get("results") or [],
        start=1,
    ):
        primary_location = (
            work.get("primary_location")
            or {}
        )

        source = (
            primary_location.get("source")
            or {}
        )

        rows.append(
            {
                "DATABASE": "OpenAlex",
                "QUERY_ID": search["query_id"],
                "THEMATIC_BLOCK": search["block"],
                "RESULT_RANK": rank,
                "TITLE": (
                    work.get("display_name")
                    or work.get("title")
                    or ""
                ),
                "AUTHORS": authors_openalex(
                    work
                ),
                "YEAR": work.get(
                    "publication_year"
                ),
                "JOURNAL_OR_SOURCE": (
                    source.get("display_name")
                    or ""
                ),
                "DOCUMENT_TYPE": (
                    work.get("type")
                    or ""
                ),
                "DOI": normalize_doi(
                    work.get("doi")
                ),
                "URL": (
                    work.get("doi")
                    or work.get("id")
                    or ""
                ),
                "ABSTRACT": invert_abstract(
                    work.get(
                        "abstract_inverted_index"
                    )
                ),
                "CITED_BY_COUNT": (
                    work.get("cited_by_count")
                ),
                "OPEN_ACCESS": (
                    (
                        work.get("open_access")
                        or {}
                    ).get("is_oa")
                ),
                "EXTERNAL_ID": (
                    work.get("id")
                    or ""
                ),
                "RAW_FILE": str(
                    raw_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    total_results = (
        data.get("meta") or {}
    ).get("count")

    return rows, total_results


def search_crossref(session, search):
    parameters = {
        "query.bibliographic": (
            search["query"]
        ),
        "filter": (
            f"from-pub-date:"
            f"{YEAR_FROM}-01-01,"
            f"until-pub-date:"
            f"{YEAR_TO}-12-31"
        ),
        "rows": RESULTS_PER_QUERY,
        "select": (
            "DOI,title,author,published,"
            "published-print,published-online,"
            "issued,container-title,type,URL,"
            "abstract,is-referenced-by-count"
        ),
    }

    response = None

    for attempt in range(1, 5):
        response = session.get(
            "https://api.crossref.org/works",
            params=parameters,
            timeout=90,
        )

        if response.status_code not in {
            429,
            500,
            502,
            503,
            504,
        }:
            break

        retry_after = response.headers.get(
            "Retry-After"
        )

        wait_seconds = (
            float(retry_after)
            if retry_after
            else float(3 * attempt)
        )

        print(
            f"  Crossref estado "
            f"{response.status_code}. "
            f"Reintento {attempt}/4 despues de "
            f"{wait_seconds:.1f} segundos."
        )

        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(
            "Crossref no produjo ninguna respuesta."
        )

    response.raise_for_status()

    data = response.json()

    raw_file = (
        RAW_DIR
        / f"{search['query_id']}_crossref.json"
    )

    save_json(
        data,
        raw_file,
    )

    message = data.get("message") or {}

    rows = []

    for rank, item in enumerate(
        message.get("items") or [],
        start=1,
    ):
        title_value = (
            item.get("title")
            or [""]
        )

        source_value = (
            item.get("container-title")
            or [""]
        )

        rows.append(
            {
                "DATABASE": "Crossref",
                "QUERY_ID": search["query_id"],
                "THEMATIC_BLOCK": search["block"],
                "RESULT_RANK": rank,
                "TITLE": title_value[0],
                "AUTHORS": authors_crossref(
                    item
                ),
                "YEAR": year_crossref(
                    item
                ),
                "JOURNAL_OR_SOURCE": (
                    source_value[0]
                ),
                "DOCUMENT_TYPE": (
                    item.get("type")
                    or ""
                ),
                "DOI": normalize_doi(
                    item.get("DOI")
                ),
                "URL": (
                    item.get("URL")
                    or ""
                ),
                "ABSTRACT": (
                    item.get("abstract")
                    or ""
                ),
                "CITED_BY_COUNT": (
                    item.get(
                        "is-referenced-by-count"
                    )
                ),
                "OPEN_ACCESS": None,
                "EXTERNAL_ID": (
                    item.get("DOI")
                    or ""
                ),
                "RAW_FILE": str(
                    raw_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    return (
        rows,
        message.get("total-results"),
    )


def score_relevance(row):
    text = " ".join(
        [
            str(row.get("TITLE") or ""),
            str(row.get("ABSTRACT") or ""),
            str(
                row.get(
                    "JOURNAL_OR_SOURCE"
                )
                or ""
            ),
        ]
    ).lower()

    positive_terms = {
        "education": 2,
        "educational": 2,
        "school": 1,
        "attainment": 3,
        "literacy": 2,
        "out-of-school": 3,
        "unesco": 2,
        "pisa": 2,
        "longitudinal": 2,
        "panel data": 3,
        "cross-country": 3,
        "international": 1,
        "system dynamics": 4,
        "dynamic system": 3,
        "differential equation": 4,
        "state space": 4,
        "cross-validation": 3,
        "prediction": 2,
        "forecasting": 2,
        "convergence": 2,
        "inequality": 2,
        "education expenditure": 3,
    }

    negative_terms = {
        "medical education": -2,
        "nursing education": -2,
        "engineering education": -1,
        "earth systems": -2,
        "business education": -1,
        "physical activity": -2,
        "dementia": -4,
        "aging": -3,
        "genetic": -3,
        "polygenic": -3,
    }

    score = 0

    for term, weight in positive_terms.items():
        if term in text:
            score += weight

    for term, weight in negative_terms.items():
        if term in text:
            score += weight

    return score


def deduplicate(records):
    data = pd.DataFrame(records)

    if data.empty:
        return data, data

    data["TITLE_NORMALIZED"] = (
        data["TITLE"].apply(
            normalize_title
        )
    )

    data["DOI_NORMALIZED"] = (
        data["DOI"].apply(
            normalize_doi
        )
    )

    data["RELEVANCE_SCORE"] = data.apply(
        score_relevance,
        axis=1,
    )

    data["DUPLICATE_KEY"] = ""

    has_doi = data[
        "DOI_NORMALIZED"
    ].ne("")

    data.loc[
        has_doi,
        "DUPLICATE_KEY",
    ] = (
        "doi:"
        + data.loc[
            has_doi,
            "DOI_NORMALIZED",
        ]
    )

    no_doi = ~has_doi

    data.loc[
        no_doi,
        "DUPLICATE_KEY",
    ] = (
        "title:"
        + data.loc[
            no_doi,
            "TITLE_NORMALIZED",
        ]
    )

    data = data.sort_values(
        [
            "RELEVANCE_SCORE",
            "CITED_BY_COUNT",
            "DATABASE",
            "RESULT_RANK",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        na_position="last",
    )

    duplicated = data.loc[
        data.duplicated(
            "DUPLICATE_KEY",
            keep="first",
        )
    ].copy()

    unique = data.drop_duplicates(
        "DUPLICATE_KEY",
        keep="first",
    ).copy()

    unique["SCREENING_DECISION"] = (
        "Pending"
    )

    unique["EXCLUSION_REASON"] = ""

    unique["REVIEWER"] = (
        "Haro and Morales"
    )

    unique["SEARCH_DATETIME"] = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    return (
        unique.reset_index(drop=True),
        duplicated.reset_index(drop=True),
    )


def main():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    all_records = []
    log_rows = []

    print(
        "===== BUSQUEDA BIBLIOGRAFICA "
        "OPENALEX Y CROSSREF ====="
    )

    for search in SEARCHES:
        for database in [
            "OpenAlex",
            "Crossref",
        ]:
            print(
                f"Consultando {database} "
                f"{search['query_id']}: "
                f"{search['block']}"
            )

            status = "OK"
            error_message = ""
            received = 0
            total_results = None

            try:
                if database == "OpenAlex":
                    rows, total_results = (
                        search_openalex(
                            session,
                            search,
                        )
                    )
                else:
                    rows, total_results = (
                        search_crossref(
                            session,
                            search,
                        )
                    )

                received = len(rows)
                all_records.extend(rows)

            except Exception as error:
                status = "ERROR"
                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                print(
                    "  ERROR:",
                    error_message,
                )

            log_rows.append(
                {
                    "SEARCH_EVENT_ID": (
                        f"{search['query_id']}-"
                        f"{database.upper()}"
                    ),
                    "DATABASE": database,
                    "QUERY_ID": (
                        search["query_id"]
                    ),
                    "THEMATIC_BLOCK": (
                        search["block"]
                    ),
                    "SEARCH_DATE": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                    "SEARCH_STRING_USED": (
                        search["query"]
                    ),
                    "YEAR_FROM": YEAR_FROM,
                    "YEAR_TO": YEAR_TO,
                    "TOTAL_RESULTS_REPORTED": (
                        total_results
                    ),
                    "RESULTS_RETRIEVED": (
                        received
                    ),
                    "STATUS": status,
                    "ERROR_MESSAGE": (
                        error_message
                    ),
                }
            )

            time.sleep(
                1.2
                if database == "OpenAlex"
                else 0.6
            )

    unique, duplicates = deduplicate(
        all_records
    )

    log = pd.DataFrame(log_rows)

    all_file = (
        OUTPUT_DIR
        / "openalex_crossref_all_records.csv"
    )

    unique_file = (
        OUTPUT_DIR
        / "openalex_crossref_unique_records.csv"
    )

    duplicate_file = (
        OUTPUT_DIR
        / "openalex_crossref_duplicates.csv"
    )

    log_file = (
        OUTPUT_DIR
        / "openalex_crossref_search_log.csv"
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "openalex_crossref_search_diagnostic.csv"
    )

    pd.DataFrame(all_records).to_csv(
        all_file,
        index=False,
        encoding="utf-8-sig",
    )

    unique.to_csv(
        unique_file,
        index=False,
        encoding="utf-8-sig",
    )

    duplicates.to_csv(
        duplicate_file,
        index=False,
        encoding="utf-8-sig",
    )

    log.to_csv(
        log_file,
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
                "SEARCH_COUNT": len(
                    SEARCHES
                ),
                "API_CALL_COUNT": len(log),
                "SUCCESSFUL_CALLS": int(
                    log["STATUS"]
                    .eq("OK")
                    .sum()
                ),
                "FAILED_CALLS": int(
                    log["STATUS"]
                    .eq("ERROR")
                    .sum()
                ),
                "RAW_RECORD_COUNT": len(
                    all_records
                ),
                "UNIQUE_RECORD_COUNT": len(
                    unique
                ),
                "DUPLICATE_RECORD_COUNT": len(
                    duplicates
                ),
                "RECORDS_SCORE_AT_LEAST_5": int(
                    (
                        unique[
                            "RELEVANCE_SCORE"
                        ] >= 5
                    ).sum()
                ),
                "RECORDS_WITH_DOI": int(
                    unique[
                        "DOI_NORMALIZED"
                    ].ne("").sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if log["STATUS"]
                    .eq("ERROR")
                    .sum()
                    == 0
                    and len(unique) > 0
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

    print()
    print("===== DIAGNOSTICO =====")
    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== REGISTROS DE MAYOR "
        "RELEVANCIA ====="
    )

    columns = [
        "RELEVANCE_SCORE",
        "YEAR",
        "TITLE",
        "DOI_NORMALIZED",
        "THEMATIC_BLOCK",
        "DATABASE",
    ]

    relevance_preview = (
        unique.sort_values(
            [
                "RELEVANCE_SCORE",
                "CITED_BY_COUNT",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .head(40)
    )

    print(
        relevance_preview[
            columns
        ].to_string(index=False)
    )

    print()
    print("Archivos creados:")
    print(all_file)
    print(unique_file)
    print(duplicate_file)
    print(log_file)
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