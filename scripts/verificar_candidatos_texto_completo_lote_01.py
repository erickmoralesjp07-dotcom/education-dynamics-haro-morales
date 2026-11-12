from pathlib import Path
from datetime import datetime
import json
import re
import time
import unicodedata

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_results"
    / "batch_01_full_text_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
)

RAW_DIR = (
    OUTPUT_DIR
    / "raw_batch_01"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

for directory in [
    OUTPUT_DIR,
    RAW_DIR,
    DIAGNOSTIC_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


USER_AGENT = (
    "EducationDynamicsHaroMorales/0.1 "
    "(academic full-text verification)"
)


def normalize_doi(value):
    if pd.isna(value):
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
    if pd.isna(value):
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


def title_similarity(left, right):
    left_terms = set(
        normalize_title(left).split()
    )

    right_terms = set(
        normalize_title(right).split()
    )

    if not left_terms or not right_terms:
        return 0.0

    return len(
        left_terms.intersection(
            right_terms
        )
    ) / len(
        left_terms.union(
            right_terms
        )
    )


def request_with_retry(
    session,
    url,
    parameters=None,
    attempts=5,
):
    response = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        response = session.get(
            url,
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
            else float(4 * attempt)
        )

        print(
            f"  Estado {response.status_code}. "
            f"Reintento {attempt}/{attempts} "
            f"despues de {wait_seconds:.1f}s."
        )

        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(
            "No se obtuvo respuesta."
        )

    response.raise_for_status()

    return response


def crossref_metadata(
    session,
    doi,
    title,
):
    if doi:
        url = (
            "https://api.crossref.org/works/"
            + requests.utils.quote(
                doi,
                safe="",
            )
        )

        response = request_with_retry(
            session,
            url,
        )

        item = (
            response.json()
            .get("message")
            or {}
        )

        mode = "DOI"
    else:
        response = request_with_retry(
            session,
            "https://api.crossref.org/works",
            parameters={
                "query.bibliographic": title,
                "rows": 5,
            },
        )

        items = (
            response.json()
            .get("message", {})
            .get("items", [])
        )

        scored = [
            (
                title_similarity(
                    title,
                    (item.get("title") or [""])[0],
                ),
                item,
            )
            for item in items
        ]

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        item = (
            scored[0][1]
            if scored
            else {}
        )

        mode = "TITLE_SEARCH"

    return item, mode


def openalex_metadata(
    session,
    doi,
    title,
):
    if doi:
        response = request_with_retry(
            session,
            "https://api.openalex.org/works/"
            + requests.utils.quote(
                "https://doi.org/" + doi,
                safe="",
            ),
        )

        work = response.json()
        mode = "DOI"

    else:
        response = request_with_retry(
            session,
            "https://api.openalex.org/works",
            parameters={
                "search": title,
                "per-page": 5,
                "sort": "relevance_score:desc",
            },
        )

        works = (
            response.json()
            .get("results", [])
        )

        scored = [
            (
                title_similarity(
                    title,
                    work.get("display_name") or "",
                ),
                work,
            )
            for work in works
        ]

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        work = (
            scored[0][1]
            if scored
            else {}
        )

        mode = "TITLE_SEARCH"

    return work, mode


def crossref_year(item):
    for field in [
        "published-print",
        "published-online",
        "published",
        "issued",
    ]:
        parts = (
            (item.get(field) or {})
            .get("date-parts")
            or []
        )

        if (
            parts
            and parts[0]
            and parts[0][0]
        ):
            return int(parts[0][0])

    return None


def select_crossref_title(item):
    titles = item.get("title") or []
    return titles[0] if titles else ""


def select_crossref_source(item):
    values = (
        item.get("container-title")
        or []
    )

    return values[0] if values else ""


def save_json(data, path):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    candidates = pd.read_csv(
        INPUT_FILE
    )

    if len(candidates) != 18:
        raise ValueError(
            "Se esperaban 18 candidatos "
            f"y se encontraron {len(candidates)}."
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    rows = []

    print(
        "===== VERIFICACION DE CANDIDATOS "
        "DEL LOTE 01 ====="
    )

    for _, candidate in candidates.iterrows():
        screening_id = str(
            candidate["SCREENING_ID"]
        )

        title = str(
            candidate["TITLE"]
        )

        doi = normalize_doi(
            candidate.get("DOI_NORMALIZED")
        )

        print(
            f"Verificando {screening_id}: "
            f"{title[:80]}"
        )

        crossref_status = "OK"
        openalex_status = "OK"

        crossref_error = ""
        openalex_error = ""

        crossref_item = {}
        openalex_work = {}

        crossref_mode = ""
        openalex_mode = ""

        try:
            (
                crossref_item,
                crossref_mode,
            ) = crossref_metadata(
                session,
                doi,
                title,
            )

            save_json(
                crossref_item,
                RAW_DIR
                / (
                    f"{screening_id}_"
                    f"crossref.json"
                ),
            )

        except Exception as error:
            crossref_status = "ERROR"
            crossref_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        time.sleep(0.5)

        try:
            (
                openalex_work,
                openalex_mode,
            ) = openalex_metadata(
                session,
                doi,
                title,
            )

            save_json(
                openalex_work,
                RAW_DIR
                / (
                    f"{screening_id}_"
                    f"openalex.json"
                ),
            )

        except Exception as error:
            openalex_status = "ERROR"
            openalex_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        time.sleep(1.0)

        crossref_title = (
            select_crossref_title(
                crossref_item
            )
        )

        openalex_title = (
            openalex_work.get(
                "display_name"
            )
            or ""
        )

        crossref_doi = normalize_doi(
            crossref_item.get("DOI")
        )

        openalex_doi = normalize_doi(
            openalex_work.get("doi")
        )

        best_title_similarity = max(
            title_similarity(
                title,
                crossref_title,
            ),
            title_similarity(
                title,
                openalex_title,
            ),
        )

        primary_location = (
            openalex_work.get(
                "primary_location"
            )
            or {}
        )

        best_oa_location = (
            openalex_work.get(
                "best_oa_location"
            )
            or {}
        )

        open_access = (
            openalex_work.get(
                "open_access"
            )
            or {}
        )

        source = (
            primary_location.get(
                "source"
            )
            or {}
        )

        rows.append(
            {
                "SCREENING_ID": screening_id,
                "ORIGINAL_DECISION": (
                    candidate[
                        "REVIEWER_1_DECISION"
                    ]
                ),
                "ORIGINAL_TITLE": title,
                "ORIGINAL_YEAR": (
                    candidate.get("YEAR")
                ),
                "ORIGINAL_DOI": doi,
                "CROSSREF_STATUS": (
                    crossref_status
                ),
                "CROSSREF_MODE": (
                    crossref_mode
                ),
                "CROSSREF_TITLE": (
                    crossref_title
                ),
                "CROSSREF_YEAR": (
                    crossref_year(
                        crossref_item
                    )
                ),
                "CROSSREF_DOI": (
                    crossref_doi
                ),
                "CROSSREF_SOURCE": (
                    select_crossref_source(
                        crossref_item
                    )
                ),
                "CROSSREF_TYPE": (
                    crossref_item.get("type")
                    or ""
                ),
                "CROSSREF_URL": (
                    crossref_item.get("URL")
                    or ""
                ),
                "CROSSREF_ERROR": (
                    crossref_error
                ),
                "OPENALEX_STATUS": (
                    openalex_status
                ),
                "OPENALEX_MODE": (
                    openalex_mode
                ),
                "OPENALEX_ID": (
                    openalex_work.get("id")
                    or ""
                ),
                "OPENALEX_TITLE": (
                    openalex_title
                ),
                "OPENALEX_YEAR": (
                    openalex_work.get(
                        "publication_year"
                    )
                ),
                "OPENALEX_DOI": (
                    openalex_doi
                ),
                "OPENALEX_TYPE": (
                    openalex_work.get("type")
                    or ""
                ),
                "OPENALEX_SOURCE": (
                    source.get("display_name")
                    or ""
                ),
                "OPENALEX_CITED_BY_COUNT": (
                    openalex_work.get(
                        "cited_by_count"
                    )
                ),
                "OPENALEX_IS_OA": (
                    open_access.get("is_oa")
                ),
                "OPENALEX_OA_STATUS": (
                    open_access.get("oa_status")
                    or ""
                ),
                "OPENALEX_OA_URL": (
                    open_access.get("oa_url")
                    or ""
                ),
                "OPENALEX_BEST_OA_LANDING_PAGE": (
                    best_oa_location.get(
                        "landing_page_url"
                    )
                    or ""
                ),
                "OPENALEX_BEST_OA_PDF_URL": (
                    best_oa_location.get(
                        "pdf_url"
                    )
                    or ""
                ),
                "OPENALEX_ERROR": (
                    openalex_error
                ),
                "BEST_TITLE_SIMILARITY": (
                    best_title_similarity
                ),
                "DOI_MATCH": (
                    bool(doi)
                    and doi
                    in {
                        crossref_doi,
                        openalex_doi,
                    }
                ),
                "POSSIBLE_PUBLISHED_VERSION": (
                    (
                        "posted-content"
                        in str(
                            candidate.get(
                                "DOCUMENT_TYPE"
                            )
                        )
                    )
                    and (
                        crossref_item.get("type")
                        == "journal-article"
                        or openalex_work.get("type")
                        == "article"
                    )
                ),
                "VERIFICATION_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            }
        )

    verification = pd.DataFrame(
        rows
    )

    verification["VERIFICATION_ACTION"] = (
        "Manual review"
    )

    verification.loc[
        verification[
            "BEST_TITLE_SIMILARITY"
        ].lt(0.5),
        "VERIFICATION_ACTION",
    ] = "Review metadata match"

    verification.loc[
        verification[
            "OPENALEX_IS_OA"
        ].eq(True),
        "VERIFICATION_ACTION",
    ] = "Open-access full text candidate"

    verification.loc[
        verification[
            "POSSIBLE_PUBLISHED_VERSION"
        ].eq(True),
        "VERIFICATION_ACTION",
    ] = "Check published version"

    verification_file = (
        OUTPUT_DIR
        / "batch_01_candidate_verification.csv"
    )

    verification.to_csv(
        verification_file,
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
                "CANDIDATE_COUNT": len(
                    verification
                ),
                "CROSSREF_SUCCESS": int(
                    verification[
                        "CROSSREF_STATUS"
                    ].eq("OK").sum()
                ),
                "OPENALEX_SUCCESS": int(
                    verification[
                        "OPENALEX_STATUS"
                    ].eq("OK").sum()
                ),
                "DOI_MATCH_COUNT": int(
                    verification[
                        "DOI_MATCH"
                    ].sum()
                ),
                "OPEN_ACCESS_COUNT": int(
                    verification[
                        "OPENALEX_IS_OA"
                    ].eq(True).sum()
                ),
                "OA_PDF_URL_COUNT": int(
                    verification[
                        "OPENALEX_BEST_OA_PDF_URL"
                    ]
                    .fillna("")
                    .str.strip()
                    .ne("")
                    .sum()
                ),
                "POSSIBLE_PUBLISHED_VERSIONS": int(
                    verification[
                        "POSSIBLE_PUBLISHED_VERSION"
                    ].sum()
                ),
                "LOW_TITLE_MATCH_COUNT": int(
                    verification[
                        "BEST_TITLE_SIMILARITY"
                    ].lt(0.5).sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(verification) == 18
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "batch_01_candidate_verification_diagnostic.csv"
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "===== DIAGNOSTICO DE VERIFICACION ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESULTADOS DE VERIFICACION ====="
    )

    display_columns = [
        "SCREENING_ID",
        "ORIGINAL_DECISION",
        "CROSSREF_YEAR",
        "OPENALEX_YEAR",
        "CROSSREF_SOURCE",
        "OPENALEX_SOURCE",
        "OPENALEX_IS_OA",
        "OPENALEX_OA_STATUS",
        "OPENALEX_BEST_OA_PDF_URL",
        "BEST_TITLE_SIMILARITY",
        "DOI_MATCH",
        "POSSIBLE_PUBLISHED_VERSION",
        "VERIFICATION_ACTION",
    ]

    print(
        verification[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(verification_file)
    print(RAW_DIR)
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