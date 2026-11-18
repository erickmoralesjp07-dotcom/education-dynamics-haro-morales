from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = (
    ROOT
    / "literature_review"
    / "screening_results"
    / "batch_02_full_text_candidates.csv"
)

PDF_DIR = (
    ROOT
    / "literature_review"
    / "full_text"
    / "screening_batch_02"
)

VERIFICATION_DIR = (
    ROOT
    / "literature_review"
    / "full_text_verification"
)

RAW_DIR = (
    VERIFICATION_DIR
    / "raw_batch_02"
)

DIAGNOSTIC_DIR = (
    ROOT
    / "results"
    / "diagnostics"
)

INVENTORY_FILE = (
    VERIFICATION_DIR
    / "batch_02_download_inventory.csv"
)

ATTEMPTS_FILE = (
    VERIFICATION_DIR
    / "batch_02_download_attempts.csv"
)

MANUAL_FILE = (
    VERIFICATION_DIR
    / "batch_02_manual_retrieval_queue.csv"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "batch_02_full_text_retrieval_diagnostic.csv"
)

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MINIMUM_PDF_BYTES = 5000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    ),
    "Accept": (
        "application/pdf,text/html,"
        "application/xhtml+xml,"
        "application/json;q=0.9,*/*;q=0.5"
    ),
}

DIRECT_URLS = {
    "SCR-0026": [
        "https://zenodo.org/api/records/23205601",
    ],
    "SCR-0033": [
        "https://www.mdpi.com/2227-7099/14/1/28/pdf",
        (
            "https://mdpi-res.com/d_attachment/"
            "economies/economies-14-00028/"
            "article_deploy/"
            "economies-14-00028-v2.pdf"
        ),
    ],
    "SCR-0038": [
        (
            "https://journals.plos.org/"
            "plosone/article/file"
            "?id=10.1371/journal.pone.0208019"
            "&type=printable"
        ),
        (
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/PMC6289438/pdf/"
            "pone.0208019.pdf"
        ),
    ],
    "SCR-0041": [
        (
            "https://www.researchsquare.com/"
            "article/rs-4066275/v1.pdf"
        ),
    ],
}


def clean_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def clean_doi(value):
    doi = clean_text(value)

    if not doi:
        return ""

    if doi.lower() == "nan":
        return ""

    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    )

    return doi.strip()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def is_valid_pdf(path):
    if not path.exists():
        return False

    if path.stat().st_size < MINIMUM_PDF_BYTES:
        return False

    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def unique_urls(urls):
    result = []
    seen = set()

    for url in urls:
        url = clean_text(url)

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)
        result.append(url)

    return result


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def request_json(session, url):
    try:
        response = session.get(
            url,
            timeout=45,
            allow_redirects=True,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def get_openalex_urls(
    session,
    screening_id,
    doi,
    title,
):
    if doi:
        url = (
            "https://api.openalex.org/works/"
            "https://doi.org/"
            f"{quote(doi, safe='')}"
        )
    else:
        url = (
            "https://api.openalex.org/works"
            f"?search={quote(title)}"
            "&per-page=5"
        )

    data = request_json(
        session,
        url,
    )

    if not isinstance(data, dict):
        return []

    save_json(
        RAW_DIR
        / f"{screening_id}_openalex.json",
        data,
    )

    if "results" in data:
        results = data.get(
            "results",
            [],
        )

        if not results:
            return []

        work = results[0]
    else:
        work = data

    urls = []

    open_access = work.get(
        "open_access"
    )

    if isinstance(open_access, dict):
        oa_url = open_access.get(
            "oa_url"
        )

        if oa_url:
            urls.append(oa_url)

    locations = []

    best_location = work.get(
        "best_oa_location"
    )

    if isinstance(best_location, dict):
        locations.append(
            best_location
        )

    primary_location = work.get(
        "primary_location"
    )

    if isinstance(primary_location, dict):
        locations.append(
            primary_location
        )

    locations.extend(
        [
            item
            for item in work.get(
                "locations",
                [],
            )
            if isinstance(item, dict)
        ]
    )

    for location in locations:
        pdf_url = location.get(
            "pdf_url"
        )

        landing_url = location.get(
            "landing_page_url"
        )

        if pdf_url:
            urls.append(pdf_url)

        if landing_url:
            urls.append(landing_url)

    return unique_urls(urls)


def get_crossref_urls(
    session,
    screening_id,
    doi,
):
    if not doi:
        return []

    url = (
        "https://api.crossref.org/works/"
        f"{quote(doi, safe='')}"
    )

    data = request_json(
        session,
        url,
    )

    if not isinstance(data, dict):
        return []

    save_json(
        RAW_DIR
        / f"{screening_id}_crossref.json",
        data,
    )

    message = data.get(
        "message",
        {},
    )

    urls = []

    for link in message.get(
        "link",
        [],
    ):
        link_url = link.get(
            "URL"
        )

        if link_url:
            urls.append(link_url)

    resource = message.get(
        "resource",
        {}
    )

    if isinstance(resource, dict):
        primary = resource.get(
            "primary",
            {}
        )

        if isinstance(primary, dict):
            primary_url = primary.get(
                "URL"
            )

            if primary_url:
                urls.append(primary_url)

    publisher_url = message.get(
        "URL"
    )

    if publisher_url:
        urls.append(publisher_url)

    return unique_urls(urls)


def get_zenodo_file_urls(
    session,
    api_url,
):
    data = request_json(
        session,
        api_url,
    )

    if not isinstance(data, dict):
        return []

    urls = []

    for file_data in data.get(
        "files",
        [],
    ):
        if not isinstance(
            file_data,
            dict,
        ):
            continue

        links = file_data.get(
            "links",
            {}
        )

        if not isinstance(links, dict):
            continue

        content_url = links.get(
            "content"
        )

        self_url = links.get(
            "self"
        )

        if content_url:
            urls.append(content_url)

        if self_url:
            urls.append(self_url)

    return unique_urls(urls)


def get_doi_variants(doi):
    if not doi:
        return []

    lower = doi.lower()

    urls = [
        f"https://doi.org/{doi}",
    ]

    if lower.startswith(
        "10.1371/journal."
    ):
        urls.append(
            "https://journals.plos.org/"
            "plosone/article/file"
            f"?id={doi}&type=printable"
        )

    if lower.startswith(
        "10.1007/"
    ):
        encoded = quote(
            doi,
            safe="",
        )

        urls.append(
            "https://link.springer.com/"
            f"content/pdf/{encoded}.pdf"
        )

    if lower.startswith(
        "10.21203/rs.3.rs-"
    ):
        match = re.search(
            r"rs-(\d+)/(v\d+)",
            lower,
        )

        if match:
            number = match.group(1)
            version = match.group(2)

            urls.append(
                "https://www.researchsquare.com/"
                f"article/rs-{number}/"
                f"{version}.pdf"
            )

    if lower.startswith(
        "10.5281/zenodo."
    ):
        record_id = lower.rsplit(
            ".",
            1,
        )[-1]

        urls.append(
            "https://zenodo.org/api/records/"
            f"{record_id}"
        )

    return unique_urls(urls)


def try_download(
    session,
    screening_id,
    url,
    output_file,
):
    attempt = {
        "SCREENING_ID": screening_id,
        "REQUESTED_URL": url,
        "FINAL_URL": "",
        "HTTP_STATUS": "",
        "CONTENT_TYPE": "",
        "FILE_SIZE_BYTES": 0,
        "PDF_SIGNATURE_VALID": False,
        "DOWNLOAD_RESULT": "",
        "ERROR_MESSAGE": "",
        "ATTEMPT_DATETIME": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    try:
        response = session.get(
            url,
            timeout=70,
            allow_redirects=True,
            headers={
                **HEADERS,
                "Referer": "https://doi.org/",
            },
        )

        content = response.content

        content_type = response.headers.get(
            "content-type",
            "",
        )

        valid = (
            response.status_code == 200
            and len(content)
            >= MINIMUM_PDF_BYTES
            and content.startswith(b"%PDF-")
        )

        attempt[
            "FINAL_URL"
        ] = response.url

        attempt[
            "HTTP_STATUS"
        ] = response.status_code

        attempt[
            "CONTENT_TYPE"
        ] = content_type

        attempt[
            "FILE_SIZE_BYTES"
        ] = len(content)

        attempt[
            "PDF_SIGNATURE_VALID"
        ] = valid

        if valid:
            output_file.write_bytes(
                content
            )

            attempt[
                "DOWNLOAD_RESULT"
            ] = "PDF_SAVED"

            return attempt, True

        if response.status_code == 403:
            result = "BLOCKED_403"

        elif response.status_code == 429:
            result = "RATE_LIMITED_429"

        elif response.status_code == 404:
            result = "NOT_FOUND_404"

        elif response.status_code != 200:
            result = (
                f"HTTP_{response.status_code}"
            )

        else:
            result = "NOT_A_VALID_PDF"

        attempt[
            "DOWNLOAD_RESULT"
        ] = result

        return attempt, False

    except Exception as error:
        attempt[
            "DOWNLOAD_RESULT"
        ] = "REQUEST_ERROR"

        attempt[
            "ERROR_MESSAGE"
        ] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return attempt, False


def main():
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            CANDIDATES_FILE
        )

    candidates = pd.read_csv(
        CANDIDATES_FILE,
        encoding="utf-8-sig",
    )

    if len(candidates) != 18:
        raise ValueError(
            "Se esperaban 18 candidatos "
            f"y se encontraron {len(candidates)}."
        )

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    inventory_rows = []
    attempt_rows = []
    manual_rows = []

    print(
        "===== RECUPERACION DE TEXTOS "
        "COMPLETOS LOTE 02 ====="
    )

    for _, row in candidates.iterrows():
        screening_id = clean_text(
            row["SCREENING_ID"]
        )

        title = clean_text(
            row.get(
                "TITLE",
                "",
            )
        )

        doi = clean_doi(
            row.get(
                "DOI_NORMALIZED",
                "",
            )
        )

        decision = clean_text(
            row.get(
                "REVIEWER_1_DECISION",
                "",
            )
        )

        output_file = (
            PDF_DIR
            / f"{screening_id}_full_text.pdf"
        )

        print()
        print(
            "Procesando:",
            screening_id,
        )

        if is_valid_pdf(
            output_file
        ):
            inventory_rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": title,
                    "DOI_NORMALIZED": doi,
                    "REVIEWER_1_DECISION": decision,
                    "DOWNLOAD_STATUS": (
                        "OK_EXISTING"
                    ),
                    "SOURCE_URL": (
                        "LOCAL_EXISTING_FILE"
                    ),
                    "FINAL_URL": "",
                    "HTTP_STATUS": "",
                    "CONTENT_TYPE": (
                        "application/pdf"
                    ),
                    "FILE_SIZE_BYTES": (
                        output_file.stat().st_size
                    ),
                    "PDF_SIGNATURE_VALID": True,
                    "SHA256": sha256_file(
                        output_file
                    ),
                    "LOCAL_RELATIVE_PATH": str(
                        output_file.relative_to(
                            ROOT
                        )
                    ),
                    "ATTEMPTS_COUNT": 0,
                    "ERROR_SUMMARY": "",
                    "DOWNLOAD_DATETIME": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                }
            )

            print(
                "  Estado: OK_EXISTING"
            )

            continue

        urls = []

        urls.extend(
            DIRECT_URLS.get(
                screening_id,
                [],
            )
        )

        urls.extend(
            get_openalex_urls(
                session,
                screening_id,
                doi,
                title,
            )
        )

        urls.extend(
            get_crossref_urls(
                session,
                screening_id,
                doi,
            )
        )

        urls.extend(
            get_doi_variants(
                doi
            )
        )

        urls = unique_urls(
            urls
        )

        expanded_urls = []

        for url in urls:
            if url.startswith(
                "https://zenodo.org/api/"
            ):
                expanded_urls.extend(
                    get_zenodo_file_urls(
                        session,
                        url,
                    )
                )
            else:
                expanded_urls.append(
                    url
                )

        urls = unique_urls(
            expanded_urls
        )

        saved = False
        successful_attempt = None
        errors = []
        attempts_count = 0

        for url in urls:
            attempts_count += 1

            attempt, success = try_download(
                session,
                screening_id,
                url,
                output_file,
            )

            attempt_rows.append(
                attempt
            )

            if success:
                saved = True
                successful_attempt = attempt
                break

            errors.append(
                (
                    f"{attempt['DOWNLOAD_RESULT']}: "
                    f"{url}"
                )
            )

            time.sleep(0.5)

        if saved:
            inventory_rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": title,
                    "DOI_NORMALIZED": doi,
                    "REVIEWER_1_DECISION": decision,
                    "DOWNLOAD_STATUS": "OK",
                    "SOURCE_URL": (
                        successful_attempt[
                            "REQUESTED_URL"
                        ]
                    ),
                    "FINAL_URL": (
                        successful_attempt[
                            "FINAL_URL"
                        ]
                    ),
                    "HTTP_STATUS": (
                        successful_attempt[
                            "HTTP_STATUS"
                        ]
                    ),
                    "CONTENT_TYPE": (
                        successful_attempt[
                            "CONTENT_TYPE"
                        ]
                    ),
                    "FILE_SIZE_BYTES": (
                        output_file.stat().st_size
                    ),
                    "PDF_SIGNATURE_VALID": True,
                    "SHA256": sha256_file(
                        output_file
                    ),
                    "LOCAL_RELATIVE_PATH": str(
                        output_file.relative_to(
                            ROOT
                        )
                    ),
                    "ATTEMPTS_COUNT": (
                        attempts_count
                    ),
                    "ERROR_SUMMARY": "",
                    "DOWNLOAD_DATETIME": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                }
            )

            print(
                "  Estado: OK"
            )

            print(
                "  Bytes:",
                output_file.stat().st_size,
            )

        else:
            error_summary = " || ".join(
                errors[-8:]
            )

            inventory_rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": title,
                    "DOI_NORMALIZED": doi,
                    "REVIEWER_1_DECISION": decision,
                    "DOWNLOAD_STATUS": (
                        "NOT_RETRIEVED"
                    ),
                    "SOURCE_URL": (
                        " | ".join(urls)
                    ),
                    "FINAL_URL": "",
                    "HTTP_STATUS": "",
                    "CONTENT_TYPE": "",
                    "FILE_SIZE_BYTES": 0,
                    "PDF_SIGNATURE_VALID": False,
                    "SHA256": "",
                    "LOCAL_RELATIVE_PATH": "",
                    "ATTEMPTS_COUNT": (
                        attempts_count
                    ),
                    "ERROR_SUMMARY": (
                        error_summary
                    ),
                    "DOWNLOAD_DATETIME": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                }
            )

            manual_rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": title,
                    "DOI_NORMALIZED": doi,
                    "REVIEWER_1_DECISION": decision,
                    "SUGGESTED_LANDING_PAGE": (
                        f"https://doi.org/{doi}"
                        if doi
                        else ""
                    ),
                    "ATTEMPTED_URLS": (
                        " | ".join(urls)
                    ),
                    "LAST_ERRORS": (
                        error_summary
                    ),
                    "MANUAL_STATUS": (
                        "PENDING_MANUAL_RETRIEVAL"
                    ),
                }
            )

            print(
                "  Estado: NOT_RETRIEVED"
            )

        time.sleep(0.5)

    inventory = pd.DataFrame(
        inventory_rows
    )

    attempts = pd.DataFrame(
        attempt_rows
    )

    manual = pd.DataFrame(
        manual_rows
    )

    inventory.to_csv(
        INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    attempts.to_csv(
        ATTEMPTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    manual.to_csv(
        MANUAL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    retrieved = int(
        inventory[
            "DOWNLOAD_STATUS"
        ].isin(
            [
                "OK",
                "OK_EXISTING",
            ]
        ).sum()
    )

    not_retrieved = int(
        inventory[
            "DOWNLOAD_STATUS"
        ].eq(
            "NOT_RETRIEVED"
        ).sum()
    )

    valid_local_pdfs = 0

    for screening_id in inventory[
        "SCREENING_ID"
    ]:
        path = (
            PDF_DIR
            / f"{screening_id}_full_text.pdf"
        )

        if is_valid_pdf(path):
            valid_local_pdfs += 1

    valid = (
        len(inventory) == 18
        and retrieved
        + not_retrieved
        == 18
        and valid_local_pdfs
        == retrieved
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
                "FULL_TEXT_CANDIDATES": (
                    len(inventory)
                ),
                "AUTOMATICALLY_RETRIEVED": (
                    retrieved
                ),
                "NOT_RETRIEVED": (
                    not_retrieved
                ),
                "VALID_LOCAL_PDFS": (
                    valid_local_pdfs
                ),
                "DOWNLOAD_ATTEMPTS": (
                    len(attempts)
                ),
                "MANUAL_QUEUE_RECORDS": (
                    len(manual)
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
        "===== RESUMEN DE RECUPERACION ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== INVENTARIO ====="
    )

    print(
        inventory[
            [
                "SCREENING_ID",
                "REVIEWER_1_DECISION",
                "DOWNLOAD_STATUS",
                "FILE_SIZE_BYTES",
                "ATTEMPTS_COUNT",
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

    if not valid:
        raise ValueError(
            "La recuperacion automatica "
            "del lote 02 requiere revision."
        )


if __name__ == "__main__":
    main()