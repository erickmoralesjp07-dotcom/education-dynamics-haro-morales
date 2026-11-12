from pathlib import Path
from datetime import datetime
import hashlib
import re
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
    / "batch_01_candidate_verification.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text"
    / "screening_batch_01"
)

INVENTORY_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
    / "batch_01_download_inventory.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "batch_01_download_diagnostic.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

USER_AGENT = (
    "Mozilla/5.0 "
    "EducationDynamicsHaroMorales/0.1 "
    "Academic full-text retrieval"
)

MINIMUM_PDF_BYTES = 5000


def safe_filename(value):
    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip("_")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def request_with_retry(
    session,
    url,
    attempts=4,
):
    response = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        response = session.get(
            url,
            timeout=120,
            allow_redirects=True,
        )

        if response.status_code not in {
            429,
            500,
            502,
            503,
            504,
        }:
            break

        retry_after = (
            response.headers.get(
                "Retry-After"
            )
        )

        wait_seconds = (
            float(retry_after)
            if retry_after
            else float(5 * attempt)
        )

        print(
            f"  Estado {response.status_code}. "
            f"Reintento {attempt}/{attempts} "
            f"en {wait_seconds:.1f}s."
        )

        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(
            "No se obtuvo respuesta."
        )

    return response


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    verification = pd.read_csv(
        INPUT_FILE
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "application/pdf,"
                "application/octet-stream;q=0.9,"
                "*/*;q=0.5"
            ),
        }
    )

    rows = []

    candidates = verification.loc[
        verification[
            "OPENALEX_BEST_OA_PDF_URL"
        ]
        .fillna("")
        .str.strip()
        .ne("")
    ].copy()

    print(
        "===== DESCARGA DE TEXTOS "
        "COMPLETOS LOTE 01 ====="
    )

    print(
        "Candidatos con URL:",
        len(candidates),
    )

    for _, row in candidates.iterrows():
        screening_id = str(
            row["SCREENING_ID"]
        )

        url = str(
            row[
                "OPENALEX_BEST_OA_PDF_URL"
            ]
        ).strip()

        title = str(
            row["ORIGINAL_TITLE"]
        )

        output_file = (
            OUTPUT_DIR
            / (
                f"{safe_filename(screening_id)}"
                f"_full_text.pdf"
            )
        )

        print()
        print(
            f"Descargando {screening_id}: "
            f"{title[:75]}"
        )

        status = "ERROR"
        error_message = ""
        http_status = None
        content_type = ""
        final_url = ""
        file_size = 0
        pdf_signature = False
        file_hash = ""

        try:
            response = request_with_retry(
                session,
                url,
            )

            http_status = (
                response.status_code
            )

            content_type = str(
                response.headers.get(
                    "Content-Type",
                    "",
                )
            )

            final_url = str(
                response.url
            )

            content = response.content

            file_size = len(content)

            pdf_signature = (
                content[:5] == b"%PDF-"
            )

            if response.status_code != 200:
                raise ValueError(
                    "HTTP status "
                    f"{response.status_code}"
                )

            if not pdf_signature:
                raise ValueError(
                    "La respuesta no tiene "
                    "firma PDF."
                )

            if file_size < MINIMUM_PDF_BYTES:
                raise ValueError(
                    "PDF demasiado pequeno: "
                    f"{file_size} bytes."
                )

            output_file.write_bytes(
                content
            )

            file_hash = sha256_file(
                output_file
            )

            status = "OK"

        except Exception as error:
            error_message = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            if output_file.exists():
                output_file.unlink()

            print(
                "  ERROR:",
                error_message,
            )

        else:
            print(
                "  OK:",
                output_file.name,
                "KB=",
                round(
                    file_size / 1024,
                    2,
                ),
            )

        rows.append(
            {
                "SCREENING_ID": (
                    screening_id
                ),
                "TITLE": title,
                "SOURCE_URL": url,
                "FINAL_URL": final_url,
                "HTTP_STATUS": (
                    http_status
                ),
                "CONTENT_TYPE": (
                    content_type
                ),
                "FILE_SIZE_BYTES": (
                    file_size
                ),
                "PDF_SIGNATURE_VALID": (
                    pdf_signature
                ),
                "SHA256": file_hash,
                "LOCAL_RELATIVE_PATH": (
                    str(
                        output_file.relative_to(
                            PROJECT_ROOT
                        )
                    )
                    if status == "OK"
                    else ""
                ),
                "DOWNLOAD_STATUS": status,
                "ERROR_MESSAGE": (
                    error_message
                ),
                "DOWNLOAD_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
            }
        )

        time.sleep(1.0)

    inventory = pd.DataFrame(
        rows
    )

    inventory.to_csv(
        INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    success_count = int(
        inventory[
            "DOWNLOAD_STATUS"
        ].eq("OK").sum()
    )

    failure_count = int(
        inventory[
            "DOWNLOAD_STATUS"
        ].eq("ERROR").sum()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "URL_CANDIDATES": (
                    len(candidates)
                ),
                "SUCCESSFUL_DOWNLOADS": (
                    success_count
                ),
                "FAILED_DOWNLOADS": (
                    failure_count
                ),
                "VALID_PDF_SIGNATURES": int(
                    inventory[
                        "PDF_SIGNATURE_VALID"
                    ].sum()
                ),
                "TOTAL_DOWNLOADED_MB": (
                    round(
                        inventory.loc[
                            inventory[
                                "DOWNLOAD_STATUS"
                            ].eq("OK"),
                            "FILE_SIZE_BYTES",
                        ].sum()
                        / 1024
                        / 1024,
                        3,
                    )
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(candidates) == 13
                    and success_count > 0
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
        "===== DIAGNOSTICO DE DESCARGA ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== INVENTARIO DE DESCARGA ====="
    )

    display_columns = [
        "SCREENING_ID",
        "HTTP_STATUS",
        "CONTENT_TYPE",
        "FILE_SIZE_BYTES",
        "PDF_SIGNATURE_VALID",
        "DOWNLOAD_STATUS",
        "ERROR_MESSAGE",
    ]

    print(
        inventory[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Inventario:")
    print(INVENTORY_FILE)

    print()
    print("PDF locales:")
    print(OUTPUT_DIR)

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