from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

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

RESULTS_DIR = (
    ROOT
    / "results"
    / "diagnostics"
)

REPORT_FILE = (
    VERIFICATION_DIR
    / "batch_02_targeted_retrieval_report.csv"
)

DIAGNOSTIC_FILE = (
    RESULTS_DIR
    / "batch_02_targeted_retrieval_diagnostic.csv"
)

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VERIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCES = {
    "SCR-0027": {
        "title": (
            "The geography of tertiary education "
            "inequality in Europe"
        ),
        "version": "OPEN_WORKING_PAPER_VERSION",
        "urls": [
            (
                "https://zenodo.org/records/10401330/files/"
                "D3.2_Geography_of_tertiary_education_"
                "inequalities_over_time_v2.1.pdf"
            ),
        ],
    },
    "SCR-0029": {
        "title": (
            "Does schooling actually perpetuate "
            "educational inequality in mathematics performance?"
        ),
        "version": "PUBLISHED_VERSION",
        "urls": [
            (
                "https://rd.springer.com/content/pdf/"
                "10.1007/s11858-018-0935-3.pdf"
            ),
        ],
    },
    "SCR-0037": {
        "title": (
            "Trends and Determinants of Intergenerational "
            "Educational Inequality in Sub-Saharan Africa"
        ),
        "version": "PUBLISHED_VERSION",
        "urls": [
            (
                "https://madoc.bib.uni-mannheim.de/"
                "66950/1/"
                "plavgo-bernardi-2023-trends-and-"
                "determinants-of-intergenerational-"
                "educational-inequality-in-sub-"
                "saharan-africa-for.pdf"
            ),
        ],
    },
    "SCR-0039": {
        "title": (
            "Differences in educational attainment "
            "by country of origin"
        ),
        "version": "OPEN_WORKING_PAPER_VERSION",
        "urls": [
            (
                "https://www.monash.edu/business/"
                "economics/our-research/working-papers/"
                "files/2017/"
                "0517differencesparasnisswan.pdf"
            ),
        ],
    },
    "SCR-0049": {
        "title": (
            "Correlation Between Corruption and "
            "Education in Developing Countries"
        ),
        "version": "CONFERENCE_POSTER",
        "urls": [
            (
                "https://core.ac.uk/download/"
                "pdf/84290765.pdf"
            ),
        ],
    },
    "SCR-0050": {
        "title": (
            "Determinants of educational attainment "
            "in Egypt and MENA"
        ),
        "version": "PHD_THESIS",
        "urls": [
            (
                "https://core.ac.uk/download/"
                "pdf/33565871.pdf"
            ),
            (
                "http://eprints.nottingham.ac.uk/"
                "12947/1/MB_THESIS_Final.pdf"
            ),
        ],
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/130.0 Safari/537.36"
    ),
    "Accept": (
        "application/pdf,"
        "application/octet-stream;q=0.9,"
        "*/*;q=0.5"
    ),
}


def valid_pdf(content):
    return (
        len(content) > 5000
        and content.startswith(b"%PDF-")
    )


def sha256_bytes(content):
    return hashlib.sha256(
        content
    ).hexdigest()


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    rows = []

    print(
        "===== RECUPERACION ESPECIFICA "
        "LOTE 02 ====="
    )

    for screening_id, source in (
        SOURCES.items()
    ):
        output_file = (
            PDF_DIR
            / f"{screening_id}_full_text.pdf"
        )

        print()
        print(
            "Procesando:",
            screening_id,
        )

        if (
            output_file.exists()
            and output_file.stat().st_size > 5000
            and output_file.read_bytes()[:5] == b"%PDF-"
        ):
            content = output_file.read_bytes()

            rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": source["title"],
                    "VERSION_STATUS": source["version"],
                    "SOURCE_URL": "LOCAL_EXISTING_FILE",
                    "FINAL_URL": "",
                    "HTTP_STATUS": "",
                    "CONTENT_TYPE": "application/pdf",
                    "FILE_SIZE_BYTES": len(content),
                    "PDF_VALID": True,
                    "SHA256": sha256_bytes(content),
                    "LOCAL_RELATIVE_PATH": str(
                        output_file.relative_to(ROOT)
                    ),
                    "RETRIEVAL_STATUS": "OK_EXISTING",
                    "ERROR_MESSAGE": "",
                    "RETRIEVAL_DATETIME": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                }
            )

            print("  Estado: OK_EXISTING")
            continue

        saved = False
        errors = []

        for url in source["urls"]:
            try:
                response = session.get(
                    url,
                    timeout=120,
                    allow_redirects=True,
                )

                content = response.content

                if (
                    response.status_code == 200
                    and valid_pdf(content)
                ):
                    output_file.write_bytes(content)

                    rows.append(
                        {
                            "SCREENING_ID": screening_id,
                            "TITLE": source["title"],
                            "VERSION_STATUS": source["version"],
                            "SOURCE_URL": url,
                            "FINAL_URL": response.url,
                            "HTTP_STATUS": response.status_code,
                            "CONTENT_TYPE": (
                                response.headers.get(
                                    "content-type",
                                    "",
                                )
                            ),
                            "FILE_SIZE_BYTES": len(content),
                            "PDF_VALID": True,
                            "SHA256": sha256_bytes(content),
                            "LOCAL_RELATIVE_PATH": str(
                                output_file.relative_to(ROOT)
                            ),
                            "RETRIEVAL_STATUS": "OK",
                            "ERROR_MESSAGE": "",
                            "RETRIEVAL_DATETIME": (
                                datetime.now()
                                .astimezone()
                                .isoformat()
                            ),
                        }
                    )

                    print("  Estado: OK")
                    print("  Bytes:", len(content))

                    saved = True
                    break

                errors.append(
                    (
                        f"HTTP={response.status_code}; "
                        f"TYPE={response.headers.get('content-type', '')}; "
                        f"BYTES={len(content)}; URL={url}"
                    )
                )

            except Exception as error:
                errors.append(
                    (
                        f"{type(error).__name__}: "
                        f"{error}; URL={url}"
                    )
                )

        if not saved:
            rows.append(
                {
                    "SCREENING_ID": screening_id,
                    "TITLE": source["title"],
                    "VERSION_STATUS": source["version"],
                    "SOURCE_URL": " | ".join(source["urls"]),
                    "FINAL_URL": "",
                    "HTTP_STATUS": "",
                    "CONTENT_TYPE": "",
                    "FILE_SIZE_BYTES": 0,
                    "PDF_VALID": False,
                    "SHA256": "",
                    "LOCAL_RELATIVE_PATH": "",
                    "RETRIEVAL_STATUS": "NOT_RETRIEVED",
                    "ERROR_MESSAGE": " || ".join(errors),
                    "RETRIEVAL_DATETIME": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                }
            )

            print("  Estado: NOT_RETRIEVED")

    report = pd.DataFrame(rows)

    report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    retrieved = int(
        report[
            "RETRIEVAL_STATUS"
        ].isin(
            [
                "OK",
                "OK_EXISTING",
            ]
        ).sum()
    )

    not_retrieved = int(
        report[
            "RETRIEVAL_STATUS"
        ].eq("NOT_RETRIEVED").sum()
    )

    valid = (
        len(report) == 6
        and retrieved + not_retrieved == 6
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "REQUESTED_DOCUMENTS": 6,
                "RETRIEVED_DOCUMENTS": retrieved,
                "NOT_RETRIEVED": not_retrieved,
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
        "===== RESUMEN DE RECUPERACION "
        "ESPECIFICA ====="
    )

    print(
        report[
            [
                "SCREENING_ID",
                "VERSION_STATUS",
                "RETRIEVAL_STATUS",
                "FILE_SIZE_BYTES",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        diagnostic.to_string(index=False)
    )

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