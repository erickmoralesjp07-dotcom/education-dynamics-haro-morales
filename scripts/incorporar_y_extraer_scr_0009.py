from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
from PyPDF2 import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_ID = "SCR-0009"

PDF_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text"
    / "screening_batch_01"
    / "SCR-0009_full_text.pdf"
)

TEXT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "extracted_text"
    / "screening_batch_01"
)

VERIFICATION_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
)

DOWNLOAD_INVENTORY_FILE = (
    VERIFICATION_DIR
    / "batch_01_download_inventory.csv"
)

EXTRACTION_SUMMARY_FILE = (
    VERIFICATION_DIR
    / "batch_01_full_text_extraction_summary.csv"
)

PAGE_INVENTORY_FILE = (
    VERIFICATION_DIR
    / "batch_01_full_text_page_inventory.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "scr_0009_preprint_extraction_diagnostic.csv"
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VERIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PREPRINT_URL = (
    "https://www.researchsquare.com/"
    "article/rs-3709698/latest.pdf"
)

PUBLISHED_DOI = "10.1111/roiw.70040"

PREPRINT_DOI = (
    "10.21203/rs.3.rs-3709698/v1"
)

TITLE = (
    "Within-School Wealth Inequality and "
    "Learning Achievement: Evidence From PISA"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def word_count(text):
    return len(
        [
            word
            for word in text.split()
            if word.strip()
        ]
    )


def upsert_by_screening_id(
    existing,
    new_row,
):
    if existing.empty:
        return pd.DataFrame(
            [new_row]
        )

    if "SCREENING_ID" not in existing.columns:
        raise ValueError(
            "El archivo no contiene "
            "SCREENING_ID."
        )

    existing = existing.loc[
        ~existing[
            "SCREENING_ID"
        ].eq(SCREENING_ID)
    ].copy()

    return pd.concat(
        [
            existing,
            pd.DataFrame(
                [new_row]
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            PDF_FILE
        )

    signature = PDF_FILE.read_bytes()[
        :5
    ]

    if signature != b"%PDF-":
        raise ValueError(
            "El archivo no tiene firma PDF."
        )

    pdf_size = PDF_FILE.stat().st_size
    pdf_sha256 = sha256_file(
        PDF_FILE
    )

    reader = PdfReader(
        str(PDF_FILE)
    )

    page_count = len(
        reader.pages
    )

    page_rows = []
    combined_parts = []

    pages_with_text = 0
    empty_pages = 0
    error_pages = 0
    total_characters = 0
    total_words = 0

    print(
        "===== INCORPORACION Y EXTRACCION "
        "SCR-0009 ====="
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_status = "OK"
        error_message = ""

        try:
            text = (
                page.extract_text()
                or ""
            ).strip()

        except Exception as error:
            text = ""
            page_status = "ERROR"
            error_message = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            error_pages += 1

        characters = len(text)
        words = word_count(text)

        if text:
            pages_with_text += 1
        elif page_status != "ERROR":
            empty_pages += 1

        total_characters += characters
        total_words += words

        text_file = (
            TEXT_DIR
            / (
                f"{SCREENING_ID}_"
                f"page_{page_number:03d}.txt"
            )
        )

        text_file.write_text(
            text,
            encoding="utf-8",
        )

        combined_parts.append(
            "\n".join(
                [
                    "=" * 72,
                    (
                        "SCREENING_ID: "
                        f"{SCREENING_ID}"
                    ),
                    (
                        "PDF_PAGE: "
                        f"{page_number}"
                    ),
                    (
                        "TOTAL_PDF_PAGES: "
                        f"{page_count}"
                    ),
                    "=" * 72,
                    "",
                    text,
                    "",
                ]
            )
        )

        page_rows.append(
            {
                "SCREENING_ID": (
                    SCREENING_ID
                ),
                "PDF_FILE_NAME": (
                    PDF_FILE.name
                ),
                "PDF_PAGE": (
                    page_number
                ),
                "TOTAL_PDF_PAGES": (
                    page_count
                ),
                "CHARACTER_COUNT": (
                    characters
                ),
                "WORD_COUNT": (
                    words
                ),
                "PAGE_STATUS": (
                    page_status
                ),
                "ERROR_MESSAGE": (
                    error_message
                ),
                "TEXT_FILE": str(
                    text_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    combined_file = (
        TEXT_DIR
        / (
            f"{SCREENING_ID}_"
            "full_text.txt"
        )
    )

    combined_file.write_text(
        "\n".join(
            combined_parts
        ),
        encoding="utf-8",
    )

    extraction_status = (
        "OK"
        if pages_with_text > 0
        and error_pages == 0
        else "REVIEW"
    )

    download_row = {
        "SCREENING_ID": SCREENING_ID,
        "TITLE": TITLE,
        "SOURCE_URL": PREPRINT_URL,
        "FINAL_URL": PREPRINT_URL,
        "HTTP_STATUS": 200,
        "CONTENT_TYPE": (
            "application/pdf"
        ),
        "FILE_SIZE_BYTES": (
            pdf_size
        ),
        "PDF_SIGNATURE_VALID": True,
        "SHA256": pdf_sha256,
        "LOCAL_RELATIVE_PATH": str(
            PDF_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "DOWNLOAD_STATUS": "OK",
        "ERROR_MESSAGE": "",
        "DOWNLOAD_DATETIME": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "VERSION_USED_FOR_ASSESSMENT": (
            "Research Square preprint v1"
        ),
        "PREPRINT_DOI": PREPRINT_DOI,
        "PUBLISHED_DOI": PUBLISHED_DOI,
        "CITATION_VERSION": (
            "Published journal article"
        ),
    }

    if (
        DOWNLOAD_INVENTORY_FILE.exists()
    ):
        download_inventory = (
            pd.read_csv(
                DOWNLOAD_INVENTORY_FILE
            )
        )
    else:
        download_inventory = (
            pd.DataFrame()
        )

    download_inventory = (
        upsert_by_screening_id(
            download_inventory,
            download_row,
        )
    )

    download_inventory.to_csv(
        DOWNLOAD_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    extraction_row = {
        "SCREENING_ID": SCREENING_ID,
        "PDF_FILE_NAME": (
            PDF_FILE.name
        ),
        "LOCAL_RELATIVE_PATH": str(
            PDF_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "PDF_SHA256": pdf_sha256,
        "PDF_PAGE_COUNT": page_count,
        "PAGES_WITH_TEXT": (
            pages_with_text
        ),
        "EMPTY_PAGES": empty_pages,
        "ERROR_PAGES": error_pages,
        "TOTAL_CHARACTER_COUNT": (
            total_characters
        ),
        "TOTAL_WORD_COUNT": (
            total_words
        ),
        "COMBINED_TEXT_FILE": str(
            combined_file.relative_to(
                PROJECT_ROOT
            )
        ),
        "EXTRACTION_STATUS": (
            extraction_status
        ),
        "EXTRACTION_DATETIME": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "VERSION_USED_FOR_ASSESSMENT": (
            "Research Square preprint v1"
        ),
        "PUBLISHED_DOI": (
            PUBLISHED_DOI
        ),
    }

    if (
        EXTRACTION_SUMMARY_FILE.exists()
    ):
        extraction_summary = (
            pd.read_csv(
                EXTRACTION_SUMMARY_FILE
            )
        )
    else:
        extraction_summary = (
            pd.DataFrame()
        )

    extraction_summary = (
        upsert_by_screening_id(
            extraction_summary,
            extraction_row,
        )
    )

    extraction_summary.to_csv(
        EXTRACTION_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if PAGE_INVENTORY_FILE.exists():
        page_inventory = pd.read_csv(
            PAGE_INVENTORY_FILE
        )
    else:
        page_inventory = pd.DataFrame()

    if (
        not page_inventory.empty
        and "SCREENING_ID"
        in page_inventory.columns
    ):
        page_inventory = (
            page_inventory.loc[
                ~page_inventory[
                    "SCREENING_ID"
                ].eq(SCREENING_ID)
            ].copy()
        )

    page_inventory = pd.concat(
        [
            page_inventory,
            pd.DataFrame(
                page_rows
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    page_inventory.to_csv(
        PAGE_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        pdf_size == 329127
        and page_count > 0
        and pages_with_text > 0
        and error_pages == 0
        and extraction_status == "OK"
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "SCREENING_ID": (
                    SCREENING_ID
                ),
                "PDF_SIZE_BYTES": (
                    pdf_size
                ),
                "PDF_SHA256": (
                    pdf_sha256
                ),
                "PDF_PAGE_COUNT": (
                    page_count
                ),
                "PAGES_WITH_TEXT": (
                    pages_with_text
                ),
                "EMPTY_PAGES": (
                    empty_pages
                ),
                "ERROR_PAGES": (
                    error_pages
                ),
                "TOTAL_WORD_COUNT": (
                    total_words
                ),
                "VERSION_USED": (
                    "Research Square "
                    "preprint v1"
                ),
                "CITATION_DOI": (
                    PUBLISHED_DOI
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

    print(
        "PDF_SIZE_BYTES:",
        pdf_size,
    )

    print(
        "PDF_SHA256:",
        pdf_sha256,
    )

    print(
        "PDF_PAGE_COUNT:",
        page_count,
    )

    print(
        "PAGES_WITH_TEXT:",
        pages_with_text,
    )

    print(
        "EMPTY_PAGES:",
        empty_pages,
    )

    print(
        "ERROR_PAGES:",
        error_pages,
    )

    print(
        "TOTAL_WORD_COUNT:",
        total_words,
    )

    print(
        "VERSION_USED:",
        "Research Square preprint v1",
    )

    print(
        "CITATION_DOI:",
        PUBLISHED_DOI,
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "La incorporacion de SCR-0009 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()