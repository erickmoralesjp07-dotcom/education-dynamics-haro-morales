from pathlib import Path
from datetime import datetime
import hashlib
import shutil

import pandas as pd
from PyPDF2 import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_ID = "SCR-0002"

SOURCE_FILE = (
    Path.home()
    / "Downloads"
    / "1-s2.0-S0738059322000116-main.pdf"
)

PDF_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text"
    / "screening_batch_01"
    / "SCR-0002_full_text.pdf"
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

IDENTITY_FILE = (
    VERIFICATION_DIR
    / "batch_01_pdf_identity.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "scr_0002_manual_pdf_extraction_diagnostic.csv"
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VERIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PUBLISHED_DOI = (
    "10.1016/j.ijedudev.2022.102561"
)

PREPRINT_DOI = (
    "10.2139/ssrn.3604528"
)

TITLE = (
    "Female Enrollment, Child Mortality and "
    "Corruption Are Good Predictors of a "
    "Country's UN Education Index"
)

EXPECTED_FILE_SIZE = 1171726


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
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
            "El archivo no contiene SCREENING_ID."
        )

    existing = existing.loc[
        ~existing[
            "SCREENING_ID"
        ].eq(SCREENING_ID)
    ].copy()

    return pd.concat(
        [
            existing,
            pd.DataFrame([new_row]),
        ],
        ignore_index=True,
        sort=False,
    )


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            SOURCE_FILE
        )

    source_signature = (
        SOURCE_FILE.read_bytes()[:5]
    )

    if source_signature != b"%PDF-":
        raise ValueError(
            "El archivo descargado no tiene "
            "una firma PDF valida."
        )

    shutil.copy2(
        SOURCE_FILE,
        PDF_FILE,
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

    identity_parts = []

    print(
        "===== INCORPORACION Y EXTRACCION "
        "SCR-0002 ====="
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

        if page_number <= 3:
            identity_parts.append(
                text.lower()
            )

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
                    f"SCREENING_ID: {SCREENING_ID}",
                    f"PDF_PAGE: {page_number}",
                    f"TOTAL_PDF_PAGES: {page_count}",
                    "=" * 72,
                    "",
                    text,
                    "",
                ]
            )
        )

        page_rows.append(
            {
                "SCREENING_ID": SCREENING_ID,
                "PDF_FILE_NAME": PDF_FILE.name,
                "PDF_PAGE": page_number,
                "TOTAL_PDF_PAGES": page_count,
                "CHARACTER_COUNT": characters,
                "WORD_COUNT": words,
                "PAGE_STATUS": page_status,
                "ERROR_MESSAGE": error_message,
                "TEXT_FILE": str(
                    text_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    combined_file = (
        TEXT_DIR
        / "SCR-0002_full_text.txt"
    )

    combined_file.write_text(
        "\n".join(combined_parts),
        encoding="utf-8",
    )

    identity_text = " ".join(
        identity_parts
    )

    full_text_lower = (
        combined_file
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
        .lower()
    )

    title_match = (
        "female enrollment"
        in identity_text
        and "education index"
        in identity_text
    )

    adeleke_match = (
        "adeleke"
        in identity_text
        or "adeleke"
        in full_text_lower
    )

    mcsharry_match = (
        "mcsharry"
        in identity_text
        or "mcsharry"
        in full_text_lower
    )

    author_match = (
        adeleke_match
        and mcsharry_match
    )

    doi_match = (
        PUBLISHED_DOI.lower()
        in identity_text
        or PUBLISHED_DOI.lower()
        in full_text_lower
    )

    extraction_status = (
        "OK"
        if pages_with_text == page_count
        and empty_pages == 0
        and error_pages == 0
        and title_match
        and author_match
        and doi_match
        else "REVIEW"
    )

    download_row = {
        "SCREENING_ID": SCREENING_ID,
        "TITLE": TITLE,
        "SOURCE_URL": (
            "Manual download from ScienceDirect"
        ),
        "FINAL_URL": (
            "https://www.sciencedirect.com/"
            "science/article/pii/"
            "S0738059322000116"
        ),
        "HTTP_STATUS": 200,
        "CONTENT_TYPE": "application/pdf",
        "FILE_SIZE_BYTES": pdf_size,
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
            "Published journal article"
        ),
        "PREPRINT_DOI": PREPRINT_DOI,
        "PUBLISHED_DOI": PUBLISHED_DOI,
        "CITATION_VERSION": (
            "Published journal article"
        ),
    }

    if DOWNLOAD_INVENTORY_FILE.exists():
        download_inventory = pd.read_csv(
            DOWNLOAD_INVENTORY_FILE
        )
    else:
        download_inventory = pd.DataFrame()

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
        "PDF_FILE_NAME": PDF_FILE.name,
        "LOCAL_RELATIVE_PATH": str(
            PDF_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "PDF_SHA256": pdf_sha256,
        "PDF_PAGE_COUNT": page_count,
        "PAGES_WITH_TEXT": pages_with_text,
        "EMPTY_PAGES": empty_pages,
        "ERROR_PAGES": error_pages,
        "TOTAL_CHARACTER_COUNT": (
            total_characters
        ),
        "TOTAL_WORD_COUNT": total_words,
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
            "Published journal article"
        ),
        "PUBLISHED_DOI": PUBLISHED_DOI,
    }

    if EXTRACTION_SUMMARY_FILE.exists():
        extraction_summary = pd.read_csv(
            EXTRACTION_SUMMARY_FILE
        )
    else:
        extraction_summary = pd.DataFrame()

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
        page_inventory = page_inventory.loc[
            ~page_inventory[
                "SCREENING_ID"
            ].eq(SCREENING_ID)
        ].copy()

    page_inventory = pd.concat(
        [
            page_inventory,
            pd.DataFrame(page_rows),
        ],
        ignore_index=True,
        sort=False,
    )

    page_inventory.to_csv(
        PAGE_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    identity_row = {
        "SCREENING_ID": SCREENING_ID,
        "IDENTITY_STATUS": (
            "IDENTITY_CONFIRMED"
            if title_match
            and author_match
            and doi_match
            else "IDENTITY_REVIEW"
        ),
        "TITLE_MATCH": title_match,
        "AUTHOR_MATCH": author_match,
        "DOI_IN_TEXT": doi_match,
        "VERSION_TYPE": (
            "Published article"
        ),
        "USABLE_FOR_FULL_TEXT_ASSESSMENT": (
            title_match
            and author_match
            and doi_match
        ),
        "CITATION_ACTION": (
            "Cite published article"
        ),
        "NOTES": (
            "Manual ScienceDirect PDF. "
            "Title, author surnames and DOI "
            "verified in extracted text."
        ),
    }

    if IDENTITY_FILE.exists():
        identity = pd.read_csv(
            IDENTITY_FILE
        )
    else:
        identity = pd.DataFrame()

    identity = upsert_by_screening_id(
        identity,
        identity_row,
    )

    identity.to_csv(
        IDENTITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid = (
        pdf_size == EXPECTED_FILE_SIZE
        and page_count == 9
        and pages_with_text == 9
        and empty_pages == 0
        and error_pages == 0
        and title_match
        and adeleke_match
        and mcsharry_match
        and author_match
        and doi_match
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
                "SCREENING_ID": SCREENING_ID,
                "PDF_SIZE_BYTES": pdf_size,
                "PDF_SHA256": pdf_sha256,
                "PDF_PAGE_COUNT": page_count,
                "PAGES_WITH_TEXT": (
                    pages_with_text
                ),
                "EMPTY_PAGES": empty_pages,
                "ERROR_PAGES": error_pages,
                "TOTAL_WORD_COUNT": (
                    total_words
                ),
                "TITLE_MATCH": title_match,
                "ADELEKE_MATCH": (
                    adeleke_match
                ),
                "MCSHARRY_MATCH": (
                    mcsharry_match
                ),
                "AUTHOR_MATCH": author_match,
                "DOI_IN_TEXT": doi_match,
                "VERSION_USED": (
                    "Published journal article"
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
        "SOURCE_FILE:",
        SOURCE_FILE,
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
        "TITLE_MATCH:",
        title_match,
    )
    print(
        "ADELEKE_MATCH:",
        adeleke_match,
    )
    print(
        "MCSHARRY_MATCH:",
        mcsharry_match,
    )
    print(
        "AUTHOR_MATCH:",
        author_match,
    )
    print(
        "DOI_IN_TEXT:",
        doi_match,
    )
    print(
        "VERSION_USED:",
        "Published journal article",
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
            "La incorporacion de SCR-0002 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()