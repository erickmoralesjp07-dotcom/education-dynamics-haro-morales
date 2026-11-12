from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
from PyPDF2 import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
    / "batch_01_download_inventory.csv"
)

TEXT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "extracted_text"
    / "screening_batch_01"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text_verification"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            INVENTORY_FILE
        )

    inventory = pd.read_csv(
        INVENTORY_FILE
    )

    successful = inventory.loc[
        inventory[
            "DOWNLOAD_STATUS"
        ].eq("OK")
    ].copy()

    if len(successful) != 6:
        raise ValueError(
            "Se esperaban 6 PDF validos "
            f"y se encontraron {len(successful)}."
        )

    document_rows = []
    page_rows = []

    print(
        "===== EXTRACCION DE TEXTOS "
        "COMPLETOS LOTE 01 ====="
    )

    for _, row in successful.iterrows():
        screening_id = str(
            row["SCREENING_ID"]
        )

        pdf_path = (
            PROJECT_ROOT
            / str(
                row[
                    "LOCAL_RELATIVE_PATH"
                ]
            )
        )

        if not pdf_path.exists():
            raise FileNotFoundError(
                pdf_path
            )

        print()
        print(
            f"Extrayendo {screening_id}: "
            f"{pdf_path.name}"
        )

        reader = PdfReader(
            str(pdf_path)
        )

        page_count = len(
            reader.pages
        )

        combined_parts = []

        pages_with_text = 0
        empty_pages = 0
        error_pages = 0

        total_characters = 0
        total_words = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            status = "OK"
            error_message = ""

            try:
                text = (
                    page.extract_text()
                    or ""
                )

            except Exception as error:
                text = ""
                status = "ERROR"
                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                error_pages += 1

            text = text.strip()

            characters = len(text)
            words = word_count(text)

            if text:
                pages_with_text += 1
            elif status != "ERROR":
                empty_pages += 1

            total_characters += (
                characters
            )

            total_words += words

            page_file = (
                TEXT_DIR
                / (
                    f"{screening_id}_"
                    f"page_{page_number:03d}.txt"
                )
            )

            page_file.write_text(
                text,
                encoding="utf-8",
            )

            combined_parts.append(
                "\n".join(
                    [
                        "=" * 72,
                        f"SCREENING_ID: {screening_id}",
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
                    "SCREENING_ID": (
                        screening_id
                    ),
                    "PDF_FILE_NAME": (
                        pdf_path.name
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
                    "PAGE_STATUS": status,
                    "ERROR_MESSAGE": (
                        error_message
                    ),
                    "TEXT_FILE": str(
                        page_file.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                }
            )

        combined_file = (
            TEXT_DIR
            / (
                f"{screening_id}_"
                f"full_text.txt"
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
            if error_pages == 0
            and pages_with_text > 0
            else "REVIEW"
        )

        document_rows.append(
            {
                "SCREENING_ID": (
                    screening_id
                ),
                "PDF_FILE_NAME": (
                    pdf_path.name
                ),
                "LOCAL_RELATIVE_PATH": str(
                    pdf_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "PDF_SHA256": (
                    sha256_file(
                        pdf_path
                    )
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
            }
        )

        print(
            "  Paginas:",
            page_count,
        )

        print(
            "  Paginas con texto:",
            pages_with_text,
        )

        print(
            "  Palabras:",
            total_words,
        )

        print(
            "  Estado:",
            extraction_status,
        )

    document_summary = pd.DataFrame(
        document_rows
    )

    page_inventory = pd.DataFrame(
        page_rows
    )

    document_file = (
        OUTPUT_DIR
        / "batch_01_full_text_extraction_summary.csv"
    )

    page_file = (
        OUTPUT_DIR
        / "batch_01_full_text_page_inventory.csv"
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "batch_01_full_text_extraction_diagnostic.csv"
    )

    document_summary.to_csv(
        document_file,
        index=False,
        encoding="utf-8-sig",
    )

    page_inventory.to_csv(
        page_file,
        index=False,
        encoding="utf-8-sig",
    )

    valid_documents = int(
        document_summary[
            "EXTRACTION_STATUS"
        ].eq("OK").sum()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "DOCUMENT_COUNT": (
                    len(document_summary)
                ),
                "VALID_DOCUMENTS": (
                    valid_documents
                ),
                "TOTAL_PAGES": int(
                    document_summary[
                        "PDF_PAGE_COUNT"
                    ].sum()
                ),
                "PAGES_WITH_TEXT": int(
                    document_summary[
                        "PAGES_WITH_TEXT"
                    ].sum()
                ),
                "EMPTY_PAGES": int(
                    document_summary[
                        "EMPTY_PAGES"
                    ].sum()
                ),
                "ERROR_PAGES": int(
                    document_summary[
                        "ERROR_PAGES"
                    ].sum()
                ),
                "TOTAL_WORDS": int(
                    document_summary[
                        "TOTAL_WORD_COUNT"
                    ].sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if len(
                        document_summary
                    ) == 6
                    and valid_documents == 6
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
    print(
        "===== RESUMEN DE EXTRACCION ====="
    )

    print(
        document_summary[
            [
                "SCREENING_ID",
                "PDF_PAGE_COUNT",
                "PAGES_WITH_TEXT",
                "EMPTY_PAGES",
                "ERROR_PAGES",
                "TOTAL_WORD_COUNT",
                "EXTRACTION_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== DIAGNOSTICO ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(document_file)
    print(page_file)
    print(diagnostic_file)
    print(TEXT_DIR)

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