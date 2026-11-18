from pathlib import Path
from datetime import datetime
import hashlib
import re

import pandas as pd
from PyPDF2 import PdfReader


ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = (
    ROOT
    / "literature_review"
    / "full_text"
    / "screening_batch_02"
)

TEXT_DIR = (
    ROOT
    / "literature_review"
    / "extracted_text"
    / "screening_batch_02"
)

VERIFICATION_DIR = (
    ROOT
    / "literature_review"
    / "full_text_verification"
)

INVENTORY_FILE = (
    VERIFICATION_DIR
    / "batch_02_download_inventory.csv"
)

SUMMARY_FILE = (
    VERIFICATION_DIR
    / "batch_02_full_text_extraction_summary.csv"
)

PAGE_FILE = (
    VERIFICATION_DIR
    / "batch_02_full_text_page_inventory.csv"
)

IDENTITY_FILE = (
    VERIFICATION_DIR
    / "batch_02_pdf_identity.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "batch_02_full_text_extraction_diagnostic.csv"
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_IDS = {
    "SCR-0026",
    "SCR-0028",
    "SCR-0030",
    "SCR-0035",
    "SCR-0038",
    "SCR-0040",
    "SCR-0041",
    "SCR-0042",
    "SCR-0044",
}


def clean_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


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
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def title_terms(title):
    words = re.findall(
        r"[A-Za-zÀ-ÿ0-9]+",
        title.lower(),
    )

    ignored = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "into",
        "using",
        "between",
        "across",
        "evidence",
        "study",
        "analysis",
        "empirical",
        "results",
        "data",
        "educational",
        "education",
    }

    useful = [
        word
        for word in words
        if len(word) >= 5
        and word not in ignored
    ]

    return useful[:8]


def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            INVENTORY_FILE
        )

    inventory = pd.read_csv(
        INVENTORY_FILE,
        encoding="utf-8-sig",
    )

    retrieved = inventory.loc[
        inventory[
            "DOWNLOAD_STATUS"
        ].isin(
            [
                "OK",
                "OK_EXISTING",
            ]
        )
    ].copy()

    retrieved_ids = set(
        retrieved[
            "SCREENING_ID"
        ].astype(str)
    )

    if retrieved_ids != EXPECTED_IDS:
        raise ValueError(
            "Los PDF recuperados no coinciden. "
            f"Esperados: {sorted(EXPECTED_IDS)}; "
            f"encontrados: {sorted(retrieved_ids)}"
        )

    summary_rows = []
    page_rows = []
    identity_rows = []

    print(
        "===== EXTRACCION DE TEXTOS "
        "RECUPERADOS LOTE 02 ====="
    )

    for _, row in retrieved.iterrows():
        screening_id = clean_text(
            row["SCREENING_ID"]
        )

        title = clean_text(
            row.get(
                "TITLE",
                "",
            )
        )

        doi = clean_text(
            row.get(
                "DOI_NORMALIZED",
                "",
            )
        )

        pdf_file = (
            PDF_DIR
            / f"{screening_id}_full_text.pdf"
        )

        if not pdf_file.exists():
            raise FileNotFoundError(
                pdf_file
            )

        if pdf_file.read_bytes()[:5] != b"%PDF-":
            raise ValueError(
                f"{pdf_file} no tiene firma PDF."
            )

        reader = PdfReader(
            str(pdf_file)
        )

        total_pages = len(
            reader.pages
        )

        all_parts = []
        first_pages = []

        pages_with_text = 0
        empty_pages = 0
        error_pages = 0
        total_characters = 0
        total_words = 0

        print()
        print(
            "Extrayendo:",
            screening_id,
        )

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
                ).strip()

            except Exception as error:
                text = ""
                status = "ERROR"
                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )
                error_pages += 1

            characters = len(text)
            words = word_count(text)

            if text:
                pages_with_text += 1
            elif status != "ERROR":
                empty_pages += 1

            total_characters += characters
            total_words += words

            if page_number <= 3:
                first_pages.append(
                    text.lower()
                )

            page_text_file = (
                TEXT_DIR
                / (
                    f"{screening_id}_"
                    f"page_{page_number:03d}.txt"
                )
            )

            page_text_file.write_text(
                text,
                encoding="utf-8",
            )

            all_parts.append(
                "\n".join(
                    [
                        "=" * 72,
                        (
                            "SCREENING_ID: "
                            f"{screening_id}"
                        ),
                        (
                            "PDF_PAGE: "
                            f"{page_number}"
                        ),
                        (
                            "TOTAL_PDF_PAGES: "
                            f"{total_pages}"
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
                        screening_id
                    ),
                    "PDF_FILE_NAME": (
                        pdf_file.name
                    ),
                    "PDF_PAGE": (
                        page_number
                    ),
                    "TOTAL_PDF_PAGES": (
                        total_pages
                    ),
                    "CHARACTER_COUNT": (
                        characters
                    ),
                    "WORD_COUNT": (
                        words
                    ),
                    "PAGE_STATUS": (
                        status
                    ),
                    "ERROR_MESSAGE": (
                        error_message
                    ),
                    "TEXT_FILE": str(
                        page_text_file.relative_to(
                            ROOT
                        )
                    ),
                }
            )

        combined_file = (
            TEXT_DIR
            / f"{screening_id}_full_text.txt"
        )

        combined_text = "\n".join(
            all_parts
        )

        combined_file.write_text(
            combined_text,
            encoding="utf-8",
        )

        opening_text = " ".join(
            first_pages
        )

        terms = title_terms(
            title
        )

        matched_terms = [
            term
            for term in terms
            if term in opening_text
            or term in combined_text.lower()
        ]

        minimum_matches = min(
            2,
            len(terms),
        )

        title_match = (
            len(matched_terms)
            >= minimum_matches
            if terms
            else False
        )

        doi_match = (
            doi.lower()
            in combined_text.lower()
            if doi
            else False
        )

        usable = (
            total_pages > 0
            and pages_with_text > 0
            and error_pages == 0
            and total_words > 500
            and title_match
        )

        extraction_status = (
            "OK"
            if usable
            else "REVIEW"
        )

        summary_rows.append(
            {
                "SCREENING_ID": (
                    screening_id
                ),
                "PDF_FILE_NAME": (
                    pdf_file.name
                ),
                "LOCAL_RELATIVE_PATH": str(
                    pdf_file.relative_to(
                        ROOT
                    )
                ),
                "PDF_SHA256": (
                    sha256_file(
                        pdf_file
                    )
                ),
                "PDF_PAGE_COUNT": (
                    total_pages
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
                        ROOT
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

        identity_rows.append(
            {
                "SCREENING_ID": (
                    screening_id
                ),
                "TITLE": title,
                "DOI_NORMALIZED": doi,
                "TITLE_TERMS_CHECKED": (
                    " | ".join(terms)
                ),
                "TITLE_TERMS_MATCHED": (
                    " | ".join(
                        matched_terms
                    )
                ),
                "TITLE_MATCH_COUNT": (
                    len(matched_terms)
                ),
                "TITLE_MATCH": (
                    title_match
                ),
                "DOI_IN_TEXT": (
                    doi_match
                ),
                "IDENTITY_STATUS": (
                    "IDENTITY_CONFIRMED"
                    if title_match
                    else "IDENTITY_REVIEW"
                ),
                "USABLE_FOR_FULL_TEXT_ASSESSMENT": (
                    usable
                ),
            }
        )

        print(
            "  Paginas:",
            total_pages,
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
            "  TITLE_MATCH:",
            title_match,
        )

        print(
            "  Estado:",
            extraction_status,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    page_inventory = pd.DataFrame(
        page_rows
    )

    identity = pd.DataFrame(
        identity_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    page_inventory.to_csv(
        PAGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    identity.to_csv(
        IDENTITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    valid_documents = int(
        summary[
            "EXTRACTION_STATUS"
        ].eq("OK").sum()
    )

    identity_confirmed = int(
        identity[
            "IDENTITY_STATUS"
        ].eq(
            "IDENTITY_CONFIRMED"
        ).sum()
    )

    total_pages = int(
        summary[
            "PDF_PAGE_COUNT"
        ].sum()
    )

    total_words = int(
        summary[
            "TOTAL_WORD_COUNT"
        ].sum()
    )

    valid = (
        len(summary) == 9
        and valid_documents == 9
        and identity_confirmed == 9
        and summary[
            "ERROR_PAGES"
        ].sum() == 0
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
                "PDF_DOCUMENTS": (
                    len(summary)
                ),
                "VALID_DOCUMENTS": (
                    valid_documents
                ),
                "IDENTITIES_CONFIRMED": (
                    identity_confirmed
                ),
                "TOTAL_PAGES": (
                    total_pages
                ),
                "TOTAL_WORDS": (
                    total_words
                ),
                "ERROR_PAGES": int(
                    summary[
                        "ERROR_PAGES"
                    ].sum()
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
        "===== RESUMEN DE EXTRACCION ====="
    )

    print(
        summary[
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
        "===== IDENTIDAD ====="
    )

    print(
        identity[
            [
                "SCREENING_ID",
                "TITLE_MATCH_COUNT",
                "TITLE_MATCH",
                "DOI_IN_TEXT",
                "IDENTITY_STATUS",
                "USABLE_FOR_FULL_TEXT_ASSESSMENT",
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
    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "La extraccion del lote 02 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()