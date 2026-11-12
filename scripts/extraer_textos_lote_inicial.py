from pathlib import Path
from datetime import datetime
import hashlib
import re
import sys

import pandas as pd
from PyPDF2 import PdfReader


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_DIR = (
    PROJECT_ROOT
    / "literature_review"
)

MANIFEST_FILE = (
    LITERATURE_DIR
    / "initial_source_manifest.csv"
)

EXTRACTED_DIR = (
    LITERATURE_DIR
    / "extracted_text"
)

PAGE_INVENTORY_FILE = (
    LITERATURE_DIR
    / "initial_pdf_page_inventory.csv"
)

DOCUMENT_INVENTORY_FILE = (
    LITERATURE_DIR
    / "initial_pdf_extraction_summary.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "initial_pdf_text_extraction_diagnostic.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_file(file_path):
    """Comprueba que exista un archivo."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def calculate_sha256(file_path, chunk_size=1024 * 1024):
    """Calcula la huella SHA-256."""

    digest = hashlib.sha256()

    with file_path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(text):
    """
    Normaliza espacios sin alterar el contenido sustantivo.

    Conserva saltos de párrafo cuando sea posible.
    """

    if text is None:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = []

    for line in text.split("\n"):
        clean_line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        lines.append(clean_line)

    normalized_lines = []
    previous_empty = False

    for line in lines:
        is_empty = line == ""

        if is_empty and previous_empty:
            continue

        normalized_lines.append(line)
        previous_empty = is_empty

    return "\n".join(
        normalized_lines
    ).strip()


def safe_metadata_value(metadata, key):
    """Obtiene metadatos sin fallar si la clave no existe."""

    if metadata is None:
        return ""

    value = metadata.get(
        key,
        "",
    )

    if value is None:
        return ""

    return str(value)


def resolve_pdf_path(manifest_row):
    """Recupera la ruta local registrada en el manifiesto."""

    relative_path = Path(
        manifest_row[
            "LOCAL_RELATIVE_PATH"
        ]
    )

    return (
        PROJECT_ROOT
        / relative_path
    )


def create_page_header(
    source_id,
    title,
    page_number,
    page_count,
):
    """Crea un encabezado trazable para cada página."""

    return (
        f"SOURCE_ID: {source_id}\n"
        f"TITLE: {title}\n"
        f"PDF_PAGE: {page_number}\n"
        f"TOTAL_PDF_PAGES: {page_count}\n"
        + "-" * 72
        + "\n"
    )


# ============================================================
# EXTRACCION DE UN PDF
# ============================================================

def extract_document(manifest_row):
    """
    Extrae un PDF página por página.

    Devuelve:
        resumen del documento
        inventario de páginas
    """

    source_id = str(
        manifest_row["SOURCE_ID"]
    )

    title = str(
        manifest_row["TITLE"]
    )

    pdf_path = resolve_pdf_path(
        manifest_row
    )

    verify_file(pdf_path)

    current_sha256 = calculate_sha256(
        pdf_path
    )

    registered_sha256 = str(
        manifest_row["SHA256"]
    )

    if current_sha256 != registered_sha256:
        raise ValueError(
            "La huella SHA-256 no coincide para:\n"
            f"{pdf_path}"
        )

    reader = PdfReader(
        str(pdf_path)
    )

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as error:
            raise ValueError(
                f"No se pudo abrir el PDF cifrado:\n{pdf_path}"
            ) from error

    metadata = reader.metadata
    page_count = len(reader.pages)

    document_output_dir = (
        EXTRACTED_DIR
        / source_id
    )

    document_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_rows = []
    combined_parts = []
    extraction_errors = 0
    empty_pages = 0

    for page_index, page in enumerate(
        reader.pages,
        start=1,
    ):
        extraction_status = "OK"
        extraction_error = ""

        try:
            raw_text = page.extract_text()
            clean_text = normalize_text(
                raw_text
            )

        except Exception as error:
            clean_text = ""
            extraction_status = "ERROR"
            extraction_error = (
                f"{type(error).__name__}: {error}"
            )
            extraction_errors += 1

        if not clean_text:
            empty_pages += 1

            if extraction_status == "OK":
                extraction_status = (
                    "EMPTY_OR_IMAGE_ONLY"
                )

        page_header = create_page_header(
            source_id=source_id,
            title=title,
            page_number=page_index,
            page_count=page_count,
        )

        page_content = (
            page_header
            + clean_text
            + "\n"
        )

        page_file_name = (
            f"{source_id}_page_{page_index:04d}.txt"
        )

        page_file = (
            document_output_dir
            / page_file_name
        )

        page_file.write_text(
            page_content,
            encoding="utf-8",
        )

        combined_parts.append(
            page_content
        )

        page_rows.append(
            {
                "SOURCE_ID": source_id,
                "TITLE": title,
                "PDF_FILE_NAME": pdf_path.name,
                "PDF_PAGE": page_index,
                "TOTAL_PDF_PAGES": page_count,
                "CHARACTER_COUNT": len(
                    clean_text
                ),
                "WORD_COUNT": len(
                    clean_text.split()
                ),
                "TEXT_AVAILABLE": bool(
                    clean_text
                ),
                "EXTRACTION_STATUS": (
                    extraction_status
                ),
                "EXTRACTION_ERROR": (
                    extraction_error
                ),
                "PAGE_TEXT_FILE": str(
                    page_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    combined_file = (
        document_output_dir
        / f"{source_id}_full_text.txt"
    )

    combined_file.write_text(
        "\n\n".join(
            combined_parts
        ),
        encoding="utf-8",
    )

    page_inventory = pd.DataFrame(
        page_rows
    )

    total_characters = int(
        page_inventory[
            "CHARACTER_COUNT"
        ].sum()
    )

    total_words = int(
        page_inventory[
            "WORD_COUNT"
        ].sum()
    )

    document_summary = {
        "SOURCE_ID": source_id,
        "TITLE": title,
        "PDF_FILE_NAME": pdf_path.name,
        "LOCAL_RELATIVE_PATH": str(
            pdf_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "PDF_SIZE_BYTES": int(
            pdf_path.stat().st_size
        ),
        "SHA256_VERIFIED": True,
        "PDF_PAGE_COUNT": page_count,
        "PAGES_WITH_TEXT": int(
            page_inventory[
                "TEXT_AVAILABLE"
            ].sum()
        ),
        "EMPTY_OR_IMAGE_ONLY_PAGES": (
            empty_pages
        ),
        "EXTRACTION_ERROR_PAGES": (
            extraction_errors
        ),
        "TOTAL_CHARACTER_COUNT": (
            total_characters
        ),
        "TOTAL_WORD_COUNT": total_words,
        "PDF_METADATA_TITLE": (
            safe_metadata_value(
                metadata,
                "/Title",
            )
        ),
        "PDF_METADATA_AUTHOR": (
            safe_metadata_value(
                metadata,
                "/Author",
            )
        ),
        "PDF_METADATA_SUBJECT": (
            safe_metadata_value(
                metadata,
                "/Subject",
            )
        ),
        "PDF_METADATA_CREATOR": (
            safe_metadata_value(
                metadata,
                "/Creator",
            )
        ),
        "PDF_METADATA_PRODUCER": (
            safe_metadata_value(
                metadata,
                "/Producer",
            )
        ),
        "COMBINED_TEXT_FILE": str(
            combined_file.relative_to(
                PROJECT_ROOT
            )
        ),
        "EXTRACTION_DATETIME": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "EXTRACTION_STATUS": (
            "OK"
            if extraction_errors == 0
            else "PARTIAL"
        ),
    }

    return (
        document_summary,
        page_inventory,
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "EXTRACCION DE TEXTO DEL LOTE BIBLIOGRAFICO INICIAL"
    )

    verify_file(
        MANIFEST_FILE
    )

    EXTRACTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTIC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST_FILE,
        dtype=str,
    )

    required_columns = {
        "SOURCE_ID",
        "TITLE",
        "LOCAL_RELATIVE_PATH",
        "SHA256",
    }

    missing_columns = sorted(
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el manifiesto: "
            f"{missing_columns}"
        )

    if manifest["SOURCE_ID"].duplicated().any():
        raise ValueError(
            "El manifiesto contiene SOURCE_ID duplicados."
        )

    print(
        "\nDocumentos por procesar:",
        len(manifest),
    )

    document_rows = []
    page_frames = []

    for _, manifest_row in manifest.iterrows():
        source_id = manifest_row[
            "SOURCE_ID"
        ]

        print(
            f"\nExtrayendo {source_id}: "
            f"{manifest_row['LOCAL_FILE_NAME']}"
        )

        (
            document_summary,
            page_inventory,
        ) = extract_document(
            manifest_row
        )

        document_rows.append(
            document_summary
        )

        page_frames.append(
            page_inventory
        )

        print(
            "  Paginas:",
            document_summary[
                "PDF_PAGE_COUNT"
            ],
        )

        print(
            "  Paginas con texto:",
            document_summary[
                "PAGES_WITH_TEXT"
            ],
        )

        print(
            "  Palabras extraidas:",
            f"{document_summary['TOTAL_WORD_COUNT']:,}",
        )

        print(
            "  Estado:",
            document_summary[
                "EXTRACTION_STATUS"
            ],
        )

    document_inventory = pd.DataFrame(
        document_rows
    )

    page_inventory = pd.concat(
        page_frames,
        ignore_index=True,
    )

    document_inventory.to_csv(
        DOCUMENT_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    page_inventory.to_csv(
        PAGE_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    total_pages = int(
        document_inventory[
            "PDF_PAGE_COUNT"
        ].sum()
    )

    pages_with_text = int(
        document_inventory[
            "PAGES_WITH_TEXT"
        ].sum()
    )

    empty_pages = int(
        document_inventory[
            "EMPTY_OR_IMAGE_ONLY_PAGES"
        ].sum()
    )

    error_pages = int(
        document_inventory[
            "EXTRACTION_ERROR_PAGES"
        ].sum()
    )

    total_words = int(
        document_inventory[
            "TOTAL_WORD_COUNT"
        ].sum()
    )

    total_characters = int(
        document_inventory[
            "TOTAL_CHARACTER_COUNT"
        ].sum()
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "DOCUMENT_COUNT": len(
                    document_inventory
                ),
                "TOTAL_PDF_PAGES": total_pages,
                "PAGES_WITH_TEXT": (
                    pages_with_text
                ),
                "EMPTY_OR_IMAGE_ONLY_PAGES": (
                    empty_pages
                ),
                "EXTRACTION_ERROR_PAGES": (
                    error_pages
                ),
                "TOTAL_WORD_COUNT": (
                    total_words
                ),
                "TOTAL_CHARACTER_COUNT": (
                    total_characters
                ),
                "SHA256_VERIFIED_DOCUMENTS": int(
                    document_inventory[
                        "SHA256_VERIFIED"
                    ].sum()
                ),
                "DOCUMENTS_WITH_OK_STATUS": int(
                    document_inventory[
                        "EXTRACTION_STATUS"
                    ].eq("OK")
                    .sum()
                ),
                "DOCUMENTS_WITH_PARTIAL_STATUS": int(
                    document_inventory[
                        "EXTRACTION_STATUS"
                    ].eq("PARTIAL")
                    .sum()
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if error_pages == 0
                    else "PARTIAL"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_section(
        "EXTRACCION DE TEXTO COMPLETADA CORRECTAMENTE"
    )

    print(
        "\nRESUMEN POR DOCUMENTO"
    )

    print(
        document_inventory[
            [
                "SOURCE_ID",
                "PDF_FILE_NAME",
                "PDF_PAGE_COUNT",
                "PAGES_WITH_TEXT",
                "EMPTY_OR_IMAGE_ONLY_PAGES",
                "TOTAL_WORD_COUNT",
                "EXTRACTION_STATUS",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESUMEN GENERAL"
    )

    print(
        "Documentos procesados:",
        len(
            document_inventory
        ),
    )

    print(
        "Paginas totales:",
        total_pages,
    )

    print(
        "Paginas con texto:",
        pages_with_text,
    )

    print(
        "Paginas vacias o basadas en imagen:",
        empty_pages,
    )

    print(
        "Paginas con error:",
        error_pages,
    )

    print(
        "Palabras extraidas:",
        f"{total_words:,}",
    )

    print(
        "Caracteres extraidos:",
        f"{total_characters:,}",
    )

    print(
        "\nInventario por documento:"
    )

    print(
        DOCUMENT_INVENTORY_FILE
    )

    print(
        "\nInventario por pagina:"
    )

    print(
        PAGE_INVENTORY_FILE
    )

    print(
        "\nTextos locales:"
    )

    print(
        EXTRACTED_DIR
    )

    print(
        "\nDiagnostico:"
    )

    print(
        DIAGNOSTIC_FILE
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE LA EXTRACCION DE TEXTO"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)