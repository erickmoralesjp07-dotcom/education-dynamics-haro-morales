from pathlib import Path
from datetime import datetime
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "literature_review"
    / "screening_batches"
    / "screening_batch_02.csv"
)

OUTPUT_DIR = (
    ROOT
    / "literature_review"
    / "screening_batches"
    / "review_packets"
)

DIAGNOSTIC_DIR = (
    ROOT
    / "results"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PACKET_FILE = (
    OUTPUT_DIR
    / "screening_batch_02_review_packet.txt"
)

TEMPLATE_FILE = (
    OUTPUT_DIR
    / "screening_batch_02_decision_template.csv"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTIC_DIR
    / "screening_batch_02_review_packet_diagnostic.csv"
)


def clean_value(value):
    if pd.isna(value):
        return ""

    text = str(value)

    text = (
        text
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def find_column(
    columns,
    candidates,
    required=False,
):
    normalized = {
        str(column).strip().upper(): column
        for column in columns
    }

    for candidate in candidates:
        key = candidate.upper()

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            "No se encontro ninguna de estas "
            f"columnas requeridas: {candidates}"
        )

    return None


def value_from_row(
    row,
    column,
):
    if column is None:
        return ""

    return clean_value(
        row.get(
            column,
            "",
        )
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError(
            "El lote 02 esta vacio."
        )

    columns = list(
        data.columns
    )

    id_column = find_column(
        columns,
        [
            "SCREENING_ID",
            "RECORD_ID",
            "ID",
        ],
        required=True,
    )

    title_column = find_column(
        columns,
        [
            "TITLE",
            "DISPLAY_NAME",
            "ARTICLE_TITLE",
        ],
        required=True,
    )

    abstract_column = find_column(
        columns,
        [
            "ABSTRACT",
            "ABSTRACT_TEXT",
            "SUMMARY",
        ],
    )

    year_column = find_column(
        columns,
        [
            "YEAR",
            "PUBLICATION_YEAR",
            "PUBLISHED_YEAR",
        ],
    )

    authors_column = find_column(
        columns,
        [
            "AUTHORS",
            "AUTHOR",
            "AUTHOR_NAMES",
        ],
    )

    source_column = find_column(
        columns,
        [
            "JOURNAL_OR_SOURCE",
            "SOURCE",
            "JOURNAL",
            "VENUE",
        ],
    )

    doi_column = find_column(
        columns,
        [
            "DOI_NORMALIZED",
            "DOI",
        ],
    )

    type_column = find_column(
        columns,
        [
            "DOCUMENT_TYPE",
            "TYPE",
            "WORK_TYPE",
        ],
    )

    thematic_column = find_column(
        columns,
        [
            "THEMATIC_BLOCK",
            "THEME",
            "TOPIC",
        ],
    )

    verification_column = find_column(
        columns,
        [
            "VERIFICATION_FLAGS",
            "FLAGS",
        ],
    )

    url_column = find_column(
        columns,
        [
            "URL",
            "LANDING_PAGE_URL",
            "PRIMARY_LOCATION",
        ],
    )

    blocks = []

    template_rows = []

    records_with_abstract = 0
    records_without_abstract = 0

    for position, (_, row) in enumerate(
        data.iterrows(),
        start=1,
    ):
        screening_id = value_from_row(
            row,
            id_column,
        )

        title = value_from_row(
            row,
            title_column,
        )

        abstract = value_from_row(
            row,
            abstract_column,
        )

        year = value_from_row(
            row,
            year_column,
        )

        authors = value_from_row(
            row,
            authors_column,
        )

        source = value_from_row(
            row,
            source_column,
        )

        doi = value_from_row(
            row,
            doi_column,
        )

        document_type = value_from_row(
            row,
            type_column,
        )

        thematic_block = value_from_row(
            row,
            thematic_column,
        )

        verification_flags = value_from_row(
            row,
            verification_column,
        )

        url = value_from_row(
            row,
            url_column,
        )

        if abstract:
            records_with_abstract += 1
        else:
            records_without_abstract += 1
            abstract = (
                "[ABSTRACT NO DISPONIBLE. "
                "DECIDIR COMO UNCLEAR SI EL TITULO "
                "NO APORTA INFORMACION SUFICIENTE.]"
            )

        block = "\n".join(
            [
                "=" * 78,
                (
                    f"REGISTRO {position:02d} "
                    f"DE {len(data):02d}"
                ),
                f"SCREENING_ID: {screening_id}",
                f"TITLE: {title}",
                f"YEAR: {year}",
                f"AUTHORS: {authors}",
                f"SOURCE: {source}",
                f"DOCUMENT_TYPE: {document_type}",
                f"DOI: {doi}",
                f"URL: {url}",
                (
                    "THEMATIC_BLOCK: "
                    f"{thematic_block}"
                ),
                (
                    "VERIFICATION_FLAGS: "
                    f"{verification_flags}"
                ),
                "",
                "ABSTRACT:",
                abstract,
                "",
                "DECISION: [Include / Exclude / Unclear]",
                "EXCLUSION_REASON: [solo si Exclude]",
                "REVIEWER_NOTES:",
                "=" * 78,
                "",
            ]
        )

        blocks.append(
            block
        )

        template_rows.append(
            {
                "SCREENING_ID": screening_id,
                "TITLE": title,
                "YEAR": year,
                "DOI_NORMALIZED": doi,
                "REVIEWER_1_DECISION": "",
                "REVIEWER_1_EXCLUSION_REASON": "",
                "REVIEWER_1_NOTES": "",
                "SCREENING_STATUS": (
                    "Awaiting_title_abstract_screening"
                ),
                "SCREENING_DATETIME": "",
            }
        )

    header = "\n".join(
        [
            "=" * 78,
            "FICHA DE REVISION POR TITULO Y RESUMEN",
            "LOTE 02",
            "=" * 78,
            "",
            f"ARCHIVO_ORIGEN: {INPUT_FILE}",
            f"TOTAL_REGISTROS: {len(data)}",
            (
                "REGISTROS_CON_RESUMEN: "
                f"{records_with_abstract}"
            ),
            (
                "REGISTROS_SIN_RESUMEN: "
                f"{records_without_abstract}"
            ),
            "",
            "CRITERIOS DE INCLUSION:",
            (
                "1. Estudio empirico o metodologico "
                "sobre educacion, logro, desigualdad, "
                "convergencia, dinamica o prediccion."
            ),
            (
                "2. Comparacion entre paises, regiones, "
                "periodos, sistemas educativos o grupos."
            ),
            (
                "3. Uso potencial de panel, series de "
                "tiempo, PISA, UNESCO, World Bank, "
                "modelos estadisticos o dinamicos."
            ),
            "",
            "CRITERIOS DE EXCLUSION:",
            (
                "1. Tema sin relacion directa con "
                "educacion o resultados educativos."
            ),
            (
                "2. Editorial, comentario, noticia, "
                "presentacion o registro sin contenido "
                "cientifico suficiente."
            ),
            (
                "3. Estudio puramente local sin aporte "
                "metodologico o comparativo relevante."
            ),
            (
                "4. Registro duplicado de una version "
                "publicada ya conservada."
            ),
            "",
            "REGLA DE SEGURIDAD:",
            (
                "Si el titulo o resumen no permite una "
                "decision firme, usar Unclear. No excluir "
                "por falta de informacion."
            ),
            "",
        ]
    )

    PACKET_FILE.write_text(
        header
        + "\n".join(blocks),
        encoding="utf-8",
    )

    template = pd.DataFrame(
        template_rows
    )

    template.to_csv(
        TEMPLATE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    unique_ids = int(
        data[
            id_column
        ].astype("string")
        .nunique(
            dropna=True
        )
    )

    duplicate_ids = int(
        data[
            id_column
        ].astype("string")
        .duplicated()
        .sum()
    )

    valid = (
        len(data) > 0
        and unique_ids == len(data)
        and duplicate_ids == 0
        and PACKET_FILE.exists()
        and TEMPLATE_FILE.exists()
        and len(template) == len(data)
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
                "TOTAL_RECORDS": (
                    len(data)
                ),
                "UNIQUE_SCREENING_IDS": (
                    unique_ids
                ),
                "DUPLICATE_SCREENING_IDS": (
                    duplicate_ids
                ),
                "RECORDS_WITH_ABSTRACT": (
                    records_with_abstract
                ),
                "RECORDS_WITHOUT_ABSTRACT": (
                    records_without_abstract
                ),
                "PACKET_CREATED": (
                    PACKET_FILE.exists()
                ),
                "TEMPLATE_CREATED": (
                    TEMPLATE_FILE.exists()
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
        "===== PREPARACION DEL LOTE 02 ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "Columnas detectadas:"
    )

    print(
        "ID:",
        id_column,
    )

    print(
        "TITLE:",
        title_column,
    )

    print(
        "ABSTRACT:",
        abstract_column,
    )

    print(
        "YEAR:",
        year_column,
    )

    print(
        "AUTHORS:",
        authors_column,
    )

    print(
        "SOURCE:",
        source_column,
    )

    print(
        "DOI:",
        doi_column,
    )

    print()
    print(
        "Archivos creados:"
    )

    print(PACKET_FILE)
    print(TEMPLATE_FILE)
    print(DIAGNOSTIC_FILE)

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
            "La preparacion del lote 02 "
            "requiere revision."
        )


if __name__ == "__main__":
    main()