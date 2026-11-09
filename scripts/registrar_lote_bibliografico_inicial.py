from pathlib import Path
from datetime import datetime
import hashlib
import sys

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text"
    / "official"
)

PEER_REVIEWED_DIR = (
    PROJECT_ROOT
    / "literature_review"
    / "full_text"
    / "peer_reviewed"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "initial_source_manifest.csv"
)

SCREENING_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "screening_register.csv"
)

EXTRACTION_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "study_extraction_matrix.csv"
)

SEARCH_LOG_FILE = (
    PROJECT_ROOT
    / "literature_review"
    / "search_log.csv"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "initial_literature_batch_diagnostic.csv"
)


# ============================================================
# METADATOS DEL LOTE INICIAL
# ============================================================

INITIAL_SOURCES = [
    {
        "SOURCE_ID": "OFF-001",
        "SOURCE_TYPE": "Official methodology",
        "TITLE": (
            "The PRISMA 2020 statement: an updated guideline "
            "for reporting systematic reviews"
        ),
        "AUTHORS_OR_INSTITUTION": (
            "Page MJ; McKenzie JE; Bossuyt PM; Boutron I; "
            "Hoffmann TC; Mulrow CD; et al."
        ),
        "YEAR": 2021,
        "JOURNAL_OR_SOURCE": "The BMJ",
        "DOCUMENT_TYPE": "Guideline article",
        "DOI": "10.1136/bmj.n71",
        "URL": (
            "https://www.bmj.com/content/372/bmj.n71"
        ),
        "LOCAL_SUBFOLDER": "official",
        "LOCAL_FILE_NAME": (
            "Page_et_al_2021_PRISMA_2020.pdf"
        ),
        "THEMATIC_BLOCKS": (
            "Predictive validation and baselines|"
            "Review methodology"
        ),
        "INITIAL_DECISION": "Include",
        "DECISION_REASON": (
            "Primary reporting guideline for the structured "
            "literature review and PRISMA flow."
        ),
        "SUPPORTS_ARTICLE_SECTION": (
            "Literature review methods"
        ),
        "PEER_REVIEWED": True,
        "OFFICIAL_OR_PRIMARY_SOURCE": True,
    },
    {
        "SOURCE_ID": "OFF-002",
        "SOURCE_TYPE": "Official indicator framework",
        "TITLE": (
            "Official List of SDG 4 Indicators"
        ),
        "AUTHORS_OR_INSTITUTION": (
            "UNESCO Institute for Statistics"
        ),
        "YEAR": 2025,
        "JOURNAL_OR_SOURCE": (
            "UNESCO Institute for Statistics"
        ),
        "DOCUMENT_TYPE": "Official technical document",
        "DOI": "",
        "URL": (
            "https://download.uis.unesco.org/bdds/202502/"
            "Official%20List%20of%20SDG%204%20Indicators.pdf"
        ),
        "LOCAL_SUBFOLDER": "official",
        "LOCAL_FILE_NAME": (
            "UNESCO_UIS_2025_Official_SDG4_Indicators.pdf"
        ),
        "THEMATIC_BLOCKS": (
            "SDG4 and educational indicators|"
            "Educational access and exclusion|"
            "Educational completion and attainment"
        ),
        "INITIAL_DECISION": "Include",
        "DECISION_REASON": (
            "Primary source for the definitions and organization "
            "of official SDG 4 indicators."
        ),
        "SUPPORTS_ARTICLE_SECTION": (
            "Data sources and indicator definitions"
        ),
        "PEER_REVIEWED": False,
        "OFFICIAL_OR_PRIMARY_SOURCE": True,
    },
    {
        "SOURCE_ID": "OFF-003",
        "SOURCE_TYPE": "Official indicator review",
        "TITLE": (
            "SDG 4 Thematic Indicator Framework Review"
        ),
        "AUTHORS_OR_INSTITUTION": (
            "UNESCO Institute for Statistics"
        ),
        "YEAR": 2025,
        "JOURNAL_OR_SOURCE": (
            "UNESCO Institute for Statistics"
        ),
        "DOCUMENT_TYPE": "Official technical review",
        "DOI": "",
        "URL": (
            "https://www.uis.unesco.org/sites/default/files/"
            "medias/fichiers/2025/08/"
            "CN_SDG4_Thematic_Indicator_Framework_Review.pdf"
        ),
        "LOCAL_SUBFOLDER": "official",
        "LOCAL_FILE_NAME": (
            "UNESCO_UIS_2025_Thematic_Framework_Review.pdf"
        ),
        "THEMATIC_BLOCKS": (
            "SDG4 and educational indicators|"
            "Missing data and unequal indicator coverage"
        ),
        "INITIAL_DECISION": "Include",
        "DECISION_REASON": (
            "Documents revisions, coverage limitations and "
            "methodological issues in the thematic framework."
        ),
        "SUPPORTS_ARTICLE_SECTION": (
            "Indicator selection and methodological limitations"
        ),
        "PEER_REVIEWED": False,
        "OFFICIAL_OR_PRIMARY_SOURCE": True,
    },
    {
        "SOURCE_ID": "OFF-004",
        "SOURCE_TYPE": "Official expenditure methodology",
        "TITLE": (
            "Education Expenditure Data: Challenges and "
            "Solutions Forward"
        ),
        "AUTHORS_OR_INSTITUTION": (
            "UNESCO Institute for Statistics"
        ),
        "YEAR": 2024,
        "JOURNAL_OR_SOURCE": (
            "UNESCO Institute for Statistics"
        ),
        "DOCUMENT_TYPE": "Official technical report",
        "DOI": "",
        "URL": (
            "https://www.uis.unesco.org/sites/default/files/"
            "medias/fichiers/2025/08/"
            "EDS-6-Expenditure-Final-WEB.pdf"
        ),
        "LOCAL_SUBFOLDER": "official",
        "LOCAL_FILE_NAME": (
            "UNESCO_UIS_2024_Education_Expenditure_Data.pdf"
        ),
        "THEMATIC_BLOCKS": (
            "Public education financing|"
            "Missing data and unequal indicator coverage"
        ),
        "INITIAL_DECISION": "Include",
        "DECISION_REASON": (
            "Primary methodological source for educational "
            "expenditure data, comparability and limitations."
        ),
        "SUPPORTS_ARTICLE_SECTION": (
            "Financing indicator and data limitations"
        ),
        "PEER_REVIEWED": False,
        "OFFICIAL_OR_PRIMARY_SOURCE": True,
    },
    {
        "SOURCE_ID": "PR-001",
        "SOURCE_TYPE": "Peer-reviewed article",
        "TITLE": (
            "Dynamic Systems Modeling in Educational System "
            "Design and Policy"
        ),
        "AUTHORS_OR_INSTITUTION": "Jennifer Groff",
        "YEAR": 2013,
        "JOURNAL_OR_SOURCE": (
            "New Approaches in Educational Research"
        ),
        "DOCUMENT_TYPE": "Journal article",
        "DOI": "10.7821/naer.2.2.72-81",
        "URL": (
            "https://files.eric.ed.gov/fulltext/EJ1128221.pdf"
        ),
        "LOCAL_SUBFOLDER": "peer_reviewed",
        "LOCAL_FILE_NAME": (
            "Groff_2013_Dynamic_Systems_Education.pdf"
        ),
        "THEMATIC_BLOCKS": (
            "Dynamic systems in education|"
            "Cross-country heterogeneity"
        ),
        "INITIAL_DECISION": "Include",
        "DECISION_REASON": (
            "Provides a theoretical and methodological basis for "
            "system-dynamics approaches to educational systems."
        ),
        "SUPPORTS_ARTICLE_SECTION": (
            "Theoretical and mathematical framework"
        ),
        "PEER_REVIEWED": True,
        "OFFICIAL_OR_PRIMARY_SOURCE": False,
    },
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def calculate_sha256(file_path, chunk_size=1024 * 1024):
    """Calcula la huella SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as source_file:
        while True:
            chunk = source_file.read(
                chunk_size
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def validate_pdf_signature(file_path):
    """
    Comprueba que el archivo empiece con la firma PDF.

    No modifica el documento.
    """

    with file_path.open("rb") as source_file:
        signature = source_file.read(5)

    return signature == b"%PDF-"


def get_local_path(source):
    """Construye la ruta local de una fuente."""

    if source["LOCAL_SUBFOLDER"] == "official":
        base_directory = OFFICIAL_DIR

    elif source["LOCAL_SUBFOLDER"] == "peer_reviewed":
        base_directory = PEER_REVIEWED_DIR

    else:
        raise ValueError(
            "Subcarpeta bibliografica desconocida: "
            f"{source['LOCAL_SUBFOLDER']}"
        )

    return (
        base_directory
        / source["LOCAL_FILE_NAME"]
    )


def build_manifest():
    """
    Verifica los documentos y construye el manifiesto.

    El proceso falla si falta un archivo o si no posee una
    firma PDF válida.
    """

    rows = []

    for source in INITIAL_SOURCES:
        file_path = get_local_path(
            source
        )

        if not file_path.exists():
            raise FileNotFoundError(
                "No se encontro el documento requerido:\n"
                f"{file_path}"
            )

        is_pdf = validate_pdf_signature(
            file_path
        )

        if not is_pdf:
            raise ValueError(
                "El archivo no tiene una firma PDF valida:\n"
                f"{file_path}"
            )

        file_size = file_path.stat().st_size
        sha256 = calculate_sha256(
            file_path
        )

        relative_path = file_path.relative_to(
            PROJECT_ROOT
        )

        row = dict(source)

        row.update(
            {
                "LOCAL_RELATIVE_PATH": str(
                    relative_path
                ),
                "FILE_SIZE_BYTES": file_size,
                "SHA256": sha256,
                "PDF_SIGNATURE_VALID": is_pdf,
                "FILE_EXISTS": True,
                "REGISTRATION_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "SCREENING_STATUS": (
                    "Initial inclusion"
                ),
                "FULL_TEXT_STATUS": "Available",
                "METADATA_VERIFICATION_STATUS": (
                    "Initial metadata registered"
                ),
            }
        )

        rows.append(row)

    manifest = pd.DataFrame(
        rows
    )

    if manifest["SOURCE_ID"].duplicated().any():
        raise ValueError(
            "Existen identificadores bibliograficos duplicados."
        )

    if manifest["SHA256"].duplicated().any():
        duplicated_sources = (
            manifest.loc[
                manifest["SHA256"].duplicated(
                    keep=False
                ),
                [
                    "SOURCE_ID",
                    "LOCAL_FILE_NAME",
                ],
            ]
        )

        raise ValueError(
            "Dos o mas archivos tienen la misma huella SHA-256:\n"
            f"{duplicated_sources.to_string(index=False)}"
        )

    return manifest


# ============================================================
# ACTUALIZACION DEL REGISTRO DE CRIBADO
# ============================================================

def update_screening_register(manifest):
    """
    Agrega el lote inicial al registro de cribado.

    Conserva cualquier registro previo y evita duplicados
    mediante RECORD_ID.
    """

    if SCREENING_FILE.exists():
        screening = pd.read_csv(
            SCREENING_FILE,
            dtype=str,
        )
    else:
        screening = pd.DataFrame()

    new_rows = []

    for _, row in manifest.iterrows():
        new_rows.append(
            {
                "RECORD_ID": row["SOURCE_ID"],
                "DATABASE": (
                    "Official source"
                    if row[
                        "OFFICIAL_OR_PRIMARY_SOURCE"
                    ]
                    else "ERIC"
                ),
                "QUERY_ID": "Initial batch",
                "TITLE": row["TITLE"],
                "AUTHORS": (
                    row[
                        "AUTHORS_OR_INSTITUTION"
                    ]
                ),
                "YEAR": str(
                    row["YEAR"]
                ),
                "JOURNAL_OR_SOURCE": (
                    row[
                        "JOURNAL_OR_SOURCE"
                    ]
                ),
                "DOCUMENT_TYPE": (
                    row[
                        "DOCUMENT_TYPE"
                    ]
                ),
                "DOI": row["DOI"],
                "URL": row["URL"],
                "ABSTRACT": "",
                "LANGUAGE": "English",
                "DUPLICATE_GROUP_ID": "",
                "IS_DUPLICATE": "False",
                "DUPLICATE_DECISION": (
                    "Unique record"
                ),
                "TITLE_ABSTRACT_DECISION": (
                    row[
                        "INITIAL_DECISION"
                    ]
                ),
                "TITLE_ABSTRACT_EXCLUSION_REASON": "",
                "FULL_TEXT_OBTAINED": "True",
                "FULL_TEXT_DECISION": (
                    row[
                        "INITIAL_DECISION"
                    ]
                ),
                "FULL_TEXT_EXCLUSION_REASON": "",
                "FINAL_INCLUDED": "Pending final appraisal",
                "THEMATIC_BLOCKS": (
                    row[
                        "THEMATIC_BLOCKS"
                    ]
                ),
                "REVIEWER": "Haro and Morales",
                "REVIEW_DATE": (
                    datetime.now().date().isoformat()
                ),
                "NOTES": (
                    row[
                        "DECISION_REASON"
                    ]
                ),
            }
        )

    new_screening = pd.DataFrame(
        new_rows
    )

    if screening.empty:
        combined = new_screening.copy()

    else:
        combined = pd.concat(
            [
                screening,
                new_screening,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = combined.drop_duplicates(
            subset=["RECORD_ID"],
            keep="last",
        )

    combined = combined.sort_values(
        by="RECORD_ID"
    ).reset_index(drop=True)

    combined.to_csv(
        SCREENING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return combined


# ============================================================
# ACTUALIZACION DE LA MATRIZ DE EXTRACCION
# ============================================================

def update_extraction_matrix(manifest):
    """
    Registra metadatos básicos en la matriz de extracción.

    Los campos que requieren lectura completa permanecen vacíos.
    """

    if EXTRACTION_FILE.exists():
        extraction = pd.read_csv(
            EXTRACTION_FILE,
            dtype=str,
        )
    else:
        extraction = pd.DataFrame()

    new_rows = []

    for _, row in manifest.iterrows():
        study_id = (
            f"ST-{row['SOURCE_ID']}"
        )

        new_rows.append(
            {
                "STUDY_ID": study_id,
                "RECORD_ID": row["SOURCE_ID"],
                "FULL_CITATION": "",
                "TITLE": row["TITLE"],
                "AUTHORS": (
                    row[
                        "AUTHORS_OR_INSTITUTION"
                    ]
                ),
                "YEAR": str(
                    row["YEAR"]
                ),
                "DOI": row["DOI"],
                "URL": row["URL"],
                "JOURNAL": (
                    row[
                        "JOURNAL_OR_SOURCE"
                    ]
                ),
                "PUBLISHER": "",
                "COUNTRY_OF_JOURNAL": "",
                "INDEXING_DATABASE": (
                    "Official institutional source"
                    if row["SOURCE_TYPE"].startswith(
                        "Official"
                    )
                    else "ERIC"
                ),
                "SJR_YEAR": "",
                "SJR_QUARTILE": "",
                "JCR_YEAR": "",
                "JCR_QUARTILE": "",
                "IMPACT_FACTOR": "",
                "CITESCORE": "",
                "OPEN_ACCESS": "Yes",
                "APC_INFORMATION": "",
                "STUDY_AIM": "",
                "RESEARCH_QUESTION": "",
                "THEORETICAL_FRAMEWORK": "",
                "THEMATIC_BLOCK": (
                    row[
                        "THEMATIC_BLOCKS"
                    ]
                ),
                "COUNTRIES_ANALYZED": "",
                "GEOGRAPHIC_SCOPE": "",
                "EDUCATION_LEVEL": "",
                "DATA_SOURCE": "",
                "DATASET_NAME": "",
                "DATA_PERIOD": "",
                "SAMPLE_SIZE": "",
                "UNIT_OF_ANALYSIS": "",
                "DEPENDENT_VARIABLES": "",
                "INDEPENDENT_VARIABLES": "",
                "INDICATORS_USED": "",
                "INDICATOR_DEFINITIONS": "",
                "MISSING_DATA_METHOD": "",
                "OUTLIER_METHOD": "",
                "MODEL_TYPE": "",
                "ESTIMATION_METHOD": "",
                "SOFTWARE": "",
                "VALIDATION_METHOD": "",
                "BASELINE_MODEL": "",
                "ROBUSTNESS_CHECKS": "",
                "MAIN_RESULTS": "",
                "EFFECT_SIZES": "",
                "UNCERTAINTY_MEASURES": "",
                "MODEL_PERFORMANCE": "",
                "LIMITATIONS_REPORTED": "",
                "CAUSAL_CLAIM": "",
                "REPRODUCIBILITY_MATERIALS": "",
                "DATA_AVAILABILITY": "",
                "CODE_AVAILABILITY": "",
                "RELEVANCE_TO_ACLF": (
                    row[
                        "DECISION_REASON"
                    ]
                ),
                "SUPPORTS_METHOD": (
                    "Yes"
                    if row[
                        "SUPPORTS_ARTICLE_SECTION"
                    ] in [
                        "Literature review methods",
                        "Data sources and indicator definitions",
                        "Indicator selection and methodological limitations",
                        "Financing indicator and data limitations",
                        "Theoretical and mathematical framework",
                    ]
                    else ""
                ),
                "SUPPORTS_DISCUSSION": "",
                "SUPPORTS_LIMITATIONS": "",
                "KEY_QUOTATION": "",
                "QUOTATION_PAGE": "",
                "REVIEWER_NOTES": (
                    "Full-text extraction pending."
                ),
            }
        )

    new_extraction = pd.DataFrame(
        new_rows
    )

    if extraction.empty:
        combined = new_extraction.copy()

    else:
        combined = pd.concat(
            [
                extraction,
                new_extraction,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = combined.drop_duplicates(
            subset=["STUDY_ID"],
            keep="last",
        )

    combined = combined.sort_values(
        by="STUDY_ID"
    ).reset_index(drop=True)

    combined.to_csv(
        EXTRACTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return combined


# ============================================================
# REGISTRO DE LA ACTIVIDAD DE BUSQUEDA
# ============================================================

def update_search_log(manifest):
    """Registra el lote inicial como actividad bibliográfica."""

    if SEARCH_LOG_FILE.exists():
        search_log = pd.read_csv(
            SEARCH_LOG_FILE,
            dtype=str,
        )
    else:
        search_log = pd.DataFrame()

    event_id = "SEARCH-INITIAL-001"

    row = {
        "SEARCH_EVENT_ID": event_id,
        "DATABASE": (
            "Official websites and ERIC"
        ),
        "QUERY_ID": "Initial primary-source batch",
        "SEARCH_DATE": (
            datetime.now().date().isoformat()
        ),
        "SEARCHER": "Haro and Morales",
        "SEARCH_STRING_USED": (
            "Targeted retrieval of PRISMA, UNESCO UIS "
            "documentation, and an educational system-dynamics article"
        ),
        "FILTERS_USED": (
            "Official source|Full text available|English"
        ),
        "RESULT_COUNT": len(
            manifest
        ),
        "EXPORT_FORMAT": "PDF and CSV manifest",
        "EXPORT_FILE": (
            MANIFEST_FILE.name
        ),
        "NOTES": (
            "Initial methodological and primary-source batch. "
            "This is not yet a database search count for PRISMA."
        ),
    }

    new_log = pd.DataFrame(
        [row]
    )

    if search_log.empty:
        combined = new_log.copy()

    else:
        combined = pd.concat(
            [
                search_log,
                new_log,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = combined.drop_duplicates(
            subset=["SEARCH_EVENT_ID"],
            keep="last",
        )

    combined.to_csv(
        SEARCH_LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return combined


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "REGISTRO DEL LOTE BIBLIOGRAFICO INICIAL"
    )

    if not OFFICIAL_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta:\n{OFFICIAL_DIR}"
        )

    if not PEER_REVIEWED_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta:\n{PEER_REVIEWED_DIR}"
        )

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTIC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nVerificando archivos PDF..."
    )

    manifest = build_manifest()

    print(
        "Actualizando registro de cribado..."
    )

    screening = update_screening_register(
        manifest
    )

    print(
        "Actualizando matriz de extraccion..."
    )

    extraction = update_extraction_matrix(
        manifest
    )

    print(
        "Actualizando registro de busqueda..."
    )

    search_log = update_search_log(
        manifest
    )

    manifest.to_csv(
        MANIFEST_FILE,
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
                "INITIAL_SOURCE_COUNT": len(
                    manifest
                ),
                "OFFICIAL_SOURCE_COUNT": int(
                    manifest[
                        "LOCAL_SUBFOLDER"
                    ].eq("official").sum()
                ),
                "PEER_REVIEWED_SOURCE_COUNT": int(
                    manifest[
                        "LOCAL_SUBFOLDER"
                    ].eq(
                        "peer_reviewed"
                    ).sum()
                ),
                "VALID_PDF_COUNT": int(
                    manifest[
                        "PDF_SIGNATURE_VALID"
                    ].sum()
                ),
                "TOTAL_SIZE_BYTES": int(
                    manifest[
                        "FILE_SIZE_BYTES"
                    ].sum()
                ),
                "UNIQUE_SHA256_COUNT": int(
                    manifest[
                        "SHA256"
                    ].nunique()
                ),
                "SCREENING_REGISTER_ROWS": len(
                    screening
                ),
                "EXTRACTION_MATRIX_ROWS": len(
                    extraction
                ),
                "SEARCH_LOG_ROWS": len(
                    search_log
                ),
                "MISSING_FILE_COUNT": 0,
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_section(
        "LOTE BIBLIOGRAFICO REGISTRADO CORRECTAMENTE"
    )

    print(
        "\nDOCUMENTOS REGISTRADOS"
    )

    print(
        manifest[
            [
                "SOURCE_ID",
                "SOURCE_TYPE",
                "YEAR",
                "LOCAL_FILE_NAME",
                "FILE_SIZE_BYTES",
                "PDF_SIGNATURE_VALID",
                "INITIAL_DECISION",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESUMEN"
    )

    print(
        "Fuentes registradas:",
        len(
            manifest
        ),
    )

    print(
        "Fuentes oficiales:",
        int(
            manifest[
                "LOCAL_SUBFOLDER"
            ].eq("official").sum()
        ),
    )

    print(
        "Articulos revisados por pares:",
        int(
            manifest[
                "LOCAL_SUBFOLDER"
            ].eq(
                "peer_reviewed"
            ).sum()
        ),
    )

    print(
        "PDF validos:",
        int(
            manifest[
                "PDF_SIGNATURE_VALID"
            ].sum()
        ),
    )

    print(
        "Tamanio total:",
        f"{manifest['FILE_SIZE_BYTES'].sum():,}",
        "bytes",
    )

    print(
        "Huellas SHA-256 unicas:",
        manifest[
            "SHA256"
        ].nunique(),
    )

    print(
        f"\nManifiesto:\n{MANIFEST_FILE}"
    )

    print(
        f"\nRegistro de cribado:\n{SCREENING_FILE}"
    )

    print(
        f"\nMatriz de extraccion:\n{EXTRACTION_FILE}"
    )

    print(
        f"\nRegistro de busqueda:\n{SEARCH_LOG_FILE}"
    )

    print(
        f"\nDiagnostico:\n{DIAGNOSTIC_FILE}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE EL REGISTRO BIBLIOGRAFICO"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)