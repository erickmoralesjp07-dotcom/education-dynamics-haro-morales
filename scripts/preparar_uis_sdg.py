from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import pandas as pd


# ============================================================
# CONFIGURACION DE RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uis"
    / "SDG_DATA_NATIONAL.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PARQUET = OUTPUT_DIR / "uis_sdg_national.parquet"

DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "diagnostics"
DIAGNOSTIC_CSV = DIAGNOSTICS_DIR / "uis_sdg_national_diagnostic.csv"

EXPECTED_COLUMNS = [
    "INDICATOR_ID",
    "COUNTRY_ID",
    "YEAR",
    "VALUE",
    "MAGNITUDE",
    "QUALIFIER",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula la huella SHA-256 del archivo original."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Comprueba que el archivo tenga exactamente las columnas esperadas."""
    actual_columns = dataframe.columns.tolist()

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    extra_columns = [
        column for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    if extra_columns:
        print(
            "ADVERTENCIA: se encontraron columnas adicionales:",
            extra_columns
        )


def main() -> None:
    print("=" * 65)
    print("PREPARACION DE DATOS NACIONALES UNESCO UIS - ODS 4")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de entrada:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nArchivo de entrada:\n{INPUT_FILE}")
    print("\nLeyendo el archivo CSV completo...")

    dataframe = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        dtype={
            "INDICATOR_ID": "string",
            "COUNTRY_ID": "string",
            "MAGNITUDE": "string",
            "QUALIFIER": "string",
        },
    )

    validate_columns(dataframe)

    original_rows = len(dataframe)
    original_columns = len(dataframe.columns)
    exact_duplicates = int(dataframe.duplicated().sum())

    print(f"Filas originales: {original_rows:,}")
    print(f"Columnas: {original_columns}")
    print(f"Duplicados exactos encontrados: {exact_duplicates:,}")

    # Normalizar identificadores de texto
    for column in [
        "INDICATOR_ID",
        "COUNTRY_ID",
        "MAGNITUDE",
        "QUALIFIER",
    ]:
        dataframe[column] = dataframe[column].str.strip()

    # Convertir las columnas numericas
    dataframe["YEAR"] = pd.to_numeric(
        dataframe["YEAR"],
        errors="coerce",
    ).astype("Int64")

    dataframe["VALUE"] = pd.to_numeric(
        dataframe["VALUE"],
        errors="coerce",
    )

    # Eliminar duplicados exactos
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    final_rows = len(dataframe)
    removed_duplicates = original_rows - final_rows

    missing_indicator = int(dataframe["INDICATOR_ID"].isna().sum())
    missing_country = int(dataframe["COUNTRY_ID"].isna().sum())
    missing_year = int(dataframe["YEAR"].isna().sum())
    missing_value = int(dataframe["VALUE"].isna().sum())

    unique_indicators = int(dataframe["INDICATOR_ID"].nunique())
    unique_countries = int(dataframe["COUNTRY_ID"].nunique())

    valid_years = dataframe["YEAR"].dropna()

    minimum_year = (
        int(valid_years.min())
        if not valid_years.empty
        else None
    )

    maximum_year = (
        int(valid_years.max())
        if not valid_years.empty
        else None
    )

    print("\nRESUMEN DESPUES DE LA LIMPIEZA")
    print(f"Filas finales: {final_rows:,}")
    print(f"Duplicados eliminados: {removed_duplicates:,}")
    print(f"Indicadores unicos: {unique_indicators:,}")
    print(f"Paises o territorios unicos: {unique_countries:,}")
    print(f"Anio minimo: {minimum_year}")
    print(f"Anio maximo: {maximum_year}")
    print(f"Valores numericos ausentes: {missing_value:,}")

    # Ordenar para facilitar analisis posteriores
    dataframe = dataframe.sort_values(
        by=["INDICATOR_ID", "COUNTRY_ID", "YEAR"],
        na_position="last",
    ).reset_index(drop=True)

    print("\nGuardando archivo Parquet...")

    dataframe.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy",
    )

    # Verificar que el Parquet pueda volver a leerse
    parquet_test = pd.read_parquet(OUTPUT_PARQUET)

    if len(parquet_test) != final_rows:
        raise RuntimeError(
            "La verificacion del Parquet fallo: "
            "el numero de filas no coincide."
        )

    source_sha256 = calculate_sha256(INPUT_FILE)
    extraction_time = datetime.now().astimezone().isoformat()

    diagnostic = pd.DataFrame(
        [
            {
                "source_file": INPUT_FILE.name,
                "output_file": OUTPUT_PARQUET.name,
                "processing_datetime": extraction_time,
                "source_sha256": source_sha256,
                "original_rows": original_rows,
                "final_rows": final_rows,
                "columns": original_columns,
                "exact_duplicates_found": exact_duplicates,
                "duplicates_removed": removed_duplicates,
                "unique_indicators": unique_indicators,
                "unique_countries": unique_countries,
                "minimum_year": minimum_year,
                "maximum_year": maximum_year,
                "missing_indicator_id": missing_indicator,
                "missing_country_id": missing_country,
                "missing_year": missing_year,
                "missing_value": missing_value,
                "parquet_size_bytes": OUTPUT_PARQUET.stat().st_size,
                "validation_status": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("PROCESAMIENTO COMPLETADO CORRECTAMENTE")
    print("=" * 65)

    print(f"\nParquet creado:\n{OUTPUT_PARQUET}")
    print(f"\nDiagnostico creado:\n{DIAGNOSTIC_CSV}")
    print(f"\nHuella SHA-256:\n{source_sha256}")
    print(f"\nTamanio del Parquet: {OUTPUT_PARQUET.stat().st_size:,} bytes")
    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 65)
        print("ERROR DURANTE EL PROCESAMIENTO")
        print("=" * 65)
        print(f"\nTipo de error: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)