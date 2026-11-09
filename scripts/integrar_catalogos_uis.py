from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NATIONAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "uis_sdg_national.parquet"
)

COUNTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uis"
    / "SDG_COUNTRY.csv"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uis"
    / "SDG_LABEL.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "uis_sdg_national_enriched.parquet"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_sdg_catalog_integration.csv"
)


# ============================================================
# FUNCIONES
# ============================================================

def verify_file(file_path: Path) -> None:
    """Comprueba que un archivo exista antes de procesarlo."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo:\n{file_path}"
        )


def main() -> None:
    print("=" * 68)
    print("INTEGRACION DE CATALOGOS UNESCO UIS")
    print("=" * 68)

    for file_path in [
        NATIONAL_FILE,
        COUNTRY_FILE,
        LABEL_FILE,
    ]:
        verify_file(file_path)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\nLeyendo la base nacional procesada...")
    national = pd.read_parquet(NATIONAL_FILE)

    print("Leyendo el catalogo de paises...")
    countries = pd.read_csv(
        COUNTRY_FILE,
        dtype={
            "COUNTRY_ID": "string",
            "COUNTRY_NAME_EN": "string",
        },
    )

    print("Leyendo el catalogo de indicadores...")
    labels = pd.read_csv(
        LABEL_FILE,
        dtype={
            "INDICATOR_ID": "string",
            "INDICATOR_LABEL_EN": "string",
        },
    )

    required_national = {
        "INDICATOR_ID",
        "COUNTRY_ID",
        "YEAR",
        "VALUE",
        "MAGNITUDE",
        "QUALIFIER",
    }

    required_countries = {
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
    }

    required_labels = {
        "INDICATOR_ID",
        "INDICATOR_LABEL_EN",
    }

    if not required_national.issubset(national.columns):
        raise ValueError(
            "La base nacional no contiene todas las columnas esperadas."
        )

    if not required_countries.issubset(countries.columns):
        raise ValueError(
            "El catalogo de paises no contiene las columnas esperadas."
        )

    if not required_labels.issubset(labels.columns):
        raise ValueError(
            "El catalogo de indicadores no contiene las columnas esperadas."
        )

    national_rows = len(national)

    duplicate_country_ids = int(
        countries["COUNTRY_ID"].duplicated().sum()
    )

    duplicate_indicator_ids = int(
        labels["INDICATOR_ID"].duplicated().sum()
    )

    print(f"\nFilas nacionales: {national_rows:,}")
    print(f"Paises en el catalogo: {len(countries):,}")
    print(f"Indicadores en el catalogo: {len(labels):,}")
    print(
        "Codigos de pais duplicados en catalogo:",
        duplicate_country_ids,
    )
    print(
        "Codigos de indicador duplicados en catalogo:",
        duplicate_indicator_ids,
    )

    if duplicate_country_ids > 0:
        raise ValueError(
            "El catalogo de paises tiene COUNTRY_ID duplicados."
        )

    if duplicate_indicator_ids > 0:
        raise ValueError(
            "El catalogo de indicadores tiene INDICATOR_ID duplicados."
        )

    print("\nIntegrando nombres de paises...")

    integrated = national.merge(
        countries[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
            ]
        ],
        on="COUNTRY_ID",
        how="left",
        validate="many_to_one",
    )

    print("Integrando nombres de indicadores...")

    integrated = integrated.merge(
        labels[
            [
                "INDICATOR_ID",
                "INDICATOR_LABEL_EN",
            ]
        ],
        on="INDICATOR_ID",
        how="left",
        validate="many_to_one",
    )

    if len(integrated) != national_rows:
        raise RuntimeError(
            "La integracion cambio inesperadamente el numero de filas."
        )

    missing_country_names = int(
        integrated["COUNTRY_NAME_EN"].isna().sum()
    )

    missing_indicator_labels = int(
        integrated["INDICATOR_LABEL_EN"].isna().sum()
    )

    unmatched_country_codes = (
        integrated.loc[
            integrated["COUNTRY_NAME_EN"].isna(),
            "COUNTRY_ID",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    unmatched_indicator_codes = (
        integrated.loc[
            integrated["INDICATOR_LABEL_EN"].isna(),
            "INDICATOR_ID",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    integrated = integrated[
        [
            "INDICATOR_ID",
            "INDICATOR_LABEL_EN",
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "YEAR",
            "VALUE",
            "MAGNITUDE",
            "QUALIFIER",
        ]
    ]

    print("\nGuardando la base enriquecida...")

    integrated.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    validation = pd.read_parquet(OUTPUT_FILE)

    if len(validation) != national_rows:
        raise RuntimeError(
            "La validacion del archivo enriquecido fallo."
        )

    diagnostic = pd.DataFrame(
        [
            {
                "processing_datetime": (
                    datetime.now().astimezone().isoformat()
                ),
                "input_file": NATIONAL_FILE.name,
                "country_catalog": COUNTRY_FILE.name,
                "indicator_catalog": LABEL_FILE.name,
                "output_file": OUTPUT_FILE.name,
                "national_rows": national_rows,
                "final_rows": len(integrated),
                "country_catalog_rows": len(countries),
                "indicator_catalog_rows": len(labels),
                "duplicate_country_ids": duplicate_country_ids,
                "duplicate_indicator_ids": duplicate_indicator_ids,
                "missing_country_names": missing_country_names,
                "missing_indicator_labels": missing_indicator_labels,
                "unmatched_country_codes": "|".join(
                    unmatched_country_codes
                ),
                "unmatched_indicator_codes": "|".join(
                    unmatched_indicator_codes
                ),
                "output_size_bytes": OUTPUT_FILE.stat().st_size,
                "validation_status": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 68)
    print("INTEGRACION COMPLETADA CORRECTAMENTE")
    print("=" * 68)

    print(f"\nFilas finales: {len(integrated):,}")
    print(
        "Registros sin nombre de pais:",
        f"{missing_country_names:,}",
    )
    print(
        "Registros sin etiqueta de indicador:",
        f"{missing_indicator_labels:,}",
    )

    if unmatched_country_codes:
        print(
            "\nCodigos de pais sin coincidencia:",
            unmatched_country_codes,
        )

    if unmatched_indicator_codes:
        print(
            "\nCodigos de indicador sin coincidencia:",
            unmatched_indicator_codes[:20],
        )

    print(f"\nBase enriquecida:\n{OUTPUT_FILE}")
    print(f"\nDiagnostico:\n{DIAGNOSTIC_FILE}")
    print(
        "\nTamanio de la base enriquecida:",
        f"{OUTPUT_FILE.stat().st_size:,} bytes",
    )
    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 68)
        print("ERROR DURANTE LA INTEGRACION")
        print("=" * 68)
        print(f"\nTipo de error: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)