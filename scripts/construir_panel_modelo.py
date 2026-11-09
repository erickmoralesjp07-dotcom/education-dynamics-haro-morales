from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NATIONAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "uis_sdg_national_enriched.parquet"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_selected_indicators.csv"
)

OUTPUT_LONG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_panel_long.parquet"
)

OUTPUT_WIDE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_panel_wide.parquet"
)

OUTPUT_COVERAGE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_panel_coverage.csv"
)

OUTPUT_COUNTRY_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_country_coverage.csv"
)

OUTPUT_INDICATOR_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_indicator_coverage.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def verify_file(file_path: Path) -> None:
    """Comprueba que el archivo exista antes de procesarlo."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def print_section(title: str) -> None:
    """Imprime un encabezado legible en la consola."""
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main() -> None:
    print_section(
        "CONSTRUCCION DEL PANEL PARA EL MODELO DE DINAMICA EDUCATIVA"
    )

    verify_file(NATIONAL_FILE)
    verify_file(SELECTION_FILE)

    OUTPUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)

    print("\nLeyendo la seleccion final de indicadores...")

    selection = pd.read_csv(
        SELECTION_FILE,
        dtype={
            "MODEL_DIMENSION": "string",
            "MODEL_VARIABLE": "string",
            "INDICATOR_ID": "string",
            "INDICATOR_LABEL_EN": "string",
            "DESCRIPTION_ES": "string",
            "DIRECTION": "string",
        },
    )

    required_selection_columns = {
        "MODEL_DIMENSION",
        "MODEL_VARIABLE",
        "INDICATOR_ID",
        "INDICATOR_LABEL_EN",
        "DESCRIPTION_ES",
        "DIRECTION",
    }

    if not required_selection_columns.issubset(selection.columns):
        missing = sorted(
            required_selection_columns - set(selection.columns)
        )

        raise ValueError(
            f"Faltan columnas en la seleccion: {missing}"
        )

    if selection["INDICATOR_ID"].duplicated().any():
        duplicated = (
            selection.loc[
                selection["INDICATOR_ID"].duplicated(
                    keep=False
                ),
                "INDICATOR_ID",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Existen indicadores seleccionados duplicados: {duplicated}"
        )

    if selection["MODEL_VARIABLE"].duplicated().any():
        duplicated = (
            selection.loc[
                selection["MODEL_VARIABLE"].duplicated(
                    keep=False
                ),
                "MODEL_VARIABLE",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Existen variables del modelo duplicadas: {duplicated}"
        )

    selected_ids = selection["INDICATOR_ID"].tolist()
    expected_variable_count = len(selection)

    print(
        "Indicadores seleccionados:",
        expected_variable_count,
    )

    print("\nLeyendo la base nacional enriquecida...")

    national = pd.read_parquet(
        NATIONAL_FILE,
        columns=[
            "INDICATOR_ID",
            "INDICATOR_LABEL_EN",
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "YEAR",
            "VALUE",
            "MAGNITUDE",
            "QUALIFIER",
        ],
    )

    print(f"Filas disponibles en la base completa: {len(national):,}")

    print("\nFiltrando los ocho indicadores seleccionados...")

    panel_long = national.loc[
        national["INDICATOR_ID"].isin(selected_ids)
    ].copy()

    if panel_long.empty:
        raise ValueError(
            "No se encontraron datos para los indicadores seleccionados."
        )

    found_ids = set(
        panel_long["INDICATOR_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    missing_ids = sorted(set(selected_ids) - found_ids)

    if missing_ids:
        raise ValueError(
            "No se encontraron datos nacionales para estos "
            f"indicadores: {missing_ids}"
        )

    print(f"Filas despues del filtro: {len(panel_long):,}")

    print("\nIntegrando variables y dimensiones del modelo...")

    selection_for_merge = selection[
        [
            "INDICATOR_ID",
            "MODEL_DIMENSION",
            "MODEL_VARIABLE",
            "DESCRIPTION_ES",
            "DIRECTION",
        ]
    ].copy()

    panel_long = panel_long.merge(
        selection_for_merge,
        on="INDICATOR_ID",
        how="left",
        validate="many_to_one",
    )

    panel_long["YEAR"] = pd.to_numeric(
        panel_long["YEAR"],
        errors="coerce",
    ).astype("Int64")

    panel_long["VALUE"] = pd.to_numeric(
        panel_long["VALUE"],
        errors="coerce",
    )

    panel_long = panel_long.dropna(
        subset=[
            "COUNTRY_ID",
            "YEAR",
            "MODEL_VARIABLE",
            "VALUE",
        ]
    ).copy()

    exact_duplicates = int(
        panel_long.duplicated().sum()
    )

    panel_long = panel_long.drop_duplicates().copy()

    # Comprobar si existe mas de un valor para el mismo
    # pais, anio y variable del modelo.
    duplicate_keys = (
        panel_long.duplicated(
            subset=[
                "COUNTRY_ID",
                "YEAR",
                "MODEL_VARIABLE",
            ],
            keep=False,
        )
    )

    duplicate_key_rows = int(duplicate_keys.sum())

    if duplicate_key_rows > 0:
        duplicate_examples = (
            panel_long.loc[
                duplicate_keys,
                [
                    "COUNTRY_ID",
                    "YEAR",
                    "MODEL_VARIABLE",
                    "INDICATOR_ID",
                    "VALUE",
                ],
            ]
            .sort_values(
                [
                    "COUNTRY_ID",
                    "YEAR",
                    "MODEL_VARIABLE",
                ]
            )
            .head(20)
        )

        print(
            "\nADVERTENCIA: se detectaron filas repetidas para "
            "pais, anio y variable."
        )

        print(duplicate_examples.to_string(index=False))

        print(
            "\nSe conservara el promedio de los valores repetidos "
            "para construir el panel ancho."
        )

    panel_long = panel_long.sort_values(
        by=[
            "COUNTRY_ID",
            "YEAR",
            "MODEL_VARIABLE",
        ]
    ).reset_index(drop=True)

    print("\nConstruyendo el panel ancho pais-anio...")

    value_panel = panel_long.pivot_table(
        index=[
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "YEAR",
        ],
        columns="MODEL_VARIABLE",
        values="VALUE",
        aggfunc="mean",
    ).reset_index()

    value_panel.columns.name = None

    expected_variables = (
        selection["MODEL_VARIABLE"]
        .dropna()
        .tolist()
    )

    for variable in expected_variables:
        if variable not in value_panel.columns:
            value_panel[variable] = pd.NA

    value_panel["AVAILABLE_VARIABLES"] = (
        value_panel[expected_variables]
        .notna()
        .sum(axis=1)
        .astype("Int64")
    )

    value_panel["TOTAL_MODEL_VARIABLES"] = (
        expected_variable_count
    )

    value_panel["COVERAGE_PERCENT"] = (
        value_panel["AVAILABLE_VARIABLES"]
        / expected_variable_count
        * 100
    )

    value_panel["COMPLETE_CASE"] = (
        value_panel["AVAILABLE_VARIABLES"]
        .eq(expected_variable_count)
    )

    value_panel["AT_LEAST_6_VARIABLES"] = (
        value_panel["AVAILABLE_VARIABLES"]
        .ge(6)
    )

    value_panel["AT_LEAST_4_VARIABLES"] = (
        value_panel["AVAILABLE_VARIABLES"]
        .ge(4)
    )

    output_columns = [
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "YEAR",
        *expected_variables,
        "AVAILABLE_VARIABLES",
        "TOTAL_MODEL_VARIABLES",
        "COVERAGE_PERCENT",
        "COMPLETE_CASE",
        "AT_LEAST_6_VARIABLES",
        "AT_LEAST_4_VARIABLES",
    ]

    value_panel = value_panel[
        output_columns
    ].sort_values(
        by=[
            "COUNTRY_ID",
            "YEAR",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # RESUMEN POR INDICADOR
    # ========================================================

    indicator_summary = (
        panel_long
        .groupby(
            [
                "MODEL_DIMENSION",
                "MODEL_VARIABLE",
                "INDICATOR_ID",
                "INDICATOR_LABEL_EN",
            ],
            dropna=False,
        )
        .agg(
            RECORD_COUNT=("VALUE", "size"),
            COUNTRY_COUNT=("COUNTRY_ID", "nunique"),
            YEAR_COUNT=("YEAR", "nunique"),
            MIN_YEAR=("YEAR", "min"),
            MAX_YEAR=("YEAR", "max"),
            MIN_VALUE=("VALUE", "min"),
            MAX_VALUE=("VALUE", "max"),
            MEAN_VALUE=("VALUE", "mean"),
        )
        .reset_index()
        .sort_values("MODEL_VARIABLE")
    )

    # ========================================================
    # RESUMEN POR PAIS
    # ========================================================

    country_summary = (
        value_panel
        .groupby(
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
            ],
            dropna=False,
        )
        .agg(
            MIN_YEAR=("YEAR", "min"),
            MAX_YEAR=("YEAR", "max"),
            OBSERVED_YEARS=("YEAR", "nunique"),
            COMPLETE_YEARS=("COMPLETE_CASE", "sum"),
            YEARS_WITH_AT_LEAST_6=(
                "AT_LEAST_6_VARIABLES",
                "sum",
            ),
            YEARS_WITH_AT_LEAST_4=(
                "AT_LEAST_4_VARIABLES",
                "sum",
            ),
            MAX_AVAILABLE_VARIABLES=(
                "AVAILABLE_VARIABLES",
                "max",
            ),
            MEAN_AVAILABLE_VARIABLES=(
                "AVAILABLE_VARIABLES",
                "mean",
            ),
        )
        .reset_index()
    )

    country_summary["MEAN_AVAILABLE_VARIABLES"] = (
        country_summary["MEAN_AVAILABLE_VARIABLES"]
        .round(2)
    )

    country_summary = country_summary.sort_values(
        by=[
            "COMPLETE_YEARS",
            "YEARS_WITH_AT_LEAST_6",
            "YEARS_WITH_AT_LEAST_4",
            "MEAN_AVAILABLE_VARIABLES",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # ========================================================
    # RESUMEN GENERAL
    # ========================================================

    complete_cases = int(
        value_panel["COMPLETE_CASE"].sum()
    )

    cases_at_least_6 = int(
        value_panel["AT_LEAST_6_VARIABLES"].sum()
    )

    cases_at_least_4 = int(
        value_panel["AT_LEAST_4_VARIABLES"].sum()
    )

    complete_case_countries = int(
        value_panel.loc[
            value_panel["COMPLETE_CASE"],
            "COUNTRY_ID",
        ].nunique()
    )

    countries_at_least_6 = int(
        value_panel.loc[
            value_panel["AT_LEAST_6_VARIABLES"],
            "COUNTRY_ID",
        ].nunique()
    )

    countries_at_least_4 = int(
        value_panel.loc[
            value_panel["AT_LEAST_4_VARIABLES"],
            "COUNTRY_ID",
        ].nunique()
    )

    coverage_summary = pd.DataFrame(
        [
            {
                "processing_datetime": (
                    datetime.now().astimezone().isoformat()
                ),
                "selected_indicator_count": (
                    expected_variable_count
                ),
                "long_panel_rows": len(panel_long),
                "country_year_rows": len(value_panel),
                "country_count": int(
                    value_panel["COUNTRY_ID"].nunique()
                ),
                "minimum_year": int(
                    value_panel["YEAR"].min()
                ),
                "maximum_year": int(
                    value_panel["YEAR"].max()
                ),
                "exact_duplicates_removed": (
                    exact_duplicates
                ),
                "duplicate_key_rows_detected": (
                    duplicate_key_rows
                ),
                "complete_country_year_rows": (
                    complete_cases
                ),
                "country_year_rows_at_least_6": (
                    cases_at_least_6
                ),
                "country_year_rows_at_least_4": (
                    cases_at_least_4
                ),
                "countries_with_complete_case": (
                    complete_case_countries
                ),
                "countries_with_at_least_6": (
                    countries_at_least_6
                ),
                "countries_with_at_least_4": (
                    countries_at_least_4
                ),
                "long_output_file": OUTPUT_LONG.name,
                "wide_output_file": OUTPUT_WIDE.name,
                "validation_status": "OK",
            }
        ]
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    print("\nGuardando el panel largo...")

    panel_long.to_parquet(
        OUTPUT_LONG,
        index=False,
        compression="snappy",
    )

    print("Guardando el panel ancho...")

    value_panel.to_parquet(
        OUTPUT_WIDE,
        index=False,
        compression="snappy",
    )

    coverage_summary.to_csv(
        OUTPUT_COVERAGE,
        index=False,
        encoding="utf-8-sig",
    )

    country_summary.to_csv(
        OUTPUT_COUNTRY_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    indicator_summary.to_csv(
        OUTPUT_INDICATOR_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # VALIDACION DE ARCHIVOS
    # ========================================================

    validated_long = pd.read_parquet(OUTPUT_LONG)
    validated_wide = pd.read_parquet(OUTPUT_WIDE)

    if len(validated_long) != len(panel_long):
        raise RuntimeError(
            "El numero de filas del panel largo no coincide."
        )

    if len(validated_wide) != len(value_panel):
        raise RuntimeError(
            "El numero de filas del panel ancho no coincide."
        )

    print_section("PANEL CONSTRUIDO CORRECTAMENTE")

    print(f"\nFilas del panel largo: {len(panel_long):,}")
    print(
        "Filas pais-anio del panel ancho:",
        f"{len(value_panel):,}",
    )
    print(
        "Paises o territorios:",
        f"{value_panel['COUNTRY_ID'].nunique():,}",
    )
    print(
        "Periodo:",
        f"{int(value_panel['YEAR'].min())} - "
        f"{int(value_panel['YEAR'].max())}",
    )

    print(
        "\nFilas pais-anio con las 8 variables:",
        f"{complete_cases:,}",
    )

    print(
        "Paises con al menos un anio completo:",
        f"{complete_case_countries:,}",
    )

    print(
        "\nFilas pais-anio con al menos 6 variables:",
        f"{cases_at_least_6:,}",
    )

    print(
        "Paises con al menos 6 variables en algun anio:",
        f"{countries_at_least_6:,}",
    )

    print(
        "\nFilas pais-anio con al menos 4 variables:",
        f"{cases_at_least_4:,}",
    )

    print(
        "Paises con al menos 4 variables en algun anio:",
        f"{countries_at_least_4:,}",
    )

    print("\nCOBERTURA POR INDICADOR")

    print(
        indicator_summary[
            [
                "MODEL_VARIABLE",
                "INDICATOR_ID",
                "RECORD_COUNT",
                "COUNTRY_COUNT",
                "MIN_YEAR",
                "MAX_YEAR",
            ]
        ].to_string(index=False)
    )

    print("\nPRIMEROS 15 PAISES CON MAYOR COBERTURA")

    print(
        country_summary[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "OBSERVED_YEARS",
                "COMPLETE_YEARS",
                "YEARS_WITH_AT_LEAST_6",
                "YEARS_WITH_AT_LEAST_4",
                "MAX_AVAILABLE_VARIABLES",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(f"\nPanel largo:\n{OUTPUT_LONG}")
    print(f"\nPanel ancho:\n{OUTPUT_WIDE}")
    print(f"\nResumen general:\n{OUTPUT_COVERAGE}")
    print(f"\nCobertura por pais:\n{OUTPUT_COUNTRY_SUMMARY}")
    print(
        f"\nCobertura por indicador:\n"
        f"{OUTPUT_INDICATOR_SUMMARY}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section("ERROR DURANTE LA CONSTRUCCION DEL PANEL")
        print(f"\nTipo de error: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)