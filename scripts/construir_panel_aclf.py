from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


# ============================================================
# CONFIGURACION DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_panel_wide.parquet"
)

OUTPUT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_complete.parquet"
)

OUTPUT_COUNTRY_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_aclf_country_summary.csv"
)

OUTPUT_GENERAL_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_aclf_summary.csv"
)

OUTPUT_GAPS = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_aclf_temporal_gaps.csv"
)


MODEL_VARIABLES = ["A", "C", "L", "F"]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title: str) -> None:
    """Muestra un encabezado en la consola."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def verify_file(file_path: Path) -> None:
    """Comprueba que un archivo exista."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def calculate_country_gaps(
    country_data: pd.DataFrame,
) -> dict:
    """
    Calcula los intervalos entre observaciones de un país.

    No interpola datos ni crea años artificiales.
    """
    ordered_years = (
        country_data["YEAR"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not ordered_years:
        return {
            "MIN_YEAR": None,
            "MAX_YEAR": None,
            "OBSERVED_YEARS": 0,
            "CALENDAR_SPAN": 0,
            "MISSING_CALENDAR_YEARS": 0,
            "MAX_YEAR_GAP": None,
            "MEAN_YEAR_GAP": None,
            "CONSECUTIVE_TRANSITIONS": 0,
            "TRANSITIONS_TOTAL": 0,
        }

    minimum_year = min(ordered_years)
    maximum_year = max(ordered_years)
    calendar_span = maximum_year - minimum_year + 1

    missing_calendar_years = (
        calendar_span - len(ordered_years)
    )

    year_gaps = [
        current_year - previous_year
        for previous_year, current_year
        in zip(ordered_years[:-1], ordered_years[1:])
    ]

    transitions_total = len(year_gaps)

    consecutive_transitions = sum(
        gap == 1 for gap in year_gaps
    )

    maximum_gap = (
        max(year_gaps)
        if year_gaps
        else 0
    )

    mean_gap = (
        sum(year_gaps) / len(year_gaps)
        if year_gaps
        else 0
    )

    return {
        "MIN_YEAR": minimum_year,
        "MAX_YEAR": maximum_year,
        "OBSERVED_YEARS": len(ordered_years),
        "CALENDAR_SPAN": calendar_span,
        "MISSING_CALENDAR_YEARS": (
            missing_calendar_years
        ),
        "MAX_YEAR_GAP": maximum_gap,
        "MEAN_YEAR_GAP": round(mean_gap, 2),
        "CONSECUTIVE_TRANSITIONS": (
            consecutive_transitions
        ),
        "TRANSITIONS_TOTAL": transitions_total,
    }


def main() -> None:
    print_section(
        "CONSTRUCCION DEL PANEL DEFINITIVO A-C-L-F"
    )

    verify_file(INPUT_PANEL)

    OUTPUT_PANEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_COUNTRY_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLeyendo el panel general...")

    panel = pd.read_parquet(INPUT_PANEL)

    required_columns = {
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "YEAR",
        *MODEL_VARIABLES,
    }

    missing_columns = sorted(
        required_columns - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el panel: {missing_columns}"
        )

    print(
        "Filas disponibles en el panel general:",
        f"{len(panel):,}",
    )

    print("\nSeleccionando casos completos A-C-L-F...")

    complete_mask = (
        panel[MODEL_VARIABLES]
        .notna()
        .all(axis=1)
    )

    aclf = panel.loc[
        complete_mask,
        [
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "YEAR",
            *MODEL_VARIABLES,
        ],
    ].copy()

    aclf["YEAR"] = pd.to_numeric(
        aclf["YEAR"],
        errors="coerce",
    ).astype("Int64")

    for variable in MODEL_VARIABLES:
        aclf[variable] = pd.to_numeric(
            aclf[variable],
            errors="coerce",
        )

    aclf = aclf.dropna(
        subset=[
            "COUNTRY_ID",
            "YEAR",
            *MODEL_VARIABLES,
        ]
    ).copy()

    duplicate_keys = int(
        aclf.duplicated(
            subset=["COUNTRY_ID", "YEAR"]
        ).sum()
    )

    if duplicate_keys > 0:
        raise ValueError(
            "Se encontraron filas duplicadas para "
            f"pais-anio: {duplicate_keys}"
        )

    aclf = aclf.sort_values(
        by=["COUNTRY_ID", "YEAR"]
    ).reset_index(drop=True)

    print(
        "Filas completas A-C-L-F:",
        f"{len(aclf):,}",
    )

    print(
        "Paises o territorios:",
        f"{aclf['COUNTRY_ID'].nunique():,}",
    )

    # ========================================================
    # RESUMEN Y CONTINUIDAD POR PAIS
    # ========================================================

    print("\nCalculando continuidad temporal por pais...")

    country_rows = []
    gap_rows = []

    for (
        country_id,
        country_name,
    ), country_data in aclf.groupby(
        [
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
        ],
        dropna=False,
    ):
        country_data = country_data.sort_values(
            "YEAR"
        )

        gap_statistics = calculate_country_gaps(
            country_data
        )

        country_rows.append(
            {
                "COUNTRY_ID": country_id,
                "COUNTRY_NAME_EN": country_name,
                **gap_statistics,
                "HAS_AT_LEAST_3_OBSERVATIONS": (
                    gap_statistics["OBSERVED_YEARS"]
                    >= 3
                ),
                "HAS_AT_LEAST_5_OBSERVATIONS": (
                    gap_statistics["OBSERVED_YEARS"]
                    >= 5
                ),
                "HAS_AT_LEAST_10_OBSERVATIONS": (
                    gap_statistics["OBSERVED_YEARS"]
                    >= 10
                ),
                "MEAN_A": country_data["A"].mean(),
                "MEAN_C": country_data["C"].mean(),
                "MEAN_L": country_data["L"].mean(),
                "MEAN_F": country_data["F"].mean(),
            }
        )

        years = (
            country_data["YEAR"]
            .astype(int)
            .sort_values()
            .tolist()
        )

        for previous_year, current_year in zip(
            years[:-1],
            years[1:],
        ):
            gap = current_year - previous_year

            gap_rows.append(
                {
                    "COUNTRY_ID": country_id,
                    "COUNTRY_NAME_EN": country_name,
                    "PREVIOUS_YEAR": previous_year,
                    "CURRENT_YEAR": current_year,
                    "YEAR_GAP": gap,
                    "IS_CONSECUTIVE": gap == 1,
                }
            )

    country_summary = pd.DataFrame(
        country_rows
    )

    temporal_gaps = pd.DataFrame(
        gap_rows
    )

    for variable in MODEL_VARIABLES:
        country_summary[f"MEAN_{variable}"] = (
            country_summary[f"MEAN_{variable}"]
            .round(4)
        )

    country_summary = country_summary.sort_values(
        by=[
            "OBSERVED_YEARS",
            "CONSECUTIVE_TRANSITIONS",
            "MAX_YEAR_GAP",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    countries_at_least_3 = int(
        country_summary[
            "HAS_AT_LEAST_3_OBSERVATIONS"
        ].sum()
    )

    countries_at_least_5 = int(
        country_summary[
            "HAS_AT_LEAST_5_OBSERVATIONS"
        ].sum()
    )

    countries_at_least_10 = int(
        country_summary[
            "HAS_AT_LEAST_10_OBSERVATIONS"
        ].sum()
    )

    total_transitions = int(
        country_summary[
            "TRANSITIONS_TOTAL"
        ].sum()
    )

    consecutive_transitions = int(
        country_summary[
            "CONSECUTIVE_TRANSITIONS"
        ].sum()
    )

    consecutive_percentage = (
        consecutive_transitions
        / total_transitions
        * 100
        if total_transitions > 0
        else 0
    )

    # ========================================================
    # RESUMEN GENERAL
    # ========================================================

    general_summary = pd.DataFrame(
        [
            {
                "processing_datetime": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "model_variables": "|".join(
                    MODEL_VARIABLES
                ),
                "country_year_rows": len(aclf),
                "country_count": int(
                    aclf["COUNTRY_ID"].nunique()
                ),
                "minimum_year": int(
                    aclf["YEAR"].min()
                ),
                "maximum_year": int(
                    aclf["YEAR"].max()
                ),
                "duplicate_country_year_rows": (
                    duplicate_keys
                ),
                "countries_at_least_3_observations": (
                    countries_at_least_3
                ),
                "countries_at_least_5_observations": (
                    countries_at_least_5
                ),
                "countries_at_least_10_observations": (
                    countries_at_least_10
                ),
                "total_temporal_transitions": (
                    total_transitions
                ),
                "consecutive_transitions": (
                    consecutive_transitions
                ),
                "consecutive_transition_percent": (
                    round(
                        consecutive_percentage,
                        2,
                    )
                ),
                "output_panel": OUTPUT_PANEL.name,
                "validation_status": "OK",
            }
        ]
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    print("\nGuardando panel A-C-L-F...")

    aclf.to_parquet(
        OUTPUT_PANEL,
        index=False,
        compression="snappy",
    )

    country_summary.to_csv(
        OUTPUT_COUNTRY_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    general_summary.to_csv(
        OUTPUT_GENERAL_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    temporal_gaps.to_csv(
        OUTPUT_GAPS,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # VALIDACION
    # ========================================================

    validation = pd.read_parquet(
        OUTPUT_PANEL
    )

    if len(validation) != len(aclf):
        raise RuntimeError(
            "La validacion del panel A-C-L-F fallo."
        )

    print_section(
        "PANEL A-C-L-F CONSTRUIDO CORRECTAMENTE"
    )

    print(
        "\nFilas pais-anio:",
        f"{len(aclf):,}",
    )

    print(
        "Paises o territorios:",
        f"{aclf['COUNTRY_ID'].nunique():,}",
    )

    print(
        "Periodo general:",
        f"{int(aclf['YEAR'].min())} - "
        f"{int(aclf['YEAR'].max())}",
    )

    print(
        "\nPaises con al menos 3 observaciones:",
        countries_at_least_3,
    )

    print(
        "Paises con al menos 5 observaciones:",
        countries_at_least_5,
    )

    print(
        "Paises con al menos 10 observaciones:",
        countries_at_least_10,
    )

    print(
        "\nTransiciones temporales totales:",
        total_transitions,
    )

    print(
        "Transiciones entre anios consecutivos:",
        consecutive_transitions,
    )

    print(
        "Porcentaje de transiciones consecutivas:",
        f"{consecutive_percentage:.2f}%",
    )

    print(
        "\nPRIMEROS 20 PAISES CON MAYOR "
        "NUMERO DE OBSERVACIONES"
    )

    print(
        country_summary[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "MIN_YEAR",
                "MAX_YEAR",
                "OBSERVED_YEARS",
                "MISSING_CALENDAR_YEARS",
                "MAX_YEAR_GAP",
                "CONSECUTIVE_TRANSITIONS",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nPanel definitivo:\n{OUTPUT_PANEL}")

    print(
        f"\nResumen por pais:\n"
        f"{OUTPUT_COUNTRY_SUMMARY}"
    )

    print(
        f"\nResumen general:\n"
        f"{OUTPUT_GENERAL_SUMMARY}"
    )

    print(
        f"\nIntervalos temporales:\n"
        f"{OUTPUT_GAPS}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE LA CONSTRUCCION DEL PANEL A-C-L-F"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(f"Detalle: {error}")

        sys.exit(1)