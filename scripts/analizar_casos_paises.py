from pathlib import Path
from datetime import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_complete.parquet"
)

RATES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

TRAJECTORY_TABLE_FILE = (
    TABLES_DIR
    / "aclf_selected_countries_trajectories.csv"
)

TREND_TABLE_FILE = (
    TABLES_DIR
    / "aclf_selected_countries_trends.csv"
)

RATE_TABLE_FILE = (
    TABLES_DIR
    / "aclf_selected_countries_rates.csv"
)

SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_country_case_analysis_summary.csv"
)

ECUADOR_FIGURE = (
    FIGURES_DIR
    / "aclf_ecuador_trajectories.png"
)

COMPARISON_FIGURE = (
    FIGURES_DIR
    / "aclf_country_comparison.png"
)

STANDARDIZED_FIGURE = (
    FIGURES_DIR
    / "aclf_country_standardized_change.png"
)

ECUADOR_RATES_FIGURE = (
    FIGURES_DIR
    / "aclf_ecuador_rates.png"
)


# ============================================================
# CONFIGURACION
# ============================================================

SELECTED_COUNTRIES = {
    "ECU": "Ecuador",
    "BRA": "Brazil",
    "BOL": "Bolivia",
    "COL": "Colombia",
    "ESP": "Spain",
    "MEX": "Mexico",
    "PER": "Peru",
}

VARIABLES = ["A", "C", "L", "F"]

VARIABLE_LABELS = {
    "A": (
        "Exclusion escolar en primaria (%)"
    ),
    "C": (
        "Finalizacion de secundaria inferior (%)"
    ),
    "L": (
        "Alfabetizacion juvenil (%)"
    ),
    "F": (
        "Gasto publico educativo (% del PIB)"
    ),
}

RATE_COLUMNS = {
    "DA_DT": "Cambio anual de A",
    "DC_DT": "Cambio anual de C",
    "DL_DT": "Cambio anual de L",
    "DF_DT": "Cambio anual de F",
}

COUNTRY_COLORS = {
    "ECU": "#1f77b4",
    "BRA": "#2ca02c",
    "BOL": "#d62728",
    "COL": "#ff7f0e",
    "ESP": "#9467bd",
    "MEX": "#8c564b",
    "PER": "#17becf",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_file(file_path):
    """Comprueba que exista un archivo requerido."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def calculate_linear_trend(
    country_data,
    variable,
):
    """
    Calcula una tendencia lineal simple respecto al año.

    La pendiente representa el cambio promedio estimado
    de la variable por cada año.
    """

    valid_data = country_data[
        ["YEAR", variable]
    ].dropna()

    if len(valid_data) < 2:
        return {
            "SLOPE_PER_YEAR": np.nan,
            "INTERCEPT": np.nan,
            "R2_TREND": np.nan,
        }

    years = (
        valid_data["YEAR"]
        .astype(float)
        .to_numpy()
    )

    values = (
        valid_data[variable]
        .astype(float)
        .to_numpy()
    )

    slope, intercept = np.polyfit(
        years,
        values,
        1,
    )

    predicted = (
        slope * years
        + intercept
    )

    residual_sum = float(
        np.sum(
            (values - predicted) ** 2
        )
    )

    total_sum = float(
        np.sum(
            (values - values.mean()) ** 2
        )
    )

    if total_sum > 0:
        r2 = (
            1
            - residual_sum / total_sum
        )
    else:
        r2 = np.nan

    return {
        "SLOPE_PER_YEAR": float(slope),
        "INTERCEPT": float(intercept),
        "R2_TREND": float(r2),
    }


def calculate_country_summary(
    country_data,
    country_id,
):
    """Calcula el resumen de cobertura de un país."""

    country_data = country_data.sort_values(
        "YEAR"
    )

    years = (
        country_data["YEAR"]
        .astype(int)
        .tolist()
    )

    year_gaps = [
        current - previous
        for previous, current
        in zip(
            years[:-1],
            years[1:],
        )
    ]

    consecutive_transitions = sum(
        gap == 1
        for gap in year_gaps
    )

    summary = {
        "PROCESSING_DATETIME": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "COUNTRY_ID": country_id,
        "COUNTRY_NAME_EN": (
            country_data[
                "COUNTRY_NAME_EN"
            ].iloc[0]
        ),
        "OBSERVATION_COUNT": len(
            country_data
        ),
        "MIN_YEAR": int(
            country_data["YEAR"].min()
        ),
        "MAX_YEAR": int(
            country_data["YEAR"].max()
        ),
        "CALENDAR_SPAN": int(
            country_data["YEAR"].max()
            - country_data["YEAR"].min()
            + 1
        ),
        "CONSECUTIVE_TRANSITIONS": int(
            consecutive_transitions
        ),
        "MAX_YEAR_GAP": (
            int(max(year_gaps))
            if year_gaps
            else 0
        ),
    }

    for variable in VARIABLES:
        initial_value = float(
            country_data[
                variable
            ].iloc[0]
        )

        final_value = float(
            country_data[
                variable
            ].iloc[-1]
        )

        summary[
            f"INITIAL_{variable}"
        ] = initial_value

        summary[
            f"FINAL_{variable}"
        ] = final_value

        summary[
            f"ABSOLUTE_CHANGE_{variable}"
        ] = (
            final_value
            - initial_value
        )

    return summary


# ============================================================
# GRAFICA DE ECUADOR
# ============================================================

def create_ecuador_trajectory_figure(
    panel,
):
    """Genera cuatro trayectorias para Ecuador."""

    ecuador = panel.loc[
        panel["COUNTRY_ID"].eq(
            "ECU"
        )
    ].sort_values(
        "YEAR"
    )

    if ecuador.empty:
        raise ValueError(
            "No se encontraron observaciones para Ecuador."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for axis, variable in zip(
        axes,
        VARIABLES,
    ):
        axis.plot(
            ecuador["YEAR"],
            ecuador[variable],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )

        axis.set_title(
            f"{variable}: "
            f"{VARIABLE_LABELS[variable]}"
        )

        axis.set_xlabel("Año")
        axis.set_ylabel(
            VARIABLE_LABELS[variable]
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    figure.suptitle(
        "Ecuador: trayectorias observadas del sistema A-C-L-F",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        ECUADOR_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# COMPARACION INTERNACIONAL
# ============================================================

def create_country_comparison_figure(
    panel,
):
    """Compara las trayectorias originales entre países."""

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    for axis, variable in zip(
        axes,
        VARIABLES,
    ):
        for country_id in SELECTED_COUNTRIES:
            country_data = panel.loc[
                panel[
                    "COUNTRY_ID"
                ].eq(country_id)
            ].sort_values(
                "YEAR"
            )

            if country_data.empty:
                continue

            line_width = (
                3
                if country_id == "ECU"
                else 1.5
            )

            marker = (
                "o"
                if country_id == "ECU"
                else None
            )

            axis.plot(
                country_data["YEAR"],
                country_data[variable],
                label=country_id,
                linewidth=line_width,
                marker=marker,
                color=COUNTRY_COLORS[
                    country_id
                ],
                alpha=0.9,
            )

        axis.set_title(
            f"{variable}: "
            f"{VARIABLE_LABELS[variable]}"
        )

        axis.set_xlabel("Año")
        axis.set_ylabel(
            VARIABLE_LABELS[variable]
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
    )

    figure.suptitle(
        "Comparación internacional de las trayectorias A-C-L-F",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0.06, 1, 0.96]
    )

    figure.savefig(
        COMPARISON_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# CAMBIOS ESTANDARIZADOS
# ============================================================

def create_standardized_change_figure(
    panel,
):
    """
    Compara el cambio relativo desde la primera observación.

    Cada trayectoria comienza en cero para facilitar la
    comparación entre variables con escalas distintas.
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    for axis, variable in zip(
        axes,
        VARIABLES,
    ):
        for country_id in SELECTED_COUNTRIES:
            country_data = panel.loc[
                panel[
                    "COUNTRY_ID"
                ].eq(country_id)
            ].sort_values(
                "YEAR"
            ).copy()

            if country_data.empty:
                continue

            initial_value = float(
                country_data[
                    variable
                ].iloc[0]
            )

            country_data[
                "STANDARDIZED_CHANGE"
            ] = (
                country_data[variable]
                - initial_value
            )

            line_width = (
                3
                if country_id == "ECU"
                else 1.5
            )

            axis.plot(
                country_data["YEAR"],
                country_data[
                    "STANDARDIZED_CHANGE"
                ],
                label=country_id,
                linewidth=line_width,
                color=COUNTRY_COLORS[
                    country_id
                ],
            )

        axis.axhline(
            y=0,
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

        axis.set_title(
            f"Cambio acumulado de {variable}"
        )

        axis.set_xlabel("Año")
        axis.set_ylabel(
            "Cambio desde la primera observación"
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
    )

    figure.suptitle(
        "Cambio acumulado desde la primera observación disponible",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0.06, 1, 0.96]
    )

    figure.savefig(
        STANDARDIZED_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# TASAS TEMPORALES DE ECUADOR
# ============================================================

def create_ecuador_rates_figure(
    rates,
):
    """Grafica las tasas anuales calculadas para Ecuador."""

    ecuador_rates = rates.loc[
        rates["COUNTRY_ID"].eq(
            "ECU"
        )
        & rates["DELTA_T"].eq(1)
    ].sort_values(
        "YEAR"
    )

    if ecuador_rates.empty:
        raise ValueError(
            "No se encontraron tasas consecutivas para Ecuador."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for axis, rate_column in zip(
        axes,
        RATE_COLUMNS,
    ):
        axis.bar(
            ecuador_rates["YEAR"],
            ecuador_rates[rate_column],
            color="#1f77b4",
            alpha=0.85,
        )

        axis.axhline(
            y=0,
            color="black",
            linewidth=0.8,
        )

        axis.set_title(
            RATE_COLUMNS[
                rate_column
            ]
        )

        axis.set_xlabel("Año final")
        axis.set_ylabel(
            "Cambio por año"
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    figure.suptitle(
        "Ecuador: tasas anuales observadas del sistema A-C-L-F",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        ECUADOR_RATES_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "ANALISIS COMPARATIVO DE CASOS NACIONALES A-C-L-F"
    )

    verify_file(PANEL_FILE)
    verify_file(RATES_FILE)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLeyendo el panel A-C-L-F...")

    panel = pd.read_parquet(
        PANEL_FILE
    )

    rates = pd.read_parquet(
        RATES_FILE
    )

    selected_ids = list(
        SELECTED_COUNTRIES.keys()
    )

    selected_panel = panel.loc[
        panel["COUNTRY_ID"].isin(
            selected_ids
        )
    ].copy()

    selected_rates = rates.loc[
        rates["COUNTRY_ID"].isin(
            selected_ids
        )
    ].copy()

    found_countries = set(
        selected_panel[
            "COUNTRY_ID"
        ].unique()
    )

    missing_countries = sorted(
        set(selected_ids)
        - found_countries
    )

    if missing_countries:
        raise ValueError(
            "Faltan paises seleccionados: "
            f"{missing_countries}"
        )

    print(
        "Filas seleccionadas:",
        f"{len(selected_panel):,}",
    )

    print(
        "Paises seleccionados:",
        f"{selected_panel['COUNTRY_ID'].nunique():,}",
    )

    # ========================================================
    # RESUMEN POR PAIS
    # ========================================================

    summary_rows = []
    trend_rows = []

    for country_id in selected_ids:
        country_data = selected_panel.loc[
            selected_panel[
                "COUNTRY_ID"
            ].eq(country_id)
        ].sort_values(
            "YEAR"
        )

        summary_rows.append(
            calculate_country_summary(
                country_data,
                country_id,
            )
        )

        for variable in VARIABLES:
            trend = calculate_linear_trend(
                country_data,
                variable,
            )

            trend_rows.append(
                {
                    "COUNTRY_ID": country_id,
                    "COUNTRY_NAME_EN": (
                        country_data[
                            "COUNTRY_NAME_EN"
                        ].iloc[0]
                    ),
                    "VARIABLE": variable,
                    "VARIABLE_LABEL": (
                        VARIABLE_LABELS[
                            variable
                        ]
                    ),
                    "OBSERVATION_COUNT": len(
                        country_data
                    ),
                    "MIN_YEAR": int(
                        country_data[
                            "YEAR"
                        ].min()
                    ),
                    "MAX_YEAR": int(
                        country_data[
                            "YEAR"
                        ].max()
                    ),
                    **trend,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    trends = pd.DataFrame(
        trend_rows
    )

    summary = summary.sort_values(
        by="OBSERVATION_COUNT",
        ascending=False,
    ).reset_index(drop=True)

    trends = trends.sort_values(
        by=[
            "VARIABLE",
            "COUNTRY_ID",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # GUARDAR TABLAS
    # ========================================================

    selected_panel.to_csv(
        TRAJECTORY_TABLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    trends.to_csv(
        TREND_TABLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    selected_rates.to_csv(
        RATE_TABLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CREAR GRAFICAS
    # ========================================================

    print("\nCreando graficas...")

    create_ecuador_trajectory_figure(
        selected_panel
    )

    create_country_comparison_figure(
        selected_panel
    )

    create_standardized_change_figure(
        selected_panel
    )

    create_ecuador_rates_figure(
        selected_rates
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    print_section(
        "ANALISIS DE CASOS COMPLETADO CORRECTAMENTE"
    )

    print(
        "\nRESUMEN DE COBERTURA"
    )

    print(
        summary[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "MIN_YEAR",
                "MAX_YEAR",
                "OBSERVATION_COUNT",
                "CONSECUTIVE_TRANSITIONS",
                "MAX_YEAR_GAP",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTENDENCIAS LINEALES ESTIMADAS"
    )

    print(
        trends[
            [
                "COUNTRY_ID",
                "VARIABLE",
                "SLOPE_PER_YEAR",
                "R2_TREND",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nTrayectorias seleccionadas:\n"
        f"{TRAJECTORY_TABLE_FILE}"
    )

    print(
        f"\nTendencias:\n"
        f"{TREND_TABLE_FILE}"
    )

    print(
        f"\nTasas seleccionadas:\n"
        f"{RATE_TABLE_FILE}"
    )

    print(
        f"\nResumen:\n"
        f"{SUMMARY_FILE}"
    )

    print(
        f"\nFigura de Ecuador:\n"
        f"{ECUADOR_FIGURE}"
    )

    print(
        f"\nComparacion internacional:\n"
        f"{COMPARISON_FIGURE}"
    )

    print(
        f"\nCambios estandarizados:\n"
        f"{STANDARDIZED_FIGURE}"
    )

    print(
        f"\nTasas de Ecuador:\n"
        f"{ECUADOR_RATES_FIGURE}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE EL ANALISIS DE CASOS"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)