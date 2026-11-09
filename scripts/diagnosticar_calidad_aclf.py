from pathlib import Path
from datetime import datetime
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "quality"
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

RANGE_FILE = (
    OUTPUT_DIR
    / "aclf_variable_ranges.csv"
)

OUTLIER_FILE = (
    OUTPUT_DIR
    / "aclf_level_outliers.csv"
)

RATE_OUTLIER_FILE = (
    OUTPUT_DIR
    / "aclf_rate_outliers.csv"
)

COUNTRY_FILE = (
    OUTPUT_DIR
    / "aclf_country_quality_summary.csv"
)

SENSITIVITY_FILE = (
    OUTPUT_DIR
    / "aclf_outlier_sensitivity_metrics.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "aclf_quality_config.json"
)

SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_quality_diagnostic_summary.csv"
)

DISTRIBUTION_FIGURE = (
    FIGURES_DIR
    / "aclf_quality_distributions.png"
)

RATE_FIGURE = (
    FIGURES_DIR
    / "aclf_quality_rate_distributions.png"
)

OUTLIER_FIGURE = (
    FIGURES_DIR
    / "aclf_quality_outliers_by_country.png"
)


# ============================================================
# CONFIGURACION
# ============================================================

VARIABLES = [
    "A",
    "C",
    "L",
    "F",
]

PREVIOUS_VARIABLES = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

RATE_COLUMNS = [
    "DA_DT",
    "DC_DT",
    "DL_DT",
    "DF_DT",
]

TARGETS = {
    "DA_DT": "dA/dt",
    "DC_DT": "dC/dt",
    "DL_DT": "dL/dt",
    "DF_DT": "dF/dt",
}

EXPECTED_RANGES = {
    "A": {
        "MINIMUM": 0.0,
        "MAXIMUM": 100.0,
    },
    "C": {
        "MINIMUM": 0.0,
        "MAXIMUM": 100.0,
    },
    "L": {
        "MINIMUM": 0.0,
        "MAXIMUM": 100.0,
    },
    "F": {
        "MINIMUM": 0.0,
        "MAXIMUM": 20.0,
    },
}

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_file(file_path):
    """Verifica la existencia de un archivo."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def calculate_iqr_limits(series):
    """
    Calcula límites de valores atípicos mediante 1,5 veces
    el rango intercuartílico.
    """

    clean_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean_series.empty:
        return {
            "Q1": np.nan,
            "MEDIAN": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "LOWER_LIMIT": np.nan,
            "UPPER_LIMIT": np.nan,
        }

    q1 = float(
        clean_series.quantile(0.25)
    )

    median = float(
        clean_series.quantile(0.50)
    )

    q3 = float(
        clean_series.quantile(0.75)
    )

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    return {
        "Q1": q1,
        "MEDIAN": median,
        "Q3": q3,
        "IQR": iqr,
        "LOWER_LIMIT": lower_limit,
        "UPPER_LIMIT": upper_limit,
    }


def calculate_metrics(observed, predicted):
    """Calcula RMSE, MAE y R2."""

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                observed,
                predicted,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            observed,
            predicted,
        )
    )

    r2 = float(
        r2_score(
            observed,
            predicted,
        )
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def create_ridge_model(alpha):
    """Crea un modelo Ridge con estandarización."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def cross_validate_by_country(
    data,
    target_column,
    alpha,
):
    """
    Valida el modelo separando países completos.

    Ningún país aparece simultáneamente en entrenamiento
    y validación dentro de una partición.
    """

    features = data[
        PREVIOUS_VARIABLES
    ].astype(float)

    target = data[
        target_column
    ].astype(float)

    groups = data[
        "COUNTRY_ID"
    ].astype(str)

    country_count = int(
        groups.nunique()
    )

    if country_count >= 10:
        fold_count = 5
    elif country_count >= 4:
        fold_count = 4
    elif country_count >= 3:
        fold_count = 3
    else:
        raise ValueError(
            "No existen suficientes paises para validar."
        )

    group_cv = GroupKFold(
        n_splits=fold_count
    )

    observed_values = []
    predicted_values = []

    for train_indices, test_indices in group_cv.split(
        features,
        target,
        groups,
    ):
        model = create_ridge_model(
            alpha
        )

        model.fit(
            features.iloc[
                train_indices
            ],
            target.iloc[
                train_indices
            ],
        )

        predictions = model.predict(
            features.iloc[
                test_indices
            ]
        )

        observed_values.extend(
            target.iloc[
                test_indices
            ].to_numpy()
        )

        predicted_values.extend(
            predictions
        )

    metrics = calculate_metrics(
        observed_values,
        predicted_values,
    )

    return metrics


def select_best_alpha(
    data,
    target_column,
):
    """Selecciona alpha usando el menor RMSE."""

    results = []

    for alpha in RIDGE_ALPHAS:
        metrics = cross_validate_by_country(
            data=data,
            target_column=target_column,
            alpha=alpha,
        )

        results.append(
            {
                "ALPHA": alpha,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
            }
        )

    best_result = min(
        results,
        key=lambda row: row["RMSE"],
    )

    return best_result


# ============================================================
# ANÁLISIS DE NIVELES
# ============================================================

def analyze_level_variables(panel):
    """Analiza rangos y valores atípicos de A, C, L y F."""

    range_rows = []
    outlier_frames = []

    for variable in VARIABLES:
        series = pd.to_numeric(
            panel[variable],
            errors="coerce",
        )

        limits = calculate_iqr_limits(
            series
        )

        expected_minimum = (
            EXPECTED_RANGES[
                variable
            ]["MINIMUM"]
        )

        expected_maximum = (
            EXPECTED_RANGES[
                variable
            ]["MAXIMUM"]
        )

        outside_expected = (
            series.lt(
                expected_minimum
            )
            | series.gt(
                expected_maximum
            )
        )

        iqr_outlier = (
            series.lt(
                limits["LOWER_LIMIT"]
            )
            | series.gt(
                limits["UPPER_LIMIT"]
            )
        )

        range_rows.append(
            {
                "VARIABLE": variable,
                "OBSERVATION_COUNT": int(
                    series.notna().sum()
                ),
                "MISSING_COUNT": int(
                    series.isna().sum()
                ),
                "MINIMUM": float(
                    series.min()
                ),
                "PERCENTILE_01": float(
                    series.quantile(0.01)
                ),
                "Q1": limits["Q1"],
                "MEDIAN": limits["MEDIAN"],
                "MEAN": float(
                    series.mean()
                ),
                "Q3": limits["Q3"],
                "PERCENTILE_99": float(
                    series.quantile(0.99)
                ),
                "MAXIMUM": float(
                    series.max()
                ),
                "STANDARD_DEVIATION": float(
                    series.std(ddof=1)
                ),
                "IQR": limits["IQR"],
                "IQR_LOWER_LIMIT": (
                    limits["LOWER_LIMIT"]
                ),
                "IQR_UPPER_LIMIT": (
                    limits["UPPER_LIMIT"]
                ),
                "EXPECTED_MINIMUM": (
                    expected_minimum
                ),
                "EXPECTED_MAXIMUM": (
                    expected_maximum
                ),
                "OUTSIDE_EXPECTED_COUNT": int(
                    outside_expected.sum()
                ),
                "IQR_OUTLIER_COUNT": int(
                    iqr_outlier.sum()
                ),
            }
        )

        variable_outliers = panel.loc[
            outside_expected
            | iqr_outlier,
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "YEAR",
                variable,
            ],
        ].copy()

        variable_outliers[
            "VARIABLE"
        ] = variable

        variable_outliers[
            "VALUE"
        ] = variable_outliers[
            variable
        ]

        variable_outliers[
            "OUTSIDE_EXPECTED_RANGE"
        ] = outside_expected.loc[
            variable_outliers.index
        ].to_numpy()

        variable_outliers[
            "IQR_OUTLIER"
        ] = iqr_outlier.loc[
            variable_outliers.index
        ].to_numpy()

        variable_outliers[
            "IQR_LOWER_LIMIT"
        ] = limits["LOWER_LIMIT"]

        variable_outliers[
            "IQR_UPPER_LIMIT"
        ] = limits["UPPER_LIMIT"]

        variable_outliers = (
            variable_outliers[
                [
                    "COUNTRY_ID",
                    "COUNTRY_NAME_EN",
                    "YEAR",
                    "VARIABLE",
                    "VALUE",
                    "OUTSIDE_EXPECTED_RANGE",
                    "IQR_OUTLIER",
                    "IQR_LOWER_LIMIT",
                    "IQR_UPPER_LIMIT",
                ]
            ]
        )

        outlier_frames.append(
            variable_outliers
        )

    ranges = pd.DataFrame(
        range_rows
    )

    level_outliers = pd.concat(
        outlier_frames,
        ignore_index=True,
    )

    return ranges, level_outliers


# ============================================================
# ANÁLISIS DE TASAS
# ============================================================

def analyze_rate_variables(rates):
    """Identifica saltos temporales atípicos."""

    rate_outlier_frames = []

    for rate_column in RATE_COLUMNS:
        series = pd.to_numeric(
            rates[rate_column],
            errors="coerce",
        )

        limits = calculate_iqr_limits(
            series
        )

        outlier_mask = (
            series.lt(
                limits["LOWER_LIMIT"]
            )
            | series.gt(
                limits["UPPER_LIMIT"]
            )
        )

        rate_outliers = rates.loc[
            outlier_mask,
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "PREVIOUS_YEAR",
                "YEAR",
                "DELTA_T",
                rate_column,
            ],
        ].copy()

        rate_outliers[
            "RATE_COLUMN"
        ] = rate_column

        rate_outliers[
            "RATE_VALUE"
        ] = rate_outliers[
            rate_column
        ]

        rate_outliers[
            "ABSOLUTE_RATE"
        ] = (
            rate_outliers[
                "RATE_VALUE"
            ].abs()
        )

        rate_outliers[
            "IQR_LOWER_LIMIT"
        ] = limits["LOWER_LIMIT"]

        rate_outliers[
            "IQR_UPPER_LIMIT"
        ] = limits["UPPER_LIMIT"]

        rate_outliers = (
            rate_outliers[
                [
                    "COUNTRY_ID",
                    "COUNTRY_NAME_EN",
                    "PREVIOUS_YEAR",
                    "YEAR",
                    "DELTA_T",
                    "RATE_COLUMN",
                    "RATE_VALUE",
                    "ABSOLUTE_RATE",
                    "IQR_LOWER_LIMIT",
                    "IQR_UPPER_LIMIT",
                ]
            ]
        )

        rate_outlier_frames.append(
            rate_outliers
        )

    rate_outliers = pd.concat(
        rate_outlier_frames,
        ignore_index=True,
    )

    rate_outliers = rate_outliers.sort_values(
        by="ABSOLUTE_RATE",
        ascending=False,
    ).reset_index(drop=True)

    return rate_outliers


# ============================================================
# RESUMEN POR PAÍS
# ============================================================

def create_country_quality_summary(
    panel,
    level_outliers,
    rate_outliers,
):
    """Resume problemas potenciales por país."""

    panel_summary = (
        panel
        .groupby(
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
            ],
            dropna=False,
        )
        .agg(
            OBSERVATION_COUNT=(
                "YEAR",
                "size",
            ),
            MIN_YEAR=(
                "YEAR",
                "min",
            ),
            MAX_YEAR=(
                "YEAR",
                "max",
            ),
        )
        .reset_index()
    )

    level_counts = (
        level_outliers
        .groupby(
            "COUNTRY_ID"
        )
        .size()
        .rename(
            "LEVEL_OUTLIER_COUNT"
        )
    )

    rate_counts = (
        rate_outliers
        .groupby(
            "COUNTRY_ID"
        )
        .size()
        .rename(
            "RATE_OUTLIER_COUNT"
        )
    )

    panel_summary = panel_summary.merge(
        level_counts,
        on="COUNTRY_ID",
        how="left",
    )

    panel_summary = panel_summary.merge(
        rate_counts,
        on="COUNTRY_ID",
        how="left",
    )

    panel_summary[
        "LEVEL_OUTLIER_COUNT"
    ] = (
        panel_summary[
            "LEVEL_OUTLIER_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    panel_summary[
        "RATE_OUTLIER_COUNT"
    ] = (
        panel_summary[
            "RATE_OUTLIER_COUNT"
        ]
        .fillna(0)
        .astype(int)
    )

    panel_summary[
        "TOTAL_FLAG_COUNT"
    ] = (
        panel_summary[
            "LEVEL_OUTLIER_COUNT"
        ]
        + panel_summary[
            "RATE_OUTLIER_COUNT"
        ]
    )

    panel_summary = panel_summary.sort_values(
        by=[
            "TOTAL_FLAG_COUNT",
            "RATE_OUTLIER_COUNT",
            "OBSERVATION_COUNT",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return panel_summary


# ============================================================
# SENSIBILIDAD DEL MODELO
# ============================================================

def run_sensitivity_analysis(
    rates,
    rate_outliers,
):
    """
    Compara el modelo usando todos los datos y excluyendo
    transiciones marcadas como atípicas.

    Se utilizan solamente transiciones consecutivas.
    """

    consecutive = rates.loc[
        rates["DELTA_T"].eq(1)
    ].copy()

    outlier_keys = (
        rate_outliers[
            [
                "COUNTRY_ID",
                "PREVIOUS_YEAR",
                "YEAR",
            ]
        ]
        .drop_duplicates()
        .assign(
            IS_RATE_OUTLIER=True
        )
    )

    consecutive = consecutive.merge(
        outlier_keys,
        on=[
            "COUNTRY_ID",
            "PREVIOUS_YEAR",
            "YEAR",
        ],
        how="left",
    )

    consecutive[
        "IS_RATE_OUTLIER"
    ] = (
        consecutive[
            "IS_RATE_OUTLIER"
        ]
        .fillna(False)
        .astype(bool)
    )

    clean_sample = consecutive.loc[
        ~consecutive[
            "IS_RATE_OUTLIER"
        ]
    ].copy()

    sample_definitions = {
        "all_consecutive": consecutive,
        "without_rate_outliers": clean_sample,
    }

    sensitivity_rows = []

    for sample_name, sample in sample_definitions.items():
        for target_column in TARGETS:
            if (
                len(sample) < 20
                or sample[
                    "COUNTRY_ID"
                ].nunique() < 3
            ):
                continue

            best_result = select_best_alpha(
                data=sample,
                target_column=target_column,
            )

            zero_predictions = np.zeros(
                len(sample),
                dtype=float,
            )

            zero_metrics = calculate_metrics(
                sample[
                    target_column
                ].astype(float),
                zero_predictions,
            )

            sensitivity_rows.append(
                {
                    "SAMPLE": sample_name,
                    "TARGET_COLUMN": (
                        target_column
                    ),
                    "EQUATION": (
                        TARGETS[
                            target_column
                        ]
                    ),
                    "OBSERVATION_COUNT": len(
                        sample
                    ),
                    "COUNTRY_COUNT": int(
                        sample[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "SELECTED_ALPHA": (
                        best_result["ALPHA"]
                    ),
                    "RIDGE_RMSE_CV": (
                        best_result["RMSE"]
                    ),
                    "RIDGE_MAE_CV": (
                        best_result["MAE"]
                    ),
                    "RIDGE_R2_CV": (
                        best_result["R2"]
                    ),
                    "ZERO_RMSE": (
                        zero_metrics["RMSE"]
                    ),
                    "RMSE_IMPROVEMENT_VS_ZERO": (
                        zero_metrics["RMSE"]
                        - best_result["RMSE"]
                    ),
                }
            )

    return pd.DataFrame(
        sensitivity_rows
    )


# ============================================================
# GRÁFICAS
# ============================================================

def create_distribution_figure(panel):
    """Grafica la distribución de los niveles."""

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
        axis.hist(
            panel[
                variable
            ].dropna(),
            bins=30,
        )

        axis.set_title(
            f"Distribucion de {variable}"
        )

        axis.set_xlabel(
            "Valor observado"
        )

        axis.set_ylabel(
            "Frecuencia"
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

    figure.suptitle(
        "Distribuciones de las variables del panel A-C-L-F",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        DISTRIBUTION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_rate_figure(rates):
    """Grafica la distribución de las tasas temporales."""

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
        axis.hist(
            rates[
                rate_column
            ].dropna(),
            bins=35,
        )

        axis.axvline(
            x=0,
            color="black",
            linewidth=0.8,
        )

        axis.set_title(
            f"Distribucion de {rate_column}"
        )

        axis.set_xlabel(
            "Cambio por año"
        )

        axis.set_ylabel(
            "Frecuencia"
        )

        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

    figure.suptitle(
        "Distribuciones de las tasas temporales A-C-L-F",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        RATE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_outlier_country_figure(
    country_summary,
):
    """Muestra los países con más observaciones señaladas."""

    top_countries = (
        country_summary
        .head(20)
        .sort_values(
            by="TOTAL_FLAG_COUNT",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(12, 8),
    )

    axis.barh(
        top_countries[
            "COUNTRY_ID"
        ],
        top_countries[
            "LEVEL_OUTLIER_COUNT"
        ],
        label="Valores de nivel",
    )

    axis.barh(
        top_countries[
            "COUNTRY_ID"
        ],
        top_countries[
            "RATE_OUTLIER_COUNT"
        ],
        left=top_countries[
            "LEVEL_OUTLIER_COUNT"
        ],
        label="Tasas temporales",
    )

    axis.set_title(
        "Países con mayor número de observaciones señaladas"
    )

    axis.set_xlabel(
        "Número de señales de revisión"
    )

    axis.set_ylabel(
        "Código de país"
    )

    axis.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTLIER_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "DIAGNOSTICO DE CALIDAD DEL PANEL A-C-L-F"
    )

    verify_file(
        PANEL_FILE
    )

    verify_file(
        RATES_FILE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLeyendo panel y tasas...")

    panel = pd.read_parquet(
        PANEL_FILE
    )

    rates = pd.read_parquet(
        RATES_FILE
    )

    print(
        "Filas del panel:",
        f"{len(panel):,}",
    )

    print(
        "Transiciones temporales:",
        f"{len(rates):,}",
    )

    print(
        "Paises o territorios:",
        f"{panel['COUNTRY_ID'].nunique():,}",
    )

    # ========================================================
    # ANÁLISIS
    # ========================================================

    print(
        "\nAnalizando rangos y valores atipicos..."
    )

    (
        ranges,
        level_outliers,
    ) = analyze_level_variables(
        panel
    )

    rate_outliers = analyze_rate_variables(
        rates
    )

    country_summary = (
        create_country_quality_summary(
            panel=panel,
            level_outliers=level_outliers,
            rate_outliers=rate_outliers,
        )
    )

    print(
        "Ejecutando analisis de sensibilidad..."
    )

    sensitivity = (
        run_sensitivity_analysis(
            rates=rates,
            rate_outliers=rate_outliers,
        )
    )

    # ========================================================
    # RESUMEN GENERAL
    # ========================================================

    outside_expected_count = int(
        ranges[
            "OUTSIDE_EXPECTED_COUNT"
        ].sum()
    )

    level_outlier_count = len(
        level_outliers
    )

    rate_outlier_count = len(
        rate_outliers
    )

    affected_countries = int(
        country_summary.loc[
            country_summary[
                "TOTAL_FLAG_COUNT"
            ].gt(0),
            "COUNTRY_ID",
        ].nunique()
    )

    summary = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "PANEL_ROWS": len(panel),
                "RATE_ROWS": len(rates),
                "COUNTRY_COUNT": int(
                    panel[
                        "COUNTRY_ID"
                    ].nunique()
                ),
                "MINIMUM_YEAR": int(
                    panel["YEAR"].min()
                ),
                "MAXIMUM_YEAR": int(
                    panel["YEAR"].max()
                ),
                "OUTSIDE_EXPECTED_RANGE_COUNT": (
                    outside_expected_count
                ),
                "LEVEL_IQR_OUTLIER_ROWS": (
                    level_outlier_count
                ),
                "RATE_IQR_OUTLIER_ROWS": (
                    rate_outlier_count
                ),
                "COUNTRIES_WITH_FLAGS": (
                    affected_countries
                ),
                "SENSITIVITY_MODEL": (
                    "Ridge con GroupKFold por pais"
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    ranges.to_csv(
        RANGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    level_outliers.to_csv(
        OUTLIER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    rate_outliers.to_csv(
        RATE_OUTLIER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    country_summary.to_csv(
        COUNTRY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity.to_csv(
        SENSITIVITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "panel_file": PANEL_FILE.name,
        "rates_file": RATES_FILE.name,
        "variables": VARIABLES,
        "rate_columns": RATE_COLUMNS,
        "expected_ranges": EXPECTED_RANGES,
        "outlier_method": (
            "1.5 veces el rango intercuartil"
        ),
        "sensitivity_sample": (
            "Transiciones consecutivas"
        ),
        "sensitivity_validation": (
            "GroupKFold agrupado por pais"
        ),
        "outlier_policy": (
            "Las observaciones se señalan para revision, "
            "pero no se eliminan de la base original."
        ),
    }

    with CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nCreando graficas...")

    create_distribution_figure(
        panel
    )

    create_rate_figure(
        rates
    )

    create_outlier_country_figure(
        country_summary
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print_section(
        "DIAGNOSTICO DE CALIDAD COMPLETADO"
    )

    print(
        "\nRANGOS DE LAS VARIABLES"
    )

    print(
        ranges[
            [
                "VARIABLE",
                "OBSERVATION_COUNT",
                "MINIMUM",
                "MEDIAN",
                "MEAN",
                "MAXIMUM",
                "OUTSIDE_EXPECTED_COUNT",
                "IQR_OUTLIER_COUNT",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESUMEN DE SEÑALES"
    )

    print(
        "Valores fuera del rango esperado:",
        outside_expected_count,
    )

    print(
        "Registros de nivel señalados:",
        level_outlier_count,
    )

    print(
        "Registros de tasas señalados:",
        rate_outlier_count,
    )

    print(
        "Paises con al menos una señal:",
        affected_countries,
    )

    print(
        "\nSENSIBILIDAD DEL MODELO"
    )

    print(
        sensitivity[
            [
                "SAMPLE",
                "EQUATION",
                "OBSERVATION_COUNT",
                "COUNTRY_COUNT",
                "SELECTED_ALPHA",
                "RIDGE_RMSE_CV",
                "RIDGE_R2_CV",
                "ZERO_RMSE",
                "RMSE_IMPROVEMENT_VS_ZERO",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nRangos:\n{RANGE_FILE}"
    )

    print(
        f"\nValores atipicos de nivel:\n"
        f"{OUTLIER_FILE}"
    )

    print(
        f"\nTasas atipicas:\n"
        f"{RATE_OUTLIER_FILE}"
    )

    print(
        f"\nResumen por pais:\n"
        f"{COUNTRY_FILE}"
    )

    print(
        f"\nSensibilidad:\n"
        f"{SENSITIVITY_FILE}"
    )

    print(
        f"\nConfiguracion:\n"
        f"{CONFIG_FILE}"
    )

    print(
        f"\nResumen general:\n"
        f"{SUMMARY_FILE}"
    )

    print(
        f"\nFigura de distribuciones:\n"
        f"{DISTRIBUTION_FIGURE}"
    )

    print(
        f"\nFigura de tasas:\n"
        f"{RATE_FIGURE}"
    )

    print(
        f"\nFigura de señales por pais:\n"
        f"{OUTLIER_FIGURE}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE EL DIAGNOSTICO DE CALIDAD"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)