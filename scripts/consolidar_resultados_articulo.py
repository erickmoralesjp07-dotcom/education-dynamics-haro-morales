from pathlib import Path
from datetime import datetime
import json
import sys

import pandas as pd


# ============================================================
# RUTAS PRINCIPALES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "model"
)

MODEL_EFFECTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_effects"
)

MODEL_ECUADOR_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_ecuador"
)

QUALITY_DIR = (
    PROJECT_ROOT
    / "results"
    / "quality"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

ARTICLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "article"
)


# ============================================================
# ARCHIVOS DE ENTRADA
# ============================================================

NATIONAL_DIAGNOSTIC = (
    DIAGNOSTICS_DIR
    / "uis_sdg_national_diagnostic.csv"
)

CATALOG_DIAGNOSTIC = (
    DIAGNOSTICS_DIR
    / "uis_sdg_catalog_integration.csv"
)

CANDIDATE_SUMMARY = (
    DIAGNOSTICS_DIR
    / "uis_candidate_indicators_summary.csv"
)

SELECTED_INDICATORS_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_selected_indicators.csv"
)

PANEL_COVERAGE_FILE = (
    DIAGNOSTICS_DIR
    / "uis_model_panel_coverage.csv"
)

PANEL_INDICATOR_COVERAGE_FILE = (
    DIAGNOSTICS_DIR
    / "uis_model_indicator_coverage.csv"
)

VARIABLE_COMBINATIONS_FILE = (
    DIAGNOSTICS_DIR
    / "uis_model_variable_combinations.csv"
)

ACLF_SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "uis_model_aclf_summary.csv"
)

ACLF_RATES_SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "uis_model_aclf_rates_summary.csv"
)

GLOBAL_MODEL_METRICS_FILE = (
    MODEL_DIR
    / "aclf_linear_ode_metrics.csv"
)

GLOBAL_MODEL_DIAGNOSTIC_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_linear_ode_diagnostic.csv"
)

EFFECTS_MODEL_METRICS_FILE = (
    MODEL_EFFECTS_DIR
    / "aclf_effects_model_metrics.csv"
)

ECUADOR_MODEL_METRICS_FILE = (
    MODEL_ECUADOR_DIR
    / "ecuador_aclf_metrics.csv"
)

ECUADOR_MODEL_COEFFICIENTS_FILE = (
    MODEL_ECUADOR_DIR
    / "ecuador_aclf_coefficients.csv"
)

COUNTRY_CASE_SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_country_case_analysis_summary.csv"
)

COUNTRY_TRENDS_FILE = (
    TABLES_DIR
    / "aclf_selected_countries_trends.csv"
)

QUALITY_SUMMARY_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_quality_diagnostic_summary.csv"
)

QUALITY_RANGES_FILE = (
    QUALITY_DIR
    / "aclf_variable_ranges.csv"
)

QUALITY_SENSITIVITY_FILE = (
    QUALITY_DIR
    / "aclf_outlier_sensitivity_metrics.csv"
)


# ============================================================
# ARCHIVOS DE SALIDA
# ============================================================

MASTER_RESULTS_FILE = (
    ARTICLE_DIR
    / "master_results_table.csv"
)

MASTER_MODELS_FILE = (
    ARTICLE_DIR
    / "master_model_comparison.csv"
)

MASTER_INDICATORS_FILE = (
    ARTICLE_DIR
    / "master_selected_indicators.csv"
)

MASTER_COUNTRY_CASES_FILE = (
    ARTICLE_DIR
    / "master_country_cases.csv"
)

MASTER_QUALITY_FILE = (
    ARTICLE_DIR
    / "master_quality_results.csv"
)

METHODOLOGICAL_DECISIONS_FILE = (
    ARTICLE_DIR
    / "methodological_decisions.csv"
)

EVIDENCE_INVENTORY_FILE = (
    ARTICLE_DIR
    / "evidence_inventory.csv"
)

ARTICLE_SUMMARY_FILE = (
    ARTICLE_DIR
    / "article_results_summary.md"
)

ARTICLE_CONFIG_FILE = (
    ARTICLE_DIR
    / "article_consolidation_config.json"
)

CONSOLIDATION_DIAGNOSTIC_FILE = (
    DIAGNOSTICS_DIR
    / "article_results_consolidation.csv"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime una sección legible en la consola."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_file(file_path):
    """Comprueba que exista un archivo requerido."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def read_single_row_csv(file_path):
    """Lee un CSV de diagnóstico que contiene una sola fila."""

    verify_file(file_path)

    data = pd.read_csv(
        file_path
    )

    if data.empty:
        raise ValueError(
            f"El archivo esta vacio:\n{file_path}"
        )

    return data.iloc[0].to_dict()


def safe_value(
    dictionary,
    key,
    default=None,
):
    """Recupera un valor sin provocar error por claves ausentes."""

    value = dictionary.get(
        key,
        default,
    )

    if pd.isna(value):
        return default

    return value


def numeric_value(
    dictionary,
    key,
    default=0,
):
    """Convierte un resultado a número cuando sea posible."""

    value = safe_value(
        dictionary,
        key,
        default,
    )

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def integer_value(
    dictionary,
    key,
    default=0,
):
    """Convierte un resultado a entero cuando sea posible."""

    value = numeric_value(
        dictionary,
        key,
        default,
    )

    return int(value)


def format_number(
    value,
    decimals=4,
):
    """Formatea números para el resumen Markdown."""

    if value is None:
        return "No disponible"

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    if pd.isna(numeric):
        return "No disponible"

    return f"{numeric:.{decimals}f}"


def relative_path(file_path):
    """Devuelve la ruta relativa al proyecto."""

    try:
        return str(
            file_path.relative_to(
                PROJECT_ROOT
            )
        )
    except ValueError:
        return str(file_path)


# ============================================================
# TABLA MAESTRA DE RESULTADOS
# ============================================================

def build_master_results():
    """Consolida las cifras principales de todo el estudio."""

    national = read_single_row_csv(
        NATIONAL_DIAGNOSTIC
    )

    catalog = read_single_row_csv(
        CATALOG_DIAGNOSTIC
    )

    panel = read_single_row_csv(
        PANEL_COVERAGE_FILE
    )

    aclf = read_single_row_csv(
        ACLF_SUMMARY_FILE
    )

    rates = read_single_row_csv(
        ACLF_RATES_SUMMARY_FILE
    )

    quality = read_single_row_csv(
        QUALITY_SUMMARY_FILE
    )

    rows = [
        {
            "SECTION": "Data source",
            "RESULT": "Original national rows",
            "VALUE": integer_value(
                national,
                "original_rows",
            ),
            "UNIT": "rows",
            "INTERPRETATION": (
                "Registros nacionales originales procesados "
                "desde UNESCO UIS."
            ),
        },
        {
            "SECTION": "Data source",
            "RESULT": "Final national rows",
            "VALUE": integer_value(
                national,
                "final_rows",
            ),
            "UNIT": "rows",
            "INTERPRETATION": (
                "Registros conservados después de validación "
                "y eliminación de duplicados exactos."
            ),
        },
        {
            "SECTION": "Data source",
            "RESULT": "Unique indicators",
            "VALUE": integer_value(
                national,
                "unique_indicators",
            ),
            "UNIT": "indicators",
            "INTERPRETATION": (
                "Indicadores distintos disponibles en la "
                "base nacional."
            ),
        },
        {
            "SECTION": "Data source",
            "RESULT": "Unique countries",
            "VALUE": integer_value(
                national,
                "unique_countries",
            ),
            "UNIT": "countries or territories",
            "INTERPRETATION": (
                "Países o territorios presentes en la base."
            ),
        },
        {
            "SECTION": "Data source",
            "RESULT": "Minimum source year",
            "VALUE": integer_value(
                national,
                "minimum_year",
            ),
            "UNIT": "year",
            "INTERPRETATION": (
                "Primer año disponible en la base nacional."
            ),
        },
        {
            "SECTION": "Data source",
            "RESULT": "Maximum source year",
            "VALUE": integer_value(
                national,
                "maximum_year",
            ),
            "UNIT": "year",
            "INTERPRETATION": (
                "Último año disponible en la base nacional."
            ),
        },
        {
            "SECTION": "Catalog integration",
            "RESULT": "Missing country names",
            "VALUE": integer_value(
                catalog,
                "missing_country_names",
            ),
            "UNIT": "rows",
            "INTERPRETATION": (
                "Registros sin coincidencia en el catálogo "
                "de nombres de países."
            ),
        },
        {
            "SECTION": "Catalog integration",
            "RESULT": "Missing indicator labels",
            "VALUE": integer_value(
                catalog,
                "missing_indicator_labels",
            ),
            "UNIT": "rows",
            "INTERPRETATION": (
                "Registros cuyos códigos no tenían etiqueta "
                "en el catálogo de indicadores."
            ),
        },
        {
            "SECTION": "General panel",
            "RESULT": "Selected indicators",
            "VALUE": integer_value(
                panel,
                "selected_indicator_count",
            ),
            "UNIT": "indicators",
            "INTERPRETATION": (
                "Indicadores inicialmente seleccionados para "
                "las dimensiones del modelo."
            ),
        },
        {
            "SECTION": "General panel",
            "RESULT": "Country-year rows",
            "VALUE": integer_value(
                panel,
                "country_year_rows",
            ),
            "UNIT": "country-year rows",
            "INTERPRETATION": (
                "Filas disponibles en el panel general ancho."
            ),
        },
        {
            "SECTION": "General panel",
            "RESULT": "Complete eight-variable rows",
            "VALUE": integer_value(
                panel,
                "complete_country_year_rows",
            ),
            "UNIT": "country-year rows",
            "INTERPRETATION": (
                "Observaciones con los ocho indicadores "
                "simultáneamente disponibles."
            ),
        },
        {
            "SECTION": "General panel",
            "RESULT": "Rows with at least six variables",
            "VALUE": integer_value(
                panel,
                "country_year_rows_at_least_6",
            ),
            "UNIT": "country-year rows",
            "INTERPRETATION": (
                "Observaciones con seis o más indicadores "
                "simultáneamente disponibles."
            ),
        },
        {
            "SECTION": "General panel",
            "RESULT": "Rows with at least four variables",
            "VALUE": integer_value(
                panel,
                "country_year_rows_at_least_4",
            ),
            "UNIT": "country-year rows",
            "INTERPRETATION": (
                "Observaciones con cuatro o más indicadores "
                "simultáneamente disponibles."
            ),
        },
        {
            "SECTION": "ACLF panel",
            "RESULT": "ACLF country-year rows",
            "VALUE": integer_value(
                aclf,
                "country_year_rows",
            ),
            "UNIT": "country-year rows",
            "INTERPRETATION": (
                "Observaciones completas para acceso, "
                "finalización, alfabetización y financiamiento."
            ),
        },
        {
            "SECTION": "ACLF panel",
            "RESULT": "ACLF countries",
            "VALUE": integer_value(
                aclf,
                "country_count",
            ),
            "UNIT": "countries or territories",
            "INTERPRETATION": (
                "Países o territorios incluidos en el panel ACLF."
            ),
        },
        {
            "SECTION": "ACLF panel",
            "RESULT": "Countries with at least 3 observations",
            "VALUE": integer_value(
                aclf,
                "countries_at_least_3_observations",
            ),
            "UNIT": "countries",
            "INTERPRETATION": (
                "Países con al menos tres observaciones ACLF."
            ),
        },
        {
            "SECTION": "ACLF panel",
            "RESULT": "Countries with at least 5 observations",
            "VALUE": integer_value(
                aclf,
                "countries_at_least_5_observations",
            ),
            "UNIT": "countries",
            "INTERPRETATION": (
                "Países con al menos cinco observaciones ACLF."
            ),
        },
        {
            "SECTION": "ACLF panel",
            "RESULT": "Countries with at least 10 observations",
            "VALUE": integer_value(
                aclf,
                "countries_at_least_10_observations",
            ),
            "UNIT": "countries",
            "INTERPRETATION": (
                "Países con al menos diez observaciones ACLF."
            ),
        },
        {
            "SECTION": "Temporal transitions",
            "RESULT": "Total transitions",
            "VALUE": integer_value(
                rates,
                "total_transitions",
            ),
            "UNIT": "transitions",
            "INTERPRETATION": (
                "Transiciones temporales construidas usando "
                "los intervalos reales entre observaciones."
            ),
        },
        {
            "SECTION": "Temporal transitions",
            "RESULT": "Consecutive transitions",
            "VALUE": integer_value(
                rates,
                "consecutive_transitions",
            ),
            "UNIT": "transitions",
            "INTERPRETATION": (
                "Transiciones separadas exactamente por un año."
            ),
        },
        {
            "SECTION": "Temporal transitions",
            "RESULT": "Transitions with gap at most 2 years",
            "VALUE": integer_value(
                rates,
                "transitions_gap_max_2",
            ),
            "UNIT": "transitions",
            "INTERPRETATION": (
                "Transiciones utilizadas como muestra de robustez."
            ),
        },
        {
            "SECTION": "Data quality",
            "RESULT": "Values outside expected ranges",
            "VALUE": integer_value(
                quality,
                "OUTSIDE_EXPECTED_RANGE_COUNT",
            ),
            "UNIT": "flags",
            "INTERPRETATION": (
                "Valores fuera de los rangos esperados para "
                "porcentajes y gasto educativo."
            ),
        },
        {
            "SECTION": "Data quality",
            "RESULT": "Level IQR outliers",
            "VALUE": integer_value(
                quality,
                "LEVEL_IQR_OUTLIER_ROWS",
            ),
            "UNIT": "flags",
            "INTERPRETATION": (
                "Observaciones de nivel señaladas por el "
                "criterio de rango intercuartílico."
            ),
        },
        {
            "SECTION": "Data quality",
            "RESULT": "Rate IQR outliers",
            "VALUE": integer_value(
                quality,
                "RATE_IQR_OUTLIER_ROWS",
            ),
            "UNIT": "flags",
            "INTERPRETATION": (
                "Cambios temporales señalados por el "
                "criterio de rango intercuartílico."
            ),
        },
        {
            "SECTION": "Data quality",
            "RESULT": "Countries with quality flags",
            "VALUE": integer_value(
                quality,
                "COUNTRIES_WITH_FLAGS",
            ),
            "UNIT": "countries",
            "INTERPRETATION": (
                "Países con al menos una observación marcada "
                "para revisión, sin eliminación automática."
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# TABLA MAESTRA DE INDICADORES
# ============================================================

def build_selected_indicators():
    """Prepara la tabla final de indicadores seleccionados."""

    verify_file(
        SELECTED_INDICATORS_FILE
    )

    indicators = pd.read_csv(
        SELECTED_INDICATORS_FILE
    )

    required_columns = [
        "MODEL_DIMENSION",
        "MODEL_VARIABLE",
        "INDICATOR_ID",
        "INDICATOR_LABEL_EN",
        "DESCRIPTION_ES",
        "DIRECTION",
        "COUNTRY_COUNT",
        "MIN_YEAR",
        "MAX_YEAR",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in indicators.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en la selección de indicadores: "
            f"{missing_columns}"
        )

    indicators = indicators[
        required_columns
    ].copy()

    indicators = indicators.sort_values(
        by="MODEL_VARIABLE"
    ).reset_index(drop=True)

    return indicators


# ============================================================
# COMPARACIÓN MAESTRA DE MODELOS
# ============================================================

def build_master_models():
    """Consolida los modelos globales, de efectos y Ecuador."""

    frames = []

    verify_file(
        GLOBAL_MODEL_METRICS_FILE
    )

    global_metrics = pd.read_csv(
        GLOBAL_MODEL_METRICS_FILE
    )

    global_output = global_metrics[
        [
            "SAMPLE",
            "TARGET_COLUMN",
            "EQUATION",
            "MODEL",
            "OBSERVATION_COUNT",
            "COUNTRY_COUNT",
            "SELECTED_ALPHA",
            "RMSE_CV",
            "MAE_CV",
            "R2_CV",
        ]
    ].copy()

    global_output[
        "ANALYSIS_LEVEL"
    ] = "Global linear ODE"

    global_output[
        "VALIDATION_METHOD"
    ] = "GroupKFold by country"

    global_output[
        "RMSE_IMPROVEMENT_VS_ZERO"
    ] = pd.NA

    frames.append(
        global_output
    )

    verify_file(
        EFFECTS_MODEL_METRICS_FILE
    )

    effects_metrics = pd.read_csv(
        EFFECTS_MODEL_METRICS_FILE
    )

    effects_output = effects_metrics[
        [
            "SAMPLE",
            "TARGET_COLUMN",
            "EQUATION",
            "MODEL",
            "OBSERVATION_COUNT",
            "COUNTRY_COUNT",
            "SELECTED_ALPHA",
            "RMSE_CV",
            "MAE_CV",
            "R2_CV",
            "RMSE_IMPROVEMENT_VS_ZERO",
        ]
    ].copy()

    effects_output[
        "ANALYSIS_LEVEL"
    ] = "Global model comparison"

    effects_output[
        "VALIDATION_METHOD"
    ] = "GroupKFold by year"

    frames.append(
        effects_output
    )

    verify_file(
        ECUADOR_MODEL_METRICS_FILE
    )

    ecuador_metrics = pd.read_csv(
        ECUADOR_MODEL_METRICS_FILE
    )

    ecuador_output = pd.DataFrame(
        {
            "SAMPLE": "Ecuador annual series",
            "TARGET_COLUMN": (
                ecuador_metrics[
                    "TARGET_COLUMN"
                ]
            ),
            "EQUATION": (
                ecuador_metrics[
                    "EQUATION"
                ]
            ),
            "MODEL": (
                ecuador_metrics[
                    "MODEL"
                ]
            ),
            "OBSERVATION_COUNT": (
                ecuador_metrics[
                    "OBSERVATION_COUNT"
                ]
            ),
            "COUNTRY_COUNT": 1,
            "SELECTED_ALPHA": (
                ecuador_metrics[
                    "SELECTED_ALPHA"
                ]
            ),
            "RMSE_CV": (
                ecuador_metrics[
                    "RMSE_LOO"
                ]
            ),
            "MAE_CV": (
                ecuador_metrics[
                    "MAE_LOO"
                ]
            ),
            "R2_CV": (
                ecuador_metrics[
                    "R2_LOO"
                ]
            ),
            "RMSE_IMPROVEMENT_VS_ZERO": (
                ecuador_metrics[
                    "RMSE_IMPROVEMENT_VS_ZERO"
                ]
            ),
            "ANALYSIS_LEVEL": (
                "Ecuador exploratory model"
            ),
            "VALIDATION_METHOD": (
                "Leave-One-Out"
            ),
        }
    )

    frames.append(
        ecuador_output
    )

    master_models = pd.concat(
        frames,
        ignore_index=True,
    )

    output_columns = [
        "ANALYSIS_LEVEL",
        "SAMPLE",
        "TARGET_COLUMN",
        "EQUATION",
        "MODEL",
        "VALIDATION_METHOD",
        "OBSERVATION_COUNT",
        "COUNTRY_COUNT",
        "SELECTED_ALPHA",
        "RMSE_CV",
        "MAE_CV",
        "R2_CV",
        "RMSE_IMPROVEMENT_VS_ZERO",
    ]

    master_models = master_models[
        output_columns
    ]

    master_models = master_models.sort_values(
        by=[
            "ANALYSIS_LEVEL",
            "SAMPLE",
            "TARGET_COLUMN",
            "RMSE_CV",
        ],
        na_position="last",
    ).reset_index(drop=True)

    return master_models


# ============================================================
# CASOS NACIONALES
# ============================================================

def build_country_cases():
    """Combina cobertura y tendencias de países seleccionados."""

    verify_file(
        COUNTRY_CASE_SUMMARY_FILE
    )

    verify_file(
        COUNTRY_TRENDS_FILE
    )

    country_summary = pd.read_csv(
        COUNTRY_CASE_SUMMARY_FILE
    )

    trends = pd.read_csv(
        COUNTRY_TRENDS_FILE
    )

    trend_wide = trends.pivot_table(
        index=[
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
        ],
        columns="VARIABLE",
        values=[
            "SLOPE_PER_YEAR",
            "R2_TREND",
        ],
        aggfunc="first",
    )

    trend_wide.columns = [
        f"{metric}_{variable}"
        for metric, variable
        in trend_wide.columns
    ]

    trend_wide = (
        trend_wide
        .reset_index()
    )

    country_cases = (
        country_summary
        .merge(
            trend_wide,
            on=[
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    country_cases = country_cases.sort_values(
        by=[
            "OBSERVATION_COUNT",
            "CONSECUTIVE_TRANSITIONS",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return country_cases


# ============================================================
# RESULTADOS DE CALIDAD
# ============================================================

def build_quality_results():
    """Combina rangos y sensibilidad del panel."""

    verify_file(
        QUALITY_RANGES_FILE
    )

    verify_file(
        QUALITY_SENSITIVITY_FILE
    )

    ranges = pd.read_csv(
        QUALITY_RANGES_FILE
    )

    sensitivity = pd.read_csv(
        QUALITY_SENSITIVITY_FILE
    )

    ranges_output = ranges.copy()

    ranges_output[
        "RESULT_TYPE"
    ] = "Variable range"

    ranges_output[
        "SAMPLE"
    ] = "Complete ACLF panel"

    sensitivity_output = sensitivity.copy()

    sensitivity_output[
        "RESULT_TYPE"
    ] = "Outlier sensitivity"

    sensitivity_output[
        "VARIABLE"
    ] = sensitivity_output[
        "EQUATION"
    ]

    quality_results = pd.concat(
        [
            ranges_output,
            sensitivity_output,
        ],
        ignore_index=True,
        sort=False,
    )

    return quality_results


# ============================================================
# DECISIONES METODOLÓGICAS
# ============================================================

def build_methodological_decisions():
    """Registra las decisiones metodológicas fundamentales."""

    rows = [
        {
            "ORDER": 1,
            "DECISION": (
                "Use official UIS bulk data"
            ),
            "JUSTIFICATION": (
                "La descarga masiva de UNESCO UIS permite "
                "procesamiento reproducible y conserva la base "
                "oficial completa."
            ),
            "IMPLEMENTATION": (
                "SDG_DATA_NATIONAL.csv y catálogos asociados."
            ),
            "ARTICLE_SECTION": (
                "Data and sources"
            ),
        },
        {
            "ORDER": 2,
            "DECISION": (
                "Keep raw data outside Git"
            ),
            "JUSTIFICATION": (
                "Los archivos originales y Parquet son grandes "
                "y pueden reconstruirse mediante scripts."
            ),
            "IMPLEMENTATION": (
                "data/raw, data/interim y data/processed "
                "protegidos mediante .gitignore."
            ),
            "ARTICLE_SECTION": (
                "Reproducibility"
            ),
        },
        {
            "ORDER": 3,
            "DECISION": (
                "Select indicators by concept and coverage"
            ),
            "JUSTIFICATION": (
                "La selección combina relevancia educativa, "
                "cobertura internacional y disponibilidad temporal."
            ),
            "IMPLEMENTATION": (
                "Ocho indicadores iniciales organizados por "
                "dimensión educativa."
            ),
            "ARTICLE_SECTION": (
                "Variable selection"
            ),
        },
        {
            "ORDER": 4,
            "DECISION": (
                "Use ACLF as the primary dynamic core"
            ),
            "JUSTIFICATION": (
                "La combinación A-C-L-F produjo la mayor cobertura "
                "conjunta entre todas las combinaciones de cuatro "
                "variables."
            ),
            "IMPLEMENTATION": (
                "A: exclusión escolar; C: finalización; "
                "L: alfabetización; F: financiamiento."
            ),
            "ARTICLE_SECTION": (
                "Dynamic model"
            ),
        },
        {
            "ORDER": 5,
            "DECISION": (
                "Use actual time intervals"
            ),
            "JUSTIFICATION": (
                "No todas las observaciones están separadas por "
                "un año. Las tasas se dividieron por DELTA_T."
            ),
            "IMPLEMENTATION": (
                "dX/dt aproximado por diferencia finita usando "
                "el intervalo real."
            ),
            "ARTICLE_SECTION": (
                "Rate construction"
            ),
        },
        {
            "ORDER": 6,
            "DECISION": (
                "Use consecutive transitions as primary sample"
            ),
            "JUSTIFICATION": (
                "Las transiciones anuales reducen la ambigüedad "
                "al aproximar derivadas temporales."
            ),
            "IMPLEMENTATION": (
                "DELTA_T igual a 1 para el ajuste principal."
            ),
            "ARTICLE_SECTION": (
                "Model estimation"
            ),
        },
        {
            "ORDER": 7,
            "DECISION": (
                "Use gaps up to two years for robustness"
            ),
            "JUSTIFICATION": (
                "Amplía la muestra sin mezclar intervalos "
                "temporales demasiado extensos."
            ),
            "IMPLEMENTATION": (
                "DELTA_T menor o igual a 2."
            ),
            "ARTICLE_SECTION": (
                "Robustness"
            ),
        },
        {
            "ORDER": 8,
            "DECISION": (
                "Compare against zero-change baseline"
            ),
            "JUSTIFICATION": (
                "Las tasas anuales suelen concentrarse cerca de "
                "cero. Un modelo debe superar esa referencia."
            ),
            "IMPLEMENTATION": (
                "Comparación de RMSE con predicción igual a cero."
            ),
            "ARTICLE_SECTION": (
                "Model evaluation"
            ),
        },
        {
            "ORDER": 9,
            "DECISION": (
                "Group cross-validation by country or year"
            ),
            "JUSTIFICATION": (
                "Evita repartir indiscriminadamente observaciones "
                "relacionadas entre entrenamiento y validación."
            ),
            "IMPLEMENTATION": (
                "GroupKFold por país en el modelo global y por "
                "año en la comparación de efectos."
            ),
            "ARTICLE_SECTION": (
                "Validation"
            ),
        },
        {
            "ORDER": 10,
            "DECISION": (
                "Treat Ecuador model as exploratory"
            ),
            "JUSTIFICATION": (
                "Ecuador tiene una serie anual limpia, pero solo "
                "14 transiciones, insuficientes para inferencia "
                "causal o modelos complejos."
            ),
            "IMPLEMENTATION": (
                "Ridge y Leave-One-Out sobre 14 transiciones."
            ),
            "ARTICLE_SECTION": (
                "Country case study"
            ),
        },
        {
            "ORDER": 11,
            "DECISION": (
                "Do not delete IQR outliers automatically"
            ),
            "JUSTIFICATION": (
                "Una observación extrema puede representar una "
                "situación educativa real, no un error."
            ),
            "IMPLEMENTATION": (
                "Señalización, revisión y análisis de sensibilidad."
            ),
            "ARTICLE_SECTION": (
                "Data quality"
            ),
        },
        {
            "ORDER": 12,
            "DECISION": (
                "Report weak predictive performance"
            ),
            "JUSTIFICATION": (
                "Los resultados negativos son evidencia científica "
                "y no deben ocultarse ni ajustarse artificialmente."
            ),
            "IMPLEMENTATION": (
                "Reporte de R2, RMSE y comparación con referencias."
            ),
            "ARTICLE_SECTION": (
                "Results and limitations"
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# INVENTARIO DE EVIDENCIAS
# ============================================================

def build_evidence_inventory():
    """Crea un inventario de tablas, figuras y resultados."""

    evidence_paths = [
        NATIONAL_DIAGNOSTIC,
        CATALOG_DIAGNOSTIC,
        CANDIDATE_SUMMARY,
        SELECTED_INDICATORS_FILE,
        PANEL_COVERAGE_FILE,
        PANEL_INDICATOR_COVERAGE_FILE,
        VARIABLE_COMBINATIONS_FILE,
        ACLF_SUMMARY_FILE,
        ACLF_RATES_SUMMARY_FILE,
        GLOBAL_MODEL_METRICS_FILE,
        GLOBAL_MODEL_DIAGNOSTIC_FILE,
        EFFECTS_MODEL_METRICS_FILE,
        ECUADOR_MODEL_METRICS_FILE,
        ECUADOR_MODEL_COEFFICIENTS_FILE,
        COUNTRY_CASE_SUMMARY_FILE,
        COUNTRY_TRENDS_FILE,
        QUALITY_SUMMARY_FILE,
        QUALITY_RANGES_FILE,
        QUALITY_SENSITIVITY_FILE,
        FIGURES_DIR / "aclf_ecuador_trajectories.png",
        FIGURES_DIR / "aclf_country_comparison.png",
        FIGURES_DIR / "aclf_country_standardized_change.png",
        FIGURES_DIR / "aclf_ecuador_rates.png",
        FIGURES_DIR / "ecuador_aclf_observed_predicted_rates.png",
        FIGURES_DIR / "ecuador_aclf_observed_simulated_states.png",
        FIGURES_DIR / "aclf_quality_distributions.png",
        FIGURES_DIR / "aclf_quality_rate_distributions.png",
        FIGURES_DIR / "aclf_quality_outliers_by_country.png",
    ]

    rows = []

    for file_path in evidence_paths:
        exists = file_path.exists()

        rows.append(
            {
                "RELATIVE_PATH": relative_path(
                    file_path
                ),
                "FILE_NAME": file_path.name,
                "FILE_TYPE": (
                    file_path.suffix.lower()
                ),
                "EXISTS": exists,
                "SIZE_BYTES": (
                    file_path.stat().st_size
                    if exists
                    else pd.NA
                ),
                "PURPOSE": infer_evidence_purpose(
                    file_path
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def infer_evidence_purpose(file_path):
    """Describe brevemente la función de cada evidencia."""

    name = file_path.name.lower()

    if "national_diagnostic" in name:
        return "Validación de la base nacional original"

    if "catalog_integration" in name:
        return "Validación de catálogos y etiquetas"

    if "candidate" in name:
        return "Búsqueda de indicadores candidatos"

    if "selected_indicators" in name:
        return "Selección final de indicadores"

    if "panel_coverage" in name:
        return "Cobertura del panel país-año"

    if "variable_combinations" in name:
        return "Comparación de combinaciones de variables"

    if "aclf_summary" in name:
        return "Resumen del panel ACLF"

    if "rates_summary" in name:
        return "Resumen de tasas temporales"

    if "model_metrics" in name:
        return "Métricas de validación de modelos"

    if "coefficients" in name:
        return "Coeficientes estimados"

    if "country_case" in name:
        return "Resumen de casos nacionales"

    if "trends" in name:
        return "Tendencias por país y variable"

    if "quality" in name:
        return "Diagnóstico de calidad"

    if file_path.suffix.lower() == ".png":
        return "Figura científica para resultados"

    return "Evidencia reproducible del análisis"


# ============================================================
# RESUMEN MARKDOWN
# ============================================================

def build_markdown_summary(
    master_results,
    master_models,
    selected_indicators,
    country_cases,
    methodological_decisions,
):
    """Genera un resumen narrativo verificable."""

    result_lookup = {
        row["RESULT"]: row["VALUE"]
        for _, row in master_results.iterrows()
    }

    primary_effects = master_models.loc[
        (
            master_models[
                "ANALYSIS_LEVEL"
            ].eq(
                "Global model comparison"
            )
        )
        & (
            master_models[
                "SAMPLE"
            ].eq(
                "consecutive"
            )
        )
    ].copy()

    best_models = (
        primary_effects
        .sort_values(
            by=[
                "TARGET_COLUMN",
                "RMSE_CV",
            ]
        )
        .drop_duplicates(
            subset=[
                "TARGET_COLUMN"
            ],
            keep="first",
        )
        .sort_values(
            by="TARGET_COLUMN"
        )
    )

    ecuador = country_cases.loc[
        country_cases[
            "COUNTRY_ID"
        ].eq(
            "ECU"
        )
    ]

    if ecuador.empty:
        raise ValueError(
            "No se encontro el caso de Ecuador."
        )

    ecuador_row = ecuador.iloc[0]

    lines = [
        "# Consolidated results for the educational dynamics article",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## 1. Data source and processing",
        "",
        (
            f"- Original national records processed: "
            f"{int(result_lookup.get('Original national rows', 0)):,}."
        ),
        (
            f"- Unique indicators in the national database: "
            f"{int(result_lookup.get('Unique indicators', 0)):,}."
        ),
        (
            f"- Countries or territories represented: "
            f"{int(result_lookup.get('Unique countries', 0)):,}."
        ),
        (
            f"- Source period: "
            f"{int(result_lookup.get('Minimum source year', 0))} to "
            f"{int(result_lookup.get('Maximum source year', 0))}."
        ),
        "",
        "## 2. Analytical panel",
        "",
        (
            f"- Complete ACLF country-year observations: "
            f"{int(result_lookup.get('ACLF country-year rows', 0)):,}."
        ),
        (
            f"- Countries or territories in the ACLF panel: "
            f"{int(result_lookup.get('ACLF countries', 0)):,}."
        ),
        (
            f"- Total temporal transitions: "
            f"{int(result_lookup.get('Total transitions', 0)):,}."
        ),
        (
            f"- Consecutive annual transitions: "
            f"{int(result_lookup.get('Consecutive transitions', 0)):,}."
        ),
        "",
        "## 3. Selected indicators",
        "",
    ]

    for _, row in selected_indicators.iterrows():
        lines.append(
            (
                f"- {row['MODEL_VARIABLE']}: "
                f"{row['DESCRIPTION_ES']} "
                f"({row['INDICATOR_ID']})."
            )
        )

    lines.extend(
        [
            "",
            "## 4. Best validated global model by equation",
            "",
        ]
    )

    for _, row in best_models.iterrows():
        alpha_text = (
            "not applicable"
            if pd.isna(
                row[
                    "SELECTED_ALPHA"
                ]
            )
            else format_number(
                row[
                    "SELECTED_ALPHA"
                ],
                4,
            )
        )

        lines.append(
            (
                f"- {row['EQUATION']}: {row['MODEL']}; "
                f"R2 = {format_number(row['R2_CV'], 4)}, "
                f"RMSE = {format_number(row['RMSE_CV'], 4)}, "
                f"alpha = {alpha_text}."
            )
        )

    lines.extend(
        [
            "",
            "## 5. Ecuador case study",
            "",
            (
                f"- Observations: "
                f"{int(ecuador_row['OBSERVATION_COUNT'])}."
            ),
            (
                f"- Period: "
                f"{int(ecuador_row['MIN_YEAR'])} to "
                f"{int(ecuador_row['MAX_YEAR'])}."
            ),
            (
                f"- Consecutive transitions: "
                f"{int(ecuador_row['CONSECUTIVE_TRANSITIONS'])}."
            ),
        ]
    )

    for variable in [
        "A",
        "C",
        "L",
        "F",
    ]:
        slope_column = (
            f"SLOPE_PER_YEAR_{variable}"
        )

        r2_column = (
            f"R2_TREND_{variable}"
        )

        if slope_column in ecuador_row.index:
            lines.append(
                (
                    f"- Trend {variable}: "
                    f"{format_number(ecuador_row[slope_column], 4)} "
                    f"units per year; trend R2 = "
                    f"{format_number(ecuador_row[r2_column], 4)}."
                )
            )

    lines.extend(
        [
            "",
            "## 6. Data quality",
            "",
            (
                f"- Values outside expected ranges: "
                f"{int(result_lookup.get('Values outside expected ranges', 0))}."
            ),
            (
                f"- Level observations flagged by IQR: "
                f"{int(result_lookup.get('Level IQR outliers', 0)):,}."
            ),
            (
                f"- Temporal rates flagged by IQR: "
                f"{int(result_lookup.get('Rate IQR outliers', 0)):,}."
            ),
            (
                "- Flagged observations were retained in the original "
                "database and evaluated through sensitivity analysis."
            ),
            "",
            "## 7. Main methodological decisions",
            "",
        ]
    )

    for _, row in methodological_decisions.iterrows():
        lines.append(
            (
                f"{int(row['ORDER'])}. {row['DECISION']}: "
                f"{row['JUSTIFICATION']}"
            )
        )

    lines.extend(
        [
            "",
            "## 8. Overall interpretation",
            "",
            (
                "The global linear system, country effects, year effects, "
                "and nonlinear specifications showed limited predictive "
                "performance for annual educational changes."
            ),
            (
                "The weak performance remained after sensitivity analyses, "
                "which suggests that the result is not explained only by "
                "extreme observations."
            ),
            (
                "The estimated equations must therefore be interpreted as "
                "exploratory dynamic descriptions and not as causal models "
                "or reliable forecasting systems."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "CONSOLIDACION DE RESULTADOS PARA EL ARTICULO"
    )

    ARTICLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nConstruyendo tabla maestra de resultados..."
    )

    master_results = build_master_results()

    print(
        "Consolidando indicadores seleccionados..."
    )

    selected_indicators = (
        build_selected_indicators()
    )

    print(
        "Consolidando comparacion de modelos..."
    )

    master_models = build_master_models()

    print(
        "Consolidando casos nacionales..."
    )

    country_cases = build_country_cases()

    print(
        "Consolidando diagnostico de calidad..."
    )

    quality_results = build_quality_results()

    print(
        "Registrando decisiones metodologicas..."
    )

    methodological_decisions = (
        build_methodological_decisions()
    )

    print(
        "Creando inventario de evidencias..."
    )

    evidence_inventory = (
        build_evidence_inventory()
    )

    missing_evidence = evidence_inventory.loc[
        ~evidence_inventory[
            "EXISTS"
        ]
    ]

    if not missing_evidence.empty:
        print(
            "\nADVERTENCIA: algunas evidencias no fueron encontradas:"
        )

        print(
            missing_evidence[
                [
                    "RELATIVE_PATH",
                    "PURPOSE",
                ]
            ].to_string(
                index=False
            )
        )

    article_summary = build_markdown_summary(
        master_results=master_results,
        master_models=master_models,
        selected_indicators=selected_indicators,
        country_cases=country_cases,
        methodological_decisions=methodological_decisions,
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    master_results.to_csv(
        MASTER_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    master_models.to_csv(
        MASTER_MODELS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    selected_indicators.to_csv(
        MASTER_INDICATORS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    country_cases.to_csv(
        MASTER_COUNTRY_CASES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    quality_results.to_csv(
        MASTER_QUALITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    methodological_decisions.to_csv(
        METHODOLOGICAL_DECISIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_inventory.to_csv(
        EVIDENCE_INVENTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    ARTICLE_SUMMARY_FILE.write_text(
        article_summary,
        encoding="utf-8",
    )

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "project_root": str(
            PROJECT_ROOT
        ),
        "output_directory": str(
            ARTICLE_DIR
        ),
        "master_results_rows": int(
            len(master_results)
        ),
        "master_models_rows": int(
            len(master_models)
        ),
        "selected_indicators_rows": int(
            len(selected_indicators)
        ),
        "country_cases_rows": int(
            len(country_cases)
        ),
        "quality_results_rows": int(
            len(quality_results)
        ),
        "methodological_decisions_rows": int(
            len(methodological_decisions)
        ),
        "evidence_inventory_rows": int(
            len(evidence_inventory)
        ),
        "missing_evidence_count": int(
            len(missing_evidence)
        ),
        "validation_status": "OK",
    }

    with ARTICLE_CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    configuration[
                        "processing_datetime"
                    ]
                ),
                "MASTER_RESULTS_ROWS": len(
                    master_results
                ),
                "MASTER_MODELS_ROWS": len(
                    master_models
                ),
                "SELECTED_INDICATORS_ROWS": len(
                    selected_indicators
                ),
                "COUNTRY_CASES_ROWS": len(
                    country_cases
                ),
                "QUALITY_RESULTS_ROWS": len(
                    quality_results
                ),
                "METHODOLOGICAL_DECISIONS_ROWS": len(
                    methodological_decisions
                ),
                "EVIDENCE_INVENTORY_ROWS": len(
                    evidence_inventory
                ),
                "MISSING_EVIDENCE_COUNT": len(
                    missing_evidence
                ),
                "SUMMARY_FILE": (
                    ARTICLE_SUMMARY_FILE.name
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        CONSOLIDATION_DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESULTADOS EN CONSOLA
    # ========================================================

    print_section(
        "CONSOLIDACION COMPLETADA CORRECTAMENTE"
    )

    print(
        "\nRESUMEN DE TABLAS GENERADAS"
    )

    print(
        f"Resultados generales: {len(master_results)} filas"
    )

    print(
        f"Comparacion de modelos: {len(master_models)} filas"
    )

    print(
        f"Indicadores seleccionados: {len(selected_indicators)} filas"
    )

    print(
        f"Casos nacionales: {len(country_cases)} filas"
    )

    print(
        f"Resultados de calidad: {len(quality_results)} filas"
    )

    print(
        "Decisiones metodologicas: "
        f"{len(methodological_decisions)} filas"
    )

    print(
        f"Evidencias inventariadas: {len(evidence_inventory)}"
    )

    print(
        f"Evidencias no encontradas: {len(missing_evidence)}"
    )

    print(
        "\nRESULTADOS CENTRALES"
    )

    print(
        master_results[
            [
                "SECTION",
                "RESULT",
                "VALUE",
                "UNIT",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nARCHIVOS CREADOS"
    )

    output_files = [
        MASTER_RESULTS_FILE,
        MASTER_MODELS_FILE,
        MASTER_INDICATORS_FILE,
        MASTER_COUNTRY_CASES_FILE,
        MASTER_QUALITY_FILE,
        METHODOLOGICAL_DECISIONS_FILE,
        EVIDENCE_INVENTORY_FILE,
        ARTICLE_SUMMARY_FILE,
        ARTICLE_CONFIG_FILE,
        CONSOLIDATION_DIAGNOSTIC_FILE,
    ]

    for file_path in output_files:
        print(
            relative_path(
                file_path
            )
        )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE LA CONSOLIDACION"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)