from pathlib import Path
from datetime import datetime
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "model"

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "aclf_linear_ode_coefficients.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "aclf_linear_ode_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "aclf_linear_ode_predictions.csv"
)

EQUATIONS_FILE = (
    OUTPUT_DIR
    / "aclf_linear_ode_equations.txt"
)

MODEL_CONFIG_FILE = (
    OUTPUT_DIR
    / "aclf_linear_ode_config.json"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_linear_ode_diagnostic.csv"
)


# ============================================================
# CONFIGURACION DEL MODELO
# ============================================================

STATE_VARIABLES = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

VARIABLE_NAMES = {
    "PREVIOUS_A": "A",
    "PREVIOUS_C": "C",
    "PREVIOUS_L": "L",
    "PREVIOUS_F": "F",
}

TARGETS = {
    "DA_DT": "dA/dt",
    "DC_DT": "dC/dt",
    "DL_DT": "dL/dt",
    "DF_DT": "dF/dt",
}

RIDGE_ALPHAS = np.logspace(
    -4,
    4,
    81,
)

RANDOM_SEED = 2026


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title: str) -> None:
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_file(file_path: Path) -> None:
    """Comprueba que exista un archivo necesario."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def select_sample(
    data: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:
    """
    Selecciona las observaciones para cada ajuste.

    consecutive:
        Usa solamente transiciones con DELTA_T igual a 1.

    gap_max_2:
        Usa transiciones con DELTA_T igual a 1 o 2.
    """

    if sample_name == "consecutive":
        sample = data.loc[
            data["DELTA_T"].eq(1)
        ].copy()

    elif sample_name == "gap_max_2":
        sample = data.loc[
            data["DELTA_T"].le(2)
        ].copy()

    else:
        raise ValueError(
            f"Muestra desconocida: {sample_name}"
        )

    required_columns = [
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "PREVIOUS_YEAR",
        "YEAR",
        "DELTA_T",
        *STATE_VARIABLES,
        *TARGETS.keys(),
    ]

    sample = sample.dropna(
        subset=required_columns
    ).copy()

    sample = sample.sort_values(
        by=[
            "COUNTRY_ID",
            "YEAR",
        ]
    ).reset_index(drop=True)

    return sample


def determine_group_folds(
    country_count: int,
) -> int:
    """
    Define un numero seguro de grupos para GroupKFold.
    """

    if country_count >= 10:
        return 5

    if country_count >= 4:
        return 4

    if country_count >= 3:
        return 3

    raise ValueError(
        "Se necesitan al menos tres paises para "
        "realizar validacion cruzada agrupada."
    )


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calcula las metricas de prediccion."""

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

    observed_std = float(
        np.std(
            observed,
            ddof=1,
        )
    )

    if observed_std > 0:
        normalized_rmse = (
            rmse / observed_std
        )
    else:
        normalized_rmse = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "OBSERVED_STD": observed_std,
        "NORMALIZED_RMSE": normalized_rmse,
    }


def format_equation(
    equation_name: str,
    intercept: float,
    coefficients: dict,
) -> str:
    """Convierte los coeficientes en una ecuacion legible."""

    equation = (
        f"{equation_name} = {intercept:.8f}"
    )

    for variable in ["A", "C", "L", "F"]:
        coefficient = coefficients[variable]

        if coefficient >= 0:
            equation += (
                f" + {coefficient:.8f}*{variable}"
            )
        else:
            equation += (
                f" - {abs(coefficient):.8f}*{variable}"
            )

    return equation


def calculate_condition_number(
    features: pd.DataFrame,
) -> float:
    """
    Calcula el numero de condicion de las variables
    explicativas estandarizadas.
    """

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(
        features
    )

    singular_values = np.linalg.svd(
        standardized_features,
        compute_uv=False,
    )

    smallest_value = singular_values.min()

    if smallest_value == 0:
        return float("inf")

    return float(
        singular_values.max()
        / smallest_value
    )


# ============================================================
# AJUSTE DE UNA ECUACION
# ============================================================

def fit_equation(
    sample: pd.DataFrame,
    sample_name: str,
    target_column: str,
):
    """
    Ajusta una ecuacion con RidgeCV.

    Tambien ajusta una regresion lineal convencional como
    referencia. La validacion cruzada se agrupa por pais.
    """

    equation_name = TARGETS[
        target_column
    ]

    features = sample[
        STATE_VARIABLES
    ].astype(float)

    target = sample[
        target_column
    ].astype(float)

    groups = sample[
        "COUNTRY_ID"
    ].astype(str)

    country_count = int(
        groups.nunique()
    )

    fold_count = determine_group_folds(
        country_count
    )

    group_cv = GroupKFold(
        n_splits=fold_count
    )

    # ========================================================
    # MODELO RIDGE
    # ========================================================

    ridge_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                RidgeCV(
                    alphas=RIDGE_ALPHAS,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ridge_cv_predictions = cross_val_predict(
            ridge_pipeline,
            features,
            target,
            groups=groups,
            cv=group_cv,
        )

    ridge_pipeline.fit(
        features,
        target,
    )

    ridge_metrics = calculate_metrics(
        target.to_numpy(),
        ridge_cv_predictions,
    )

    # ========================================================
    # REGRESION LINEAL DE REFERENCIA
    # ========================================================

    linear_model = LinearRegression()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        linear_cv_predictions = cross_val_predict(
            linear_model,
            features,
            target,
            groups=groups,
            cv=group_cv,
        )

    linear_model.fit(
        features,
        target,
    )

    linear_metrics = calculate_metrics(
        target.to_numpy(),
        linear_cv_predictions,
    )

    # ========================================================
    # RECUPERAR COEFICIENTES EN ESCALA ORIGINAL
    # ========================================================

    fitted_scaler = ridge_pipeline.named_steps[
        "scaler"
    ]

    fitted_ridge = ridge_pipeline.named_steps[
        "ridge"
    ]

    standardized_coefficients = np.asarray(
        fitted_ridge.coef_,
        dtype=float,
    )

    original_coefficients = (
        standardized_coefficients
        / fitted_scaler.scale_
    )

    original_intercept = float(
        fitted_ridge.intercept_
        - np.sum(
            standardized_coefficients
            * fitted_scaler.mean_
            / fitted_scaler.scale_
        )
    )

    coefficient_map = {
        VARIABLE_NAMES[column]: float(coefficient)
        for column, coefficient in zip(
            STATE_VARIABLES,
  original_coefficients,
        )
    }

    selected_alpha = float(
        fitted_ridge.alpha_
    )

    equation_text = format_equation(
        equation_name=equation_name,
        intercept=original_intercept,
        coefficients=coefficient_map,
    )

    condition_number = calculate_condition_number(
        features
    )

    # ========================================================
    # METRICAS
    # ========================================================

    metric_row = {
        "SAMPLE": sample_name,
        "TARGET_COLUMN": target_column,
        "EQUATION": equation_name,
        "MODEL": "RidgeCV",
        "OBSERVATION_COUNT": len(sample),
        "COUNTRY_COUNT": country_count,
        "GROUP_FOLDS": fold_count,
        "SELECTED_ALPHA": selected_alpha,
        "RMSE_CV": ridge_metrics["RMSE"],
        "MAE_CV": ridge_metrics["MAE"],
        "R2_CV": ridge_metrics["R2"],
        "OBSERVED_STD": (
            ridge_metrics["OBSERVED_STD"]
        ),
        "NORMALIZED_RMSE": (
            ridge_metrics["NORMALIZED_RMSE"]
        ),
        "LINEAR_RMSE_CV": (
            linear_metrics["RMSE"]
        ),
        "LINEAR_MAE_CV": (
            linear_metrics["MAE"]
        ),
        "LINEAR_R2_CV": (
            linear_metrics["R2"]
        ),
        "CONDITION_NUMBER_STANDARDIZED_X": (
            condition_number
        ),
    }

    # ========================================================
    # COEFICIENTES
    # ========================================================

    coefficient_rows = [
        {
            "SAMPLE": sample_name,
            "TARGET_COLUMN": target_column,
            "EQUATION": equation_name,
            "TERM": "INTERCEPT",
            "VARIABLE": "1",
            "COEFFICIENT_ORIGINAL_SCALE": (
                original_intercept
            ),
            "COEFFICIENT_STANDARDIZED": np.nan,
            "SELECTED_ALPHA": selected_alpha,
        }
    ]

    for (
        state_column,
        original_coefficient,
        standardized_coefficient,
    ) in zip(
        STATE_VARIABLES,
        original_coefficients,
        standardized_coefficients,
    ):
        coefficient_rows.append(
            {
                "SAMPLE": sample_name,
                "TARGET_COLUMN": target_column,
                "EQUATION": equation_name,
                "TERM": state_column,
                "VARIABLE": VARIABLE_NAMES[
                    state_column
                ],
                "COEFFICIENT_ORIGINAL_SCALE": float(
                    original_coefficient
                ),
                "COEFFICIENT_STANDARDIZED": float(
                    standardized_coefficient
                ),
                "SELECTED_ALPHA": selected_alpha,
            }
        )

    # ========================================================
    # PREDICCIONES DE VALIDACION
    # ========================================================

    predictions = sample[
        [
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "PREVIOUS_YEAR",
            "YEAR",
            "DELTA_T",
        ]
    ].copy()

    predictions["SAMPLE"] = sample_name
    predictions["TARGET_COLUMN"] = target_column
    predictions["EQUATION"] = equation_name

    predictions["OBSERVED_RATE"] = (
        target.to_numpy()
    )

    predictions["PREDICTED_RATE_CV"] = (
        ridge_cv_predictions
    )

    predictions["RESIDUAL_CV"] = (
        predictions["OBSERVED_RATE"]
        - predictions["PREDICTED_RATE_CV"]
    )

    predictions["ABSOLUTE_RESIDUAL_CV"] = (
        predictions["RESIDUAL_CV"].abs()
    )

    return (
        metric_row,
        coefficient_rows,
        predictions,
        equation_text,
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main() -> None:
    print_section(
        "AJUSTE DEL SISTEMA LINEAL DE ECUACIONES A-C-L-F"
    )

    verify_file(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLeyendo las tasas temporales...")

    data = pd.read_parquet(
        INPUT_FILE
    )

    required_columns = {
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "PREVIOUS_YEAR",
        "YEAR",
        "DELTA_T",
        *STATE_VARIABLES,
        *TARGETS.keys(),
    }

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el archivo: {missing_columns}"
        )

    print(
        "Transiciones disponibles:",
        f"{len(data):,}",
    )

    all_metrics = []
    all_coefficients = []
    all_predictions = []
    all_equations = []

    samples = [
        "consecutive",
        "gap_max_2",
    ]

    for sample_name in samples:
        sample = select_sample(
            data=data,
            sample_name=sample_name,
        )

        print_section(
            f"MUESTRA: {sample_name}"
        )

        print(
            "Observaciones:",
            f"{len(sample):,}",
        )

        print(
            "Paises o territorios:",
            f"{sample['COUNTRY_ID'].nunique():,}",
        )

        for target_column in TARGETS:
            equation_name = TARGETS[
                target_column
            ]

            print(
                f"\nAjustando {equation_name}..."
            )

            (
                metric_row,
                coefficient_rows,
                predictions,
                equation_text,
            ) = fit_equation(
                sample=sample,
                sample_name=sample_name,
                target_column=target_column,
            )

            all_metrics.append(
                metric_row
            )

            all_coefficients.extend(
                coefficient_rows
            )

            all_predictions.append(
                predictions
            )

            all_equations.append(
                {
                    "SAMPLE": sample_name,
                    "TARGET_COLUMN": target_column,
                    "EQUATION_TEXT": equation_text,
                }
            )

            print(
                "R2 de validacion agrupada:",
                f"{metric_row['R2_CV']:.4f}",
            )

            print(
                "RMSE de validacion agrupada:",
                f"{metric_row['RMSE_CV']:.4f}",
            )

            print(
                "Alpha Ridge seleccionado:",
                f"{metric_row['SELECTED_ALPHA']:.8f}",
            )

    # ========================================================
    # TABLAS DE RESULTADOS
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    coefficients_df = pd.DataFrame(
        all_coefficients
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    equations_df = pd.DataFrame(
        all_equations
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients_df.to_csv(
        COEFFICIENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # ARCHIVO DE ECUACIONES
    # ========================================================

    with EQUATIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "SISTEMA LINEAL DE ECUACIONES A-C-L-F\n"
        )

        file.write(
            "Estimacion mediante RidgeCV con validacion "
            "cruzada agrupada por pais.\n\n"
        )

        for sample_name in samples:
            file.write(
                f"MUESTRA: {sample_name}\n"
            )

            subset = equations_df.loc[
                equations_df["SAMPLE"].eq(
                    sample_name
                )
            ]

            for equation_text in subset[
                "EQUATION_TEXT"
            ]:
                file.write(
                    f"{equation_text}\n"
                )

            file.write("\n")

    # ========================================================
    # CONFIGURACION DEL MODELO
    # ========================================================

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "random_seed": RANDOM_SEED,
        "state_variables": [
            "A",
            "C",
            "L",
            "F",
        ],
        "targets": TARGETS,
        "samples": {
            "consecutive": (
                "DELTA_T igual a 1"
            ),
            "gap_max_2": (
                "DELTA_T menor o igual a 2"
            ),
        },
        "model": "RidgeCV",
        "feature_scaling": "StandardScaler",
        "cross_validation": (
            "GroupKFold grouped by COUNTRY_ID"
        ),
        "alpha_grid_minimum": float(
            RIDGE_ALPHAS.min()
        ),
        "alpha_grid_maximum": float(
            RIDGE_ALPHAS.max()
        ),
        "alpha_grid_count": int(
            len(RIDGE_ALPHAS)
        ),
        "primary_sample": "consecutive",
        "robustness_sample": "gap_max_2",
    }

    with MODEL_CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # COMPARACION PRINCIPAL Y ROBUSTEZ
    # ========================================================

    primary_metrics = metrics_df.loc[
        metrics_df["SAMPLE"].eq(
            "consecutive"
        )
    ].copy()

    robustness_metrics = metrics_df.loc[
        metrics_df["SAMPLE"].eq(
            "gap_max_2"
        )
    ].copy()

    diagnostic_rows = []

    for target_column in TARGETS:
        primary = primary_metrics.loc[
            primary_metrics[
                "TARGET_COLUMN"
            ].eq(target_column)
        ].iloc[0]

        robustness = robustness_metrics.loc[
            robustness_metrics[
                "TARGET_COLUMN"
            ].eq(target_column)
        ].iloc[0]

        diagnostic_rows.append(
            {
                "TARGET_COLUMN": target_column,
                "EQUATION": TARGETS[
                    target_column
                ],
                "PRIMARY_OBSERVATIONS": int(
                    primary[
                        "OBSERVATION_COUNT"
                    ]
                ),
                "PRIMARY_COUNTRIES": int(
                    primary["COUNTRY_COUNT"]
                ),
                "PRIMARY_R2_CV": float(
                    primary["R2_CV"]
                ),
                "PRIMARY_RMSE_CV": float(
                    primary["RMSE_CV"]
                ),
                "PRIMARY_ALPHA": float(
                    primary["SELECTED_ALPHA"]
                ),
                "ROBUSTNESS_OBSERVATIONS": int(
                    robustness[
                        "OBSERVATION_COUNT"
                    ]
                ),
                "ROBUSTNESS_COUNTRIES": int(
                    robustness["COUNTRY_COUNT"]
                ),
                "ROBUSTNESS_R2_CV": float(
                    robustness["R2_CV"]
                ),
                "ROBUSTNESS_RMSE_CV": float(
                    robustness["RMSE_CV"]
                ),
                "ROBUSTNESS_ALPHA": float(
                    robustness[
                        "SELECTED_ALPHA"
                    ]
                ),
                "R2_DIFFERENCE": float(
                    robustness["R2_CV"]
                    - primary["R2_CV"]
                ),
                "VALIDATION_STATUS": "OK",
            }
        )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_df.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESULTADOS EN CONSOLA
    # ========================================================

    print_section(
        "MODELO A-C-L-F AJUSTADO CORRECTAMENTE"
    )

    print(
        "\nMETRICAS DE LA MUESTRA PRINCIPAL"
    )

    print(
        primary_metrics[
            [
                "EQUATION",
                "OBSERVATION_COUNT",
                "COUNTRY_COUNT",
                "SELECTED_ALPHA",
                "RMSE_CV",
                "MAE_CV",
                "R2_CV",
                "NORMALIZED_RMSE",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nECUACIONES DE LA MUESTRA PRINCIPAL"
    )

    primary_equations = equations_df.loc[
        equations_df["SAMPLE"].eq(
            "consecutive"
        ),
        "EQUATION_TEXT",
    ]

    for equation_text in primary_equations:
        print(equation_text)

    print(
        "\nCOMPARACION CON LA MUESTRA DE "
        "BRECHA MAXIMA DE 2 ANIOS"
    )

    comparison = diagnostic_df[
        [
            "EQUATION",
            "PRIMARY_R2_CV",
            "ROBUSTNESS_R2_CV",
            "R2_DIFFERENCE",
            "PRIMARY_RMSE_CV",
            "ROBUSTNESS_RMSE_CV",
        ]
    ]

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nCoeficientes:\n{COEFFICIENTS_FILE}"
    )

    print(
        f"\nMetricas:\n{METRICS_FILE}"
    )

    print(
        f"\nPredicciones:\n{PREDICTIONS_FILE}"
    )

    print(
        f"\nEcuaciones:\n{EQUATIONS_FILE}"
    )

    print(
        f"\nConfiguracion:\n{MODEL_CONFIG_FILE}"
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
            "ERROR DURANTE EL AJUSTE DEL MODELO"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)