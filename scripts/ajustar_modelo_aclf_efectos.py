from pathlib import Path
from datetime import datetime
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_effects"
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "aclf_effects_model_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "aclf_effects_model_predictions.csv"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "aclf_effects_model_coefficients.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "aclf_effects_model_config.json"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTICS_DIR
    / "aclf_effects_model_diagnostic.csv"
)


# ============================================================
# CONFIGURACION
# ============================================================

STATE_VARIABLES = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

TARGETS = {
    "DA_DT": "dA/dt",
    "DC_DT": "dC/dt",
    "DL_DT": "dL/dt",
    "DF_DT": "dF/dt",
}

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]

MODEL_NAMES = [
    "global_linear",
    "country_effects",
    "country_year_effects",
    "nonlinear_country_year",
]

SAMPLE_NAMES = [
    "consecutive",
    "gap_max_2",
]


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def print_section(title):
    """Imprime un encabezado."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def verify_file(file_path):
    """Comprueba que exista un archivo."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo:\n{file_path}"
        )


def calculate_metrics(observed, predicted):
    """Calcula RMSE, MAE, R2 y RMSE normalizado."""

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


def select_sample(data, sample_name):
    """Selecciona la muestra principal o de robustez."""

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

    sample["COUNTRY_ID"] = (
        sample["COUNTRY_ID"]
        .astype(str)
    )

    sample["YEAR"] = pd.to_numeric(
        sample["YEAR"],
        errors="coerce",
    ).astype(int)

    sample["YEAR_CATEGORY"] = (
        sample["YEAR"]
        .astype(str)
    )

    sample = sample.sort_values(
        by=[
            "YEAR",
            "COUNTRY_ID",
        ]
    ).reset_index(drop=True)

    return sample


def determine_year_folds(year_count):
    """Determina el numero de particiones por año."""

    if year_count >= 10:
        return 5

    if year_count >= 6:
        return 4

    if year_count >= 3:
        return 3

    raise ValueError(
        "No existen suficientes anios para "
        "la validacion cruzada."
    )


# ============================================================
# CONSTRUCCION DE MODELOS
# ============================================================

def create_model(model_name, alpha):
    """Construye una especificacion Ridge."""

    numeric_columns = list(
        STATE_VARIABLES
    )

    if model_name == "global_linear":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    numeric_columns,
                ),
            ],
            remainder="drop",
        )

        feature_columns = numeric_columns

    elif model_name == "country_effects":
        categorical_columns = [
            "COUNTRY_ID",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    numeric_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        feature_columns = (
            numeric_columns
            + categorical_columns
        )

    elif model_name == "country_year_effects":
        categorical_columns = [
            "COUNTRY_ID",
            "YEAR_CATEGORY",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    numeric_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        feature_columns = (
            numeric_columns
            + categorical_columns
        )

    elif model_name == "nonlinear_country_year":
        categorical_columns = [
            "COUNTRY_ID",
            "YEAR_CATEGORY",
        ]

        nonlinear_pipeline = Pipeline(
            steps=[
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    nonlinear_pipeline,
                    numeric_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        feature_columns = (
            numeric_columns
            + categorical_columns
        )

    else:
        raise ValueError(
            f"Modelo desconocido: {model_name}"
        )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )

    return model, feature_columns


# ============================================================
# VALIDACION CRUZADA
# ============================================================

def cross_validate_model(
    sample,
    target_column,
    model_name,
    alpha,
):
    """Evalua un modelo usando años completos como grupos."""

    year_groups = sample[
        "YEAR"
    ].astype(int)

    year_count = int(
        year_groups.nunique()
    )

    fold_count = determine_year_folds(
        year_count
    )

    group_cv = GroupKFold(
        n_splits=fold_count
    )

    observed_values = []
    predicted_values = []
    prediction_frames = []

    split_iterator = group_cv.split(
        sample,
        groups=year_groups,
    )

    for fold_number, split in enumerate(
        split_iterator,
        start=1,
    ):
        train_indices, test_indices = split

        train_data = sample.iloc[
            train_indices
        ].copy()

        test_data = sample.iloc[
            test_indices
        ].copy()

        model, feature_columns = create_model(
            model_name=model_name,
            alpha=alpha,
        )

        x_train = train_data[
            feature_columns
        ]

        y_train = train_data[
            target_column
        ].astype(float)

        x_test = test_data[
            feature_columns
        ]

        y_test = test_data[
            target_column
        ].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                x_train,
                y_train,
            )

            fold_predictions = model.predict(
                x_test
            )

        observed_values.extend(
            y_test.to_numpy()
        )

        predicted_values.extend(
            fold_predictions
        )

        prediction_frame = test_data[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "PREVIOUS_YEAR",
                "YEAR",
                "DELTA_T",
            ]
        ].copy()

        prediction_frame["FOLD"] = (
            fold_number
        )

        prediction_frame[
            "TARGET_COLUMN"
        ] = target_column

        prediction_frame["MODEL"] = (
            model_name
        )

        prediction_frame["ALPHA"] = (
            alpha
        )

        prediction_frame[
            "OBSERVED_RATE"
        ] = y_test.to_numpy()

        prediction_frame[
            "PREDICTED_RATE"
        ] = fold_predictions

        prediction_frame["RESIDUAL"] = (
            prediction_frame["OBSERVED_RATE"]
            - prediction_frame["PREDICTED_RATE"]
        )

        prediction_frames.append(
            prediction_frame
        )

    metrics = calculate_metrics(
        observed_values,
        predicted_values,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return (
        metrics,
        predictions,
        fold_count,
        year_count,
    )


def choose_best_alpha(
    sample,
    target_column,
    model_name,
):
    """Escoge el alpha con el menor RMSE."""

    alpha_results = []

    for alpha in ALPHAS:
        (
            metrics,
            predictions,
            fold_count,
            year_count,
        ) = cross_validate_model(
            sample=sample,
            target_column=target_column,
            model_name=model_name,
            alpha=alpha,
        )

        alpha_results.append(
            {
                "ALPHA": alpha,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "NORMALIZED_RMSE": (
                    metrics[
                        "NORMALIZED_RMSE"
                    ]
                ),
                "PREDICTIONS": predictions,
                "FOLD_COUNT": fold_count,
                "YEAR_COUNT": year_count,
            }
        )

    best_result = min(
        alpha_results,
        key=lambda row: row["RMSE"],
    )

    return best_result, alpha_results


# ============================================================
# BASELINE CERO
# ============================================================

def evaluate_zero_baseline(
    sample,
    target_column,
):
    """Predice cambio igual a cero."""

    observed = sample[
        target_column
    ].astype(float).to_numpy()

    predicted = np.zeros(
        len(sample),
        dtype=float,
    )

    metrics = calculate_metrics(
        observed,
        predicted,
    )

    predictions = sample[
        [
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "PREVIOUS_YEAR",
            "YEAR",
            "DELTA_T",
        ]
    ].copy()

    predictions["FOLD"] = 0
    predictions["TARGET_COLUMN"] = target_column
    predictions["MODEL"] = "zero_baseline"
    predictions["ALPHA"] = np.nan
    predictions["OBSERVED_RATE"] = observed
    predictions["PREDICTED_RATE"] = predicted

    predictions["RESIDUAL"] = (
        predictions["OBSERVED_RATE"]
        - predictions["PREDICTED_RATE"]
    )

    return metrics, predictions


# ============================================================
# BASELINE PROMEDIO POR PAIS
# ============================================================

def evaluate_country_mean_baseline(
    sample,
    target_column,
):
    """Predice usando el promedio historico del pais."""

    year_groups = sample[
        "YEAR"
    ].astype(int)

    fold_count = determine_year_folds(
        int(year_groups.nunique())
    )

    group_cv = GroupKFold(
        n_splits=fold_count
    )

    observed_values = []
    predicted_values = []
    prediction_frames = []

    split_iterator = group_cv.split(
        sample,
        groups=year_groups,
    )

    for fold_number, split in enumerate(
        split_iterator,
        start=1,
    ):
        train_indices, test_indices = split

        train_data = sample.iloc[
            train_indices
        ].copy()

        test_data = sample.iloc[
            test_indices
        ].copy()

        global_train_mean = float(
            train_data[
                target_column
            ].mean()
        )

        country_means = (
            train_data
            .groupby("COUNTRY_ID")[
                target_column
            ]
            .mean()
        )

        predicted = (
            test_data["COUNTRY_ID"]
            .map(country_means)
            .fillna(global_train_mean)
            .astype(float)
            .to_numpy()
        )

        observed = (
            test_data[target_column]
            .astype(float)
            .to_numpy()
        )

        observed_values.extend(
            observed
        )

        predicted_values.extend(
            predicted
        )

        prediction_frame = test_data[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "PREVIOUS_YEAR",
                "YEAR",
                "DELTA_T",
            ]
        ].copy()

        prediction_frame["FOLD"] = (
            fold_number
        )

        prediction_frame[
            "TARGET_COLUMN"
        ] = target_column

        prediction_frame["MODEL"] = (
            "country_mean_baseline"
        )

        prediction_frame["ALPHA"] = np.nan
        prediction_frame["OBSERVED_RATE"] = observed

        prediction_frame[
            "PREDICTED_RATE"
        ] = predicted

        prediction_frame["RESIDUAL"] = (
            prediction_frame["OBSERVED_RATE"]
            - prediction_frame["PREDICTED_RATE"]
        )

        prediction_frames.append(
            prediction_frame
        )

    metrics = calculate_metrics(
        observed_values,
        predicted_values,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return metrics, predictions


# ============================================================
# COEFICIENTES DEL MODELO FINAL
# ============================================================

def fit_final_model(
    sample,
    target_column,
    model_name,
    alpha,
):
    """Ajusta toda la muestra y extrae coeficientes."""

    model, feature_columns = create_model(
        model_name=model_name,
        alpha=alpha,
    )

    x_data = sample[
        feature_columns
    ]

    y_data = sample[
        target_column
    ].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.fit(
            x_data,
            y_data,
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    ridge_model = model.named_steps[
        "ridge"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = np.asarray(
        ridge_model.coef_,
        dtype=float,
    )

    rows = [
        {
            "TARGET_COLUMN": target_column,
            "MODEL": model_name,
            "ALPHA": alpha,
            "TERM": "INTERCEPT",
            "COEFFICIENT": float(
                ridge_model.intercept_
            ),
        }
    ]

    for feature_name, coefficient in zip(
        feature_names,
        coefficients,
    ):
        rows.append(
            {
                "TARGET_COLUMN": target_column,
                "MODEL": model_name,
                "ALPHA": alpha,
                "TERM": feature_name,
                "COEFFICIENT": float(
                    coefficient
                ),
            }
        )

    return rows


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "COMPARACION DE MODELOS A-C-L-F CON EFECTOS"
    )

    verify_file(
        INPUT_FILE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_parquet(
        INPUT_FILE
    )

    print(
        "\nTransiciones disponibles:",
        f"{len(data):,}",
    )

    metrics_rows = []
    prediction_frames = []
    coefficient_rows = []
    alpha_diagnostic_rows = []

    for sample_name in SAMPLE_NAMES:
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
            "Paises:",
            f"{sample['COUNTRY_ID'].nunique():,}",
        )

        print(
            "Anios de transicion:",
            f"{sample['YEAR'].nunique():,}",
        )

        for target_column, equation_name in TARGETS.items():
            print(
                f"\nEvaluando {equation_name}..."
            )

            # Baseline cero
            zero_metrics, zero_predictions = (
                evaluate_zero_baseline(
                    sample=sample,
                    target_column=target_column,
                )
            )

            metrics_rows.append(
                {
                    "SAMPLE": sample_name,
                    "TARGET_COLUMN": target_column,
                    "EQUATION": equation_name,
                    "MODEL": "zero_baseline",
                    "SELECTED_ALPHA": np.nan,
                    "OBSERVATION_COUNT": len(sample),
                    "COUNTRY_COUNT": int(
                        sample[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "YEAR_COUNT": int(
                        sample[
                            "YEAR"
                        ].nunique()
                    ),
                    "RMSE_CV": zero_metrics["RMSE"],
                    "MAE_CV": zero_metrics["MAE"],
                    "R2_CV": zero_metrics["R2"],
                    "NORMALIZED_RMSE": (
                        zero_metrics[
                            "NORMALIZED_RMSE"
                        ]
                    ),
                }
            )

            zero_predictions["SAMPLE"] = (
                sample_name
            )

            prediction_frames.append(
                zero_predictions
            )

            # Baseline promedio por pais
            (
                mean_metrics,
                mean_predictions,
            ) = evaluate_country_mean_baseline(
                sample=sample,
                target_column=target_column,
            )

            metrics_rows.append(
                {
                    "SAMPLE": sample_name,
                    "TARGET_COLUMN": target_column,
                    "EQUATION": equation_name,
                    "MODEL": (
                        "country_mean_baseline"
                    ),
                    "SELECTED_ALPHA": np.nan,
                    "OBSERVATION_COUNT": len(sample),
                    "COUNTRY_COUNT": int(
                        sample[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "YEAR_COUNT": int(
                        sample[
                            "YEAR"
                        ].nunique()
                    ),
                    "RMSE_CV": mean_metrics["RMSE"],
                    "MAE_CV": mean_metrics["MAE"],
                    "R2_CV": mean_metrics["R2"],
                    "NORMALIZED_RMSE": (
                        mean_metrics[
                            "NORMALIZED_RMSE"
                        ]
                    ),
                }
            )

            mean_predictions["SAMPLE"] = (
                sample_name
            )

            prediction_frames.append(
                mean_predictions
            )

            # Modelos Ridge
            for model_name in MODEL_NAMES:
                (
                    best_result,
                    alpha_results,
                ) = choose_best_alpha(
                    sample=sample,
                    target_column=target_column,
                    model_name=model_name,
                )

                selected_alpha = float(
                    best_result["ALPHA"]
                )

                metrics_rows.append(
                    {
                        "SAMPLE": sample_name,
                        "TARGET_COLUMN": target_column,
                        "EQUATION": equation_name,
                        "MODEL": model_name,
                        "SELECTED_ALPHA": (
                            selected_alpha
                        ),
                        "OBSERVATION_COUNT": len(
                            sample
                        ),
                        "COUNTRY_COUNT": int(
                            sample[
                                "COUNTRY_ID"
                            ].nunique()
                        ),
                        "YEAR_COUNT": int(
                            sample[
                                "YEAR"
                            ].nunique()
                        ),
                        "RMSE_CV": (
                            best_result["RMSE"]
                        ),
                        "MAE_CV": (
                            best_result["MAE"]
                        ),
                        "R2_CV": (
                            best_result["R2"]
                        ),
                        "NORMALIZED_RMSE": (
                            best_result[
                                "NORMALIZED_RMSE"
                            ]
                        ),
                    }
                )

                selected_predictions = (
                    best_result[
                        "PREDICTIONS"
                    ].copy()
                )

                selected_predictions[
                    "SAMPLE"
                ] = sample_name

                prediction_frames.append(
                    selected_predictions
                )

                for alpha_result in alpha_results:
                    alpha_diagnostic_rows.append(
                        {
                            "SAMPLE": sample_name,
                            "TARGET_COLUMN": (
                                target_column
                            ),
                            "MODEL": model_name,
                            "ALPHA": (
                                alpha_result[
                                    "ALPHA"
                                ]
                            ),
                            "RMSE_CV": (
                                alpha_result[
                                    "RMSE"
                                ]
                            ),
                            "MAE_CV": (
                                alpha_result[
                                    "MAE"
                                ]
                            ),
                            "R2_CV": (
                                alpha_result[
                                    "R2"
                                ]
                            ),
                        }
                    )

                coefficient_rows.extend(
                    fit_final_model(
                        sample=sample,
                        target_column=target_column,
                        model_name=model_name,
                        alpha=selected_alpha,
                    )
                )

                print(
                    f"  {model_name}: "
                    f"R2={best_result['R2']:.4f}, "
                    f"RMSE={best_result['RMSE']:.4f}, "
                    f"alpha={selected_alpha:g}"
                )

    # Crear tablas
    metrics_df = pd.DataFrame(
        metrics_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
    )

    alpha_diagnostic_df = pd.DataFrame(
        alpha_diagnostic_rows
    )

    # Obtener el RMSE del baseline cero
    zero_rmse = (
        metrics_df.loc[
            metrics_df["MODEL"].eq(
                "zero_baseline"
            ),
            [
                "SAMPLE",
                "TARGET_COLUMN",
                "RMSE_CV",
            ],
        ]
        .rename(
            columns={
                "RMSE_CV": (
                    "ZERO_BASELINE_RMSE"
                )
            }
        )
    )

    metrics_df = metrics_df.merge(
        zero_rmse,
        on=[
            "SAMPLE",
            "TARGET_COLUMN",
        ],
        how="left",
        validate="many_to_one",
    )

    metrics_df[
        "RMSE_IMPROVEMENT_VS_ZERO"
    ] = (
        metrics_df["ZERO_BASELINE_RMSE"]
        - metrics_df["RMSE_CV"]
    )

    metrics_df = metrics_df.sort_values(
        by=[
            "SAMPLE",
            "TARGET_COLUMN",
            "RMSE_CV",
        ]
    ).reset_index(drop=True)

    # Guardar archivos
    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    coefficients_df.to_csv(
        COEFFICIENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    alpha_diagnostic_df.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "input_file": INPUT_FILE.name,
        "primary_sample": "consecutive",
        "robustness_sample": "gap_max_2",
        "validation": (
            "GroupKFold agrupado por YEAR"
        ),
        "state_variables": STATE_VARIABLES,
        "targets": TARGETS,
        "models": [
            "zero_baseline",
            "country_mean_baseline",
            *MODEL_NAMES,
        ],
        "ridge_alphas": ALPHAS,
        "polynomial_degree": 2,
        "validation_status": "OK",
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

    # Mostrar resultados
    print_section(
        "COMPARACION DE MODELOS COMPLETADA"
    )

    primary_metrics = metrics_df.loc[
        metrics_df["SAMPLE"].eq(
            "consecutive"
        )
    ].copy()

    for target_column, equation_name in TARGETS.items():
        print(
            f"\nRESULTADOS PARA {equation_name}"
        )

        result_table = (
            primary_metrics.loc[
                primary_metrics[
                    "TARGET_COLUMN"
                ].eq(target_column),
                [
                    "MODEL",
                    "SELECTED_ALPHA",
                    "RMSE_CV",
                    "MAE_CV",
                    "R2_CV",
                    "RMSE_IMPROVEMENT_VS_ZERO",
                ],
            ]
            .sort_values(
                by="RMSE_CV"
            )
        )

        print(
            result_table.to_string(
                index=False
            )
        )

    # Mantener la fila completa del ganador
    best_models = (
        primary_metrics
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
        .reset_index(drop=True)
    )

    print(
        "\nMEJOR MODELO POR ECUACION"
    )

    print(
        best_models[
            [
                "EQUATION",
                "MODEL",
                "SELECTED_ALPHA",
                "RMSE_CV",
                "MAE_CV",
                "R2_CV",
                "RMSE_IMPROVEMENT_VS_ZERO",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nMetricas:\n{METRICS_FILE}"
    )

    print(
        f"\nPredicciones:\n{PREDICTIONS_FILE}"
    )

    print(
        f"\nCoeficientes:\n{COEFFICIENTS_FILE}"
    )

    print(
        f"\nConfiguracion:\n{CONFIG_FILE}"
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
            "ERROR DURANTE LA COMPARACION DE MODELOS"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)