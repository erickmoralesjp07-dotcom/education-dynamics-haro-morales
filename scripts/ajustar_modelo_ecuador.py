from pathlib import Path
from datetime import datetime
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
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
    / "model_ecuador"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_metrics.csv"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_coefficients.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_predictions.csv"
)

SIMULATION_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_simulation.csv"
)

EQUATIONS_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_equations.txt"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "ecuador_aclf_config.json"
)

DIAGNOSTIC_FILE = (
    DIAGNOSTICS_DIR
    / "ecuador_aclf_model_diagnostic.csv"
)

PREDICTION_FIGURE = (
    FIGURES_DIR
    / "ecuador_aclf_observed_predicted_rates.png"
)

SIMULATION_FIGURE = (
    FIGURES_DIR
    / "ecuador_aclf_observed_simulated_states.png"
)


# ============================================================
# CONFIGURACION
# ============================================================

COUNTRY_ID = "ECU"

STATE_COLUMNS = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

STATE_VARIABLES = [
    "A",
    "C",
    "L",
    "F",
]

TARGETS = {
    "DA_DT": "dA/dt",
    "DC_DT": "dC/dt",
    "DL_DT": "dL/dt",
    "DF_DT": "dF/dt",
}

TARGET_TO_STATE = {
    "DA_DT": "A",
    "DC_DT": "C",
    "DL_DT": "L",
    "DF_DT": "F",
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
    """Construye el modelo Ridge estandarizado."""

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


def leave_one_out_predictions(
    features,
    target,
    alpha,
):
    """
    Genera predicciones Leave-One-Out.

    Cada transición se valida usando todas las demás
    transiciones como entrenamiento.
    """

    loo = LeaveOneOut()

    predictions = np.zeros(
        len(target),
        dtype=float,
    )

    for train_indices, test_indices in loo.split(
        features
    ):
        model = create_ridge_model(
            alpha
        )

        x_train = features.iloc[
            train_indices
        ]

        y_train = target.iloc[
            train_indices
        ]

        x_test = features.iloc[
            test_indices
        ]

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            model.fit(
                x_train,
                y_train,
            )

            predictions[
                test_indices
            ] = model.predict(
                x_test
            )

    return predictions


def choose_alpha(
    features,
    target,
):
    """Selecciona el alpha con menor RMSE Leave-One-Out."""

    results = []

    for alpha in ALPHAS:
        predictions = leave_one_out_predictions(
            features=features,
            target=target,
            alpha=alpha,
        )

        metrics = calculate_metrics(
            target,
            predictions,
        )

        results.append(
            {
                "ALPHA": alpha,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "PREDICTIONS": predictions,
            }
        )

    best_result = min(
        results,
        key=lambda row: row["RMSE"],
    )

    return best_result, results


def recover_original_coefficients(
    fitted_pipeline,
):
    """
    Recupera intercepto y coeficientes en la escala original.
    """

    scaler = fitted_pipeline.named_steps[
        "scaler"
    ]

    ridge = fitted_pipeline.named_steps[
        "ridge"
    ]

    standardized_coefficients = np.asarray(
        ridge.coef_,
        dtype=float,
    )

    original_coefficients = (
        standardized_coefficients
        / scaler.scale_
    )

    original_intercept = float(
        ridge.intercept_
        - np.sum(
            standardized_coefficients
            * scaler.mean_
            / scaler.scale_
        )
    )

    return (
        original_intercept,
        original_coefficients,
        standardized_coefficients,
    )


def format_equation(
    equation_name,
    intercept,
    coefficients,
):
    """Construye una ecuación lista para el informe."""

    equation = (
        f"{equation_name} = {intercept:.8f}"
    )

    for variable, coefficient in zip(
        STATE_VARIABLES,
        coefficients,
    ):
        if coefficient >= 0:
            equation += (
                f" + {coefficient:.8f}*{variable}"
            )
        else:
            equation += (
                f" - {abs(coefficient):.8f}*{variable}"
            )

    return equation


# ============================================================
# GRAFICAS
# ============================================================

def create_prediction_figure(
    predictions_df,
):
    """Compara tasas observadas y predichas."""

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9),
    )

    axes = axes.flatten()

    for axis, target_column in zip(
        axes,
        TARGETS,
    ):
        data = predictions_df.loc[
            predictions_df[
                "TARGET_COLUMN"
            ].eq(target_column)
        ].sort_values(
            "YEAR"
        )

        axis.plot(
            data["YEAR"],
            data["OBSERVED_RATE"],
            marker="o",
            linewidth=2,
            label="Observada",
        )

        axis.plot(
            data["YEAR"],
            data["PREDICTED_RATE_LOO"],
            marker="s",
            linestyle="--",
            linewidth=1.7,
            label="Predicha LOO",
        )

        axis.axhline(
            y=0,
            color="black",
            linewidth=0.8,
        )

        axis.set_title(
            TARGETS[target_column]
        )

        axis.set_xlabel(
            "Año final"
        )

        axis.set_ylabel(
            "Cambio por año"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend()

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    figure.suptitle(
        "Ecuador: tasas observadas y predichas con Leave-One-Out",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        PREDICTION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_simulation_figure(
    simulation_df,
):
    """Compara estados observados y simulados a un paso."""

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9),
    )

    axes = axes.flatten()

    for axis, variable in zip(
        axes,
        STATE_VARIABLES,
    ):
        axis.plot(
            simulation_df["YEAR"],
            simulation_df[
                f"OBSERVED_{variable}"
            ],
            marker="o",
            linewidth=2,
            label="Observado",
        )

        axis.plot(
            simulation_df["YEAR"],
            simulation_df[
                f"SIMULATED_{variable}"
            ],
            marker="s",
            linestyle="--",
            linewidth=1.7,
            label="Simulado a un paso",
        )

        axis.set_title(
            f"Variable {variable}"
        )

        axis.set_xlabel(
            "Año"
        )

        axis.set_ylabel(
            "Valor"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend()

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    figure.suptitle(
        "Ecuador: valores observados y simulados a un paso",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        SIMULATION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "AJUSTE EXPLORATORIO DEL MODELO A-C-L-F PARA ECUADOR"
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

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel = pd.read_parquet(
        PANEL_FILE
    )

    rates = pd.read_parquet(
        RATES_FILE
    )

    ecuador_panel = panel.loc[
        panel["COUNTRY_ID"].eq(
            COUNTRY_ID
        )
    ].sort_values(
        "YEAR"
    ).reset_index(drop=True)

    ecuador_rates = rates.loc[
        rates["COUNTRY_ID"].eq(
            COUNTRY_ID
        )
        & rates["DELTA_T"].eq(1)
    ].sort_values(
        "YEAR"
    ).reset_index(drop=True)

    if len(ecuador_panel) != 15:
        print(
            "\nADVERTENCIA: se esperaban 15 observaciones "
            f"para Ecuador y se encontraron {len(ecuador_panel)}."
        )

    if len(ecuador_rates) != 14:
        print(
            "\nADVERTENCIA: se esperaban 14 transiciones "
            f"y se encontraron {len(ecuador_rates)}."
        )

    print(
        "\nObservaciones de Ecuador:",
        len(ecuador_panel),
    )

    print(
        "Transiciones anuales:",
        len(ecuador_rates),
    )

    print(
        "Periodo:",
        f"{int(ecuador_panel['YEAR'].min())} - "
        f"{int(ecuador_panel['YEAR'].max())}",
    )

    features = ecuador_rates[
        STATE_COLUMNS
    ].astype(float)

    metrics_rows = []
    coefficient_rows = []
    prediction_frames = []
    diagnostic_rows = []
    fitted_models = {}
    equations = {}

    for target_column, equation_name in TARGETS.items():
        print(
            f"\nAjustando {equation_name}..."
        )

        target = ecuador_rates[
            target_column
        ].astype(float)

        best_result, alpha_results = choose_alpha(
            features=features,
            target=target,
        )

        selected_alpha = float(
            best_result["ALPHA"]
        )

        loo_predictions = np.asarray(
            best_result["PREDICTIONS"],
            dtype=float,
        )

        ridge_metrics = calculate_metrics(
            target,
            loo_predictions,
        )

        zero_predictions = np.zeros(
            len(target),
            dtype=float,
        )

        zero_metrics = calculate_metrics(
            target,
            zero_predictions,
        )

        mean_predictions = np.full(
            len(target),
            float(target.mean()),
            dtype=float,
        )

        mean_metrics = calculate_metrics(
            target,
            mean_predictions,
        )

        final_model = create_ridge_model(
            selected_alpha
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            final_model.fit(
                features,
                target,
            )

        fitted_models[
            target_column
        ] = final_model

        (
            original_intercept,
            original_coefficients,
            standardized_coefficients,
        ) = recover_original_coefficients(
            final_model
        )

        equation_text = format_equation(
            equation_name=equation_name,
            intercept=original_intercept,
            coefficients=original_coefficients,
        )

        equations[
            target_column
        ] = equation_text

        metrics_rows.append(
            {
                "TARGET_COLUMN": target_column,
                "EQUATION": equation_name,
                "MODEL": "Ridge_LOO",
                "OBSERVATION_COUNT": len(
                    target
                ),
                "SELECTED_ALPHA": (
                    selected_alpha
                ),
                "RMSE_LOO": (
                    ridge_metrics["RMSE"]
                ),
                "MAE_LOO": (
                    ridge_metrics["MAE"]
                ),
                "R2_LOO": (
                    ridge_metrics["R2"]
                ),
                "ZERO_RMSE": (
                    zero_metrics["RMSE"]
                ),
                "ZERO_MAE": (
                    zero_metrics["MAE"]
                ),
                "ZERO_R2": (
                    zero_metrics["R2"]
                ),
                "MEAN_RMSE": (
                    mean_metrics["RMSE"]
                ),
                "MEAN_MAE": (
                    mean_metrics["MAE"]
                ),
                "MEAN_R2": (
                    mean_metrics["R2"]
                ),
                "RMSE_IMPROVEMENT_VS_ZERO": (
                    zero_metrics["RMSE"]
                    - ridge_metrics["RMSE"]
                ),
            }
        )

        coefficient_rows.append(
            {
                "TARGET_COLUMN": target_column,
                "EQUATION": equation_name,
                "TERM": "INTERCEPT",
                "VARIABLE": "1",
                "COEFFICIENT_ORIGINAL": (
                    original_intercept
                ),
                "COEFFICIENT_STANDARDIZED": (
                    np.nan
                ),
                "SELECTED_ALPHA": (
                    selected_alpha
                ),
            }
        )

        for (
            state_column,
            state_variable,
            original_coefficient,
            standardized_coefficient,
        ) in zip(
            STATE_COLUMNS,
            STATE_VARIABLES,
            original_coefficients,
            standardized_coefficients,
        ):
            coefficient_rows.append(
                {
                    "TARGET_COLUMN": target_column,
                    "EQUATION": equation_name,
                    "TERM": state_column,
                    "VARIABLE": state_variable,
                    "COEFFICIENT_ORIGINAL": float(
                        original_coefficient
                    ),
                    "COEFFICIENT_STANDARDIZED": float(
                        standardized_coefficient
                    ),
                    "SELECTED_ALPHA": (
                        selected_alpha
                    ),
                }
            )

        prediction_frame = ecuador_rates[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "PREVIOUS_YEAR",
                "YEAR",
                "DELTA_T",
            ]
        ].copy()

        prediction_frame[
            "TARGET_COLUMN"
        ] = target_column

        prediction_frame[
            "EQUATION"
        ] = equation_name

        prediction_frame[
            "SELECTED_ALPHA"
        ] = selected_alpha

        prediction_frame[
            "OBSERVED_RATE"
        ] = target.to_numpy()

        prediction_frame[
            "PREDICTED_RATE_LOO"
        ] = loo_predictions

        prediction_frame[
            "ZERO_PREDICTION"
        ] = zero_predictions

        prediction_frame[
            "RESIDUAL_LOO"
        ] = (
            prediction_frame[
                "OBSERVED_RATE"
            ]
            - prediction_frame[
                "PREDICTED_RATE_LOO"
            ]
        )

        prediction_frames.append(
            prediction_frame
        )

        for alpha_result in alpha_results:
            diagnostic_rows.append(
                {
                    "TARGET_COLUMN": (
                        target_column
                    ),
                    "EQUATION": equation_name,
                    "ALPHA": (
                        alpha_result["ALPHA"]
                    ),
                    "RMSE_LOO": (
                        alpha_result["RMSE"]
                    ),
                    "MAE_LOO": (
                        alpha_result["MAE"]
                    ),
                    "R2_LOO": (
                        alpha_result["R2"]
                    ),
                }
            )

        print(
            "Alpha seleccionado:",
            selected_alpha,
        )

        print(
            "R2 Leave-One-Out:",
            f"{ridge_metrics['R2']:.4f}",
        )

        print(
            "RMSE Leave-One-Out:",
            f"{ridge_metrics['RMSE']:.4f}",
        )

        print(
            "Mejora RMSE frente a cero:",
            f"{zero_metrics['RMSE'] - ridge_metrics['RMSE']:.4f}",
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    # ========================================================
    # SIMULACION DE UN PASO
    # ========================================================

    simulation_rows = []

    for _, row in ecuador_rates.iterrows():
        state_vector = pd.DataFrame(
            [
                {
                    "PREVIOUS_A": row[
                        "PREVIOUS_A"
                    ],
                    "PREVIOUS_C": row[
                        "PREVIOUS_C"
                    ],
                    "PREVIOUS_L": row[
                        "PREVIOUS_L"
                    ],
                    "PREVIOUS_F": row[
                        "PREVIOUS_F"
                    ],
                }
            ]
        )

        predicted_rates = {}

        for target_column in TARGETS:
            predicted_rates[
                target_column
            ] = float(
                fitted_models[
                    target_column
                ].predict(
                    state_vector
                )[0]
            )

        simulation_rows.append(
            {
                "COUNTRY_ID": COUNTRY_ID,
                "COUNTRY_NAME_EN": "Ecuador",
                "PREVIOUS_YEAR": int(
                    row["PREVIOUS_YEAR"]
                ),
                "YEAR": int(
                    row["YEAR"]
                ),
                "OBSERVED_A": float(
                    row["A"]
                ),
                "OBSERVED_C": float(
                    row["C"]
                ),
                "OBSERVED_L": float(
                    row["L"]
                ),
                "OBSERVED_F": float(
                    row["F"]
                ),
                "SIMULATED_A": float(
                    row["PREVIOUS_A"]
                    + predicted_rates[
                        "DA_DT"
                    ]
                ),
                "SIMULATED_C": float(
                    row["PREVIOUS_C"]
                    + predicted_rates[
                        "DC_DT"
                    ]
                ),
                "SIMULATED_L": float(
                    row["PREVIOUS_L"]
                    + predicted_rates[
                        "DL_DT"
                    ]
                ),
                "SIMULATED_F": float(
                    row["PREVIOUS_F"]
                    + predicted_rates[
                        "DF_DT"
                    ]
                ),
                "PREDICTED_DA_DT": (
                    predicted_rates[
                        "DA_DT"
                    ]
                ),
                "PREDICTED_DC_DT": (
                    predicted_rates[
                        "DC_DT"
                    ]
                ),
                "PREDICTED_DL_DT": (
                    predicted_rates[
                        "DL_DT"
                    ]
                ),
                "PREDICTED_DF_DT": (
                    predicted_rates[
                        "DF_DT"
                    ]
                ),
            }
        )

    simulation_df = pd.DataFrame(
        simulation_rows
    )

    for variable in STATE_VARIABLES:
        simulation_df[
            f"ERROR_{variable}"
        ] = (
            simulation_df[
                f"OBSERVED_{variable}"
            ]
            - simulation_df[
                f"SIMULATED_{variable}"
            ]
        )

        simulation_df[
            f"ABS_ERROR_{variable}"
        ] = (
            simulation_df[
                f"ERROR_{variable}"
            ].abs()
        )

    # ========================================================
    # GUARDADO
    # ========================================================

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

    simulation_df.to_csv(
        SIMULATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic_df.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with EQUATIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "MODELO EXPLORATORIO A-C-L-F PARA ECUADOR\n"
        )

        file.write(
            "Validacion Leave-One-Out sobre 14 "
            "transiciones anuales.\n\n"
        )

        for target_column in TARGETS:
            file.write(
                equations[
                    target_column
                ]
                + "\n"
            )

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "country_id": COUNTRY_ID,
        "country_name": "Ecuador",
        "observation_count": int(
            len(ecuador_panel)
        ),
        "transition_count": int(
            len(ecuador_rates)
        ),
        "minimum_year": int(
            ecuador_panel["YEAR"].min()
        ),
        "maximum_year": int(
            ecuador_panel["YEAR"].max()
        ),
        "validation_method": (
            "LeaveOneOut"
        ),
        "model": (
            "Ridge con StandardScaler"
        ),
        "alphas": ALPHAS,
        "state_variables": (
            STATE_VARIABLES
        ),
        "targets": TARGETS,
        "simulation": (
            "Prediccion de un paso usando el "
            "estado observado del anio anterior"
        ),
        "interpretation": (
            "Modelo exploratorio y descriptivo. "
            "No permite inferencia causal."
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

    create_prediction_figure(
        predictions_df
    )

    create_simulation_figure(
        simulation_df
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print_section(
        "MODELO DE ECUADOR AJUSTADO CORRECTAMENTE"
    )

    print(
        "\nMETRICAS LEAVE-ONE-OUT"
    )

    print(
        metrics_df[
            [
                "EQUATION",
                "SELECTED_ALPHA",
                "RMSE_LOO",
                "MAE_LOO",
                "R2_LOO",
                "ZERO_RMSE",
                "RMSE_IMPROVEMENT_VS_ZERO",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nECUACIONES ESTIMADAS"
    )

    for target_column in TARGETS:
        print(
            equations[
                target_column
            ]
        )

    print(
        "\nERROR MEDIO ABSOLUTO DE LA "
        "SIMULACION DE UN PASO"
    )

    simulation_mae_rows = []

    for variable in STATE_VARIABLES:
        simulation_mae_rows.append(
            {
                "VARIABLE": variable,
                "MAE_ONE_STEP": float(
                    simulation_df[
                        f"ABS_ERROR_{variable}"
                    ].mean()
                ),
                "MAX_ABS_ERROR": float(
                    simulation_df[
                        f"ABS_ERROR_{variable}"
                    ].max()
                ),
            }
        )

    simulation_mae = pd.DataFrame(
        simulation_mae_rows
    )

    print(
        simulation_mae.to_string(
            index=False
        )
    )

    print(
        f"\nMetricas:\n{METRICS_FILE}"
    )

    print(
        f"\nCoeficientes:\n{COEFFICIENTS_FILE}"
    )

    print(
        f"\nPredicciones:\n{PREDICTIONS_FILE}"
    )

    print(
        f"\nSimulacion:\n{SIMULATION_FILE}"
    )

    print(
        f"\nEcuaciones:\n{EQUATIONS_FILE}"
    )

    print(
        f"\nConfiguracion:\n{CONFIG_FILE}"
    )

    print(
        f"\nDiagnostico:\n{DIAGNOSTIC_FILE}"
    )

    print(
        f"\nFigura de tasas:\n{PREDICTION_FIGURE}"
    )

    print(
        f"\nFigura de simulacion:\n{SIMULATION_FIGURE}"
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE EL MODELO DE ECUADOR"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)