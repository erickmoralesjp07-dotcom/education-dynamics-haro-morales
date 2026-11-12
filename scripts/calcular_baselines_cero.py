from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COUNTRY_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "model"
    / "aclf_linear_ode_predictions.csv"
)

ECUADOR_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "model_ecuador"
    / "ecuador_aclf_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmark_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(observed):
    observed = pd.to_numeric(
        observed,
        errors="coerce",
    ).dropna()

    predicted = np.zeros(len(observed))

    return {
        "OBSERVATION_COUNT": len(observed),
        "RMSE_CV": mean_squared_error(
            observed,
            predicted,
        )
        ** 0.5,
        "MAE_CV": mean_absolute_error(
            observed,
            predicted,
        ),
        "R2_CV": r2_score(
            observed,
            predicted,
        ),
    }


def calculate_country_baselines():
    data = pd.read_csv(COUNTRY_PREDICTIONS)

    required = {
        "SAMPLE",
        "TARGET_COLUMN",
        "EQUATION",
        "COUNTRY_ID",
        "OBSERVED_RATE",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en predicciones globales: "
            f"{sorted(missing)}"
        )

    rows = []

    for keys, group in data.groupby(
        [
            "SAMPLE",
            "TARGET_COLUMN",
            "EQUATION",
        ],
        dropna=False,
    ):
        metrics = calculate_metrics(
            group["OBSERVED_RATE"]
        )

        valid_mask = pd.to_numeric(
            group["OBSERVED_RATE"],
            errors="coerce",
        ).notna()

        rows.append(
            {
                "SAMPLE": keys[0],
                "TARGET_COLUMN": keys[1],
                "EQUATION": keys[2],
                "VALIDATION_METHOD": (
                    "Same rows as GroupKFold by country"
                ),
                "MODEL": "zero_baseline",
                "COUNTRY_COUNT": (
                    group.loc[
                        valid_mask,
                        "COUNTRY_ID",
                    ].nunique()
                ),
                **metrics,
            }
        )

    output = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "SAMPLE",
                "TARGET_COLUMN",
            ]
        )
        .reset_index(drop=True)
    )

    output_file = (
        OUTPUT_DIR
        / "zero_baseline_country_validation.csv"
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== BASELINE CERO SOBRE "
        "LAS MISMAS FILAS DE LA ODE ====="
    )
    print(output.to_string(index=False))
    print()
    print(f"GUARDADO: {output_file}")

    return output


def calculate_ecuador_baselines():
    data = pd.read_csv(ECUADOR_PREDICTIONS)

    required = {
        "TARGET_COLUMN",
        "EQUATION",
        "COUNTRY_ID",
        "OBSERVED_RATE",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en predicciones de Ecuador: "
            f"{sorted(missing)}"
        )

    rows = []

    for keys, group in data.groupby(
        [
            "TARGET_COLUMN",
            "EQUATION",
        ],
        dropna=False,
    ):
        metrics = calculate_metrics(
            group["OBSERVED_RATE"]
        )

        rows.append(
            {
                "SAMPLE": "Ecuador annual series",
                "TARGET_COLUMN": keys[0],
                "EQUATION": keys[1],
                "VALIDATION_METHOD": (
                    "Leave-One-Out same rows"
                ),
                "MODEL": "zero_baseline",
                "COUNTRY_COUNT": 1,
                **metrics,
            }
        )

    output = (
        pd.DataFrame(rows)
        .sort_values("TARGET_COLUMN")
        .reset_index(drop=True)
    )

    output_file = (
        OUTPUT_DIR
        / "zero_baseline_ecuador.csv"
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("===== BASELINE CERO ECUADOR =====")
    print(output.to_string(index=False))
    print()
    print(f"GUARDADO: {output_file}")

    return output


def main():
    if not COUNTRY_PREDICTIONS.exists():
        raise FileNotFoundError(
            COUNTRY_PREDICTIONS
        )

    if not ECUADOR_PREDICTIONS.exists():
        raise FileNotFoundError(
            ECUADOR_PREDICTIONS
        )

    country_output = calculate_country_baselines()
    ecuador_output = calculate_ecuador_baselines()

    print()
    print("===== RESUMEN =====")
    print(
        "Filas baseline global:",
        len(country_output),
    )
    print(
        "Filas baseline Ecuador:",
        len(ecuador_output),
    )
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()