from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from validar_temporalmente_dc import (
    FEATURES,
    TARGET,
    choose_alpha_by_country,
    fit_ridge,
    prepare_matrices,
    predict_ridge,
    rmse,
)


INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal_validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dc_temporal_window_sensitivity.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "dc_temporal_window_sensitivity_summary.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "dc_temporal_window_sensitivity_diagnostic.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ORIGINS = [2009, 2014, 2019]
WINDOWS = [3, 5, 7]

NOW = datetime.now().astimezone().isoformat()


def main():
    data = pd.read_parquet(
        INPUT_FILE
    )

    samples = {
        "consecutive": data.loc[
            data["DELTA_T"].eq(1)
        ].copy(),
        "gap_max_2": data.loc[
            data["DELTA_T"].le(2)
        ].copy(),
    }

    rows = []

    for sample_name, sample in samples.items():
        sample = (
            sample
            .dropna(
                subset=[
                    TARGET,
                    "YEAR",
                    "COUNTRY_ID",
                ]
            )
            .sort_values(
                [
                    "YEAR",
                    "COUNTRY_ID",
                ]
            )
            .reset_index(drop=True)
        )

        for origin in ORIGINS:
            train = sample.loc[
                sample["YEAR"].le(origin)
            ].copy()

            for window in WINDOWS:
                test = sample.loc[
                    sample["YEAR"].gt(origin)
                    & sample["YEAR"].le(
                        origin + window
                    )
                ].copy()

                if (
                    len(train) < 40
                    or len(test) < 10
                    or train[
                        "COUNTRY_ID"
                    ].nunique() < 5
                ):
                    print(
                        sample_name,
                        origin,
                        window,
                        "OMITIDO",
                    )
                    continue

                if int(
                    test["YEAR"].min()
                ) <= int(
                    train["YEAR"].max()
                ):
                    raise ValueError(
                        "Fuga temporal detectada."
                    )

                alpha = (
                    choose_alpha_by_country(
                        train
                    )
                )

                train_x, test_x = (
                    prepare_matrices(
                        train,
                        test,
                    )
                )

                train_y = (
                    train[TARGET]
                    .to_numpy(dtype=float)
                )

                observed = (
                    test[TARGET]
                    .to_numpy(dtype=float)
                )

                mean_y, coefficients = (
                    fit_ridge(
                        train_x,
                        train_y,
                        alpha,
                    )
                )

                ridge_prediction = (
                    predict_ridge(
                        test_x,
                        mean_y,
                        coefficients,
                    )
                )

                zero_prediction = np.zeros(
                    len(test)
                )

                ridge_error = rmse(
                    observed,
                    ridge_prediction,
                )

                zero_error = rmse(
                    observed,
                    zero_prediction,
                )

                gain = float(
                    100
                    * (
                        zero_error
                        - ridge_error
                    )
                    / zero_error
                )

                rows.append(
                    {
                        "SAMPLE": sample_name,
                        "TEMPORAL_ORIGIN": origin,
                        "WINDOW_YEARS": window,
                        "TEST_START_YEAR": int(
                            test["YEAR"].min()
                        ),
                        "TEST_END_YEAR": int(
                            test["YEAR"].max()
                        ),
                        "TRAIN_OBSERVATIONS": len(
                            train
                        ),
                        "TEST_OBSERVATIONS": len(
                            test
                        ),
                        "BEST_ALPHA": alpha,
                        "RIDGE_RMSE": ridge_error,
                        "ZERO_RMSE": zero_error,
                        "RMSE_GAIN_PERCENT": gain,
                        "RIDGE_BEATS_ZERO": bool(
                            gain > 0
                        ),
                        "NO_FUTURE_LEAKAGE": True,
                    }
                )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        results.groupby(
            [
                "SAMPLE",
                "WINDOW_YEARS",
            ],
            as_index=False,
        )
        .agg(
            TESTS=(
                "RIDGE_BEATS_ZERO",
                "size",
            ),
            POSITIVE_TESTS=(
                "RIDGE_BEATS_ZERO",
                "sum",
            ),
            MEAN_GAIN_PERCENT=(
                "RMSE_GAIN_PERCENT",
                "mean",
            ),
            MIN_GAIN_PERCENT=(
                "RMSE_GAIN_PERCENT",
                "min",
            ),
            MAX_GAIN_PERCENT=(
                "RMSE_GAIN_PERCENT",
                "max",
            ),
        )
    )

    summary[
        "POSITIVE_PERCENT"
    ] = (
        100
        * summary[
            "POSITIVE_TESTS"
        ]
        / summary["TESTS"]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    total_tests = len(results)

    positive_tests = int(
        results[
            "RIDGE_BEATS_ZERO"
        ].sum()
    )

    no_leakage = bool(
        results[
            "NO_FUTURE_LEAKAGE"
        ].all()
    )

    valid = (
        total_tests >= 12
        and no_leakage
        and len(summary) == 6
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "WINDOWS_TESTED": 3,
                "TEMPORAL_TESTS": total_tests,
                "POSITIVE_TESTS": positive_tests,
                "POSITIVE_PERCENT": (
                    100
                    * positive_tests
                    / total_tests
                ),
                "NO_FUTURE_LEAKAGE": no_leakage,
                "VALIDATION_STATUS": (
                    "OK"
                    if valid
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== SENSIBILIDAD DE "
        "VENTANAS TEMPORALES ====="
    )

    print(
        results[
            [
                "SAMPLE",
                "TEMPORAL_ORIGIN",
                "WINDOW_YEARS",
                "TEST_OBSERVATIONS",
                "BEST_ALPHA",
                "RMSE_GAIN_PERCENT",
                "RIDGE_BEATS_ZERO",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESUMEN ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "PRUEBAS:",
        total_tests,
    )

    print(
        "POSITIVAS:",
        positive_tests,
        "de",
        total_tests,
    )

    print(
        "FUGA TEMPORAL:",
        "NO"
        if no_leakage
        else "SÍ",
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    if not valid:
        raise ValueError(
            "La sensibilidad temporal "
            "requiere revisión."
        )


if __name__ == "__main__":
    main()