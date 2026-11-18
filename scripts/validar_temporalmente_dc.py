from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

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

RESULTS_FILE = (
    OUTPUT_DIR
    / "dc_temporal_origin_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "dc_temporal_origin_predictions.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "dc_temporal_origin_validation_diagnostic.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURES = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

TARGET = "DC_DT"

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]

TEMPORAL_ORIGINS = [
    2009,
    2014,
    2019,
]

TEST_WINDOW_YEARS = 5

NOW = datetime.now().astimezone().isoformat()


def rmse(observed, predicted):
    return float(
        np.sqrt(
            np.mean(
                (
                    observed
                    - predicted
                ) ** 2
            )
        )
    )


def mae(observed, predicted):
    return float(
        np.mean(
            np.abs(
                observed
                - predicted
            )
        )
    )


def r2_score_manual(
    observed,
    predicted,
):
    denominator = float(
        np.sum(
            (
                observed
                - np.mean(observed)
            ) ** 2
        )
    )

    if denominator == 0:
        return np.nan

    numerator = float(
        np.sum(
            (
                observed
                - predicted
            ) ** 2
        )
    )

    return float(
        1
        - numerator / denominator
    )


def prepare_matrices(
    train,
    test,
):
    train_x = (
        train[FEATURES]
        .to_numpy(dtype=float)
    )

    test_x = (
        test[FEATURES]
        .to_numpy(dtype=float)
    )

    medians = np.nanmedian(
        train_x,
        axis=0,
    )

    medians = np.where(
        np.isnan(medians),
        0.0,
        medians,
    )

    train_x = np.where(
        np.isnan(train_x),
        medians,
        train_x,
    )

    test_x = np.where(
        np.isnan(test_x),
        medians,
        test_x,
    )

    means = np.mean(
        train_x,
        axis=0,
    )

    scales = np.std(
        train_x,
        axis=0,
        ddof=0,
    )

    scales = np.where(
        scales == 0,
        1.0,
        scales,
    )

    train_scaled = (
        train_x
        - means
    ) / scales

    test_scaled = (
        test_x
        - means
    ) / scales

    return (
        train_scaled,
        test_scaled,
    )


def fit_ridge(
    train_x,
    train_y,
    alpha,
):
    target_mean = float(
        np.mean(train_y)
    )

    centered_y = (
        train_y
        - target_mean
    )

    identity = np.eye(
        train_x.shape[1]
    )

    coefficients = np.linalg.solve(
        train_x.T @ train_x
        + alpha * identity,
        train_x.T @ centered_y,
    )

    return (
        target_mean,
        coefficients,
    )


def predict_ridge(
    test_x,
    target_mean,
    coefficients,
):
    return (
        target_mean
        + test_x @ coefficients
    )


def choose_alpha_by_country(
    train,
):
    countries = sorted(
        train[
            "COUNTRY_ID"
        ].astype(str).unique()
    )

    fold_count = min(
        4,
        len(countries),
    )

    country_folds = [
        countries[
            fold_number::fold_count
        ]
        for fold_number in range(
            fold_count
        )
    ]

    scores = []

    for alpha in ALPHAS:
        fold_errors = []

        for validation_countries in (
            country_folds
        ):
            validation_mask = (
                train[
                    "COUNTRY_ID"
                ].astype(str)
                .isin(
                    validation_countries
                )
            )

            inner_train = train.loc[
                ~validation_mask
            ].copy()

            inner_validation = train.loc[
                validation_mask
            ].copy()

            if (
                inner_train.empty
                or inner_validation.empty
            ):
                continue

            (
                train_x,
                validation_x,
            ) = prepare_matrices(
                inner_train,
                inner_validation,
            )

            train_y = (
                inner_train[
                    TARGET
                ].to_numpy(
                    dtype=float
                )
            )

            validation_y = (
                inner_validation[
                    TARGET
                ].to_numpy(
                    dtype=float
                )
            )

            (
                target_mean,
                coefficients,
            ) = fit_ridge(
                train_x,
                train_y,
                alpha,
            )

            predictions = predict_ridge(
                validation_x,
                target_mean,
                coefficients,
            )

            fold_errors.append(
                rmse(
                    validation_y,
                    predictions,
                )
            )

        if fold_errors:
            scores.append(
                (
                    float(
                        np.mean(
                            fold_errors
                        )
                    ),
                    alpha,
                )
            )

    if not scores:
        raise ValueError(
            "No se pudo seleccionar alpha."
        )

    scores.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return float(
        scores[0][1]
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

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

    result_rows = []
    prediction_rows = []

    for sample_name, sample in (
        samples.items()
    ):
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

        for origin in TEMPORAL_ORIGINS:
            test_end = (
                origin
                + TEST_WINDOW_YEARS
            )

            train = sample.loc[
                sample["YEAR"].le(origin)
            ].copy()

            test = sample.loc[
                sample["YEAR"].gt(origin)
                & sample["YEAR"].le(
                    test_end
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
                    "OMITIDO:",
                    "train=",
                    len(train),
                    "test=",
                    len(test),
                )
                continue

            last_train_year = int(
                train["YEAR"].max()
            )

            first_test_year = int(
                test["YEAR"].min()
            )

            if (
                first_test_year
                <= last_train_year
            ):
                raise ValueError(
                    "Se detectó fuga temporal."
                )

            best_alpha = (
                choose_alpha_by_country(
                    train
                )
            )

            (
                train_x,
                test_x,
            ) = prepare_matrices(
                train,
                test,
            )

            train_y = (
                train[TARGET]
                .to_numpy(dtype=float)
            )

            observed = (
                test[TARGET]
                .to_numpy(dtype=float)
            )

            (
                target_mean,
                coefficients,
            ) = fit_ridge(
                train_x,
                train_y,
                best_alpha,
            )

            ridge_prediction = (
                predict_ridge(
                    test_x,
                    target_mean,
                    coefficients,
                )
            )

            zero_prediction = np.zeros(
                len(test),
                dtype=float,
            )

            ridge_rmse = rmse(
                observed,
                ridge_prediction,
            )

            zero_rmse = rmse(
                observed,
                zero_prediction,
            )

            gain_percent = float(
                100
                * (
                    zero_rmse
                    - ridge_rmse
                )
                / zero_rmse
            )

            result_rows.append(
                {
                    "SAMPLE": sample_name,
                    "TEMPORAL_ORIGIN": (
                        origin
                    ),
                    "TEST_START_YEAR": (
                        first_test_year
                    ),
                    "TEST_END_YEAR": int(
                        test["YEAR"].max()
                    ),
                    "TRAIN_OBSERVATIONS": (
                        len(train)
                    ),
                    "TEST_OBSERVATIONS": (
                        len(test)
                    ),
                    "TRAIN_COUNTRIES": int(
                        train[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "TEST_COUNTRIES": int(
                        test[
                            "COUNTRY_ID"
                        ].nunique()
                    ),
                    "BEST_ALPHA": (
                        best_alpha
                    ),
                    "RIDGE_RMSE": (
                        ridge_rmse
                    ),
                    "ZERO_RMSE": (
                        zero_rmse
                    ),
                    "RMSE_GAIN_PERCENT": (
                        gain_percent
                    ),
                    "RIDGE_MAE": mae(
                        observed,
                        ridge_prediction,
                    ),
                    "ZERO_MAE": mae(
                        observed,
                        zero_prediction,
                    ),
                    "RIDGE_R2": (
                        r2_score_manual(
                            observed,
                            ridge_prediction,
                        )
                    ),
                    "RIDGE_BEATS_ZERO": bool(
                        ridge_rmse
                        < zero_rmse
                    ),
                    "NO_FUTURE_LEAKAGE": (
                        True
                    ),
                }
            )

            for position, (_, row) in (
                enumerate(
                    test.iterrows()
                )
            ):
                prediction_rows.append(
                    {
                        "SAMPLE": (
                            sample_name
                        ),
                        "TEMPORAL_ORIGIN": (
                            origin
                        ),
                        "COUNTRY_ID": row[
                            "COUNTRY_ID"
                        ],
                        "COUNTRY_NAME_EN": (
                            row[
                                "COUNTRY_NAME_EN"
                            ]
                        ),
                        "PREVIOUS_YEAR": int(
                            row[
                                "PREVIOUS_YEAR"
                            ]
                        ),
                        "YEAR": int(
                            row["YEAR"]
                        ),
                        "OBSERVED_DC_DT": float(
                            observed[
                                position
                            ]
                        ),
                        "RIDGE_PREDICTION": float(
                            ridge_prediction[
                                position
                            ]
                        ),
                        "ZERO_PREDICTION": (
                            0.0
                        ),
                    }
                )

    results = pd.DataFrame(
        result_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if results.empty:
        raise ValueError(
            "No se generaron pruebas."
        )

    results.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

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
        len(results) >= 4
        and no_leakage
        and len(predictions) > 0
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "TEMPORAL_TESTS": len(
                    results
                ),
                "POSITIVE_TEMPORAL_TESTS": (
                    positive_tests
                ),
                "NO_FUTURE_LEAKAGE": (
                    no_leakage
                ),
                "PREDICTIONS": len(
                    predictions
                ),
                "IMPLEMENTATION": (
                    "NumPy Ridge; "
                    "training-only scaling; "
                    "country-grouped alpha selection"
                ),
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
        "===== VALIDACIÓN TEMPORAL "
        "DE dC/dt ====="
    )

    print(
        results[
            [
                "SAMPLE",
                "TEMPORAL_ORIGIN",
                "TEST_START_YEAR",
                "TEST_END_YEAR",
                "TRAIN_OBSERVATIONS",
                "TEST_OBSERVATIONS",
                "BEST_ALPHA",
                "RIDGE_RMSE",
                "ZERO_RMSE",
                "RMSE_GAIN_PERCENT",
                "RIDGE_BEATS_ZERO",
                "NO_FUTURE_LEAKAGE",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PRUEBAS TEMPORALES:",
        len(results),
    )

    print(
        "RIDGE SUPERA CAMBIO CERO:",
        positive_tests,
        "de",
        len(results),
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
            "La validación temporal "
            "requiere revisión."
        )


if __name__ == "__main__":
    main()