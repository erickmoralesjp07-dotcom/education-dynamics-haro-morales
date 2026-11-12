from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    / "ml_benchmark"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 20261112

FEATURE_COLUMNS = [
    "PREVIOUS_A",
    "PREVIOUS_C",
    "PREVIOUS_L",
    "PREVIOUS_F",
]

TARGET_COLUMN = "DC_DT"

OUTER_MAX_FOLDS = 5
INNER_MAX_FOLDS = 4


def rmse(observed, predicted):
    return float(
        mean_squared_error(
            observed,
            predicted,
        )
        ** 0.5
    )


def safe_r2(observed, predicted):
    if len(observed) < 2:
        return np.nan

    if np.isclose(
        np.var(observed),
        0.0,
    ):
        return np.nan

    return float(
        r2_score(
            observed,
            predicted,
        )
    )


def determine_folds(
    groups,
    maximum,
):
    group_count = pd.Series(
        groups
    ).nunique()

    if group_count < 2:
        raise ValueError(
            "Se requieren al menos dos paises."
        )

    return int(
        min(
            maximum,
            group_count,
        )
    )


def make_models():
    ridge_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )

    elastic_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                ElasticNet(
                    max_iter=50000,
                    random_state=RANDOM_SEED,
                    selection="cyclic",
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    hist_gradient_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_SEED,
                    early_stopping=False,
                ),
            ),
        ]
    )

    return {
        "ridge": {
            "estimator": ridge_pipeline,
            "parameters": {
                "model__alpha": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                    1000.0,
                    10000.0,
                ],
            },
        },
        "elastic_net": {
            "estimator": elastic_pipeline,
            "parameters": {
                "model__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
                "model__l1_ratio": [
                    0.1,
                    0.5,
                    0.9,
                ],
            },
        },
        "random_forest": {
            "estimator": random_forest_pipeline,
            "parameters": {
                "model__n_estimators": [
                    300,
                ],
                "model__max_depth": [
                    2,
                    4,
                    None,
                ],
                "model__min_samples_leaf": [
                    2,
                    5,
                    10,
                ],
                "model__max_features": [
                    0.5,
                    1.0,
                ],
            },
        },
        "hist_gradient_boosting": {
            "estimator": hist_gradient_pipeline,
            "parameters": {
                "model__learning_rate": [
                    0.03,
                    0.1,
                ],
                "model__max_iter": [
                    200,
                ],
                "model__max_leaf_nodes": [
                    7,
                    15,
                    31,
                ],
                "model__min_samples_leaf": [
                    5,
                    10,
                    20,
                ],
                "model__l2_regularization": [
                    0.0,
                    1.0,
                    10.0,
                ],
            },
        },
    }


def prepare_sample(
    data,
    sample_name,
):
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

    required = (
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "COUNTRY_ID",
            "COUNTRY_NAME_EN",
            "PREVIOUS_YEAR",
            "YEAR",
            "DELTA_T",
        ]
    )

    sample = sample[
        required
    ].copy()

    for column in (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    ):
        sample[column] = pd.to_numeric(
            sample[column],
            errors="coerce",
        )

    sample = sample.dropna(
        subset=[
            TARGET_COLUMN,
            "COUNTRY_ID",
        ]
    ).reset_index(drop=True)

    if sample.empty:
        raise ValueError(
            f"La muestra {sample_name} esta vacia."
        )

    duplicate_count = int(
        sample.duplicated(
            [
                "COUNTRY_ID",
                "PREVIOUS_YEAR",
                "YEAR",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"La muestra {sample_name} "
            f"contiene {duplicate_count} duplicados."
        )

    return sample


def predict_baselines(
    y_train,
    y_test,
):
    zero_prediction = np.zeros(
        len(y_test),
        dtype=float,
    )

    training_mean_prediction = np.full(
        len(y_test),
        float(np.mean(y_train)),
        dtype=float,
    )

    return {
        "zero_baseline": zero_prediction,
        "training_mean_baseline": (
            training_mean_prediction
        ),
    }


def nested_group_predictions(
    sample,
    sample_name,
):
    features = sample[
        FEATURE_COLUMNS
    ].copy()

    target = sample[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    groups = sample[
        "COUNTRY_ID"
    ].astype(str).to_numpy()

    outer_fold_count = determine_folds(
        groups,
        OUTER_MAX_FOLDS,
    )

    outer_cv = GroupKFold(
        n_splits=outer_fold_count,
    )

    model_definitions = make_models()

    prediction_rows = []
    tuning_rows = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        outer_cv.split(
            features,
            target,
            groups,
        ),
        start=1,
    ):
        train_features = features.iloc[
            train_index
        ].copy()

        test_features = features.iloc[
            test_index
        ].copy()

        train_target = target[
            train_index
        ]

        test_target = target[
            test_index
        ]

        train_groups = groups[
            train_index
        ]

        test_groups = groups[
            test_index
        ]

        overlap = set(
            train_groups
        ).intersection(
            set(test_groups)
        )

        if overlap:
            raise ValueError(
                "Se detecto fuga de paises "
                f"en el pliegue {fold_number}: "
                f"{sorted(overlap)}"
            )

        baseline_predictions = (
            predict_baselines(
                train_target,
                test_target,
            )
        )

        for (
            baseline_name,
            predictions,
        ) in baseline_predictions.items():
            for local_position, row_index in enumerate(
                test_index
            ):
                prediction_rows.append(
                    {
                        "SAMPLE": sample_name,
                        "FOLD": fold_number,
                        "MODEL": baseline_name,
                        "COUNTRY_ID": (
                            sample.loc[
                                row_index,
                                "COUNTRY_ID",
                            ]
                        ),
                        "COUNTRY_NAME_EN": (
                            sample.loc[
                                row_index,
                                "COUNTRY_NAME_EN",
                            ]
                        ),
                        "PREVIOUS_YEAR": int(
                            sample.loc[
                                row_index,
                                "PREVIOUS_YEAR",
                            ]
                        ),
                        "YEAR": int(
                            sample.loc[
                                row_index,
                                "YEAR",
                            ]
                        ),
                        "DELTA_T": int(
                            sample.loc[
                                row_index,
                                "DELTA_T",
                            ]
                        ),
                        "OBSERVED_RATE": float(
                            test_target[
                                local_position
                            ]
                        ),
                        "PREDICTED_RATE": float(
                            predictions[
                                local_position
                            ]
                        ),
                    }
                )

        inner_fold_count = determine_folds(
            train_groups,
            INNER_MAX_FOLDS,
        )

        inner_cv = GroupKFold(
            n_splits=inner_fold_count,
        )

        for model_name, definition in (
            model_definitions.items()
        ):
            estimator = clone(
                definition["estimator"]
            )

            search = GridSearchCV(
                estimator=estimator,
                param_grid=(
                    definition[
                        "parameters"
                    ]
                ),
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                search.fit(
                    train_features,
                    train_target,
                    groups=train_groups,
                )

            predictions = search.predict(
                test_features
            )

            tuning_rows.append(
                {
                    "SAMPLE": sample_name,
                    "FOLD": fold_number,
                    "MODEL": model_name,
                    "TRAIN_OBSERVATIONS": int(
                        len(train_index)
                    ),
                    "TEST_OBSERVATIONS": int(
                        len(test_index)
                    ),
                    "TRAIN_COUNTRIES": int(
                        pd.Series(
                            train_groups
                        ).nunique()
                    ),
                    "TEST_COUNTRIES": int(
                        pd.Series(
                            test_groups
                        ).nunique()
                    ),
                    "INNER_FOLDS": (
                        inner_fold_count
                    ),
                    "BEST_INNER_RMSE": float(
                        -search.best_score_
                    ),
                    "BEST_PARAMETERS": str(
                        search.best_params_
                    ),
                }
            )

            for local_position, row_index in enumerate(
                test_index
            ):
                prediction_rows.append(
                    {
                        "SAMPLE": sample_name,
                        "FOLD": fold_number,
                        "MODEL": model_name,
                        "COUNTRY_ID": (
                            sample.loc[
                                row_index,
                                "COUNTRY_ID",
                            ]
                        ),
                        "COUNTRY_NAME_EN": (
                            sample.loc[
                                row_index,
                                "COUNTRY_NAME_EN",
                            ]
                        ),
                        "PREVIOUS_YEAR": int(
                            sample.loc[
                                row_index,
                                "PREVIOUS_YEAR",
                            ]
                        ),
                        "YEAR": int(
                            sample.loc[
                                row_index,
                                "YEAR",
                            ]
                        ),
                        "DELTA_T": int(
                            sample.loc[
                                row_index,
                                "DELTA_T",
                            ]
                        ),
                        "OBSERVED_RATE": float(
                            test_target[
                                local_position
                            ]
                        ),
                        "PREDICTED_RATE": float(
                            predictions[
                                local_position
                            ]
                        ),
                    }
                )

    predictions = pd.DataFrame(
        prediction_rows
    )

    tuning = pd.DataFrame(
        tuning_rows
    )

    predictions["RESIDUAL"] = (
        predictions["OBSERVED_RATE"]
        - predictions["PREDICTED_RATE"]
    )

    predictions[
        "ABSOLUTE_RESIDUAL"
    ] = predictions["RESIDUAL"].abs()

    predictions[
        "SQUARED_RESIDUAL"
    ] = predictions["RESIDUAL"] ** 2

    return (
        predictions,
        tuning,
        outer_fold_count,
    )


def calculate_model_metrics(
    predictions,
):
    rows = []

    for keys, group in predictions.groupby(
        [
            "SAMPLE",
            "MODEL",
        ],
        sort=True,
    ):
        observed = group[
            "OBSERVED_RATE"
        ].to_numpy(dtype=float)

        predicted = group[
            "PREDICTED_RATE"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "SAMPLE": keys[0],
                "MODEL": keys[1],
                "OBSERVATION_COUNT": int(
                    len(group)
                ),
                "COUNTRY_COUNT": int(
                    group[
                        "COUNTRY_ID"
                    ].nunique()
                ),
                "RMSE_CV": rmse(
                    observed,
                    predicted,
                ),
                "MAE_CV": float(
                    mean_absolute_error(
                        observed,
                        predicted,
                    )
                ),
                "R2_CV": safe_r2(
                    observed,
                    predicted,
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    zero = (
        metrics.loc[
            metrics[
                "MODEL"
            ].eq("zero_baseline"),
            [
                "SAMPLE",
                "RMSE_CV",
            ],
        ]
        .rename(
            columns={
                "RMSE_CV": (
                    "ZERO_RMSE_CV"
                )
            }
        )
    )

    metrics = metrics.merge(
        zero,
        on="SAMPLE",
        how="left",
        validate="many_to_one",
    )

    metrics[
        "RMSE_GAIN_PERCENT_VS_ZERO"
    ] = (
        100
        * (
            metrics["ZERO_RMSE_CV"]
            - metrics["RMSE_CV"]
        )
        / metrics["ZERO_RMSE_CV"]
    )

    return metrics.sort_values(
        [
            "SAMPLE",
            "RMSE_CV",
        ]
    ).reset_index(drop=True)


def create_summary(
    predictions,
    metrics,
    tuning,
    fold_counts,
):
    winner_rows = (
        metrics.sort_values(
            [
                "SAMPLE",
                "RMSE_CV",
            ]
        )
        .groupby(
            "SAMPLE",
            as_index=False,
        )
        .first()
    )

    summary = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "TARGET_COLUMN": (
                    TARGET_COLUMN
                ),
                "FEATURE_COLUMNS": (
                    ", ".join(
                        FEATURE_COLUMNS
                    )
                ),
                "RANDOM_SEED": (
                    RANDOM_SEED
                ),
                "SAMPLE_COUNT": int(
                    metrics[
                        "SAMPLE"
                    ].nunique()
                ),
                "MODEL_COUNT": int(
                    metrics[
                        "MODEL"
                    ].nunique()
                ),
                "PREDICTION_ROWS": int(
                    len(predictions)
                ),
                "TUNING_ROWS": int(
                    len(tuning)
                ),
                "CONSECUTIVE_OUTER_FOLDS": int(
                    fold_counts[
                        "consecutive"
                    ]
                ),
                "GAP_MAX_2_OUTER_FOLDS": int(
                    fold_counts[
                        "gap_max_2"
                    ]
                ),
                "CONSECUTIVE_WINNER": str(
                    winner_rows.loc[
                        winner_rows[
                            "SAMPLE"
                        ].eq("consecutive"),
                        "MODEL",
                    ].iloc[0]
                ),
                "GAP_MAX_2_WINNER": str(
                    winner_rows.loc[
                        winner_rows[
                            "SAMPLE"
                        ].eq("gap_max_2"),
                        "MODEL",
                    ].iloc[0]
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    return summary


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_parquet(
        INPUT_FILE
    )

    predictions_all = []
    tuning_all = []
    fold_counts = {}

    for sample_name in [
        "consecutive",
        "gap_max_2",
    ]:
        sample = prepare_sample(
            data,
            sample_name,
        )

        (
            predictions,
            tuning,
            fold_count,
        ) = nested_group_predictions(
            sample,
            sample_name,
        )

        predictions_all.append(
            predictions
        )

        tuning_all.append(
            tuning
        )

        fold_counts[
            sample_name
        ] = fold_count

    predictions = pd.concat(
        predictions_all,
        ignore_index=True,
    )

    tuning = pd.concat(
        tuning_all,
        ignore_index=True,
    )

    metrics = calculate_model_metrics(
        predictions
    )

    summary = create_summary(
        predictions,
        metrics,
        tuning,
        fold_counts,
    )

    predictions_file = (
        OUTPUT_DIR
        / "dc_ml_nested_predictions.csv"
    )

    metrics_file = (
        OUTPUT_DIR
        / "dc_ml_nested_metrics.csv"
    )

    tuning_file = (
        OUTPUT_DIR
        / "dc_ml_nested_tuning.csv"
    )

    summary_file = (
        DIAGNOSTIC_DIR
        / "dc_ml_nested_summary.csv"
    )

    predictions.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig",
    )

    tuning.to_csv(
        tuning_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== BENCHMARK ML ANIDADO DC_DT ====="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "===== HIPERPARAMETROS "
        "SELECCIONADOS ====="
    )

    print(
        tuning.to_string(
            index=False
        )
    )

    print()
    print("===== RESUMEN =====")

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(predictions_file)
    print(metrics_file)
    print(tuning_file)
    print(summary_file)

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()