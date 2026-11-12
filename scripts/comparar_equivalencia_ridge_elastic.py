from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ml_benchmark"
    / "dc_ml_nested_predictions.csv"
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
BOOTSTRAP_REPETITIONS = 5000

PRACTICAL_EQUIVALENCE_PERCENT = 1.0

KEY_COLUMNS = [
    "SAMPLE",
    "COUNTRY_ID",
    "COUNTRY_NAME_EN",
    "PREVIOUS_YEAR",
    "YEAR",
    "DELTA_T",
    "FOLD",
]


def rmse(observed, predicted):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (observed - predicted) ** 2
            )
        )
    )


def prepare_paired_data(data):
    selected = data.loc[
        data["MODEL"].isin(
            [
                "ridge",
                "elastic_net",
            ]
        )
    ].copy()

    duplicate_count = int(
        selected.duplicated(
            KEY_COLUMNS + ["MODEL"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Se detectaron "
            f"{duplicate_count} predicciones "
            "duplicadas."
        )

    ridge = (
        selected.loc[
            selected["MODEL"].eq("ridge"),
            KEY_COLUMNS
            + [
                "OBSERVED_RATE",
                "PREDICTED_RATE",
            ],
        ]
        .rename(
            columns={
                "PREDICTED_RATE": (
                    "RIDGE_PREDICTION"
                ),
            }
        )
    )

    elastic = (
        selected.loc[
            selected["MODEL"].eq(
                "elastic_net"
            ),
            KEY_COLUMNS
            + [
                "OBSERVED_RATE",
                "PREDICTED_RATE",
            ],
        ]
        .rename(
            columns={
                "OBSERVED_RATE": (
                    "ELASTIC_OBSERVED_RATE"
                ),
                "PREDICTED_RATE": (
                    "ELASTIC_PREDICTION"
                ),
            }
        )
    )

    paired = ridge.merge(
        elastic,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(ridge):
        raise ValueError(
            "No todas las predicciones Ridge "
            "tienen correspondencia Elastic Net."
        )

    observed_difference = np.abs(
        paired["OBSERVED_RATE"]
        - paired["ELASTIC_OBSERVED_RATE"]
    )

    if not np.allclose(
        observed_difference,
        0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError(
            "Los valores observados no coinciden "
            "entre Ridge y Elastic Net."
        )

    paired = paired.drop(
        columns=[
            "ELASTIC_OBSERVED_RATE",
        ]
    )

    paired["RIDGE_RESIDUAL"] = (
        paired["OBSERVED_RATE"]
        - paired["RIDGE_PREDICTION"]
    )

    paired["ELASTIC_RESIDUAL"] = (
        paired["OBSERVED_RATE"]
        - paired["ELASTIC_PREDICTION"]
    )

    paired["RIDGE_SE"] = (
        paired["RIDGE_RESIDUAL"] ** 2
    )

    paired["ELASTIC_SE"] = (
        paired["ELASTIC_RESIDUAL"] ** 2
    )

    paired["SE_DIFFERENCE_RIDGE_MINUS_ELASTIC"] = (
        paired["RIDGE_SE"]
        - paired["ELASTIC_SE"]
    )

    return paired


def calculate_comparison(data):
    observed = data[
        "OBSERVED_RATE"
    ].to_numpy(dtype=float)

    ridge_prediction = data[
        "RIDGE_PREDICTION"
    ].to_numpy(dtype=float)

    elastic_prediction = data[
        "ELASTIC_PREDICTION"
    ].to_numpy(dtype=float)

    ridge_rmse = rmse(
        observed,
        ridge_prediction,
    )

    elastic_rmse = rmse(
        observed,
        elastic_prediction,
    )

    mean_rmse = (
        ridge_rmse + elastic_rmse
    ) / 2

    difference = (
        ridge_rmse - elastic_rmse
    )

    relative_difference = (
        100
        * difference
        / mean_rmse
    )

    return {
        "OBSERVATION_COUNT": int(
            len(data)
        ),
        "COUNTRY_COUNT": int(
            data["COUNTRY_ID"].nunique()
        ),
        "RIDGE_RMSE": ridge_rmse,
        "ELASTIC_RMSE": elastic_rmse,
        "RIDGE_MINUS_ELASTIC_RMSE": (
            difference
        ),
        "RELATIVE_DIFFERENCE_PERCENT": (
            relative_difference
        ),
        "ABSOLUTE_RELATIVE_DIFFERENCE_PERCENT": (
            abs(relative_difference)
        ),
        "PRACTICALLY_EQUIVALENT": bool(
            abs(relative_difference)
            < PRACTICAL_EQUIVALENCE_PERCENT
        ),
    }


def cluster_bootstrap(paired):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for sample_name, sample in paired.groupby(
        "SAMPLE",
        sort=True,
    ):
        countries = np.array(
            sorted(
                sample["COUNTRY_ID"]
                .dropna()
                .unique()
            )
        )

        country_frames = {
            country: sample.loc[
                sample["COUNTRY_ID"].eq(
                    country
                )
            ].copy()
            for country in countries
        }

        for iteration in range(
            BOOTSTRAP_REPETITIONS
        ):
            selected_countries = rng.choice(
                countries,
                size=len(countries),
                replace=True,
            )

            bootstrap_sample = pd.concat(
                [
                    country_frames[country]
                    for country
                    in selected_countries
                ],
                ignore_index=True,
            )

            metrics = calculate_comparison(
                bootstrap_sample
            )

            rows.append(
                {
                    "SAMPLE": sample_name,
                    "ITERATION": (
                        iteration + 1
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_bootstrap(bootstrap):
    rows = []

    for sample_name, sample in bootstrap.groupby(
        "SAMPLE",
        sort=True,
    ):
        difference = sample[
            "RELATIVE_DIFFERENCE_PERCENT"
        ].to_numpy(dtype=float)

        absolute_difference = np.abs(
            difference
        )

        rows.append(
            {
                "SAMPLE": sample_name,
                "BOOTSTRAP_REPETITIONS": int(
                    len(sample)
                ),
                "DIFFERENCE_MEAN_PERCENT": float(
                    np.mean(difference)
                ),
                "DIFFERENCE_MEDIAN_PERCENT": float(
                    np.median(difference)
                ),
                "DIFFERENCE_CI_2_5": float(
                    np.quantile(
                        difference,
                        0.025,
                    )
                ),
                "DIFFERENCE_CI_97_5": float(
                    np.quantile(
                        difference,
                        0.975,
                    )
                ),
                "PROBABILITY_RIDGE_BETTER": float(
                    np.mean(difference < 0)
                ),
                "PROBABILITY_ELASTIC_BETTER": float(
                    np.mean(difference > 0)
                ),
                "PROBABILITY_PRACTICAL_EQUIVALENCE": (
                    float(
                        np.mean(
                            absolute_difference
                            < PRACTICAL_EQUIVALENCE_PERCENT
                        )
                    )
                ),
                "EQUIVALENCE_THRESHOLD_PERCENT": (
                    PRACTICAL_EQUIVALENCE_PERCENT
                ),
            }
        )

    return pd.DataFrame(rows)


def leave_one_country_out(paired):
    rows = []

    for sample_name, sample in paired.groupby(
        "SAMPLE",
        sort=True,
    ):
        countries = sorted(
            sample["COUNTRY_ID"]
            .dropna()
            .unique()
        )

        full_metrics = calculate_comparison(
            sample
        )

        for country_id in countries:
            country_rows = sample.loc[
                sample["COUNTRY_ID"].eq(
                    country_id
                )
            ]

            reduced = sample.loc[
                ~sample["COUNTRY_ID"].eq(
                    country_id
                )
            ].copy()

            if reduced.empty:
                continue

            metrics = calculate_comparison(
                reduced
            )

            rows.append(
                {
                    "SAMPLE": sample_name,
                    "REMOVED_COUNTRY_ID": (
                        country_id
                    ),
                    "REMOVED_COUNTRY_NAME": str(
                        country_rows[
                            "COUNTRY_NAME_EN"
                        ].iloc[0]
                    ),
                    "FULL_RELATIVE_DIFFERENCE_PERCENT": (
                        full_metrics[
                            "RELATIVE_DIFFERENCE_PERCENT"
                        ]
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def decade_analysis(paired):
    working = paired.copy()

    working["DECADE"] = (
        working["YEAR"] // 10
    ) * 10

    rows = []

    for keys, group in working.groupby(
        [
            "SAMPLE",
            "DECADE",
        ],
        sort=True,
    ):
        metrics = calculate_comparison(
            group
        )

        rows.append(
            {
                "SAMPLE": keys[0],
                "DECADE": int(keys[1]),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def build_overall_summary(
    paired,
    bootstrap_summary,
    loo,
    decade,
):
    rows = []

    for sample_name, sample in paired.groupby(
        "SAMPLE",
        sort=True,
    ):
        overall = calculate_comparison(
            sample
        )

        bootstrap_row = (
            bootstrap_summary.loc[
                bootstrap_summary[
                    "SAMPLE"
                ].eq(sample_name)
            ]
            .iloc[0]
        )

        loo_sample = loo.loc[
            loo["SAMPLE"].eq(
                sample_name
            )
        ]

        decade_sample = decade.loc[
            decade["SAMPLE"].eq(
                sample_name
            )
        ]

        rows.append(
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "SAMPLE": sample_name,
                **overall,
                "BOOTSTRAP_CI_2_5": float(
                    bootstrap_row[
                        "DIFFERENCE_CI_2_5"
                    ]
                ),
                "BOOTSTRAP_CI_97_5": float(
                    bootstrap_row[
                        "DIFFERENCE_CI_97_5"
                    ]
                ),
                "PROBABILITY_RIDGE_BETTER": float(
                    bootstrap_row[
                        "PROBABILITY_RIDGE_BETTER"
                    ]
                ),
                "PROBABILITY_ELASTIC_BETTER": float(
                    bootstrap_row[
                        "PROBABILITY_ELASTIC_BETTER"
                    ]
                ),
                "PROBABILITY_EQUIVALENT": float(
                    bootstrap_row[
                        "PROBABILITY_PRACTICAL_EQUIVALENCE"
                    ]
                ),
                "LOO_MIN_DIFFERENCE_PERCENT": float(
                    loo_sample[
                        "RELATIVE_DIFFERENCE_PERCENT"
                    ].min()
                ),
                "LOO_MAX_DIFFERENCE_PERCENT": float(
                    loo_sample[
                        "RELATIVE_DIFFERENCE_PERCENT"
                    ].max()
                ),
                "LOO_EQUIVALENT_PERCENT": float(
                    100
                    * loo_sample[
                        "PRACTICALLY_EQUIVALENT"
                    ].mean()
                ),
                "DECADES_EVALUATED": int(
                    decade_sample[
                        "DECADE"
                    ].nunique()
                ),
                "VALIDATION_STATUS": "OK",
            }
        )

    return pd.DataFrame(rows)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    required_columns = set(
        KEY_COLUMNS
        + [
            "MODEL",
            "OBSERVED_RATE",
            "PREDICTED_RATE",
        ]
    )

    missing_columns = (
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas: "
            f"{sorted(missing_columns)}"
        )

    paired = prepare_paired_data(
        data
    )

    bootstrap = cluster_bootstrap(
        paired
    )

    bootstrap_results = (
        summarize_bootstrap(
            bootstrap
        )
    )

    loo = leave_one_country_out(
        paired
    )

    decade = decade_analysis(
        paired
    )

    summary = build_overall_summary(
        paired,
        bootstrap_results,
        loo,
        decade,
    )

    paired_file = (
        OUTPUT_DIR
        / "dc_ridge_elastic_paired_predictions.csv"
    )

    bootstrap_file = (
        OUTPUT_DIR
        / "dc_ridge_elastic_bootstrap_iterations.csv"
    )

    bootstrap_summary_file = (
        OUTPUT_DIR
        / "dc_ridge_elastic_bootstrap_summary.csv"
    )

    loo_file = (
        OUTPUT_DIR
        / "dc_ridge_elastic_leave_one_country_out.csv"
    )

    decade_file = (
        OUTPUT_DIR
        / "dc_ridge_elastic_decade.csv"
    )

    summary_file = (
        DIAGNOSTIC_DIR
        / "dc_ridge_elastic_equivalence_summary.csv"
    )

    paired.to_csv(
        paired_file,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap.to_csv(
        bootstrap_file,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap_results.to_csv(
        bootstrap_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    loo.to_csv(
        loo_file,
        index=False,
        encoding="utf-8-sig",
    )

    decade.to_csv(
        decade_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== EQUIVALENCIA RIDGE VS "
        "ELASTIC NET ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== BOOTSTRAP PAREADO "
        "POR PAIS ====="
    )

    print(
        bootstrap_results.to_string(
            index=False
        )
    )

    print()
    print(
        "===== COMPARACION POR DECADA ====="
    )

    print(
        decade.to_string(
            index=False
        )
    )

    print()
    print(
        "===== RANGO AL RETIRAR "
        "UN PAIS ====="
    )

    loo_summary = (
        loo.groupby(
            "SAMPLE",
            as_index=False,
        )
        .agg(
            MIN_DIFFERENCE_PERCENT=(
                "RELATIVE_DIFFERENCE_PERCENT",
                "min",
            ),
            MAX_DIFFERENCE_PERCENT=(
                "RELATIVE_DIFFERENCE_PERCENT",
                "max",
            ),
            EQUIVALENT_PERCENT=(
                "PRACTICALLY_EQUIVALENT",
                lambda values: (
                    100
                    * values.mean()
                ),
            ),
        )
    )

    print(
        loo_summary.to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")

    for file in [
        paired_file,
        bootstrap_file,
        bootstrap_summary_file,
        loo_file,
        decade_file,
        summary_file,
    ]:
        print(file)

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()