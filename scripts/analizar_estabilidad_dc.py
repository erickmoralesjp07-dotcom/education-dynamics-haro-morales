from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "model"
    / "aclf_linear_ode_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmark_audit"
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


def rmse(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(values ** 2)
        )
    )


def calculate_metrics(data):
    observed = data[
        "OBSERVED_RATE"
    ].to_numpy(dtype=float)

    predicted = data[
        "PREDICTED_RATE_CV"
    ].to_numpy(dtype=float)

    ridge_errors = observed - predicted
    zero_errors = observed

    ridge_rmse = rmse(ridge_errors)
    zero_rmse = rmse(zero_errors)

    gain_percent = (
        100
        * (zero_rmse - ridge_rmse)
        / zero_rmse
    )

    return {
        "OBSERVATION_COUNT": int(len(data)),
        "COUNTRY_COUNT": int(
            data["COUNTRY_ID"].nunique()
        ),
        "RIDGE_RMSE": ridge_rmse,
        "ZERO_RMSE": zero_rmse,
        "RMSE_GAIN_PERCENT": gain_percent,
        "RIDGE_BEATS_ZERO": bool(
            ridge_rmse < zero_rmse
        ),
    }


def country_contributions(data):
    working = data.copy()

    working["RIDGE_SE"] = (
        working["OBSERVED_RATE"]
        - working["PREDICTED_RATE_CV"]
    ) ** 2

    working["ZERO_SE"] = (
        working["OBSERVED_RATE"] ** 2
    )

    working["GAIN_SSE"] = (
        working["ZERO_SE"]
        - working["RIDGE_SE"]
    )

    output = (
        working.groupby(
            [
                "SAMPLE",
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
            ],
            as_index=False,
        )
        .agg(
            OBSERVATION_COUNT=(
                "GAIN_SSE",
                "size",
            ),
            RIDGE_SSE=(
                "RIDGE_SE",
                "sum",
            ),
            ZERO_SSE=(
                "ZERO_SE",
                "sum",
            ),
            GAIN_SSE=(
                "GAIN_SSE",
                "sum",
            ),
        )
    )

    output["GAIN_PERCENT_SSE"] = np.where(
        output["ZERO_SSE"].ne(0),
        100
        * output["GAIN_SSE"]
        / output["ZERO_SSE"],
        np.nan,
    )

    return output


def leave_one_country_out(data):
    rows = []

    for sample_name, sample in data.groupby(
        "SAMPLE",
        sort=True,
    ):
        base_metrics = calculate_metrics(
            sample
        )

        countries = sorted(
            sample["COUNTRY_ID"]
            .dropna()
            .unique()
        )

        for country_id in countries:
            removed_rows = sample.loc[
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

            metrics = calculate_metrics(
                reduced
            )

            if removed_rows.empty:
                country_name = str(country_id)
            else:
                country_name = str(
                    removed_rows[
                        "COUNTRY_NAME_EN"
                    ].iloc[0]
                )

            row = {
                "SAMPLE": sample_name,
                "REMOVED_COUNTRY_ID": (
                    country_id
                ),
                "REMOVED_COUNTRY_NAME": (
                    country_name
                ),
                "FULL_GAIN_PERCENT": (
                    base_metrics[
                        "RMSE_GAIN_PERCENT"
                    ]
                ),
                "OBSERVATION_COUNT": (
                    metrics[
                        "OBSERVATION_COUNT"
                    ]
                ),
                "COUNTRY_COUNT": (
                    metrics[
                        "COUNTRY_COUNT"
                    ]
                ),
                "RIDGE_RMSE": (
                    metrics[
                        "RIDGE_RMSE"
                    ]
                ),
                "ZERO_RMSE": (
                    metrics[
                        "ZERO_RMSE"
                    ]
                ),
                "RMSE_GAIN_PERCENT": (
                    metrics[
                        "RMSE_GAIN_PERCENT"
                    ]
                ),
                "RIDGE_BEATS_ZERO": (
                    metrics[
                        "RIDGE_BEATS_ZERO"
                    ]
                ),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def cluster_bootstrap(data):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for sample_name, sample in data.groupby(
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

            sampled_frames = [
                country_frames[country]
                for country
                in selected_countries
            ]

            bootstrap_sample = pd.concat(
                sampled_frames,
                ignore_index=True,
            )

            metrics = calculate_metrics(
                bootstrap_sample
            )

            row = {
                "SAMPLE": sample_name,
                "ITERATION": (
                    iteration + 1
                ),
                "OBSERVATION_COUNT": (
                    metrics[
                        "OBSERVATION_COUNT"
                    ]
                ),
                "COUNTRY_DRAWS": int(
                    len(selected_countries)
                ),
                "RIDGE_RMSE": (
                    metrics[
                        "RIDGE_RMSE"
                    ]
                ),
                "ZERO_RMSE": (
                    metrics[
                        "ZERO_RMSE"
                    ]
                ),
                "RMSE_GAIN_PERCENT": (
                    metrics[
                        "RMSE_GAIN_PERCENT"
                    ]
                ),
                "RIDGE_BEATS_ZERO": (
                    metrics[
                        "RIDGE_BEATS_ZERO"
                    ]
                ),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_summary(bootstrap):
    rows = []

    for sample_name, sample in bootstrap.groupby(
        "SAMPLE",
        sort=True,
    ):
        gains = sample[
            "RMSE_GAIN_PERCENT"
        ].to_numpy(dtype=float)

        row = {
            "SAMPLE": sample_name,
            "BOOTSTRAP_REPETITIONS": int(
                len(sample)
            ),
            "GAIN_MEAN": float(
                np.mean(gains)
            ),
            "GAIN_MEDIAN": float(
                np.median(gains)
            ),
            "GAIN_CI_2_5": float(
                np.quantile(
                    gains,
                    0.025,
                )
            ),
            "GAIN_CI_97_5": float(
                np.quantile(
                    gains,
                    0.975,
                )
            ),
            "PROBABILITY_RIDGE_BEATS_ZERO": (
                float(
                    np.mean(gains > 0)
                )
            ),
            "PROBABILITY_GAIN_AT_LEAST_1": (
                float(
                    np.mean(gains >= 1)
                )
            ),
            "PROBABILITY_GAIN_AT_LEAST_5": (
                float(
                    np.mean(gains >= 5)
                )
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def decade_analysis(data):
    working = data.copy()

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
        metrics = calculate_metrics(
            group
        )

        row = {
            "SAMPLE": keys[0],
            "DECADE": int(keys[1]),
            "OBSERVATION_COUNT": (
                metrics[
                    "OBSERVATION_COUNT"
                ]
            ),
            "COUNTRY_COUNT": (
                metrics[
                    "COUNTRY_COUNT"
                ]
            ),
            "RIDGE_RMSE": (
                metrics[
                    "RIDGE_RMSE"
                ]
            ),
            "ZERO_RMSE": (
                metrics[
                    "ZERO_RMSE"
                ]
            ),
            "RMSE_GAIN_PERCENT": (
                metrics[
                    "RMSE_GAIN_PERCENT"
                ]
            ),
            "RIDGE_BEATS_ZERO": (
                metrics[
                    "RIDGE_BEATS_ZERO"
                ]
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def create_summary(
    data,
    loo,
    bootstrap_results,
    decade_results,
):
    rows = []

    for sample_name, sample in data.groupby(
        "SAMPLE",
        sort=True,
    ):
        overall = calculate_metrics(
            sample
        )

        loo_sample = loo.loc[
            loo["SAMPLE"].eq(
                sample_name
            )
        ].copy()

        bootstrap_sample = (
            bootstrap_results.loc[
                bootstrap_results[
                    "SAMPLE"
                ].eq(sample_name)
            ]
        )

        decade_sample = (
            decade_results.loc[
                decade_results[
                    "SAMPLE"
                ].eq(sample_name)
            ]
        )

        if bootstrap_sample.empty:
            raise ValueError(
                "No se encontro resumen bootstrap "
                f"para {sample_name}."
            )

        bootstrap_row = (
            bootstrap_sample.iloc[0]
        )

        row = {
            "PROCESSING_DATETIME": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            "SAMPLE": sample_name,
            "OBSERVATION_COUNT": (
                overall[
                    "OBSERVATION_COUNT"
                ]
            ),
            "COUNTRY_COUNT": (
                overall[
                    "COUNTRY_COUNT"
                ]
            ),
            "OVERALL_GAIN_PERCENT": (
                overall[
                    "RMSE_GAIN_PERCENT"
                ]
            ),
            "LOO_MIN_GAIN_PERCENT": float(
                loo_sample[
                    "RMSE_GAIN_PERCENT"
                ].min()
            ),
            "LOO_MAX_GAIN_PERCENT": float(
                loo_sample[
                    "RMSE_GAIN_PERCENT"
                ].max()
            ),
            "LOO_POSITIVE_PERCENT": float(
                100
                * loo_sample[
                    "RIDGE_BEATS_ZERO"
                ].mean()
            ),
            "BOOTSTRAP_CI_2_5": float(
                bootstrap_row[
                    "GAIN_CI_2_5"
                ]
            ),
            "BOOTSTRAP_CI_97_5": float(
                bootstrap_row[
                    "GAIN_CI_97_5"
                ]
            ),
            "BOOTSTRAP_PROBABILITY_WIN": float(
                bootstrap_row[
                    "PROBABILITY_RIDGE_BEATS_ZERO"
                ]
            ),
            "DECADES_EVALUATED": int(
                decade_sample[
                    "DECADE"
                ].nunique()
            ),
            "DECADES_WITH_POSITIVE_GAIN": int(
                (
                    decade_sample[
                        "RMSE_GAIN_PERCENT"
                    ] > 0
                ).sum()
            ),
            "VALIDATION_STATUS": "OK",
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "YEAR",
        "SAMPLE",
        "TARGET_COLUMN",
        "OBSERVED_RATE",
        "PREDICTED_RATE_CV",
    }

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

    data = data.loc[
        data["TARGET_COLUMN"].eq(
            "DC_DT"
        )
    ].copy()

    if data.empty:
        raise ValueError(
            "No existen predicciones para DC_DT."
        )

    numeric_columns = [
        "OBSERVED_RATE",
        "PREDICTED_RATE_CV",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Existen valores faltantes "
            "en observaciones o predicciones."
        )

    duplicate_count = int(
        data.duplicated(
            [
                "SAMPLE",
                "COUNTRY_ID",
                "PREVIOUS_YEAR",
                "YEAR",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Se detectaron "
            f"{duplicate_count} duplicados."
        )

    contributions = country_contributions(
        data
    )

    loo = leave_one_country_out(
        data
    )

    bootstrap = cluster_bootstrap(
        data
    )

    bootstrap_results = (
        bootstrap_summary(
            bootstrap
        )
    )

    decade_results = decade_analysis(
        data
    )

    summary = create_summary(
        data,
        loo,
        bootstrap_results,
        decade_results,
    )

    contributions_file = (
        OUTPUT_DIR
        / "dc_country_error_contributions.csv"
    )

    loo_file = (
        OUTPUT_DIR
        / "dc_leave_one_country_out.csv"
    )

    bootstrap_file = (
        OUTPUT_DIR
        / "dc_cluster_bootstrap_iterations.csv"
    )

    bootstrap_summary_file = (
        OUTPUT_DIR
        / "dc_cluster_bootstrap_summary.csv"
    )

    decade_file = (
        OUTPUT_DIR
        / "dc_decade_stability.csv"
    )

    summary_file = (
        DIAGNOSTIC_DIR
        / "dc_stability_summary.csv"
    )

    contributions.to_csv(
        contributions_file,
        index=False,
        encoding="utf-8-sig",
    )

    loo.to_csv(
        loo_file,
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

    decade_results.to_csv(
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
        "===== ESTABILIDAD DE DC_DT ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== BOOTSTRAP AGRUPADO "
        "POR PAIS ====="
    )

    print(
        bootstrap_results.to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESULTADOS POR DECADA ====="
    )

    print(
        decade_results.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PAISES CUYA EXCLUSION "
        "MAS REDUCE LA VENTAJA ====="
    )

    influential = (
        loo.sort_values(
            [
                "SAMPLE",
                "RMSE_GAIN_PERCENT",
            ]
        )
        .groupby(
            "SAMPLE",
            as_index=False,
            group_keys=False,
        )
        .head(10)
    )

    influential_columns = [
        "SAMPLE",
        "REMOVED_COUNTRY_ID",
        "REMOVED_COUNTRY_NAME",
        "RMSE_GAIN_PERCENT",
        "RIDGE_BEATS_ZERO",
    ]

    print(
        influential[
            influential_columns
        ].to_string(index=False)
    )

    print()
    print("Archivos creados:")

    for file in [
        contributions_file,
        loo_file,
        bootstrap_file,
        bootstrap_summary_file,
        decade_file,
        summary_file,
    ]:
        print(file)

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()