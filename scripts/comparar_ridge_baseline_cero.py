from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLOBAL_RIDGE_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "aclf_linear_ode_diagnostic.csv"
)

GLOBAL_ZERO_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmark_audit"
    / "zero_baseline_country_validation.csv"
)

ECUADOR_RIDGE_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "ecuador_aclf_model_diagnostic.csv"
)

ECUADOR_ZERO_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmark_audit"
    / "zero_baseline_ecuador.csv"
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


def classify_gain(value):
    if pd.isna(value):
        return "not_available"
    if value <= 0:
        return "no_improvement"
    if value < 1:
        return "negligible"
    if value < 5:
        return "small"
    if value < 10:
        return "moderate"
    return "substantial"


def prepare_global_comparison():
    ridge = pd.read_csv(GLOBAL_RIDGE_FILE)
    zero = pd.read_csv(GLOBAL_ZERO_FILE)

    ridge_long = pd.concat(
        [
            ridge[
                [
                    "TARGET_COLUMN",
                    "EQUATION",
                    "PRIMARY_OBSERVATIONS",
                    "PRIMARY_COUNTRIES",
                    "PRIMARY_RMSE_CV",
                    "PRIMARY_R2_CV",
                    "PRIMARY_ALPHA",
                ]
            ]
            .rename(
                columns={
                    "PRIMARY_OBSERVATIONS": "OBSERVATION_COUNT",
                    "PRIMARY_COUNTRIES": "COUNTRY_COUNT",
                    "PRIMARY_RMSE_CV": "RIDGE_RMSE",
                    "PRIMARY_R2_CV": "RIDGE_R2",
                    "PRIMARY_ALPHA": "RIDGE_ALPHA",
                }
            )
            .assign(SAMPLE="consecutive"),
            ridge[
                [
                    "TARGET_COLUMN",
                    "EQUATION",
                    "ROBUSTNESS_OBSERVATIONS",
                    "ROBUSTNESS_COUNTRIES",
                    "ROBUSTNESS_RMSE_CV",
                    "ROBUSTNESS_R2_CV",
                    "ROBUSTNESS_ALPHA",
                ]
            ]
            .rename(
                columns={
                    "ROBUSTNESS_OBSERVATIONS": "OBSERVATION_COUNT",
                    "ROBUSTNESS_COUNTRIES": "COUNTRY_COUNT",
                    "ROBUSTNESS_RMSE_CV": "RIDGE_RMSE",
                    "ROBUSTNESS_R2_CV": "RIDGE_R2",
                    "ROBUSTNESS_ALPHA": "RIDGE_ALPHA",
                }
            )
            .assign(SAMPLE="gap_max_2"),
        ],
        ignore_index=True,
    )

    zero = zero.rename(
        columns={
            "RMSE_CV": "ZERO_RMSE",
            "MAE_CV": "ZERO_MAE",
            "R2_CV": "ZERO_R2",
        }
    )

    comparison = ridge_long.merge(
        zero[
            [
                "SAMPLE",
                "TARGET_COLUMN",
                "OBSERVATION_COUNT",
                "COUNTRY_COUNT",
                "ZERO_RMSE",
                "ZERO_MAE",
                "ZERO_R2",
            ]
        ],
        on=[
            "SAMPLE",
            "TARGET_COLUMN",
            "OBSERVATION_COUNT",
            "COUNTRY_COUNT",
        ],
        how="left",
        validate="one_to_one",
    )

    comparison["RMSE_GAIN"] = (
        comparison["ZERO_RMSE"]
        - comparison["RIDGE_RMSE"]
    )

    comparison["RMSE_GAIN_PERCENT"] = (
        100
        * comparison["RMSE_GAIN"]
        / comparison["ZERO_RMSE"]
    )

    comparison["IMPROVEMENT_CLASS"] = (
        comparison["RMSE_GAIN_PERCENT"]
        .apply(classify_gain)
    )

    comparison["RIDGE_BEATS_ZERO"] = (
        comparison["RIDGE_RMSE"]
        < comparison["ZERO_RMSE"]
    )

    comparison["ANALYSIS_LEVEL"] = (
        "Global GroupKFold by country"
    )

    return comparison.sort_values(
        [
            "SAMPLE",
            "TARGET_COLUMN",
        ]
    ).reset_index(drop=True)


def prepare_ecuador_comparison():
    ridge = pd.read_csv(ECUADOR_RIDGE_FILE)
    zero = pd.read_csv(ECUADOR_ZERO_FILE)

    best_ridge = (
        ridge.sort_values(
            [
                "TARGET_COLUMN",
                "RMSE_LOO",
            ]
        )
        .groupby(
            "TARGET_COLUMN",
            as_index=False,
        )
        .first()
    )

    zero = zero.rename(
        columns={
            "RMSE_CV": "ZERO_RMSE",
            "MAE_CV": "ZERO_MAE",
            "R2_CV": "ZERO_R2",
        }
    )

    comparison = best_ridge.merge(
        zero[
            [
                "TARGET_COLUMN",
                "OBSERVATION_COUNT",
                "ZERO_RMSE",
                "ZERO_MAE",
                "ZERO_R2",
            ]
        ],
        on="TARGET_COLUMN",
        how="left",
        validate="one_to_one",
    )

    comparison = comparison.rename(
        columns={
            "RMSE_LOO": "RIDGE_RMSE",
            "MAE_LOO": "RIDGE_MAE",
            "R2_LOO": "RIDGE_R2",
            "ALPHA": "RIDGE_ALPHA",
        }
    )

    comparison["RMSE_GAIN"] = (
        comparison["ZERO_RMSE"]
        - comparison["RIDGE_RMSE"]
    )

    comparison["RMSE_GAIN_PERCENT"] = (
        100
        * comparison["RMSE_GAIN"]
        / comparison["ZERO_RMSE"]
    )

    comparison["IMPROVEMENT_CLASS"] = (
        comparison["RMSE_GAIN_PERCENT"]
        .apply(classify_gain)
    )

    comparison["RIDGE_BEATS_ZERO"] = (
        comparison["RIDGE_RMSE"]
        < comparison["ZERO_RMSE"]
    )

    comparison["SAMPLE"] = (
        "Ecuador annual series"
    )

    comparison["COUNTRY_COUNT"] = 1

    comparison["ANALYSIS_LEVEL"] = (
        "Ecuador Leave-One-Out"
    )

    return comparison.sort_values(
        "TARGET_COLUMN"
    ).reset_index(drop=True)


def create_summary(global_result, ecuador_result):
    combined = pd.concat(
        [
            global_result[
                [
                    "ANALYSIS_LEVEL",
                    "SAMPLE",
                    "TARGET_COLUMN",
                    "EQUATION",
                    "OBSERVATION_COUNT",
                    "COUNTRY_COUNT",
                    "RIDGE_ALPHA",
                    "RIDGE_RMSE",
                    "RIDGE_R2",
                    "ZERO_RMSE",
                    "ZERO_R2",
                    "RMSE_GAIN",
                    "RMSE_GAIN_PERCENT",
                    "IMPROVEMENT_CLASS",
                    "RIDGE_BEATS_ZERO",
                ]
            ],
            ecuador_result[
                [
                    "ANALYSIS_LEVEL",
                    "SAMPLE",
                    "TARGET_COLUMN",
                    "EQUATION",
                    "OBSERVATION_COUNT",
                    "COUNTRY_COUNT",
                    "RIDGE_ALPHA",
                    "RIDGE_RMSE",
                    "RIDGE_R2",
                    "ZERO_RMSE",
                    "ZERO_R2",
                    "RMSE_GAIN",
                    "RMSE_GAIN_PERCENT",
                    "IMPROVEMENT_CLASS",
                    "RIDGE_BEATS_ZERO",
                ]
            ],
        ],
        ignore_index=True,
    )

    summary = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "COMPARISON_ROWS": len(combined),
                "GLOBAL_ROWS": len(global_result),
                "ECUADOR_ROWS": len(ecuador_result),
                "RIDGE_WINS": int(
                    combined[
                        "RIDGE_BEATS_ZERO"
                    ].sum()
                ),
                "NO_IMPROVEMENT_ROWS": int(
                    (
                        combined[
                            "IMPROVEMENT_CLASS"
                        ]
                        == "no_improvement"
                    ).sum()
                ),
                "NEGLIGIBLE_ROWS": int(
                    (
                        combined[
                            "IMPROVEMENT_CLASS"
                        ]
                        == "negligible"
                    ).sum()
                ),
                "SMALL_ROWS": int(
                    (
                        combined[
                            "IMPROVEMENT_CLASS"
                        ]
                        == "small"
                    ).sum()
                ),
                "MODERATE_ROWS": int(
                    (
                        combined[
                            "IMPROVEMENT_CLASS"
                        ]
                        == "moderate"
                    ).sum()
                ),
                "SUBSTANTIAL_ROWS": int(
                    (
                        combined[
                            "IMPROVEMENT_CLASS"
                        ]
                        == "substantial"
                    ).sum()
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    return combined, summary


def main():
    required_files = [
        GLOBAL_RIDGE_FILE,
        GLOBAL_ZERO_FILE,
        ECUADOR_RIDGE_FILE,
        ECUADOR_ZERO_FILE,
    ]

    missing = [
        str(file)
        for file in required_files
        if not file.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos: "
            + "; ".join(missing)
        )

    global_result = prepare_global_comparison()
    ecuador_result = prepare_ecuador_comparison()

    combined, summary = create_summary(
        global_result,
        ecuador_result,
    )

    global_file = (
        OUTPUT_DIR
        / "global_ridge_vs_zero.csv"
    )

    ecuador_file = (
        OUTPUT_DIR
        / "ecuador_ridge_vs_zero.csv"
    )

    combined_file = (
        OUTPUT_DIR
        / "ridge_vs_zero_all_comparisons.csv"
    )

    summary_file = (
        DIAGNOSTIC_DIR
        / "ridge_vs_zero_summary.csv"
    )

    global_result.to_csv(
        global_file,
        index=False,
        encoding="utf-8-sig",
    )

    ecuador_result.to_csv(
        ecuador_file,
        index=False,
        encoding="utf-8-sig",
    )

    combined.to_csv(
        combined_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "ANALYSIS_LEVEL",
        "SAMPLE",
        "TARGET_COLUMN",
        "EQUATION",
        "OBSERVATION_COUNT",
        "COUNTRY_COUNT",
        "RIDGE_RMSE",
        "ZERO_RMSE",
        "RMSE_GAIN_PERCENT",
        "IMPROVEMENT_CLASS",
        "RIDGE_BEATS_ZERO",
    ]

    print(
        "===== RIDGE VS BASELINE CERO ====="
    )

    print(
        combined[display_columns]
        .to_string(index=False)
    )

    print()
    print("===== RESUMEN =====")
    print(summary.to_string(index=False))

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()