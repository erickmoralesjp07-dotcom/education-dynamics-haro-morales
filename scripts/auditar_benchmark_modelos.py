from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "article"
    / "master_model_comparison.csv"
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


REQUIRED_COLUMNS = [
    "ANALYSIS_LEVEL",
    "SAMPLE",
    "TARGET_COLUMN",
    "EQUATION",
    "MODEL",
    "VALIDATION_METHOD",
    "OBSERVATION_COUNT",
    "COUNTRY_COUNT",
    "RMSE_CV",
    "MAE_CV",
    "R2_CV",
]


def classify_improvement(value):
    if pd.isna(value):
        return "not_available"
    if value <= 0:
        return "no_improvement"
    if value < 1.0:
        return "negligible"
    if value < 5.0:
        return "small"
    if value < 10.0:
        return "moderate"
    return "substantial"


def validate_input(data):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if data.empty:
        raise ValueError("The model comparison table is empty.")


def add_zero_baseline_comparison(data):
    comparison = data.copy()

    baseline = (
        comparison.loc[
            comparison["MODEL"].eq("zero_baseline"),
            [
                "SAMPLE",
                "TARGET_COLUMN",
                "VALIDATION_METHOD",
                "RMSE_CV",
                "MAE_CV",
                "R2_CV",
            ],
        ]
        .rename(
            columns={
                "RMSE_CV": "ZERO_RMSE_CV",
                "MAE_CV": "ZERO_MAE_CV",
                "R2_CV": "ZERO_R2_CV",
            }
        )
        .drop_duplicates(
            subset=[
                "SAMPLE",
                "TARGET_COLUMN",
                "VALIDATION_METHOD",
            ]
        )
    )

    comparison = comparison.merge(
        baseline,
        on=[
            "SAMPLE",
            "TARGET_COLUMN",
            "VALIDATION_METHOD",
        ],
        how="left",
        validate="many_to_one",
    )

    comparison["RMSE_GAIN_VS_ZERO"] = (
        comparison["ZERO_RMSE_CV"]
        - comparison["RMSE_CV"]
    )

    comparison["RMSE_GAIN_PERCENT_VS_ZERO"] = (
        100
        * comparison["RMSE_GAIN_VS_ZERO"]
        / comparison["ZERO_RMSE_CV"]
    )

    comparison["R2_GAIN_VS_ZERO"] = (
        comparison["R2_CV"]
        - comparison["ZERO_R2_CV"]
    )

    comparison["IMPROVEMENT_CLASS"] = (
        comparison["RMSE_GAIN_PERCENT_VS_ZERO"]
        .apply(classify_improvement)
    )

    return comparison


def select_fair_winners(comparison):
    valid = comparison.dropna(
        subset=[
            "RMSE_CV",
            "R2_CV",
        ]
    ).copy()

    grouping = [
        "SAMPLE",
        "TARGET_COLUMN",
        "VALIDATION_METHOD",
    ]

    winners = (
        valid.sort_values(
            by=[
                "SAMPLE",
                "TARGET_COLUMN",
                "VALIDATION_METHOD",
                "RMSE_CV",
                "R2_CV",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
            ],
        )
        .groupby(
            grouping,
            as_index=False,
            dropna=False,
        )
        .first()
    )

    return winners


def create_summary(comparison, winners):
    summary = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now().astimezone().isoformat()
                ),
                "INPUT_ROWS": len(comparison),
                "SAMPLE_COUNT": (
                    comparison["SAMPLE"].nunique()
                ),
                "TARGET_COUNT": (
                    comparison["TARGET_COLUMN"].nunique()
                ),
                "MODEL_COUNT": (
                    comparison["MODEL"].nunique()
                ),
                "VALIDATION_METHOD_COUNT": (
                    comparison[
                        "VALIDATION_METHOD"
                    ].nunique()
                ),
                "FAIR_COMPARISON_GROUPS": len(winners),
                "WINNERS_BEATING_ZERO": int(
                    (
                        winners[
                            "RMSE_GAIN_VS_ZERO"
                        ] > 0
                    ).sum()
                ),
                "WINNERS_WITH_POSITIVE_R2": int(
                    (
                        winners["R2_CV"] > 0
                    ).sum()
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    return summary


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)
    validate_input(data)

    comparison = add_zero_baseline_comparison(data)
    winners = select_fair_winners(comparison)
    summary = create_summary(comparison, winners)

    all_output = (
        OUTPUT_DIR
        / "model_comparison_with_zero_baseline.csv"
    )

    winners_output = (
        OUTPUT_DIR
        / "fair_model_winners.csv"
    )

    summary_output = (
        DIAGNOSTIC_DIR
        / "benchmark_audit_summary.csv"
    )

    comparison.to_csv(
        all_output,
        index=False,
        encoding="utf-8-sig",
    )

    winners.to_csv(
        winners_output,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "SAMPLE",
        "TARGET_COLUMN",
        "VALIDATION_METHOD",
        "MODEL",
        "OBSERVATION_COUNT",
        "COUNTRY_COUNT",
        "RMSE_CV",
        "MAE_CV",
        "R2_CV",
        "ZERO_RMSE_CV",
        "RMSE_GAIN_PERCENT_VS_ZERO",
        "IMPROVEMENT_CLASS",
    ]

    print("===== FAIR MODEL WINNERS =====")
    print(
        winners[display_columns]
        .sort_values(
            by=[
                "SAMPLE",
                "TARGET_COLUMN",
                "VALIDATION_METHOD",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("===== AUDIT SUMMARY =====")
    print(summary.to_string(index=False))

    print()
    print("Outputs:")
    print(all_output)
    print(winners_output)
    print(summary_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise