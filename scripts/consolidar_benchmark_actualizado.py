from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTICLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "article"
)

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
)

ARTICLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {
    "panel": (
        PROJECT_ROOT
        / "results"
        / "diagnostics"
        / "uis_model_panel_coverage.csv"
    ),
    "indicators": (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "uis_selected_indicators.csv"
    ),
    "ridge_zero": (
        PROJECT_ROOT
        / "results"
        / "benchmark_audit"
        / "ridge_vs_zero_all_comparisons.csv"
    ),
    "stability": (
        PROJECT_ROOT
        / "results"
        / "diagnostics"
        / "dc_stability_summary.csv"
    ),
    "ml_metrics": (
        PROJECT_ROOT
        / "results"
        / "ml_benchmark"
        / "dc_ml_nested_metrics.csv"
    ),
    "ml_summary": (
        PROJECT_ROOT
        / "results"
        / "diagnostics"
        / "dc_ml_nested_summary.csv"
    ),
    "equivalence": (
        PROJECT_ROOT
        / "results"
        / "diagnostics"
        / "dc_ridge_elastic_equivalence_summary.csv"
    ),
}


def require_files():
    missing = [
        str(path)
        for path in FILES.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos: "
            + "; ".join(missing)
        )


def load_data():
    return {
        name: pd.read_csv(path)
        for name, path in FILES.items()
    }


def format_number(value, decimals=2):
    return f"{float(value):.{decimals}f}"


def build_key_results(data):
    ridge_zero = data["ridge_zero"].copy()

    global_rows = ridge_zero.loc[
        ridge_zero[
            "ANALYSIS_LEVEL"
        ].eq(
            "Global GroupKFold by country"
        )
    ].copy()

    ecuador_rows = ridge_zero.loc[
        ridge_zero[
            "ANALYSIS_LEVEL"
        ].eq(
            "Ecuador Leave-One-Out"
        )
    ].copy()

    rows = []

    for _, row in global_rows.iterrows():
        rows.append(
            {
                "RESULT_GROUP": (
                    "Global Ridge vs zero"
                ),
                "SAMPLE": row["SAMPLE"],
                "TARGET_COLUMN": (
                    row["TARGET_COLUMN"]
                ),
                "EQUATION": row["EQUATION"],
                "MODEL": "Ridge",
                "RMSE": row["RIDGE_RMSE"],
                "REFERENCE_RMSE": (
                    row["ZERO_RMSE"]
                ),
                "GAIN_PERCENT": (
                    row["RMSE_GAIN_PERCENT"]
                ),
                "CLASSIFICATION": (
                    row["IMPROVEMENT_CLASS"]
                ),
            }
        )

    for _, row in ecuador_rows.iterrows():
        rows.append(
            {
                "RESULT_GROUP": (
                    "Ecuador Ridge vs zero"
                ),
                "SAMPLE": row["SAMPLE"],
                "TARGET_COLUMN": (
                    row["TARGET_COLUMN"]
                ),
                "EQUATION": row["EQUATION"],
                "MODEL": "Ridge",
                "RMSE": row["RIDGE_RMSE"],
                "REFERENCE_RMSE": (
                    row["ZERO_RMSE"]
                ),
                "GAIN_PERCENT": (
                    row["RMSE_GAIN_PERCENT"]
                ),
                "CLASSIFICATION": (
                    row["IMPROVEMENT_CLASS"]
                ),
            }
        )

    ml_metrics = data["ml_metrics"].copy()

    for sample_name, sample in ml_metrics.groupby(
        "SAMPLE"
    ):
        winner = sample.sort_values(
            "RMSE_CV"
        ).iloc[0]

        rows.append(
            {
                "RESULT_GROUP": (
                    "Nested ML benchmark"
                ),
                "SAMPLE": sample_name,
                "TARGET_COLUMN": "DC_DT",
                "EQUATION": "dC/dt",
                "MODEL": winner["MODEL"],
                "RMSE": winner["RMSE_CV"],
                "REFERENCE_RMSE": (
                    winner["ZERO_RMSE_CV"]
                ),
                "GAIN_PERCENT": (
                    winner[
                        "RMSE_GAIN_PERCENT_VS_ZERO"
                    ]
                ),
                "CLASSIFICATION": (
                    "best_nested_model"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_markdown(data, key_results):
    panel = data["panel"].iloc[0]
    indicators = data["indicators"]
    stability = data["stability"]
    ml_metrics = data["ml_metrics"]
    equivalence = data["equivalence"]

    generated = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    lines = []

    lines.append(
        "# Updated Scientific Benchmark Summary"
    )
    lines.append("")
    lines.append(
        f"Generated: {generated}"
    )
    lines.append("")

    lines.append("## 1. Current study scope")
    lines.append("")
    lines.append(
        "The project evaluates whether a "
        "structured educational dynamics model "
        "can predict changes in access, "
        "adult educational attainment, youth literacy, and "
        "education finance using UNESCO UIS data."
    )
    lines.append("")

    lines.append("## 2. Data coverage")
    lines.append("")
    lines.append(
        f"- Selected indicators: "
        f"{int(panel['selected_indicator_count'])}."
    )
    lines.append(
        f"- Long-format observations: "
        f"{int(panel['long_panel_rows']):,}."
    )
    lines.append(
        f"- Country-year rows: "
        f"{int(panel['country_year_rows']):,}."
    )
    lines.append(
        f"- Countries or territories: "
        f"{int(panel['country_count'])}."
    )
    lines.append(
        f"- General period: "
        f"{int(panel['minimum_year'])}-"
        f"{int(panel['maximum_year'])}."
    )
    lines.append(
        f"- Complete country-year rows: "
        f"{int(panel['complete_country_year_rows'])}."
    )
    lines.append(
        f"- Rows with at least four indicators: "
        f"{int(panel['country_year_rows_at_least_4']):,}."
    )
    lines.append("")

    lines.append("## 3. Selected indicators")
    lines.append("")
    lines.append(
        "| Variable | Dimension | Indicator | "
        "Direction | Countries | Period |"
    )
    lines.append(
        "|---|---|---|---|---:|---|"
    )

    for _, row in indicators.iterrows():
        period = (
            f"{int(row['MIN_YEAR'])}-"
            f"{int(row['MAX_YEAR'])}"
        )

        lines.append(
            f"| {row['MODEL_VARIABLE']} "
            f"| {row['MODEL_DIMENSION']} "
            f"| {row['INDICATOR_LABEL_EN']} "
            f"| {row['DIRECTION']} "
            f"| {int(row['COUNTRY_COUNT'])} "
            f"| {period} |"
        )

    lines.append("")
    lines.append("## 4. ACLF model")
    lines.append("")
    lines.append(
        "- A: out-of-school rate at primary age."
    )
    lines.append(
        "- C: share of the population aged 25 years or more "
        "that attained at least lower-secondary education."
    )
    lines.append(
        "- L: youth literacy rate, ages 15-24."
    )
    lines.append(
        "- F: government education expenditure "
        "as a percentage of GDP."
    )
    lines.append("")

    lines.append(
        "The linear system predicts dA/dt, dC/dt, "
        "dL/dt, and dF/dt from the previous ACLF "
        "state."
    )
    lines.append("")

    lines.append(
        "## 5. Ridge compared with zero-change baseline"
    )
    lines.append("")
    lines.append(
        "| Level | Sample | Equation | "
        "Ridge RMSE | Zero RMSE | Gain (%) | Result |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---|"
    )

    for _, row in key_results.loc[
        key_results["RESULT_GROUP"].str.contains(
            "Ridge vs zero"
        )
    ].iterrows():
        lines.append(
            f"| {row['RESULT_GROUP']} "
            f"| {row['SAMPLE']} "
            f"| {row['EQUATION']} "
            f"| {format_number(row['RMSE'], 4)} "
            f"| {format_number(row['REFERENCE_RMSE'], 4)} "
            f"| {format_number(row['GAIN_PERCENT'], 2)} "
            f"| {row['CLASSIFICATION']} |"
        )

    lines.append("")
    lines.append(
        "Only the adult educational-attainment equation, dC/dt, "
        "consistently outperformed the zero-change "
        "baseline."
    )
    lines.append("")

    lines.append(
        "## 6. Stability of the adult educational-attainment signal"
    )
    lines.append("")

    for _, row in stability.iterrows():
        lines.append(
            f"- {row['SAMPLE']}: overall RMSE "
            f"reduction of "
            f"{format_number(row['OVERALL_GAIN_PERCENT'], 2)}%; "
            f"leave-one-country-out range "
            f"{format_number(row['LOO_MIN_GAIN_PERCENT'], 2)}% "
            f"to "
            f"{format_number(row['LOO_MAX_GAIN_PERCENT'], 2)}%; "
            f"cluster-bootstrap 95% interval "
            f"{format_number(row['BOOTSTRAP_CI_2_5'], 2)}% "
            f"to "
            f"{format_number(row['BOOTSTRAP_CI_97_5'], 2)}%."
        )

    lines.append("")
    lines.append(
        "The advantage remained positive after "
        "removing each country and in all evaluated "
        "decades."
    )
    lines.append("")

    lines.append(
        "## 7. Nested machine-learning benchmark"
    )
    lines.append("")
    lines.append(
        "| Sample | Model | RMSE | MAE | R2 | "
        "Gain vs zero (%) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for _, row in ml_metrics.iterrows():
        lines.append(
            f"| {row['SAMPLE']} "
            f"| {row['MODEL']} "
            f"| {format_number(row['RMSE_CV'], 4)} "
            f"| {format_number(row['MAE_CV'], 4)} "
            f"| {format_number(row['R2_CV'], 4)} "
            f"| "
            f"{format_number(row['RMSE_GAIN_PERCENT_VS_ZERO'], 2)} "
            f"|"
        )

    lines.append("")
    lines.append(
        "Ridge was the nominal winner for annual "
        "transitions. Elastic Net was the nominal "
        "winner for transitions of up to two years."
    )
    lines.append("")

    lines.append(
        "## 8. Ridge-Elastic Net equivalence"
    )
    lines.append("")

    for _, row in equivalence.iterrows():
        lines.append(
            f"- {row['SAMPLE']}: absolute relative "
            f"RMSE difference "
            f"{format_number(row['ABSOLUTE_RELATIVE_DIFFERENCE_PERCENT'], 3)}%; "
            f"bootstrap interval "
            f"{format_number(row['BOOTSTRAP_CI_2_5'], 3)}% "
            f"to "
            f"{format_number(row['BOOTSTRAP_CI_97_5'], 3)}%; "
            f"practical equivalence probability "
            f"{format_number(row['PROBABILITY_EQUIVALENT'] * 100, 2)}%."
        )

    lines.append("")
    lines.append(
        "Ridge and Elastic Net are practically "
        "equivalent under the provisional 1% RMSE "
        "equivalence margin."
    )
    lines.append("")

    lines.append(
        "## 9. Current scientific interpretation"
    )
    lines.append("")
    lines.append(
        "The ACLF state contains a small but stable "
        "predictive signal for changes in "
        "adult educational attainment. The signal is "
        "captured by low-complexity regularized "
        "linear models. Random Forest and "
        "HistGradientBoosting did not provide a "
        "consistent improvement."
    )
    lines.append("")
    lines.append(
        "The remaining ACLF equations did not "
        "outperform a zero-change prediction. "
        "Therefore, the available evidence does not "
        "support the general superiority of the "
        "linear dynamic system."
    )
    lines.append("")

    lines.append("## 10. Limitations")
    lines.append("")
    lines.append(
        "- The panel is sparse and unbalanced."
    )
    lines.append(
        "- Only a small number of country-year rows "
        "contain all eight selected indicators."
    )
    lines.append(
        "- The adult educational-attainment prediction R2 remains low."
    )
    lines.append(
        "- The bootstrap reuses previously generated "
        "out-of-sample predictions."
    )
    lines.append(
        "- The analysis is predictive and "
        "associational, not causal."
    )
    lines.append(
        "- The 1% equivalence margin is provisional."
    )
    lines.append(
        "- PISA has not yet been integrated into "
        "the ACLF benchmark."
    )
    lines.append("")

    lines.append("## 11. Next scientific steps")
    lines.append("")
    lines.append(
        "1. Evaluate sensitivity to alternative "
        "equivalence margins."
    )
    lines.append(
        "2. Examine coefficients and predictor "
        "stability for adult educational attainment."
    )
    lines.append(
        "3. Evaluate temporal-origin validation."
    )
    lines.append(
        "4. Test a state-space formulation."
    )
    lines.append(
        "5. Integrate PISA country-cycle estimates "
        "with sampling uncertainty."
    )
    lines.append(
        "6. Expand the peer-reviewed literature review."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    require_files()

    data = load_data()

    key_results = build_key_results(
        data
    )

    markdown = build_markdown(
        data,
        key_results,
    )

    key_results_file = (
        ARTICLE_DIR
        / "updated_key_results.csv"
    )

    markdown_file = (
        ARTICLE_DIR
        / "updated_benchmark_summary.md"
    )

    diagnostic_file = (
        DIAGNOSTIC_DIR
        / "updated_benchmark_consolidation.csv"
    )

    key_results.to_csv(
        key_results_file,
        index=False,
        encoding="utf-8-sig",
    )

    markdown_file.write_text(
        markdown,
        encoding="utf-8",
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "KEY_RESULT_ROWS": len(key_results),
                "INDICATOR_ROWS": len(
                    data["indicators"]
                ),
                "ML_METRIC_ROWS": len(
                    data["ml_metrics"]
                ),
                "STABILITY_ROWS": len(
                    data["stability"]
                ),
                "EQUIVALENCE_ROWS": len(
                    data["equivalence"]
                ),
                "MARKDOWN_FILE": str(
                    markdown_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        diagnostic_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== CONSOLIDACION ACTUALIZADA ====="
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print(
        "===== RESULTADOS CLAVE ====="
    )

    print(
        key_results.to_string(
            index=False
        )
    )

    print()
    print("Archivos creados:")
    print(key_results_file)
    print(markdown_file)
    print(diagnostic_file)

    print()
    print("VALIDACION_STATUS: OK")


if __name__ == "__main__":
    main()