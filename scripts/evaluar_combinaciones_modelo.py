from pathlib import Path
from itertools import combinations
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_panel_wide.parquet"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_selected_indicators.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_variable_combinations.csv"
)


def evaluate_combination(
    panel: pd.DataFrame,
    variables: tuple[str, ...],
) -> dict:
    """Calcula la cobertura conjunta de una combinación."""

    complete_mask = panel[list(variables)].notna().all(axis=1)
    complete_data = panel.loc[complete_mask].copy()

    country_count = int(
        complete_data["COUNTRY_ID"].nunique()
    )

    country_year_rows = len(complete_data)

    if complete_data.empty:
        minimum_year = None
        maximum_year = None
        countries_with_3_years = 0
        countries_with_5_years = 0
        countries_with_10_years = 0

    else:
        minimum_year = int(complete_data["YEAR"].min())
        maximum_year = int(complete_data["YEAR"].max())

        observations_by_country = (
            complete_data
            .groupby("COUNTRY_ID")
            .size()
        )

        countries_with_3_years = int(
            observations_by_country.ge(3).sum()
        )

        countries_with_5_years = int(
            observations_by_country.ge(5).sum()
        )

        countries_with_10_years = int(
            observations_by_country.ge(10).sum()
        )

    return {
        "VARIABLE_COUNT": len(variables),
        "VARIABLES": "|".join(variables),
        "COUNTRY_YEAR_ROWS": country_year_rows,
        "COUNTRY_COUNT": country_count,
        "MIN_YEAR": minimum_year,
        "MAX_YEAR": maximum_year,
        "COUNTRIES_WITH_AT_LEAST_3_YEARS": (
            countries_with_3_years
        ),
        "COUNTRIES_WITH_AT_LEAST_5_YEARS": (
            countries_with_5_years
        ),
        "COUNTRIES_WITH_AT_LEAST_10_YEARS": (
            countries_with_10_years
        ),
    }


def main() -> None:
    print("=" * 74)
    print("EVALUACION DE COMBINACIONES DE VARIABLES DEL MODELO")
    print("=" * 74)

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro el panel:\n{PANEL_FILE}"
        )

    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro la seleccion:\n{SELECTION_FILE}"
        )

    print("\nLeyendo el panel ancho...")

    panel = pd.read_parquet(PANEL_FILE)

    selection = pd.read_csv(
        SELECTION_FILE,
        dtype={
            "MODEL_VARIABLE": "string",
            "MODEL_DIMENSION": "string",
            "INDICATOR_ID": "string",
        },
    )

    variables = (
        selection["MODEL_VARIABLE"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print("Variables disponibles:", variables)

    missing_variables = [
        variable
        for variable in variables
        if variable not in panel.columns
    ]

    if missing_variables:
        raise ValueError(
            "Faltan variables en el panel: "
            f"{missing_variables}"
        )

    results = []

    print("\nEvaluando todas las combinaciones de 4 variables...")

    for combination in combinations(variables, 4):
        results.append(
            evaluate_combination(
                panel,
                combination,
            )
        )

    print("Evaluando todas las combinaciones de 5 variables...")

    for combination in combinations(variables, 5):
        results.append(
            evaluate_combination(
                panel,
                combination,
            )
        )

    print("Evaluando todas las combinaciones de 6 variables...")

    for combination in combinations(variables, 6):
        results.append(
            evaluate_combination(
                panel,
                combination,
            )
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "VARIABLE_COUNT",
            "COUNTRY_YEAR_ROWS",
            "COUNTRIES_WITH_AT_LEAST_5_YEARS",
            "COUNTRY_COUNT",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 74)
    print("EVALUACION COMPLETADA CORRECTAMENTE")
    print("=" * 74)

    for variable_count in [4, 5, 6]:
        print(
            f"\nMEJORES 10 COMBINACIONES DE "
            f"{variable_count} VARIABLES"
        )

        subset = (
            results_df.loc[
                results_df["VARIABLE_COUNT"]
                .eq(variable_count)
            ]
            .head(10)
        )

        print(
            subset[
                [
                    "VARIABLES",
                    "COUNTRY_YEAR_ROWS",
                    "COUNTRY_COUNT",
                    "MIN_YEAR",
                    "MAX_YEAR",
                    "COUNTRIES_WITH_AT_LEAST_3_YEARS",
                    "COUNTRIES_WITH_AT_LEAST_5_YEARS",
                    "COUNTRIES_WITH_AT_LEAST_10_YEARS",
                ]
            ].to_string(index=False)
        )

    print(f"\nResultados guardados:\n{OUTPUT_FILE}")
    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR DURANTE LA EVALUACION")
        print(f"Tipo: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)