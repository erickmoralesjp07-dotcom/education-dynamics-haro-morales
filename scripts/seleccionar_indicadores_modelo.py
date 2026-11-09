from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_candidate_indicators.csv"
)

OUTPUT_SELECTION = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_selected_indicators.csv"
)

OUTPUT_DIAGNOSTIC = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_selected_indicators_validation.csv"
)


SELECTED_INDICATORS = {
    "access": {
        "indicator_id": "ROFST.1.CP",
        "model_variable": "A",
        "direction": "negative",
        "description_es": (
            "Tasa de exclusion escolar en edad de primaria"
        ),
    },
    "completion": {
        "indicator_id": "EA.2T8.AG25T99",
        "model_variable": "C",
        "direction": "positive",
        "description_es": (
            "Poblacion adulta que completo secundaria inferior o superior"
        ),
    },
    "mathematics_learning": {
        "indicator_id": "MATH.PRIMARY",
        "model_variable": "M",
        "direction": "positive",
        "description_es": (
            "Competencia minima en matematicas al finalizar primaria"
        ),
    },
    "reading_learning": {
        "indicator_id": "READ.PRIMARY",
        "model_variable": "R",
        "direction": "positive",
        "description_es": (
            "Competencia minima en lectura al finalizar primaria"
        ),
    },
    "youth_literacy": {
        "indicator_id": "LR.AG15T24",
        "model_variable": "L",
        "direction": "positive",
        "description_es": (
            "Tasa de alfabetizacion juvenil de 15 a 24 anios"
        ),
    },
    "qualified_teachers": {
        "indicator_id": "TRTP.1",
        "model_variable": "T",
        "direction": "positive",
        "description_es": (
            "Docentes de primaria con cualificaciones minimas"
        ),
    },
    "education_finance": {
        "indicator_id": "XGDP.FSGOV",
        "model_variable": "F",
        "direction": "positive",
        "description_es": (
            "Gasto publico en educacion como porcentaje del PIB"
        ),
    },
    "wealth_equity": {
        "indicator_id": "ROFST.H.1.WPIA",
        "model_variable": "E",
        "direction": "contextual",
        "description_es": (
            "Paridad por riqueza en exclusion escolar de primaria"
        ),
    },
}


def main() -> None:
    print("=" * 72)
    print("SELECCION FINAL DE INDICADORES PARA EL MODELO")
    print("=" * 72)

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro:\n{CANDIDATES_FILE}"
        )

    candidates = pd.read_csv(CANDIDATES_FILE)

    selected_rows = []

    for dimension, configuration in SELECTED_INDICATORS.items():
        indicator_id = configuration["indicator_id"]

        match = candidates.loc[
            candidates["INDICATOR_ID"].eq(indicator_id)
        ].copy()

        if match.empty:
            raise ValueError(
                f"No se encontro el indicador {indicator_id}"
            )

        row = match.iloc[0].to_dict()

        row["MODEL_DIMENSION"] = dimension
        row["MODEL_VARIABLE"] = configuration["model_variable"]
        row["DIRECTION"] = configuration["direction"]
        row["DESCRIPTION_ES"] = configuration["description_es"]

        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)

    selected = selected[
        [
            "MODEL_DIMENSION",
            "MODEL_VARIABLE",
            "INDICATOR_ID",
            "INDICATOR_LABEL_EN",
            "DESCRIPTION_ES",
            "DIRECTION",
            "COUNTRY_COUNT",
            "MIN_YEAR",
            "MAX_YEAR",
            "YEAR_SPAN",
            "RECORD_COUNT",
            "NON_MISSING_VALUES",
        ]
    ]

    selected["SELECTION_DATETIME"] = (
        datetime.now().astimezone().isoformat()
    )

    selected.to_csv(
        OUTPUT_SELECTION,
        index=False,
        encoding="utf-8-sig",
    )

    validation = pd.DataFrame(
        [
            {
                "selected_indicator_count": len(selected),
                "unique_indicator_count": (
                    selected["INDICATOR_ID"].nunique()
                ),
                "unique_model_variables": (
                    selected["MODEL_VARIABLE"].nunique()
                ),
                "minimum_country_coverage": int(
                    selected["COUNTRY_COUNT"].min()
                ),
                "maximum_country_coverage": int(
                    selected["COUNTRY_COUNT"].max()
                ),
                "earliest_available_year": int(
                    selected["MIN_YEAR"].min()
                ),
                "latest_available_year": int(
                    selected["MAX_YEAR"].max()
                ),
                "validation_status": "OK",
            }
        ]
    )

    validation.to_csv(
        OUTPUT_DIAGNOSTIC,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nINDICADORES SELECCIONADOS")

    print(
        selected[
            [
                "MODEL_DIMENSION",
                "MODEL_VARIABLE",
                "INDICATOR_ID",
                "COUNTRY_COUNT",
                "MIN_YEAR",
                "MAX_YEAR",
            ]
        ].to_string(index=False)
    )

    print(f"\nSeleccion guardada:\n{OUTPUT_SELECTION}")
    print(f"\nDiagnostico guardado:\n{OUTPUT_DIAGNOSTIC}")
    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR DURANTE LA SELECCION")
        print(f"Tipo: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)