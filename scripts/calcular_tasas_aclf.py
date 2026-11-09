from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_complete.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uis_model_aclf_rates.parquet"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_aclf_rates_summary.csv"
)

COUNTRY_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_model_aclf_rates_country_summary.csv"
)


VARIABLES = ["A", "C", "L", "F"]


def main() -> None:
    print("=" * 72)
    print("CALCULO DE TASAS TEMPORALES DEL MODELO A-C-L-F")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro el panel:\n{INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\nLeyendo el panel A-C-L-F...")

    panel = pd.read_parquet(INPUT_FILE)

    required_columns = {
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "YEAR",
        *VARIABLES,
    }

    missing_columns = sorted(
        required_columns - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas: {missing_columns}"
        )

    panel = panel.sort_values(
        ["COUNTRY_ID", "YEAR"]
    ).reset_index(drop=True)

    print(f"Filas originales: {len(panel):,}")

    grouped = panel.groupby(
        "COUNTRY_ID",
        sort=False,
    )

    panel["PREVIOUS_YEAR"] = grouped["YEAR"].shift(1)

    panel["DELTA_T"] = (
        panel["YEAR"] - panel["PREVIOUS_YEAR"]
    )

    for variable in VARIABLES:
        panel[f"PREVIOUS_{variable}"] = (
            grouped[variable].shift(1)
        )

        panel[f"DELTA_{variable}"] = (
            panel[variable]
            - panel[f"PREVIOUS_{variable}"]
        )

        panel[f"D{variable}_DT"] = (
            panel[f"DELTA_{variable}"]
            / panel["DELTA_T"]
        )

    rates = panel.dropna(
        subset=[
            "PREVIOUS_YEAR",
            "DELTA_T",
            "DA_DT",
            "DC_DT",
            "DL_DT",
            "DF_DT",
        ]
    ).copy()

    rates = rates.loc[
        rates["DELTA_T"].gt(0)
    ].copy()

    rates["DELTA_T"] = rates["DELTA_T"].astype(int)

    rates["IS_CONSECUTIVE"] = (
        rates["DELTA_T"].eq(1)
    )

    rates["MAX_GAP_2_YEARS"] = (
        rates["DELTA_T"].le(2)
    )

    rates["MAX_GAP_3_YEARS"] = (
        rates["DELTA_T"].le(3)
    )

    output_columns = [
        "COUNTRY_ID",
        "COUNTRY_NAME_EN",
        "PREVIOUS_YEAR",
        "YEAR",
        "DELTA_T",
        "IS_CONSECUTIVE",
        "MAX_GAP_2_YEARS",
        "MAX_GAP_3_YEARS",
        "PREVIOUS_A",
        "PREVIOUS_C",
        "PREVIOUS_L",
        "PREVIOUS_F",
        "A",
        "C",
        "L",
        "F",
        "DELTA_A",
        "DELTA_C",
        "DELTA_L",
        "DELTA_F",
        "DA_DT",
        "DC_DT",
        "DL_DT",
        "DF_DT",
    ]

    rates = rates[
        output_columns
    ].sort_values(
        ["COUNTRY_ID", "YEAR"]
    ).reset_index(drop=True)

    total_transitions = len(rates)
    consecutive = int(rates["IS_CONSECUTIVE"].sum())
    maximum_gap_2 = int(rates["MAX_GAP_2_YEARS"].sum())
    maximum_gap_3 = int(rates["MAX_GAP_3_YEARS"].sum())

    country_summary = (
        rates
        .groupby(
            ["COUNTRY_ID", "COUNTRY_NAME_EN"],
            dropna=False,
        )
        .agg(
            TRANSITION_COUNT=("DELTA_T", "size"),
            CONSECUTIVE_TRANSITIONS=(
                "IS_CONSECUTIVE",
                "sum",
            ),
            TRANSITIONS_GAP_MAX_2=(
                "MAX_GAP_2_YEARS",
                "sum",
            ),
            TRANSITIONS_GAP_MAX_3=(
                "MAX_GAP_3_YEARS",
                "sum",
            ),
            MIN_YEAR=("PREVIOUS_YEAR", "min"),
            MAX_YEAR=("YEAR", "max"),
            MEAN_GAP=("DELTA_T", "mean"),
            MAX_GAP=("DELTA_T", "max"),
        )
        .reset_index()
    )

    country_summary["MEAN_GAP"] = (
        country_summary["MEAN_GAP"].round(2)
    )

    country_summary = country_summary.sort_values(
        by=[
            "CONSECUTIVE_TRANSITIONS",
            "TRANSITIONS_GAP_MAX_2",
            "TRANSITION_COUNT",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "processing_datetime": (
                    datetime.now().astimezone().isoformat()
                ),
                "model_variables": "A|C|L|F",
                "total_transitions": total_transitions,
                "country_count": int(
                    rates["COUNTRY_ID"].nunique()
                ),
                "consecutive_transitions": consecutive,
                "transitions_gap_max_2": maximum_gap_2,
                "transitions_gap_max_3": maximum_gap_3,
                "consecutive_percent": round(
                    consecutive / total_transitions * 100,
                    2,
                ),
                "gap_max_2_percent": round(
                    maximum_gap_2 / total_transitions * 100,
                    2,
                ),
                "gap_max_3_percent": round(
                    maximum_gap_3 / total_transitions * 100,
                    2,
                ),
                "minimum_gap": int(
                    rates["DELTA_T"].min()
                ),
                "maximum_gap": int(
                    rates["DELTA_T"].max()
                ),
                "output_file": OUTPUT_FILE.name,
                "validation_status": "OK",
            }
        ]
    )

    print("\nGuardando tasas temporales...")

    rates.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    country_summary.to_csv(
        COUNTRY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    validation = pd.read_parquet(OUTPUT_FILE)

    if len(validation) != total_transitions:
        raise RuntimeError(
            "La validacion del archivo de tasas fallo."
        )

    print("\n" + "=" * 72)
    print("TASAS TEMPORALES CALCULADAS CORRECTAMENTE")
    print("=" * 72)

    print(
        "\nTransiciones totales:",
        f"{total_transitions:,}",
    )

    print(
        "Paises o territorios:",
        f"{rates['COUNTRY_ID'].nunique():,}",
    )

    print(
        "Transiciones consecutivas:",
        f"{consecutive:,}",
    )

    print(
        "Transiciones con intervalo maximo de 2 anios:",
        f"{maximum_gap_2:,}",
    )

    print(
        "Transiciones con intervalo maximo de 3 anios:",
        f"{maximum_gap_3:,}",
    )

    print("\nPRIMEROS 15 PAISES CON MAYOR CONTINUIDAD")

    print(
        country_summary[
            [
                "COUNTRY_ID",
                "COUNTRY_NAME_EN",
                "TRANSITION_COUNT",
                "CONSECUTIVE_TRANSITIONS",
                "TRANSITIONS_GAP_MAX_2",
                "MEAN_GAP",
                "MAX_GAP",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(f"\nArchivo de tasas:\n{OUTPUT_FILE}")
    print(f"\nResumen general:\n{SUMMARY_FILE}")
    print(f"\nResumen por pais:\n{COUNTRY_FILE}")
    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR DURANTE EL CALCULO DE TASAS")
        print(f"Tipo: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)