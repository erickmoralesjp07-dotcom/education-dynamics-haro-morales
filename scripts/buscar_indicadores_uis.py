from pathlib import Path
from datetime import datetime
import sys

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uis"
    / "SDG_LABEL.csv"
)

NATIONAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "uis_sdg_national_enriched.parquet"
)

OUTPUT_CATALOG = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "uis_candidate_indicators.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "uis_candidate_indicators_summary.csv"
)


# ============================================================
# DIMENSIONES Y PALABRAS DE BUSQUEDA
# ============================================================

SEARCH_GROUPS = {
    "access_enrolment": [
        "enrolment",
        "enrollment",
        "attendance",
        "out-of-school",
        "out of school",
        "participation rate",
        "admission rate",
    ],
    "completion_transition": [
        "completion rate",
        "graduation",
        "transition rate",
        "survival rate",
        "last grade",
        "completed primary",
        "completed lower secondary",
        "completed upper secondary",
    ],
    "learning_quality": [
        "minimum proficiency",
        "learning achievement",
        "reading proficiency",
        "mathematics proficiency",
        "learning assessment",
        "learning outcomes",
    ],
    "literacy_numeracy": [
        "literacy rate",
        "literate",
        "functional literacy",
        "numeracy rate",
        "adult literacy",
        "youth literacy",
    ],
    "teachers_resources": [
        "trained teachers",
        "qualified teachers",
        "pupil-teacher ratio",
        "pupil teacher ratio",
        "student-teacher ratio",
        "student teacher ratio",
        "teacher attrition",
        "teacher training",
        "teachers with",
    ],
    "finance_infrastructure": [
        "expenditure on education",
        "government expenditure",
        "education expenditure",
        "spending on education",
        "electricity",
        "internet",
        "computers",
        "basic drinking water",
        "handwashing facilities",
        "school infrastructure",
    ],
    "equity_gender": [
        "gender parity",
        "parity index",
        "female to male",
        "wealth parity",
        "location parity",
        "rural to urban",
        "disability parity",
        "adjusted gender parity",
    ],
}


def verify_file(file_path: Path) -> None:
    """Comprueba que un archivo necesario exista."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo requerido:\n{file_path}"
        )


def classify_indicator(label: str) -> tuple[str, str]:
    """
    Clasifica una etiqueta según las dimensiones definidas.

    Devuelve:
        dimensiones encontradas
        palabras clave encontradas
    """
    normalized_label = label.lower()

    matched_dimensions = []
    matched_keywords = []

    for dimension, keywords in SEARCH_GROUPS.items():
        dimension_matches = [
            keyword
            for keyword in keywords
            if keyword in normalized_label
        ]

        if dimension_matches:
            matched_dimensions.append(dimension)
            matched_keywords.extend(dimension_matches)

    return (
        "|".join(matched_dimensions),
        "|".join(sorted(set(matched_keywords))),
    )


def main() -> None:
    print("=" * 72)
    print("BUSQUEDA DE INDICADORES CANDIDATOS UNESCO UIS")
    print("=" * 72)

    verify_file(LABEL_FILE)
    verify_file(NATIONAL_FILE)

    OUTPUT_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    print("\nLeyendo el catalogo de indicadores...")

    labels = pd.read_csv(
        LABEL_FILE,
        dtype={
            "INDICATOR_ID": "string",
            "INDICATOR_LABEL_EN": "string",
        },
    )

    labels = labels.drop_duplicates(
        subset=["INDICATOR_ID"]
    ).copy()

    labels = labels.dropna(
        subset=["INDICATOR_ID", "INDICATOR_LABEL_EN"]
    ).copy()

    print(f"Indicadores disponibles en el catalogo: {len(labels):,}")

    print("\nClasificando indicadores por dimensiones...")

    classification = labels["INDICATOR_LABEL_EN"].apply(
        classify_indicator
    )

    labels["DIMENSION"] = classification.apply(
        lambda result: result[0]
    )

    labels["MATCHED_KEYWORDS"] = classification.apply(
        lambda result: result[1]
    )

    candidates = labels.loc[
        labels["DIMENSION"].ne("")
    ].copy()

    print(
        "Indicadores encontrados mediante palabras clave:",
        f"{len(candidates):,}",
    )

    print("\nCalculando cobertura disponible en la base nacional...")

    national = pd.read_parquet(
        NATIONAL_FILE,
        columns=[
            "INDICATOR_ID",
            "COUNTRY_ID",
            "YEAR",
            "VALUE",
        ],
    )

    coverage = (
        national
        .groupby("INDICATOR_ID", dropna=False)
        .agg(
            RECORD_COUNT=("VALUE", "size"),
            NON_MISSING_VALUES=("VALUE", "count"),
            COUNTRY_COUNT=("COUNTRY_ID", "nunique"),
            MIN_YEAR=("YEAR", "min"),
            MAX_YEAR=("YEAR", "max"),
        )
        .reset_index()
    )

    candidates = candidates.merge(
        coverage,
        on="INDICATOR_ID",
        how="left",
        validate="one_to_one",
    )

    numeric_columns = [
        "RECORD_COUNT",
        "NON_MISSING_VALUES",
        "COUNTRY_COUNT",
        "MIN_YEAR",
        "MAX_YEAR",
    ]

    for column in numeric_columns:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        ).astype("Int64")

    candidates["YEAR_SPAN"] = (
        candidates["MAX_YEAR"]
        - candidates["MIN_YEAR"]
        + 1
    ).astype("Int64")

    candidates["HAS_NATIONAL_DATA"] = (
        candidates["RECORD_COUNT"]
        .fillna(0)
        .gt(0)
    )

    candidates["PRIORITY_SCORE"] = (
        candidates["COUNTRY_COUNT"]
        .fillna(0)
        .astype(float)
        + candidates["YEAR_SPAN"]
        .fillna(0)
        .astype(float)
    )

    candidates = candidates[
        [
            "DIMENSION",
            "INDICATOR_ID",
            "INDICATOR_LABEL_EN",
            "MATCHED_KEYWORDS",
            "RECORD_COUNT",
            "NON_MISSING_VALUES",
            "COUNTRY_COUNT",
            "MIN_YEAR",
            "MAX_YEAR",
            "YEAR_SPAN",
            "HAS_NATIONAL_DATA",
            "PRIORITY_SCORE",
        ]
    ]

    candidates = candidates.sort_values(
        by=[
            "DIMENSION",
            "HAS_NATIONAL_DATA",
            "COUNTRY_COUNT",
            "YEAR_SPAN",
            "INDICATOR_ID",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    candidates.to_csv(
        OUTPUT_CATALOG,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for dimension in SEARCH_GROUPS:
        dimension_mask = candidates["DIMENSION"].str.contains(
            dimension,
            regex=False,
            na=False,
        )

        dimension_data = candidates.loc[dimension_mask]

        summary_rows.append(
            {
                "processing_datetime": (
                    datetime.now().astimezone().isoformat()
                ),
                "dimension": dimension,
                "candidate_indicator_count": len(dimension_data),
                "indicators_with_national_data": int(
                    dimension_data["HAS_NATIONAL_DATA"].sum()
                ),
                "maximum_country_coverage": (
                    int(dimension_data["COUNTRY_COUNT"].max())
                    if not dimension_data.empty
                    and dimension_data["COUNTRY_COUNT"].notna().any()
                    else 0
                ),
                "earliest_year": (
                    int(dimension_data["MIN_YEAR"].min())
                    if not dimension_data.empty
                    and dimension_data["MIN_YEAR"].notna().any()
                    else None
                ),
                "latest_year": (
                    int(dimension_data["MAX_YEAR"].max())
                    if not dimension_data.empty
                    and dimension_data["MAX_YEAR"].notna().any()
                    else None
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("BUSQUEDA COMPLETADA CORRECTAMENTE")
    print("=" * 72)

    print(f"\nCatalogo creado:\n{OUTPUT_CATALOG}")
    print(f"\nResumen creado:\n{OUTPUT_SUMMARY}")

    print("\nRESUMEN POR DIMENSION")
    print(
        summary[
            [
                "dimension",
                "candidate_indicator_count",
                "indicators_with_national_data",
                "maximum_country_coverage",
                "earliest_year",
                "latest_year",
            ]
        ].to_string(index=False)
    )

    print("\nPRIMEROS 20 INDICADORES CON MAYOR COBERTURA")

    top_candidates = candidates.loc[
        candidates["HAS_NATIONAL_DATA"]
    ].sort_values(
        by=[
            "COUNTRY_COUNT",
            "YEAR_SPAN",
            "RECORD_COUNT",
        ],
        ascending=False,
    ).head(20)

    print(
        top_candidates[
            [
                "DIMENSION",
                "INDICATOR_ID",
                "INDICATOR_LABEL_EN",
                "COUNTRY_COUNT",
                "MIN_YEAR",
                "MAX_YEAR",
            ]
        ].to_string(index=False)
    )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\n" + "=" * 72)
        print("ERROR DURANTE LA BUSQUEDA")
        print("=" * 72)
        print(f"\nTipo de error: {type(error).__name__}")
        print(f"Detalle: {error}")
        sys.exit(1)