from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_FILE = (
    ROOT
    / "results"
    / "temporal_validation"
    / "dc_temporal_origin_results.csv"
)

WINDOW_FILE = (
    ROOT
    / "results"
    / "temporal_validation"
    / "dc_temporal_window_sensitivity.csv"
)

COEFFICIENT_FILE = (
    ROOT
    / "results"
    / "coefficient_stability"
    / "dc_temporal_coefficient_sign_summary.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "article"
    / "final_validation_summary.csv"
)

SUMMARY_FILE = (
    ROOT
    / "results"
    / "article"
    / "final_scientific_validation.md"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "final_scientific_validation_diagnostic.csv"
)

NOW = datetime.now().astimezone().isoformat()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def main():
    required_files = [
        TEMPORAL_FILE,
        WINDOW_FILE,
        COEFFICIENT_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    temporal = pd.read_csv(
        TEMPORAL_FILE,
        encoding="utf-8-sig",
    )

    windows = pd.read_csv(
        WINDOW_FILE,
        encoding="utf-8-sig",
    )

    coefficients = pd.read_csv(
        COEFFICIENT_FILE,
        encoding="utf-8-sig",
    )

    temporal_tests = len(temporal)

    temporal_positive = int(
        temporal[
            "RIDGE_BEATS_ZERO"
        ].sum()
    )

    window_tests = len(windows)

    window_positive = int(
        windows[
            "RIDGE_BEATS_ZERO"
        ].sum()
    )

    gap_rows = windows.loc[
        windows[
            "SAMPLE"
        ].eq("gap_max_2")
    ].copy()

    gap_tests = len(gap_rows)

    gap_positive = int(
        gap_rows[
            "RIDGE_BEATS_ZERO"
        ].sum()
    )

    annual_rows = windows.loc[
        windows[
            "SAMPLE"
        ].eq("consecutive")
    ].copy()

    annual_tests = len(annual_rows)

    annual_positive = int(
        annual_rows[
            "RIDGE_BEATS_ZERO"
        ].sum()
    )

    stable_terms = coefficients.loc[
        coefficients[
            "SIGN_CONSISTENCY_PERCENT"
        ].eq(100.0),
        "TERM",
    ].astype(str).tolist()

    coefficient_consistency = float(
        coefficients[
            "EXPECTED_SIGN_MATCHES"
        ].sum()
        / coefficients[
            "MODELS_EVALUATED"
        ].sum()
        * 100
    )

    no_future_leakage = bool(
        temporal[
            "NO_FUTURE_LEAKAGE"
        ].all()
        and windows[
            "NO_FUTURE_LEAKAGE"
        ].all()
    )

    summary_rows = [
        {
            "VALIDATION_COMPONENT": (
                "Temporal-origin validation"
            ),
            "TESTS": temporal_tests,
            "POSITIVE_TESTS": temporal_positive,
            "POSITIVE_PERCENT": round(
                100
                * temporal_positive
                / temporal_tests,
                2,
            ),
            "MAIN_RESULT": (
                "Ridge supera cambio cero "
                "en 4 de 5 pruebas"
            ),
            "STATUS": "OK",
        },
        {
            "VALIDATION_COMPONENT": (
                "Temporal-window sensitivity"
            ),
            "TESTS": window_tests,
            "POSITIVE_TESTS": window_positive,
            "POSITIVE_PERCENT": round(
                100
                * window_positive
                / window_tests,
                2,
            ),
            "MAIN_RESULT": (
                "12 de 15 pruebas positivas"
            ),
            "STATUS": "OK",
        },
        {
            "VALIDATION_COMPONENT": (
                "Gap maximum two years"
            ),
            "TESTS": gap_tests,
            "POSITIVE_TESTS": gap_positive,
            "POSITIVE_PERCENT": round(
                100
                * gap_positive
                / gap_tests,
                2,
            ),
            "MAIN_RESULT": (
                "9 de 9 pruebas positivas"
            ),
            "STATUS": "OK",
        },
        {
            "VALIDATION_COMPONENT": (
                "Annual transitions"
            ),
            "TESTS": annual_tests,
            "POSITIVE_TESTS": annual_positive,
            "POSITIVE_PERCENT": round(
                100
                * annual_positive
                / annual_tests,
                2,
            ),
            "MAIN_RESULT": (
                "3 de 6 pruebas positivas; "
                "estabilidad limitada"
            ),
            "STATUS": "CAUTION",
        },
        {
            "VALIDATION_COMPONENT": (
                "Coefficient sign stability"
            ),
            "TESTS": int(
                coefficients[
                    "MODELS_EVALUATED"
                ].sum()
            ),
            "POSITIVE_TESTS": int(
                coefficients[
                    "EXPECTED_SIGN_MATCHES"
                ].sum()
            ),
            "POSITIVE_PERCENT": round(
                coefficient_consistency,
                2,
            ),
            "MAIN_RESULT": (
                "C y L mantienen el signo "
                "en todos los modelos"
            ),
            "STATUS": "OK_WITH_LIMITATIONS",
        },
    ]

    final_table = pd.DataFrame(
        summary_rows
    )

    final_table.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    stable_text = (
        ", ".join(stable_terms)
        if stable_terms
        else "ninguna"
    )

    markdown = f"""# Validación científica final

Generado: {NOW}

## Resultado general

La señal predictiva de `dC/dt` fue evaluada mediante validación agrupada por país, validación temporal, sensibilidad a ventanas futuras y estabilidad de coeficientes.

## Validación temporal

- Pruebas con origen temporal: {temporal_tests}.
- Pruebas positivas: {temporal_positive} de {temporal_tests}.
- No se utilizó información futura para el entrenamiento.
- La señal fue positiva en cuatro de cinco pruebas, pero falló en una prueba anual con origen 2014.

## Sensibilidad de ventanas

- Pruebas totales: {window_tests}.
- Pruebas positivas: {window_positive} de {window_tests}.
- Transiciones anuales: {annual_positive} de {annual_tests} positivas.
- Intervalos de hasta dos años: {gap_positive} de {gap_tests} positivas.

La señal es más estable cuando se permiten intervalos de hasta dos años. La muestra estrictamente anual presenta mayor variabilidad temporal.

## Estabilidad de coeficientes

- Coincidencia general de signos: {coefficient_consistency:.1f}%.
- Variables con estabilidad de signo del 100%: {stable_text}.
- `PREVIOUS_C` mantuvo signo negativo.
- `PREVIOUS_L` mantuvo signo positivo.
- `PREVIOUS_A` y `PREVIOUS_F` fueron menos estables.

## Interpretación

Los resultados respaldan una señal predictiva pequeña para el cambio en logro educativo adulto. La señal no es uniforme en todos los periodos y no debe interpretarse causalmente.

La evidencia más sólida corresponde a:

1. validación agrupada por país;
2. validación temporal sin información futura;
3. resultados positivos en 9 de 9 pruebas con intervalos de hasta dos años;
4. estabilidad completa de los signos de `PREVIOUS_C` y `PREVIOUS_L`.

## Limitaciones

1. Las transiciones anuales fueron positivas solo en tres de seis pruebas de sensibilidad.
2. El corte anual con origen 2014 produjo un resultado negativo.
3. Los signos de `PREVIOUS_A` y `PREVIOUS_F` no fueron completamente estables.
4. El rendimiento predictivo sigue siendo bajo.
5. Los coeficientes representan asociaciones predictivas, no efectos causales.

## Conclusión

La validación científica final respalda el uso prudente de Ridge para modelar `dC/dt`, especialmente con intervalos de hasta dos años. El resultado debe presentarse como una señal predictiva moderada y parcialmente estable, no como una ley general de dinámica educativa.
"""

    SUMMARY_FILE.write_text(
        markdown,
        encoding="utf-8",
    )

    valid = (
        temporal_tests == 5
        and temporal_positive == 4
        and window_tests == 15
        and window_positive == 12
        and gap_tests == 9
        and gap_positive == 9
        and annual_tests == 6
        and annual_positive == 3
        and round(
            coefficient_consistency,
            1,
        ) == 80.0
        and set(stable_terms)
        == {
            "PREVIOUS_C",
            "PREVIOUS_L",
        }
        and no_future_leakage
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "TEMPORAL_TESTS": temporal_tests,
                "TEMPORAL_POSITIVE": (
                    temporal_positive
                ),
                "WINDOW_TESTS": window_tests,
                "WINDOW_POSITIVE": (
                    window_positive
                ),
                "GAP_MAX_2_TESTS": gap_tests,
                "GAP_MAX_2_POSITIVE": (
                    gap_positive
                ),
                "ANNUAL_TESTS": annual_tests,
                "ANNUAL_POSITIVE": (
                    annual_positive
                ),
                "COEFFICIENT_SIGN_CONSISTENCY": (
                    coefficient_consistency
                ),
                "FULLY_STABLE_TERMS": (
                    stable_text
                ),
                "NO_FUTURE_LEAKAGE": (
                    no_future_leakage
                ),
                "VALIDATION_STATUS": (
                    "OK"
                    if valid
                    else "REVIEW"
                ),
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "===== INTEGRACIÓN DE "
        "VALIDACIONES FINALES ====="
    )

    print(
        final_table.to_string(
            index=False
        )
    )

    print()
    print(
        "PRUEBAS TEMPORALES:",
        temporal_tests,
    )

    print(
        "SENSIBILIDADES POSITIVAS:",
        window_positive,
        "de",
        window_tests,
    )

    print(
        "GAP_MAX_2 POSITIVAS:",
        gap_positive,
        "de",
        gap_tests,
    )

    print(
        "CONSISTENCIA DE SIGNOS:",
        f"{coefficient_consistency:.1f}%",
    )

    print(
        "VARIABLES ESTABLES:",
        stable_text,
    )

    print(
        "FUGA TEMPORAL:",
        "NO"
        if no_future_leakage
        else "SÍ",
    )

    print(
        "VALIDACION_STATUS:",
        diagnostic.loc[
            0,
            "VALIDATION_STATUS",
        ],
    )

    print(
        "ARCHIVO:",
        OUTPUT_FILE,
    )

    if not valid:
        raise ValueError(
            "La integración final "
            "requiere revisión."
        )


if __name__ == "__main__":
    main()