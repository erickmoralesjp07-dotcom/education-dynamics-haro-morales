from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_EVIDENCE_FILE = (
    ROOT
    / "literature_review"
    / "evidence_synthesis"
    / "final_candidates_master_evidence_table.csv"
)

ASSESSMENT_FILE = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "all_batches_full_text_assessment_consolidated.csv"
)

UNRETRIEVED_FILE = (
    ROOT
    / "literature_review"
    / "full_text_assessment"
    / "batch_02_unretrieved_full_text_closure.csv"
)

BENCHMARK_FILE = (
    ROOT
    / "results"
    / "article"
    / "updated_benchmark_summary.md"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "article"
    / "integrated_article_draft.md"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "results"
    / "diagnostics"
    / "integrated_article_draft_diagnostic.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

NOW = datetime.now().astimezone().isoformat()


def clean(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def yes_count(data, column):
    values = (
        data[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return int(
        values.isin(
            [
                "true",
                "1",
                "yes",
            ]
        ).sum()
    )


def main():
    required_files = [
        MASTER_EVIDENCE_FILE,
        ASSESSMENT_FILE,
        UNRETRIEVED_FILE,
        BENCHMARK_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    evidence = pd.read_csv(
        MASTER_EVIDENCE_FILE,
        encoding="utf-8-sig",
    )

    assessments = pd.read_csv(
        ASSESSMENT_FILE,
        encoding="utf-8-sig",
    )

    unretrieved = pd.read_csv(
        UNRETRIEVED_FILE,
        encoding="utf-8-sig",
    )

    final_count = int(
        assessments[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_final_candidate"
        ).sum()
    )

    limited_count = int(
        assessments[
            "FULL_TEXT_DECISION"
        ].eq(
            "Include_with_major_limitations"
        ).sum()
    )

    excluded_count = int(
        assessments[
            "FULL_TEXT_DECISION"
        ].eq(
            "Exclude_full_text"
        ).sum()
    )

    design_counts = (
        evidence[
            "STUDY_DESIGN_CATEGORY"
        ]
        .value_counts()
    )

    dataset_counts = (
        evidence[
            "DATASET_CATEGORY"
        ]
        .value_counts()
    )

    sample_adequate = yes_count(
        evidence,
        "SAMPLE_DESIGN_ADEQUATE",
    )

    missing_adequate = yes_count(
        evidence,
        "MISSING_DATA_TREATMENT_ADEQUATE",
    )

    validation_adequate = yes_count(
        evidence,
        "VALIDATION_ADEQUATE",
    )

    causal_supported = yes_count(
        evidence,
        "CAUSAL_CLAIM_SUPPORTED",
    )

    code_available = yes_count(
        evidence,
        "CODE_AVAILABLE",
    )

    study_rows = []

    for _, row in evidence.iterrows():
        study_rows.append(
            "| "
            + " | ".join(
                [
                    clean(
                        row.get(
                            "SCREENING_ID",
                            "",
                        )
                    ),
                    clean(
                        row.get(
                            "TITLE",
                            "",
                        )
                    ).replace(
                        "|",
                        "/",
                    ),
                    clean(
                        row.get(
                            "STUDY_DESIGN_CATEGORY",
                            "",
                        )
                    ),
                    clean(
                        row.get(
                            "DATASET_CATEGORY",
                            "",
                        )
                    ),
                    clean(
                        row.get(
                            "FULL_TEXT_REASON",
                            "",
                        )
                    ).replace(
                        "|",
                        "/",
                    ),
                ]
            )
            + " |"
        )

    design_lines = [
        (
            f"- {name}: {count} estudio(s)."
        )
        for name, count
        in design_counts.items()
    ]

    dataset_lines = [
        (
            f"- {name}: {count} estudio(s)."
        )
        for name, count
        in dataset_counts.items()
    ]

    draft = f"""# Borrador integrado del artículo sobre dinámica educativa

Generado: {NOW}

## 1. Propósito del estudio

Este estudio evalúa si un sistema dinámico estructurado puede predecir cambios en dimensiones educativas nacionales mediante información oficial de UNESCO UIS. El núcleo ACLF representa cuatro estados: exclusión escolar en primaria (A), logro educativo de la población adulta (C), alfabetización juvenil (L) y gasto público en educación como porcentaje del PIB (F). El objetivo no es atribuir efectos causales, sino determinar si el estado educativo previo contiene una señal predictiva reproducible para los cambios posteriores.

## 2. Datos y cobertura

La base analítica actual contiene 16.647 observaciones en formato largo, 7.467 filas país-año y 222 países o territorios para el periodo general 1970-2025. Los ocho indicadores seleccionados cubren acceso, logro educativo adulto, aprendizaje en matemáticas, aprendizaje en lectura, alfabetización juvenil, docentes cualificados, financiamiento educativo y equidad por riqueza.

El panel es disperso y no balanceado. Solo diez filas país-año contienen simultáneamente los ocho indicadores, mientras que 1.233 filas contienen al menos cuatro. Por esta razón, el análisis dinámico principal utiliza el subconjunto ACLF, que ofrece la cobertura conjunta más sólida.

## 3. Formulación dinámica

Para cada país y transición temporal se aproximaron las tasas:

- dA/dt: cambio anual en exclusión escolar.
- dC/dt: cambio anual en logro educativo adulto.
- dL/dt: cambio anual en alfabetización juvenil.
- dF/dt: cambio anual en gasto público educativo.

Las tasas se dividieron por el intervalo temporal real. Las transiciones consecutivas constituyen la muestra principal y las brechas de hasta dos años se utilizaron como análisis de robustez.

Los modelos se evaluaron fuera de muestra y se compararon con una referencia de cambio cero. Esta referencia es exigente porque muchas tasas educativas anuales se concentran cerca de cero.

## 4. Resultados predictivos

La ecuación de logro educativo adulto, dC/dt, fue la única que superó consistentemente la predicción de cambio cero. Ridge redujo el RMSE en 5,34 % para transiciones anuales y en 6,73 % para transiciones de hasta dos años.

La mejora permaneció positiva al retirar individualmente cada país. Para transiciones consecutivas, el intervalo bootstrap agrupado de la mejora fue de 3,00 % a 10,01 %. Para brechas de hasta dos años, el intervalo fue de 4,13 % a 12,12 %.

Ridge y Elastic Net presentaron resultados prácticamente equivalentes. La diferencia relativa absoluta de RMSE fue de 0,042 % en las transiciones anuales y de 0,130 % en la muestra con brechas de hasta dos años. Random Forest e HistGradientBoosting no lograron una mejora consistente.

Las ecuaciones dA/dt, dL/dt y dF/dt no superaron de manera estable la referencia de cambio cero. Por tanto, la evidencia no respalda la superioridad general del sistema lineal para todas las dimensiones educativas.

## 5. Revisión de literatura

Se evaluaron {len(assessments)} textos completos entre los lotes 01 y 02. La evaluación produjo:

- {final_count} candidatos finales.
- {limited_count} estudios incluidos con limitaciones metodológicas importantes.
- {excluded_count} exclusiones en texto completo.
- {len(unretrieved)} documentos no recuperados después de búsqueda exhaustiva en el lote 02.

Los {len(evidence)} candidatos finales fueron organizados en una tabla maestra y todos quedaron listos para síntesis. Todos fueron clasificados con riesgo moderado de sesgo, por lo que la literatura se utiliza como apoyo metodológico y contextual, no como evidencia causal definitiva.

### 5.1 Diseños identificados

{chr(10).join(design_lines)}

### 5.2 Fuentes de datos identificadas

{chr(10).join(dataset_lines)}

### 5.3 Calidad metodológica

- Diseño muestral considerado adecuado: {sample_adequate} de {len(evidence)} estudios.
- Tratamiento de datos faltantes considerado adecuado: {missing_adequate} de {len(evidence)}.
- Validación considerada adecuada: {validation_adequate} de {len(evidence)}.
- Afirmaciones causales respaldadas: {causal_supported} de {len(evidence)}.
- Código disponible identificado: {code_available} de {len(evidence)}.

El predominio de diseños observacionales y la ausencia de respaldo causal refuerzan la decisión de interpretar el modelo ACLF como una herramienta predictiva y exploratoria.

## 6. Tabla de los candidatos finales

| ID | Título | Diseño | Datos | Relevancia principal |
|---|---|---|---|---|
{chr(10).join(study_rows)}

## 7. Síntesis entre literatura y resultados

La literatura seleccionada respalda cinco decisiones centrales del estudio.

Primero, las desigualdades educativas presentan una estructura multinivel. Los resultados varían entre estudiantes, escuelas, regiones y países. Esto justifica la validación agrupada y evita tratar las observaciones nacionales como independientes.

Segundo, PISA y otras evaluaciones internacionales requieren el uso correcto de valores plausibles, ponderaciones y errores de muestreo. Debido a que estas condiciones todavía no están integradas plenamente al núcleo ACLF, PISA debe incorporarse en una extensión separada y no como una variable ordinaria.

Tercero, la medición del logro educativo contiene errores, datos agrupados e incertidumbre histórica. La estabilidad de dC/dt debe interpretarse considerando estos problemas de medición, aunque la señal permaneció positiva en las pruebas de sensibilidad realizadas.

Cuarto, los diseños longitudinales muestran que los resultados educativos se acumulan durante periodos prolongados. Esto ayuda a explicar por qué el logro educativo adulto presenta una señal predictiva más estable que las variables con fluctuaciones administrativas o presupuestarias de corto plazo.

Quinto, la literatura no respalda una interpretación causal automática de asociaciones entre financiamiento, acceso, alfabetización y logro. En consecuencia, los coeficientes del sistema dinámico no deben interpretarse como efectos de intervención.

## 8. Caso de Ecuador

La serie de Ecuador contiene 15 observaciones para 2009-2023 y 14 transiciones consecutivas. El modelo Ridge superó la referencia de cambio cero únicamente para dC/dt, con una reducción del RMSE de 11,52 %. Las demás ecuaciones no mejoraron la referencia.

Este resultado es coherente con el patrón global, pero el tamaño de la serie ecuatoriana es insuficiente para inferencia causal o para modelos de alta complejidad. El caso nacional debe presentarse como una comprobación exploratoria.

## 9. Limitaciones

1. El panel internacional es disperso y no balanceado.
2. La cobertura conjunta de los ocho indicadores es reducida.
3. El R2 predictivo de dC/dt sigue siendo bajo.
4. El análisis es predictivo y asociacional, no causal.
5. Los estudios seleccionados tienen riesgo moderado de sesgo.
6. Cinco documentos del lote 02 no pudieron recuperarse.
7. Las evaluaciones PISA todavía no están integradas con sus ponderaciones y valores plausibles en el benchmark ACLF.
8. La equivalencia práctica del 1 % de RMSE es un margen provisional.
9. La disponibilidad de código fue limitada en la literatura final.
10. El caso de Ecuador contiene únicamente 14 transiciones anuales.

## 10. Conclusión

El estado ACLF contiene una señal predictiva pequeña pero estable para el cambio en el logro educativo adulto. La señal es capturada por modelos lineales regularizados de baja complejidad, mientras que los modelos no lineales no aportan una mejora consistente.

La ausencia de mejoras para acceso, alfabetización juvenil y financiamiento muestra que el sistema no debe presentarse como un modelo predictivo general de toda la dinámica educativa. El principal aporte es más específico: demostrar que el logro educativo adulto posee una persistencia estructurada y parcialmente predecible, pero con capacidad explicativa limitada.

La revisión de literatura apoya una interpretación prudente. Los resultados deben utilizarse para describir regularidades, comparar especificaciones y formular hipótesis, no para afirmar relaciones causales ni realizar pronósticos de alta confianza.

## 11. Próximos análisis

1. Examinar estabilidad temporal y de coeficientes para dC/dt.
2. Aplicar validación con origen temporal.
3. Evaluar formulaciones de espacio de estados.
4. Integrar PISA con valores plausibles, ponderaciones y errores de muestreo.
5. Probar márgenes alternativos de equivalencia Ridge-Elastic Net.
6. Preparar tablas y figuras definitivas para el manuscrito.
"""

    OUTPUT_FILE.write_text(
        draft,
        encoding="utf-8",
    )

    required_sections = [
        "## 1. Propósito del estudio",
        "## 4. Resultados predictivos",
        "## 5. Revisión de literatura",
        "## 6. Tabla de los candidatos finales",
        "## 8. Caso de Ecuador",
        "## 9. Limitaciones",
        "## 10. Conclusión",
    ]

    valid = (
        len(evidence) == 12
        and len(assessments) == 27
        and final_count == 12
        and limited_count == 10
        and excluded_count == 5
        and len(unretrieved) == 5
        and all(
            section in draft
            for section in required_sections
        )
        and len(draft) > 10000
    )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": NOW,
                "FINAL_CANDIDATES": len(
                    evidence
                ),
                "FULL_TEXTS_ASSESSED": len(
                    assessments
                ),
                "UNRETRIEVED_FULL_TEXTS": len(
                    unretrieved
                ),
                "DRAFT_CHARACTERS": len(
                    draft
                ),
                "REQUIRED_SECTIONS": len(
                    required_sections
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
        "===== BORRADOR INTEGRADO "
        "DEL ARTÍCULO ====="
    )

    print(
        "TEXTOS EVALUADOS:",
        len(assessments),
    )

    print(
        "CANDIDATOS FINALES:",
        len(evidence),
    )

    print(
        "NO RECUPERADOS:",
        len(unretrieved),
    )

    print(
        "CARACTERES DEL BORRADOR:",
        len(draft),
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
            "El borrador integrado requiere revisión."
        )


if __name__ == "__main__":
    main()