# Borrador integrado del artículo sobre dinámica educativa

Generado: 2026-11-18T07:39:39.984747-05:00

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

Se evaluaron 27 textos completos entre los lotes 01 y 02. La evaluación produjo:

- 12 candidatos finales.
- 10 estudios incluidos con limitaciones metodológicas importantes.
- 5 exclusiones en texto completo.
- 5 documentos no recuperados después de búsqueda exhaustiva en el lote 02.

Los 12 candidatos finales fueron organizados en una tabla maestra y todos quedaron listos para síntesis. Todos fueron clasificados con riesgo moderado de sesgo, por lo que la literatura se utiliza como apoyo metodológico y contextual, no como evidencia causal definitiva.

### 5.1 Diseños identificados

- Measurement/validation: 4 estudio(s).
- Multilevel/comparative: 3 estudio(s).
- Longitudinal/cohort: 2 estudio(s).
- International assessment analysis: 2 estudio(s).
- Panel/fixed-effects: 1 estudio(s).

### 5.2 Fuentes de datos identificadas

- PISA: 7 estudio(s).
- Survey data: 2 estudio(s).
- European Social Survey: 1 estudio(s).
- Other/constructed dataset: 1 estudio(s).
- PISA; LSAY: 1 estudio(s).

### 5.3 Calidad metodológica

- Diseño muestral considerado adecuado: 12 de 12 estudios.
- Tratamiento de datos faltantes considerado adecuado: 8 de 12.
- Validación considerada adecuada: 12 de 12.
- Afirmaciones causales respaldadas: 0 de 12.
- Código disponible identificado: 0 de 12.

El predominio de diseños observacionales y la ausencia de respaldo causal refuerzan la decisión de interpretar el modelo ACLF como una herramienta predictiva y exploratoria.

## 6. Tabla de los candidatos finales

| ID | Título | Diseño | Datos | Relevancia principal |
|---|---|---|---|---|
| SCR-0009 | Within-School Wealth Inequality and Learning Achievement: Evidence From PISA | Panel/fixed-effects | PISA | Methodologically adequate multi-cycle PISA inequality study |
| SCR-0010 | Performance decline in a low‐stakes test at age 15 and educational attainment at age 25: Cross‐country longitudinal evidence | Longitudinal/cohort | PISA | Methodologically adequate longitudinal multi-country study |
| SCR-0012 | Факторы, влияющие на достижения Казахстанских учащихся в международном исследовании PISA-2018 | Multilevel/comparative | PISA | Adequate multilevel PISA analysis using plausible values and complex sample design |
| SCR-0017 | Education for Growth: Why and For Whom? | Measurement/validation | Survey data | Strong methodological analysis of measurement error and heterogeneity in education-growth models |
| SCR-0021 | Who Makes It Into PISA? Understanding the Impact of PISA Sample Eligibility Using Turkey as a Case Study (PISA 2003 - PISA 2012) | International assessment analysis | PISA | Strong analysis of PISA eligibility, coverage and changing sample composition |
| SCR-0027 | The geography of tertiary education inequality in Europe: a within- and cross-country analysis over time | Multilevel/comparative | European Social Survey | Large harmonized European Social Survey analysis with multilevel regional and country comparisons |
| SCR-0028 | The Century of Education | Measurement/validation | Other/constructed dataset | Strong long-run attainment database with explicit measurement-error and robustness analysis |
| SCR-0029 | Does schooling actually perpetuate educational inequality in mathematics performance? A validity question on the measures of opportunity to learn in PISA | Measurement/validation | PISA | Strong methodological assessment of construct validity in PISA opportunity-to-learn measurement |
| SCR-0038 | Improving the estimation of educational attainment: New methods for assessing average years of schooling from binned data | Measurement/validation | Survey data | Strong out-of-sample validation of educational-attainment measurement methods |
| SCR-0039 | Differences in educational attainment by country of origin: Evidence from Australia | Longitudinal/cohort | PISA; LSAY | National longitudinal cohort analysis with sampling and attrition weights, probit models and robustness checks |
| SCR-0041 | The Intraclass Correlation Coefficient as a Measure of Educational Inequality: An Empirical Study with Data from PISA 2018 | Multilevel/comparative | PISA | Strong multilevel PISA analysis with plausible values, weights, missing-data treatment and meta-analysis |
| SCR-0044 | Socioeconomic Inequalities in Reading Literacy and Learning-Related Characteristics: Evidence from PISA 2022 | International assessment analysis | PISA | Technically adequate PISA 2022 comparative analysis with plausible values and replicate weights |

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
