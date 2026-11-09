# Consolidated results for the educational dynamics article

Generated: 2026-11-09T15:29:13.631343-05:00

## 1. Data source and processing

- Original national records processed: 1,460,179.
- Unique indicators in the national database: 2,941.
- Countries or territories represented: 236.
- Source period: 1970 to 2025.

## 2. Analytical panel

- Complete ACLF country-year observations: 487.
- Countries or territories in the ACLF panel: 112.
- Total temporal transitions: 375.
- Consecutive annual transitions: 207.

## 3. Selected indicators

- A: Tasa de exclusion escolar en edad de primaria (ROFST.1.CP).
- C: Poblacion adulta que completo secundaria inferior o superior (EA.2T8.AG25T99).
- E: Paridad por riqueza en exclusion escolar de primaria (ROFST.H.1.WPIA).
- F: Gasto publico en educacion como porcentaje del PIB (XGDP.FSGOV).
- L: Tasa de alfabetizacion juvenil de 15 a 24 anios (LR.AG15T24).
- M: Competencia minima en matematicas al finalizar primaria (MATH.PRIMARY).
- R: Competencia minima en lectura al finalizar primaria (READ.PRIMARY).
- T: Docentes de primaria con cualificaciones minimas (TRTP.1).

## 4. Best validated global model by equation

- dA/dt: country_effects; R2 = -0.0152, RMSE = 1.5291, alpha = 10000.0000.
- dC/dt: global_linear; R2 = 0.0123, RMSE = 2.8175, alpha = 10.0000.
- dF/dt: zero_baseline; R2 = -0.0006, RMSE = 0.3622, alpha = not applicable.
- dL/dt: zero_baseline; R2 = -0.0025, RMSE = 1.0147, alpha = not applicable.

## 5. Ecuador case study

- Observations: 15.
- Period: 2009 to 2023.
- Consecutive transitions: 14.
- Trend A: 0.1349 units per year; trend R2 = 0.3272.
- Trend C: 0.8108 units per year; trend R2 = 0.8871.
- Trend L: 0.0320 units per year; trend R2 = 0.2385.
- Trend F: -0.0769 units per year; trend R2 = 0.5102.

## 6. Data quality

- Values outside expected ranges: 0.
- Level observations flagged by IQR: 149.
- Temporal rates flagged by IQR: 192.
- Flagged observations were retained in the original database and evaluated through sensitivity analysis.

## 7. Main methodological decisions

1. Use official UIS bulk data: La descarga masiva de UNESCO UIS permite procesamiento reproducible y conserva la base oficial completa.
2. Keep raw data outside Git: Los archivos originales y Parquet son grandes y pueden reconstruirse mediante scripts.
3. Select indicators by concept and coverage: La selección combina relevancia educativa, cobertura internacional y disponibilidad temporal.
4. Use ACLF as the primary dynamic core: La combinación A-C-L-F produjo la mayor cobertura conjunta entre todas las combinaciones de cuatro variables.
5. Use actual time intervals: No todas las observaciones están separadas por un año. Las tasas se dividieron por DELTA_T.
6. Use consecutive transitions as primary sample: Las transiciones anuales reducen la ambigüedad al aproximar derivadas temporales.
7. Use gaps up to two years for robustness: Amplía la muestra sin mezclar intervalos temporales demasiado extensos.
8. Compare against zero-change baseline: Las tasas anuales suelen concentrarse cerca de cero. Un modelo debe superar esa referencia.
9. Group cross-validation by country or year: Evita repartir indiscriminadamente observaciones relacionadas entre entrenamiento y validación.
10. Treat Ecuador model as exploratory: Ecuador tiene una serie anual limpia, pero solo 14 transiciones, insuficientes para inferencia causal o modelos complejos.
11. Do not delete IQR outliers automatically: Una observación extrema puede representar una situación educativa real, no un error.
12. Report weak predictive performance: Los resultados negativos son evidencia científica y no deben ocultarse ni ajustarse artificialmente.

## 8. Overall interpretation

The global linear system, country effects, year effects, and nonlinear specifications showed limited predictive performance for annual educational changes.
The weak performance remained after sensitivity analyses, which suggests that the result is not explained only by extreme observations.
The estimated equations must therefore be interpreted as exploratory dynamic descriptions and not as causal models or reliable forecasting systems.
