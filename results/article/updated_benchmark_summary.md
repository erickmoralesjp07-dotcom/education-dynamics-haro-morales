# Updated Scientific Benchmark Summary

Generated: 2026-11-12T10:01:08.304285-05:00

## 1. Current study scope

The project evaluates whether a structured educational dynamics model can predict changes in access, completion, youth literacy, and education finance using UNESCO UIS data.

## 2. Data coverage

- Selected indicators: 8.
- Long-format observations: 16,647.
- Country-year rows: 7,467.
- Countries or territories: 222.
- General period: 1970-2025.
- Complete country-year rows: 10.
- Rows with at least four indicators: 1,233.

## 3. Selected indicators

| Variable | Dimension | Indicator | Direction | Countries | Period |
|---|---|---|---|---:|---|
| A | access | Out-of-school rate for children of primary school age, both sexes (%) | negative | 217 | 1970-2025 |
| C | completion | Educational attainment rate, completed lower secondary education or higher, population 25+ years, both sexes (%) | positive | 211 | 1970-2024 |
| M | mathematics_learning | Proportion of students at the end of primary education achieving at least a minimum proficiency level in mathematics, both sexes (%) | positive | 133 | 1995-2023 |
| R | reading_learning | Proportion of students at the end of primary education achieving at least a minimum proficiency level in reading, both sexes (%) | positive | 123 | 2001-2023 |
| L | youth_literacy | Youth literacy rate, population 15-24 years, both sexes (%) | positive | 182 | 1970-2024 |
| T | qualified_teachers | Proportion of teachers with the minimum required qualifications in primary education, both sexes (%) | positive | 162 | 1998-2025 |
| F | education_finance | Government expenditure on education as a percentage of GDP (%) | positive | 206 | 1970-2025 |
| E | wealth_equity | Out-of-school rate for children of primary school age, adjusted wealth parity index (household survey data) (WPIA) | contextual | 122 | 2000-2025 |

## 4. ACLF model

- A: out-of-school rate at primary age.
- C: lower-secondary completion or higher among the population aged 25 years or more.
- L: youth literacy rate, ages 15-24.
- F: government education expenditure as a percentage of GDP.

The linear system predicts dA/dt, dC/dt, dL/dt, and dF/dt from the previous ACLF state.

## 5. Ridge compared with zero-change baseline

| Level | Sample | Equation | Ridge RMSE | Zero RMSE | Gain (%) | Result |
|---|---|---|---:|---:|---:|---|
| Global Ridge vs zero | consecutive | dA/dt | 1.5502 | 1.5322 | -1.18 | no_improvement |
| Global Ridge vs zero | consecutive | dC/dt | 2.8166 | 2.9756 | 5.34 | moderate |
| Global Ridge vs zero | consecutive | dF/dt | 0.3635 | 0.3622 | -0.36 | no_improvement |
| Global Ridge vs zero | consecutive | dL/dt | 1.4967 | 1.0147 | -47.50 | no_improvement |
| Global Ridge vs zero | gap_max_2 | dA/dt | 1.5667 | 1.5488 | -1.16 | no_improvement |
| Global Ridge vs zero | gap_max_2 | dC/dt | 2.5478 | 2.7315 | 6.73 | moderate |
| Global Ridge vs zero | gap_max_2 | dF/dt | 0.3663 | 0.3639 | -0.68 | no_improvement |
| Global Ridge vs zero | gap_max_2 | dL/dt | 1.2720 | 1.2456 | -2.12 | no_improvement |
| Ecuador Ridge vs zero | Ecuador annual series | dA/dt | 1.0623 | 1.0288 | -3.26 | no_improvement |
| Ecuador Ridge vs zero | Ecuador annual series | dC/dt | 1.3804 | 1.5602 | 11.52 | substantial |
| Ecuador Ridge vs zero | Ecuador annual series | dF/dt | 0.3092 | 0.2914 | -6.12 | no_improvement |
| Ecuador Ridge vs zero | Ecuador annual series | dL/dt | 0.3385 | 0.3280 | -3.21 | no_improvement |

Only the completion-rate equation, dC/dt, consistently outperformed the zero-change baseline.

## 6. Stability of the adult educational-attainment signal

- consecutive: overall RMSE reduction of 5.34%; leave-one-country-out range 4.80% to 6.17%; cluster-bootstrap 95% interval 3.00% to 10.01%.
- gap_max_2: overall RMSE reduction of 6.73%; leave-one-country-out range 6.21% to 7.74%; cluster-bootstrap 95% interval 4.13% to 12.12%.

The advantage remained positive after removing each country and in all evaluated decades.

## 7. Nested machine-learning benchmark

| Sample | Model | RMSE | MAE | R2 | Gain vs zero (%) |
|---|---|---:|---:|---:|---:|
| consecutive | ridge | 2.8157 | 1.6131 | 0.0136 | 5.37 |
| consecutive | elastic_net | 2.8168 | 1.6169 | 0.0127 | 5.33 |
| consecutive | training_mean_baseline | 2.8376 | 1.6491 | -0.0019 | 4.64 |
| consecutive | random_forest | 2.9162 | 1.7689 | -0.0581 | 2.00 |
| consecutive | zero_baseline | 2.9756 | 1.8765 | -0.1017 | 0.00 |
| consecutive | hist_gradient_boosting | 3.0258 | 1.9711 | -0.1392 | -1.69 |
| gap_max_2 | elastic_net | 2.5447 | 1.4472 | 0.0325 | 6.84 |
| gap_max_2 | ridge | 2.5480 | 1.4579 | 0.0300 | 6.72 |
| gap_max_2 | random_forest | 2.5666 | 1.5128 | 0.0159 | 6.04 |
| gap_max_2 | training_mean_baseline | 2.5907 | 1.5055 | -0.0027 | 5.16 |
| gap_max_2 | zero_baseline | 2.7315 | 1.7417 | -0.1147 | 0.00 |
| gap_max_2 | hist_gradient_boosting | 2.7601 | 1.7937 | -0.1381 | -1.05 |

Ridge was the nominal winner for annual transitions. Elastic Net was the nominal winner for transitions of up to two years.

## 8. Ridge-Elastic Net equivalence

- consecutive: absolute relative RMSE difference 0.042%; bootstrap interval -0.415% to 0.386%; practical equivalence probability 99.86%.
- gap_max_2: absolute relative RMSE difference 0.130%; bootstrap interval -0.074% to 0.502%; practical equivalence probability 99.96%.

Ridge and Elastic Net are practically equivalent under the provisional 1% RMSE equivalence margin.

## 9. Current scientific interpretation

The ACLF state contains a small but stable predictive signal for the evolution of educational completion. The signal is captured by low-complexity regularized linear models. Random Forest and HistGradientBoosting did not provide a consistent improvement.

The remaining ACLF equations did not outperform a zero-change prediction. Therefore, the available evidence does not support the general superiority of the linear dynamic system.

## 10. Limitations

- The panel is sparse and unbalanced.
- Only a small number of country-year rows contain all eight selected indicators.
- The adult educational-attainment prediction R2 remains low.
- The bootstrap reuses previously generated out-of-sample predictions.
- The analysis is predictive and associational, not causal.
- The 1% equivalence margin is provisional.
- PISA has not yet been integrated into the ACLF benchmark.

## 11. Next scientific steps

1. Evaluate sensitivity to alternative equivalence margins.
2. Examine coefficients and predictor stability for completion.
3. Evaluate temporal-origin validation.
4. Test a state-space formulation.
5. Integrate PISA country-cycle estimates with sampling uncertainty.
6. Expand the peer-reviewed literature review.
