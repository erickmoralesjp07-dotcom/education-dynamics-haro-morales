\# 1. Introduction



Education systems evolve through interactions among school access, accumulated educational attainment, literacy, learning outcomes, institutional capacity, and public expenditure. These dimensions do not change at the same speed. School participation and public expenditure may respond rapidly to policy or economic shocks, whereas adult educational attainment changes gradually as successive cohorts move through the education system. This difference creates a methodological challenge for educational planning: the variables used to describe an education system may be related, but they may not be equally predictable over time.



International open-data initiatives have made long-run comparative analysis increasingly feasible. The UNESCO Institute for Statistics provides country-level indicators covering participation, attainment, literacy, learning, teachers, equity, and education finance. These data support international monitoring, but the resulting panel is sparse, unbalanced, and characterized by irregular time intervals. Consequently, conventional models that assume complete annual observations may produce misleading estimates or use only a small and unrepresentative subset of countries.



Dynamic-system formulations offer a transparent way to represent changes in education indicators. A state vector can summarize the educational condition of a country at a given time, while state equations describe expected changes between observations. However, the availability of a mathematical formulation does not guarantee useful prediction. Annual changes in national education indicators are often small, noisy, and affected by measurement revisions, administrative processes, sampling uncertainty, and omitted contextual factors. Any proposed dynamic model must therefore be compared with a simple and demanding reference, such as predicting no change.



This study evaluates whether a compact educational state contains reproducible information for predicting subsequent national change. The primary state, denoted ACLF, contains four variables: the out-of-school rate for children of primary-school age (A), adult educational attainment at lower-secondary level or above (C), youth literacy (L), and government expenditure on education as a percentage of gross domestic product (F). The corresponding equations predict the annualized changes dA/dt, dC/dt, dL/dt, and dF/dt.



The analysis emphasizes out-of-sample performance rather than in-sample fit. Ridge regression, Elastic Net, Random Forest, and HistGradientBoosting are compared under country-grouped validation. The study also evaluates a zero-change baseline, leave-one-country-out stability, cluster bootstrap uncertainty, temporal-origin validation, sensitivity to future test windows, and coefficient-sign stability. Ecuador is examined as an exploratory national case.



The study addresses four research questions:



1\. Does the previous ACLF state predict subsequent changes better than a zero-change baseline?

2\. Are nonlinear machine-learning models more accurate than regularized linear models?

3\. Does any predictive signal remain under country-grouped and forward temporal validation?

4\. Are the direction and interpretation of the estimated coefficients stable across temporal specifications?



The contribution is methodological and planning-oriented. The purpose is not to estimate causal effects or produce high-confidence national forecasts. Instead, the study determines which educational dimensions contain a small but reproducible predictive signal, documents when that signal fails, and establishes a transparent validation framework for international education planning with open data.



\# 2. Literature Review and Research Gap



\## 2.1 Educational inequality and multilevel structure



Educational outcomes are structured at several levels. Students are nested within schools, regions, and national systems, while the relevant socioeconomic conditions operate at household, institutional, territorial, and policy levels (Bernardi et al., 2026; Yardım et al., 2024). Research based on PISA and other international assessments shows that inequality may appear both within and between schools and countries (Sempé & Esposito, 2025; Yang, 2026). Consequently, observations from the same country or system cannot be treated as fully independent.



This multilevel structure has direct implications for predictive validation. Randomly distributing country-year observations between training and testing sets would allow the same national context to appear in both samples. Such a procedure could overestimate generalization to countries not seen during model fitting. Country-grouped validation is therefore necessary when the intended interpretation concerns international transferability.



\## 2.2 Longitudinal change and accumulated attainment



Adult educational attainment is an accumulated stock. Changes reflect cohort replacement, completion patterns, demographic composition, migration, and historical expansion of education. This process differs from short-term administrative indicators, which may fluctuate rapidly. Longitudinal studies suggest that educational trajectories and later attainment depend on conditions observed years earlier, but predictive relationships are generally modest and context-dependent (Borgonovi et al., 2021; Parasnis & Swan, 2020).



The cumulative character of attainment suggests that dC/dt may be more predictable than short-run changes in expenditure, literacy, or exclusion. At the same time, persistence can create a strong zero-change baseline. A useful model must demonstrate improvement beyond this baseline, not merely produce a positive coefficient of determination in the training sample.



\## 2.3 International assessments and measurement uncertainty



International assessments require specialized treatment, including attention to sample eligibility and population coverage (Spaull, 2017). PISA and TIMSS use complex sampling designs, student and school weights, replicate weights, and plausible values. Analyses that ignore these elements may underestimate uncertainty or misinterpret achievement measures as directly observed scores. Construct validity also matters because survey indicators may combine the intended concept with response behavior, self-perception, cultural interpretation, or other irrelevant variation (Hansen & Strietholt, 2018).



These limitations influenced the present design. PISA evidence is used to motivate the methodological framework and interpret inequality, but PISA country-cycle estimates are not inserted into the ACLF state as ordinary error-free variables. A future extension should integrate plausible-value and sampling uncertainty explicitly.



\## 2.4 Measurement of educational attainment



Historical and cross-national attainment data contain missing observations, grouped categories, imputation, and measurement error (Friedman et al., 2018; Morrisson & Murtin, 2009). Methods for estimating average years of schooling from binned data show that alternative measurement procedures can materially affect comparative results (Friedman et al., 2018). Validation outside the estimation sample is therefore essential.



This literature supports two decisions in the present study. First, the model is evaluated predictively rather than justified solely by theoretical plausibility. Second, the conclusions focus on robustness across samples and time rather than on a single fitted coefficient matrix.



\## 2.5 Predictive validation in educational planning



Educational planning studies frequently use panel models, fixed effects, multilevel regression, efficiency models, or machine learning (Krueger & Lindahl, 2000; Bernardi et al., 2026). Yet model comparison is often weakened by non-nested feature selection, random observation-level splitting, insufficient baseline comparison, or limited temporal validation. Complex models may appear attractive without producing meaningful improvement over simple references.



The main research gap is therefore not the absence of models, but the limited evidence that compact national education states can predict future changes under strict validation. This study addresses that gap by combining country-grouped validation, nested tuning, zero-change comparison, temporal-origin testing, and sensitivity analysis.



\# 3. Data and Methods



\## 3.1 Data source and coverage



The study uses open national data from the UNESCO Institute for Statistics (UNESCO Institute for Statistics, 2026a, 2026b). The processed database contains 16,647 observations in long format, 7,467 country-year rows, and 222 countries or territories over the general period 1970 to 2025.



Eight indicators were selected according to conceptual relevance, international coverage, and temporal availability:



\- A: out-of-school rate for children of primary-school age.

\- C: adult educational attainment at lower-secondary level or above.

\- M: minimum mathematics proficiency at the end of primary education.

\- R: minimum reading proficiency at the end of primary education.

\- L: youth literacy rate for ages 15 to 24.

\- T: primary teachers with minimum required qualifications.

\- F: government expenditure on education as a percentage of GDP.

\- E: wealth parity in the primary out-of-school rate.



The complete eight-indicator panel is highly sparse. Only 10 country-year rows contain all eight indicators, while 1,233 rows contain at least four. The ACLF combination was therefore selected as the principal dynamic core because it provided the strongest joint temporal coverage while representing access, accumulated attainment, literacy, and finance.



Table 1 presents the selected indicators and their coverage.



\## 3.2 Dynamic state formulation



For country \\(i\\) at time \\(t\\), the educational state is defined as:



\\\[

\\mathbf{x}\_{i,t} =

\\begin{bmatrix}

A\_{i,t} \\\\

C\_{i,t} \\\\

L\_{i,t} \\\\

F\_{i,t}

\\end{bmatrix}.

\\]



For two observed years \\(t\_0\\) and \\(t\_1\\), with interval \\(\\Delta t = t\_1-t\_0\\), the annualized change for variable \\(V\\) is:



\\\[

\\frac{dV}{dt}

\\approx

\\frac{V\_{i,t\_1}-V\_{i,t\_0}}{\\Delta t}.

\\]



The linear predictive system is:



\\\[

\\frac{d\\mathbf{x}\_{i,t}}{dt}

=

\\mathbf{b}

\+

\\mathbf{B}\\mathbf{x}\_{i,t\_0}

\+

\\boldsymbol{\\varepsilon}\_{i,t},

\\]



where \\(\\mathbf{b}\\) is an intercept vector, \\(\\mathbf{B}\\) is the coefficient matrix, and \\(\\boldsymbol{\\varepsilon}\_{i,t}\\) represents unexplained change.



The full rates database contains 375 transitions from 84 countries between 1980 and 2024. The primary annual sample contains 207 consecutive transitions from 37 countries. The robustness sample contains 266 transitions from 49 countries with intervals of one or two years.



\## 3.3 Models



Ridge regression was used as the principal model because the ACLF predictors may be correlated and the sample is limited (Hoerl & Kennard, 1970). The Ridge objective is:



\\\[

\\widehat{\\boldsymbol{\\beta}}

=

\\arg\\min\_{\\boldsymbol{\\beta}}

\\left\[

\\sum\_{j=1}^{n}

\\left(

y\_j-\\beta\_0-\\mathbf{x}\_j^\\top\\boldsymbol{\\beta}

\\right)^2

\+

\\alpha

\\sum\_{k=1}^{p}\\beta\_k^2

\\right].

\\]



The benchmark additionally evaluated Elastic Net (Zou & Hastie, 2005), Random Forest (Breiman, 2001), and the following alternatives:



\- Elastic Net;

\- Random Forest;

\- HistGradientBoosting;

\- a training-mean baseline;

\- a zero-change baseline.



The zero-change prediction is:



\\\[

\\widehat{dV/dt}=0.

\\]



Performance was measured using RMSE, MAE, and out-of-sample \\(R^2\\). Improvement over zero change was calculated as:



\\\[

\\text{Gain}(\\%)

=

100

\\left(

\\frac{\\text{RMSE}\_{zero}-\\text{RMSE}\_{model}}

{\\text{RMSE}\_{zero}}

\\right).

\\]



Positive values indicate improvement over the zero-change baseline.



\## 3.4 Missing values and scaling



For each training split, missing predictor values were replaced using medians calculated only from the training data. Predictors were standardized using training means and standard deviations. The fitted transformations were then applied to the test set. This sequence prevents information from the test observations from influencing preprocessing.



\## 3.5 Country-grouped nested validation



The main global validation used GroupKFold with COUNTRY\_ID as the grouping variable. All transitions from a country were assigned exclusively to either training or testing within each outer fold. The implementation explicitly checked for overlap between training and test countries and stopped execution if any overlap was detected.



Hyperparameters were selected in an inner GroupKFold loop using only the outer training sample. Ridge penalties were evaluated from 0.01 to 10,000. Elastic Net penalties and mixing parameters were tuned similarly. Random Forest and HistGradientBoosting were fitted with fixed random seeds for reproducibility.



\## 3.6 Stability analyses



The dC/dt signal was examined through:



1\. leave-one-country-out analysis;

2\. 5,000-repetition cluster bootstrap by country;

3\. stability by decade;

4\. paired Ridge and Elastic Net comparison;

5\. temporal-origin validation;

6\. temporal-window sensitivity;

7\. coefficient-sign stability.



Temporal-origin validation trained on observations up to a selected origin year and tested only on subsequent years. Origins of 2009, 2014, and 2019 were examined when minimum sample requirements were met. Test windows of three, five, and seven years were evaluated. No future observation was used for training, imputation, scaling, or hyperparameter selection.



\## 3.7 Ecuador case study



Ecuador was analyzed separately using 15 observations from 2009 to 2023 and 14 consecutive annual transitions. Leave-one-out evaluation was used because the national series was small. The national analysis is exploratory and is not used to claim causal effects or broad forecasting capacity.



\## 3.8 Literature review



A structured literature review was conducted to identify empirical and methodological studies on educational dynamics, attainment, inequality, international assessments, validation, and prediction. Twenty-seven full texts were evaluated across two batches. Twelve studies were retained as final evidence candidates, 10 were included with major limitations, and five were excluded at full-text assessment. Five additional documents in the second batch could not be retrieved after an exhaustive search.



The final evidence set included longitudinal, multilevel, panel, international-assessment, and measurement-validation studies. All 12 final candidates were classified as having moderate risk of bias. The literature supports interpretation and methodological choices but is not treated as definitive causal evidence.



\# 4. Results



\## 4.1 Global predictive performance



Ridge did not provide a general improvement across all ACLF equations. For consecutive annual transitions, the RMSE gains relative to zero change were:



\- dA/dt: -1.18%;

\- dC/dt: 5.34%;

\- dF/dt: -0.36%;

\- dL/dt: -47.50%.



For intervals of up to two years, the corresponding gains were:



\- dA/dt: -1.16%;

\- dC/dt: 6.73%;

\- dF/dt: -0.68%;

\- dL/dt: -2.12%.



Only dC/dt improved over zero change in both global samples. Figure 1 summarizes these comparisons, and Table 2 reports the main predictive results.



\## 4.2 Benchmark for adult educational attainment



For consecutive transitions, Ridge achieved an RMSE of 2.8157 and a 5.37% improvement over zero change. Elastic Net achieved an RMSE of 2.8168 and a 5.33% improvement. The practical difference was negligible.



For intervals of up to two years, Elastic Net obtained an RMSE of 2.5447 and a 6.84% improvement, while Ridge obtained an RMSE of 2.5480 and a 6.72% improvement.



Random Forest and HistGradientBoosting did not produce a consistent advantage. Ridge was the nominal winner for annual transitions, and Elastic Net was the nominal winner for intervals of up to two years. However, paired bootstrap analysis indicated practical equivalence between the two regularized linear models under a provisional 1% RMSE margin.



Figure 2 presents the nested benchmark for dC/dt.



\## 4.3 Country and bootstrap stability



The dC/dt improvement remained positive after individually removing each country. For annual transitions, the leave-one-country-out gain ranged from 4.80% to 6.17%. For intervals of up to two years, the range was 6.21% to 7.74%.



The country-cluster bootstrap 95% interval was 3.00% to 10.01% for annual transitions and 4.13% to 12.12% for intervals of up to two years. The gain was positive in all evaluated decades.



Table 3 presents the stability results.



\## 4.4 Temporal-origin validation



Five valid forward temporal tests were obtained. Ridge outperformed zero change in four of the five tests.



For consecutive annual transitions:



\- origin 2014, test period 2015-2019: -31.74%;

\- origin 2019, test period 2020-2024: 5.60%.



For intervals of up to two years:



\- origin 2009, test period 2010-2014: 6.57%;

\- origin 2014, test period 2015-2019: 6.98%;

\- origin 2019, test period 2020-2024: 7.34%.



The negative annual result for the 2014 origin demonstrates that the signal is not uniformly stable. However, all temporal tests allowing intervals of up to two years were positive.



\## 4.5 Sensitivity to temporal windows



Fifteen combinations of sample, origin, and future test-window length were evaluated. Twelve produced positive gains.



For strictly annual transitions, three of six tests were positive. Mean gains varied by window and were affected by the negative 2014-origin results.



For intervals of up to two years, all nine tests were positive:



\- three-year windows: gains from 4.43% to 13.28%;

\- five-year windows: gains from 6.57% to 7.34%;

\- seven-year windows: gains from 7.18% to 8.54%.



The signal is therefore more robust when moderate temporal gaps are accepted.



\## 4.6 Coefficient-sign stability



Twenty predictor coefficients were evaluated across five temporal models. The overall agreement with the signs estimated in the complete samples was 80%.



\- Previous adult attainment retained a negative coefficient in five of five models.

\- Previous youth literacy retained a positive coefficient in five of five models.

\- Previous education expenditure retained a positive coefficient in four of five models.

\- Previous exclusion retained a negative coefficient in two of five models.



The stable negative coefficient for previous attainment may reflect convergence, ceiling constraints, or slower growth among countries already at higher attainment levels. The positive literacy coefficient may indicate that accumulated foundational human capital helps predict subsequent attainment growth. These associations are predictive and should not be interpreted as causal effects.



\## 4.7 Ecuador



For Ecuador, Ridge improved over zero change only for dC/dt. The Ecuadorian dC/dt gain was 11.52%, while the remaining equations showed negative gains. Figure 3 presents the observed ACLF trajectories, and Figure 4 compares observed and predicted annual rates.



The Ecuador result is consistent with the global pattern, but the small national sample limits inference. The case provides an exploratory illustration rather than independent confirmation of a general forecasting model.



\# 5. Discussion



The primary result is selective rather than general. A compact ACLF state does not reliably predict all dimensions of national educational change. Access, literacy, and education finance were not consistently more predictable than a zero-change assumption. Adult educational attainment was the exception.



This result is substantively plausible. Adult attainment is a slowly changing stock shaped by accumulated schooling histories and cohort replacement. The previous educational state may therefore contain more useful information for dC/dt than for short-run administrative or financial changes. Nevertheless, the predictive gain remained modest, and the out-of-sample \\(R^2\\) values were low.



The comparison of models is also informative. Random Forest and HistGradientBoosting did not consistently outperform Ridge or Elastic Net. This suggests that, under the present sample size and predictor structure, additional nonlinear flexibility mainly increases variance rather than capturing a stronger reproducible signal. For planning purposes, the lower-complexity models provide better transparency and comparable or superior accuracy.



The temporal results qualify the overall conclusion. The signal was positive in four of five temporal-origin tests and 12 of 15 temporal-window tests, but annual transitions were positive in only three of six sensitivity tests. The strongest consistency occurred when intervals of up to two years were permitted. This may reflect the low signal-to-noise ratio of annual national changes, reporting irregularities, and delayed measurement.



Coefficient stability provides further context. Previous adult attainment and youth literacy retained the same signs across all temporal models. In contrast, access and expenditure coefficients changed signs. The unstable coefficients should not be used to support direct policy mechanisms. Even the stable coefficients represent predictive associations, not intervention effects.



The literature review supports this cautious interpretation. Educational inequality is multilevel, attainment measures contain uncertainty, and international assessments require specialized variance treatment. These conditions make simple causal narratives inappropriate (Ospanova et al., 2023; Hansen & Strietholt, 2018). The present framework is more defensible as a monitoring and model-comparison tool.



\# 6. Planning and Policy Implications



The results have four implications for socio-economic and educational planning.



First, planning systems should compare complex models with simple baselines. A model that does not outperform zero change should not be presented as operationally useful, regardless of in-sample fit.



Second, adult educational attainment may provide a more stable planning target than high-frequency changes in expenditure or access. The dC/dt equation may support scenario monitoring, identification of unusual deviations, and comparison of expected versus observed progress.



Third, one-year changes should be interpreted cautiously. The superior stability of intervals up to two years suggests that annual monitoring may contain substantial noise. Planning dashboards can retain annual reporting while also presenting smoothed or multi-year change indicators.



Fourth, model outputs should not be converted directly into policy prescriptions. The framework does not estimate the causal effect of expenditure, literacy, or exclusion on attainment. Policy evaluation requires stronger research designs, institutional context, and intervention-specific evidence.



\# 7. Limitations



The study has several limitations.



1\. The international panel is sparse and unbalanced.

2\. Joint coverage of the full eight-indicator system is limited.

3\. The predictive improvement for dC/dt is modest.

4\. Strictly annual temporal results are not stable across all periods.

5\. The 2014-origin annual test produced a substantial negative gain.

6\. Coefficient signs for access and expenditure were unstable.

7\. The study is predictive and associational, not causal.

8\. National aggregates conceal within-country inequality.

9\. Measurement revisions and reporting practices may affect observed changes.

10\. PISA uncertainty has not yet been integrated directly into the ACLF state.

11\. The Ecuador analysis contains only 14 annual transitions.

12\. Five full-text documents from the second review batch could not be retrieved.

13\. All final literature candidates were classified as having moderate risk of bias.

14\. The practical-equivalence margin between Ridge and Elastic Net is provisional.

15\. External validation with a separate international database remains necessary.



\# 8. Conclusions



This study tested whether a compact national educational state can predict subsequent changes using open UNESCO UIS data and strict out-of-sample validation. The general ACLF system did not outperform zero change across all equations.



Adult educational attainment was the only dimension with a small but reproducible predictive signal. Ridge improved RMSE by approximately 5% to 7% in the global grouped analyses. The signal was positive in four of five temporal-origin tests and all nine temporal-window tests that permitted intervals of up to two years.



The result is useful but limited. Regularized linear models are preferable to more complex nonlinear alternatives for the present data, yet the predictive capacity remains modest. The framework should be used for comparative monitoring, robustness assessment, and hypothesis generation, not causal attribution or high-confidence forecasting.



The broader methodological conclusion is that educational dynamic models must be evaluated against strong simple baselines, separated by country, and tested forward in time. Under these conditions, only selected educational states may retain reproducible predictive information.



\# 9. Data and Code Availability



The analysis uses open data from the UNESCO Institute for Statistics. Processing, modeling, validation, literature-review outputs, diagnostic files, tables, and figures are documented in the project repository:



https://github.com/erickmoralesjp07-dotcom/education-dynamics-haro-morales



Large raw and processed data files may be excluded from Git because of file size and licensing or redistribution considerations. The repository includes environment specifications, scripts, methodological decisions, validation diagnostics, and instructions required to reconstruct the analytical workflow from the original source data.



\# 10. Author Contributions



Erick Jair Morales Portilla: Conceptualization, data curation, formal analysis, investigation, methodology, software, validation, visualization, and writing.



\[Add the real contributions of the remaining authors using the CRediT taxonomy.]



\# 11. Declaration of Competing Interests



The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.



\# 12. Funding



\[State the real funding source. If there was no external funding, write: “This research received no external funding.”]



\# 13. Acknowledgements



\[Add only real acknowledgements, including academic supervision or institutional support.]



\# 14. References



\\



Bernardi, F., Lievore, I., \& Valdes, M. T. (2026). The geography of tertiary education inequality in Europe: A within- and cross-country analysis over time. \*European Societies\*, 1–24. https://doi.org/10.1162/euso.a.37



Borgonovi, F., Ferrara, A., \& Piacentini, M. (2021). Performance decline in a low-stakes test at age 15 and educational attainment at age 25: Cross-country longitudinal evidence. \*Journal of Adolescence, 92\*(1), 114–125. https://doi.org/10.1016/j.adolescence.2021.08.011



Breiman, L. (2001). Random forests. \*Machine Learning, 45\*, 5–32. https://doi.org/10.1023/A:1010933404324



Friedman, J. R., Graetz, N., \& Gakidou, E. E. (2018). Improving the estimation of educational attainment: New methods for assessing average years of schooling from binned data. \*PLOS ONE, 13\*(11), e0208019. https://doi.org/10.1371/journal.pone.0208019



Hansen, K. Y., \& Strietholt, R. (2018). Does schooling actually perpetuate educational inequality in mathematics performance? A validity question on the measures of opportunity to learn in PISA. \*ZDM, 50\*(4), 643–658. https://doi.org/10.1007/s11858-018-0935-3



Hoerl, A. E., \& Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. \*Technometrics, 12\*(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634



Krueger, A. B., \& Lindahl, M. (2000). \*Education for growth: Why and for whom?\* NBER Working Paper No. 7591. National Bureau of Economic Research. https://doi.org/10.3386/w7591



Morrisson, C., \& Murtin, F. (2009). The century of education. \*Journal of Human Capital, 3\*(1), 1–42. https://doi.org/10.1086/600102



Ospanova, U., Nurumov, K., \& Atanaeva, M. (2023). Factors influencing the achievements of Kazakhstani students in the PISA-2018 international study. \*Journal of Educational Sciences, 76\*(3), 116–126. https://doi.org/10.26577/jes.2023.v76.i3.010



Parasnis, J., \& Swan, J. (2020). Differences in educational attainment by country of origin: Evidence from Australia. \*Migration Studies, 8\*(4), 530–553. https://doi.org/10.1093/migration/mnz008



Sempé, L. N., \& Esposito, L. (2025). Within-school wealth inequality and learning achievement: Evidence from PISA. \*Review of Income and Wealth, 71\*(4), e70040. https://doi.org/10.1111/roiw.70040



Spaull, N. (2017). \*Who makes it into PISA? Understanding the impact of PISA sample eligibility using Turkey as a case study (PISA 2003–PISA 2012).\* OECD Education Working Papers No. 154. OECD Publishing. https://doi.org/10.1787/41d175fc-en



UNESCO Institute for Statistics. (2026). \*Background information on statistics in the UIS database: 2026 data refresh.\* UNESCO Institute for Statistics. https://download.uis.unesco.org/bdds/202602/UIS%20-%20Data%20Release%20Background%20Doc%20-EN-Feb%202026.pdf



UNESCO Institute for Statistics. (2026). \*UIS Data Browser: Bulk data download.\* https://databrowser.uis.unesco.org/resources/bulk



Yardım, Y., Cüvitoğlu, G., \& Aydın, B. (2024). The intraclass correlation coefficient as a measure of educational inequality: An empirical study with data from PISA 2018. \*Research Square\* \[Preprint]. https://doi.org/10.21203/rs.3.rs-4066275/v1



Yang, R. (2026). Socioeconomic inequalities in reading literacy and learning-related characteristics: Evidence from PISA 2022. \*Journal of Education Research, 7\*(3), 2731–2742. https://doi.org/10.37985/jer.v7i3.3480



Zou, H., \& Hastie, T. (2005). Regularization and variable selection via the Elastic Net. \*Journal of the Royal Statistical Society: Series B, 67\*(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

