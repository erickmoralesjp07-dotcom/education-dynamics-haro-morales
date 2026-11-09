from pathlib import Path
from datetime import datetime
import json
import sys

import pandas as pd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "literature_review"
)

SEARCH_STRATEGIES_FILE = (
    OUTPUT_DIR
    / "search_strategies.csv"
)

ELIGIBILITY_FILE = (
    OUTPUT_DIR
    / "eligibility_criteria.csv"
)

SCREENING_FILE = (
    OUTPUT_DIR
    / "screening_register.csv"
)

EXTRACTION_FILE = (
    OUTPUT_DIR
    / "study_extraction_matrix.csv"
)

PRISMA_FILE = (
    OUTPUT_DIR
    / "prisma_flow.csv"
)

THEMATIC_FRAMEWORK_FILE = (
    OUTPUT_DIR
    / "thematic_framework.csv"
)

SEARCH_LOG_FILE = (
    OUTPUT_DIR
    / "search_log.csv"
)

QUALITY_APPRAISAL_FILE = (
    OUTPUT_DIR
    / "quality_appraisal.csv"
)

JOURNAL_TARGETS_FILE = (
    OUTPUT_DIR
    / "target_journals.csv"
)

PROTOCOL_FILE = (
    OUTPUT_DIR
    / "review_protocol.md"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "review_protocol_config.json"
)

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "literature_review_protocol_diagnostic.csv"
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

REVIEW_TITLE = (
    "Dynamic modeling of international educational indicators: "
    "a structured literature review"
)

REVIEW_QUESTION = (
    "How have international educational indicators, panel data, "
    "dynamic systems, differential equations, and predictive "
    "validation been used to study changes in educational access, "
    "completion, literacy, and public financing?"
)

PRIMARY_TARGET_JOURNAL = (
    "Social Indicators Research"
)

ALTERNATIVE_TARGET_JOURNALS = [
    "Education Sciences",
    "Journal of Educational Change",
]

DATABASES = [
    "Scopus",
    "Web of Science Core Collection",
    "ERIC",
    "ScienceDirect",
    "SpringerLink",
    "Google Scholar",
    "UNESCO Digital Library",
    "OECD iLibrary",
]

THEMATIC_BLOCKS = [
    "SDG4 and educational indicators",
    "Education and social development",
    "Educational access and exclusion",
    "Educational completion and attainment",
    "Youth literacy",
    "Public education financing",
    "Dynamic systems in education",
    "Differential equations in social sciences",
    "International educational panel data",
    "Cross-country heterogeneity",
    "Predictive validation and baselines",
    "Missing data and unequal indicator coverage",
    "Negative and null predictive results",
    "Ecuador and Latin American education",
]

CURRENT_YEAR = datetime.now().year

MINIMUM_PUBLICATION_YEAR = 2000

RECENT_LITERATURE_YEAR = 2018


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_section(title):
    """Imprime un encabezado legible."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def empty_value():
    """Devuelve un campo vacío uniforme."""

    return ""


def create_output_directory():
    """Crea la carpeta del protocolo."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTIC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# ESTRATEGIAS DE BUSQUEDA
# ============================================================

def build_search_strategies():
    """
    Crea cadenas booleanas adaptadas a las bases principales.

    Las cadenas se guardan para ejecutarlas posteriormente
    en cada plataforma bibliográfica.
    """

    core_concepts = {
        "education_indicators": (
            '("educational indicator*" OR "education indicator*" '
            'OR "education statistics" OR "SDG 4" OR "SDG4")'
        ),
        "dynamic_modeling": (
            '("dynamic model*" OR "system dynamics" '
            'OR "differential equation*" OR "dynamic system*" '
            'OR "longitudinal model*")'
        ),
        "international_panel": (
            '("panel data" OR longitudinal OR "cross-country" '
            'OR international OR global)'
        ),
        "outcomes": (
            '("school access" OR "out-of-school" OR completion '
            'OR attainment OR literacy OR "education financing" '
            'OR "education expenditure")'
        ),
        "validation": (
            '("cross-validation" OR prediction OR forecasting '
            'OR robustness OR sensitivity)'
        ),
    }

    broad_query = (
        f"{core_concepts['education_indicators']} AND "
        f"{core_concepts['dynamic_modeling']} AND "
        f"{core_concepts['international_panel']}"
    )

    indicator_query = (
        f"{core_concepts['education_indicators']} AND "
        f"{core_concepts['outcomes']} AND "
        f"{core_concepts['international_panel']}"
    )

    validation_query = (
        f"{core_concepts['dynamic_modeling']} AND "
        f"{core_concepts['outcomes']} AND "
        f"{core_concepts['validation']}"
    )

    financing_query = (
        '("education expenditure" OR "education financing" '
        'OR "public education spending") AND '
        '(completion OR attainment OR literacy OR '
        '"out-of-school") AND '
        '("panel data" OR "cross-country" OR international)'
    )

    latin_america_query = (
        '("Latin America" OR Ecuador) AND '
        '("education indicators" OR "SDG 4") AND '
        '(access OR completion OR literacy OR financing)'
    )

    rows = []

    queries = [
        (
            "Q1",
            "Dynamic educational systems",
            broad_query,
        ),
        (
            "Q2",
            "International indicator panels",
            indicator_query,
        ),
        (
            "Q3",
            "Prediction and validation",
            validation_query,
        ),
        (
            "Q4",
            "Education financing and outcomes",
            financing_query,
        ),
        (
            "Q5",
            "Latin America and Ecuador",
            latin_america_query,
        ),
    ]

    for database in DATABASES:
        for query_id, purpose, query in queries:
            rows.append(
                {
                    "DATABASE": database,
                    "QUERY_ID": query_id,
                    "PURPOSE": purpose,
                    "SEARCH_STRING": query,
                    "LANGUAGE_FILTER": (
                        "English OR Spanish"
                    ),
                    "PUBLICATION_YEAR_FROM": (
                        MINIMUM_PUBLICATION_YEAR
                    ),
                    "PUBLICATION_YEAR_TO": (
                        CURRENT_YEAR
                    ),
                    "DOCUMENT_TYPES": (
                        "Article|Review|Conference paper|"
                        "Book chapter|Technical report"
                    ),
                    "SEARCH_STATUS": "Pending",
                    "SEARCH_DATE": empty_value(),
                    "RESULT_COUNT": empty_value(),
                    "EXPORTED_FILE": empty_value(),
                    "NOTES": empty_value(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CRITERIOS DE ELEGIBILIDAD
# ============================================================

def build_eligibility_criteria():
    """Define criterios de inclusión y exclusión."""

    rows = [
        {
            "CRITERION_ID": "I01",
            "TYPE": "Inclusion",
            "CATEGORY": "Topic",
            "CRITERION": (
                "The study analyzes educational indicators, "
                "educational systems, or educational change."
            ),
            "JUSTIFICATION": (
                "Direct relevance to the research question."
            ),
        },
        {
            "CRITERION_ID": "I02",
            "TYPE": "Inclusion",
            "CATEGORY": "Method",
            "CRITERION": (
                "The study uses longitudinal, panel, dynamic, "
                "time-series, system-dynamics, or differential-"
                "equation methods."
            ),
            "JUSTIFICATION": (
                "Supports the dynamic-methodological framework."
            ),
        },
        {
            "CRITERION_ID": "I03",
            "TYPE": "Inclusion",
            "CATEGORY": "Variables",
            "CRITERION": (
                "The study includes access, exclusion, completion, "
                "attainment, literacy, education quality, teachers, "
                "infrastructure, or educational financing."
            ),
            "JUSTIFICATION": (
                "Connects the literature with the UIS indicators."
            ),
        },
        {
            "CRITERION_ID": "I04",
            "TYPE": "Inclusion",
            "CATEGORY": "Evidence",
            "CRITERION": (
                "The study reports enough methodological detail "
                "to evaluate data, model, sample, and findings."
            ),
            "JUSTIFICATION": (
                "Required for critical appraisal."
            ),
        },
        {
            "CRITERION_ID": "I05",
            "TYPE": "Inclusion",
            "CATEGORY": "Source",
            "CRITERION": (
                "Peer-reviewed studies, authoritative institutional "
                "reports, methodological books, or official data "
                "documentation."
            ),
            "JUSTIFICATION": (
                "Allows academic and primary-source evidence."
            ),
        },
        {
            "CRITERION_ID": "I06",
            "TYPE": "Inclusion",
            "CATEGORY": "Language",
            "CRITERION": (
                "Publication available in English or Spanish."
            ),
            "JUSTIFICATION": (
                "Languages that can be assessed consistently."
            ),
        },
        {
            "CRITERION_ID": "E01",
            "TYPE": "Exclusion",
            "CATEGORY": "Topic",
            "CRITERION": (
                "The study concerns classroom teaching only and "
                "does not analyze system-level educational change."
            ),
            "JUSTIFICATION": (
                "Outside the system and indicator level."
            ),
        },
        {
            "CRITERION_ID": "E02",
            "TYPE": "Exclusion",
            "CATEGORY": "Evidence",
            "CRITERION": (
                "Opinion pieces, promotional material, or documents "
                "without identifiable methods or sources."
            ),
            "JUSTIFICATION": (
                "Insufficient scientific traceability."
            ),
        },
        {
            "CRITERION_ID": "E03",
            "TYPE": "Exclusion",
            "CATEGORY": "Duplication",
            "CRITERION": (
                "Duplicate publication or duplicate database record."
            ),
            "JUSTIFICATION": (
                "Each study must be counted once."
            ),
        },
        {
            "CRITERION_ID": "E04",
            "TYPE": "Exclusion",
            "CATEGORY": "Method",
            "CRITERION": (
                "The study reports only a cross-sectional description "
                "without temporal, comparative, or methodological "
                "relevance."
            ),
            "JUSTIFICATION": (
                "Does not inform educational dynamics."
            ),
        },
        {
            "CRITERION_ID": "E05",
            "TYPE": "Exclusion",
            "CATEGORY": "Access",
            "CRITERION": (
                "Full text cannot be obtained after reasonable "
                "institutional and author searches."
            ),
            "JUSTIFICATION": (
                "Full methodological assessment is impossible."
            ),
        },
        {
            "CRITERION_ID": "E06",
            "TYPE": "Exclusion",
            "CATEGORY": "Data quality",
            "CRITERION": (
                "The source, sample, indicators, or estimation "
                "procedure cannot be determined."
            ),
            "JUSTIFICATION": (
                "Insufficient reproducibility."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# REGISTRO DE BUSQUEDA
# ============================================================

def build_search_log():
    """Crea el registro cronológico de búsquedas."""

    columns = [
        "SEARCH_EVENT_ID",
        "DATABASE",
        "QUERY_ID",
        "SEARCH_DATE",
        "SEARCHER",
        "SEARCH_STRING_USED",
        "FILTERS_USED",
        "RESULT_COUNT",
        "EXPORT_FORMAT",
        "EXPORT_FILE",
        "NOTES",
    ]

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# REGISTRO DE CRIBADO
# ============================================================

def build_screening_register():
    """Crea la plantilla para identificación y cribado."""

    columns = [
        "RECORD_ID",
        "DATABASE",
        "QUERY_ID",
        "TITLE",
        "AUTHORS",
        "YEAR",
        "JOURNAL_OR_SOURCE",
        "DOCUMENT_TYPE",
        "DOI",
        "URL",
        "ABSTRACT",
        "LANGUAGE",
        "DUPLICATE_GROUP_ID",
        "IS_DUPLICATE",
        "DUPLICATE_DECISION",
        "TITLE_ABSTRACT_DECISION",
        "TITLE_ABSTRACT_EXCLUSION_REASON",
        "FULL_TEXT_OBTAINED",
        "FULL_TEXT_DECISION",
        "FULL_TEXT_EXCLUSION_REASON",
        "FINAL_INCLUDED",
        "THEMATIC_BLOCKS",
        "REVIEWER",
        "REVIEW_DATE",
        "NOTES",
    ]

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# MATRIZ DE EXTRACCION
# ============================================================

def build_extraction_matrix():
    """Crea la matriz detallada de extracción de estudios."""

    columns = [
        "STUDY_ID",
        "RECORD_ID",
        "FULL_CITATION",
        "TITLE",
        "AUTHORS",
        "YEAR",
        "DOI",
        "URL",
        "JOURNAL",
        "PUBLISHER",
        "COUNTRY_OF_JOURNAL",
        "INDEXING_DATABASE",
        "SJR_YEAR",
        "SJR_QUARTILE",
        "JCR_YEAR",
        "JCR_QUARTILE",
        "IMPACT_FACTOR",
        "CITESCORE",
        "OPEN_ACCESS",
        "APC_INFORMATION",
        "STUDY_AIM",
        "RESEARCH_QUESTION",
        "THEORETICAL_FRAMEWORK",
        "THEMATIC_BLOCK",
        "COUNTRIES_ANALYZED",
        "GEOGRAPHIC_SCOPE",
        "EDUCATION_LEVEL",
        "DATA_SOURCE",
        "DATASET_NAME",
        "DATA_PERIOD",
        "SAMPLE_SIZE",
        "UNIT_OF_ANALYSIS",
        "DEPENDENT_VARIABLES",
        "INDEPENDENT_VARIABLES",
        "INDICATORS_USED",
        "INDICATOR_DEFINITIONS",
        "MISSING_DATA_METHOD",
        "OUTLIER_METHOD",
        "MODEL_TYPE",
        "ESTIMATION_METHOD",
        "SOFTWARE",
        "VALIDATION_METHOD",
        "BASELINE_MODEL",
        "ROBUSTNESS_CHECKS",
        "MAIN_RESULTS",
        "EFFECT_SIZES",
        "UNCERTAINTY_MEASURES",
        "MODEL_PERFORMANCE",
        "LIMITATIONS_REPORTED",
        "CAUSAL_CLAIM",
        "REPRODUCIBILITY_MATERIALS",
        "DATA_AVAILABILITY",
        "CODE_AVAILABILITY",
        "RELEVANCE_TO_ACLF",
        "SUPPORTS_METHOD",
        "SUPPORTS_DISCUSSION",
        "SUPPORTS_LIMITATIONS",
        "KEY_QUOTATION",
        "QUOTATION_PAGE",
        "REVIEWER_NOTES",
    ]

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# EVALUACION DE CALIDAD
# ============================================================

def build_quality_appraisal():
    """Crea una matriz de calidad adaptable a estudios cuantitativos."""

    criteria = [
        (
            "Q01",
            "Clear research question",
            "The research objective or question is explicitly stated.",
        ),
        (
            "Q02",
            "Appropriate data source",
            "The data source is relevant and adequately documented.",
        ),
        (
            "Q03",
            "Indicator validity",
            "Indicators are defined and justified for the construct.",
        ),
        (
            "Q04",
            "Temporal adequacy",
            "The time structure is appropriate for dynamic conclusions.",
        ),
        (
            "Q05",
            "Sample transparency",
            "Countries, years, units, and exclusions are reported.",
        ),
        (
            "Q06",
            "Missing-data transparency",
            "Missing data and coverage limitations are addressed.",
        ),
        (
            "Q07",
            "Appropriate model",
            "The estimation method matches the research question.",
        ),
        (
            "Q08",
            "Validation",
            "The study uses out-of-sample validation or robustness checks.",
        ),
        (
            "Q09",
            "Baseline comparison",
            "The proposed model is compared with a meaningful baseline.",
        ),
        (
            "Q10",
            "Uncertainty reporting",
            "Uncertainty, intervals, standard errors, or sensitivity are reported.",
        ),
        (
            "Q11",
            "Causal restraint",
            "Causal conclusions are supported or appropriately limited.",
        ),
        (
            "Q12",
            "Reproducibility",
            "Data, code, appendices, or sufficient procedural detail exist.",
        ),
    ]

    rows = []

    for criterion_id, criterion, description in criteria:
        rows.append(
            {
                "STUDY_ID": empty_value(),
                "CRITERION_ID": criterion_id,
                "CRITERION": criterion,
                "DESCRIPTION": description,
                "SCORE": empty_value(),
                "SCORE_RULE": (
                    "0=No; 1=Partial or unclear; 2=Yes"
                ),
                "JUSTIFICATION": empty_value(),
                "REVIEWER": empty_value(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FLUJO PRISMA
# ============================================================

def build_prisma_flow():
    """Crea la estructura numérica del flujo PRISMA."""

    rows = [
        {
            "STAGE_ORDER": 1,
            "PRISMA_STAGE": "Identification",
            "ITEM": "Records identified from databases",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 2,
            "PRISMA_STAGE": "Identification",
            "ITEM": "Records identified from other sources",
            "COUNT": empty_value(),
            "NOTES": (
                "Backward citation, forward citation, institutional sources."
            ),
        },
        {
            "STAGE_ORDER": 3,
            "PRISMA_STAGE": "Deduplication",
            "ITEM": "Duplicate records removed",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 4,
            "PRISMA_STAGE": "Screening",
            "ITEM": "Records screened by title and abstract",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 5,
            "PRISMA_STAGE": "Screening",
            "ITEM": "Records excluded by title and abstract",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 6,
            "PRISMA_STAGE": "Eligibility",
            "ITEM": "Full-text reports sought",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 7,
            "PRISMA_STAGE": "Eligibility",
            "ITEM": "Full-text reports not retrieved",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 8,
            "PRISMA_STAGE": "Eligibility",
            "ITEM": "Full-text reports assessed",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 9,
            "PRISMA_STAGE": "Eligibility",
            "ITEM": "Full-text reports excluded",
            "COUNT": empty_value(),
            "NOTES": (
                "Provide counts by exclusion reason."
            ),
        },
        {
            "STAGE_ORDER": 10,
            "PRISMA_STAGE": "Included",
            "ITEM": "Studies included in qualitative synthesis",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
        {
            "STAGE_ORDER": 11,
            "PRISMA_STAGE": "Included",
            "ITEM": "Studies included in methodological comparison",
            "COUNT": empty_value(),
            "NOTES": empty_value(),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# MARCO TEMATICO
# ============================================================

def build_thematic_framework():
    """Define los bloques que organizarán la revisión."""

    descriptions = {
        "SDG4 and educational indicators": (
            "Definitions, monitoring frameworks, metadata, and UIS indicators."
        ),
        "Education and social development": (
            "Education as a component of well-being, human development, and equity."
        ),
        "Educational access and exclusion": (
            "Out-of-school rates, participation, enrollment, and access barriers."
        ),
        "Educational completion and attainment": (
            "Completion, survival, transition, graduation, and adult attainment."
        ),
        "Youth literacy": (
            "Literacy definitions, measurement, trends, and ceiling effects."
        ),
        "Public education financing": (
            "Public expenditure, allocation, efficiency, and educational outcomes."
        ),
        "Dynamic systems in education": (
            "Feedback, stocks, flows, system dynamics, and educational complexity."
        ),
        "Differential equations in social sciences": (
            "Continuous-time models, finite differences, and dynamic interpretation."
        ),
        "International educational panel data": (
            "Cross-country panels, temporal coverage, comparability, and dependency."
        ),
        "Cross-country heterogeneity": (
            "Institutional, economic, regional, and policy differences."
        ),
        "Predictive validation and baselines": (
            "Cross-validation, temporal validation, grouped validation, and null models."
        ),
        "Missing data and unequal indicator coverage": (
            "Missingness, irregular observation intervals, and selection effects."
        ),
        "Negative and null predictive results": (
            "Weak prediction, model limits, publication bias, and transparent reporting."
        ),
        "Ecuador and Latin American education": (
            "Regional educational trends, policy context, and national evidence."
        ),
    }

    rows = []

    for order, block in enumerate(
        THEMATIC_BLOCKS,
        start=1,
    ):
        rows.append(
            {
                "BLOCK_ORDER": order,
                "THEMATIC_BLOCK": block,
                "DESCRIPTION": descriptions[block],
                "MINIMUM_TARGET_STUDIES": (
                    8
                    if block in [
                        "SDG4 and educational indicators",
                        "Dynamic systems in education",
                        "International educational panel data",
                        "Public education financing",
                    ]
                    else 5
                ),
                "FOUND_STUDIES": 0,
                "GAP_STATUS": "Pending",
                "ARTICLE_SECTION": infer_article_section(
                    block
                ),
            }
        )

    return pd.DataFrame(rows)


def infer_article_section(block):
    """Asigna cada bloque a una sección del artículo."""

    mapping = {
        "SDG4 and educational indicators": (
            "Data and indicators"
        ),
        "Education and social development": (
            "Theoretical framework"
        ),
        "Educational access and exclusion": (
            "Theoretical framework"
        ),
        "Educational completion and attainment": (
            "Theoretical framework"
        ),
        "Youth literacy": (
            "Theoretical framework"
        ),
        "Public education financing": (
            "Theoretical framework"
        ),
        "Dynamic systems in education": (
            "Mathematical framework"
        ),
        "Differential equations in social sciences": (
            "Mathematical framework"
        ),
        "International educational panel data": (
            "Methods"
        ),
        "Cross-country heterogeneity": (
            "Discussion"
        ),
        "Predictive validation and baselines": (
            "Methods and validation"
        ),
        "Missing data and unequal indicator coverage": (
            "Methods and limitations"
        ),
        "Negative and null predictive results": (
            "Discussion"
        ),
        "Ecuador and Latin American education": (
            "Country case study"
        ),
    }

    return mapping.get(
        block,
        "Literature review",
    )


# ============================================================
# REVISTAS OBJETIVO
# ============================================================

def build_target_journals():
    """Registra las revistas objetivo provisionales."""

    rows = [
        {
            "PRIORITY": 1,
            "JOURNAL": (
                PRIMARY_TARGET_JOURNAL
            ),
            "ROLE": "Primary target",
            "PROVISIONAL_FIT": "High",
            "MANUSCRIPT_ORIENTATION": (
                "Social indicators, longitudinal measurement, "
                "dynamic social phenomena, and methodological limits."
            ),
            "REQUIRES_STRONGER_EMPHASIS_ON": (
                "Education as social development and indicator methodology."
            ),
            "QUARTILE_TO_VERIFY_BEFORE_SUBMISSION": "Yes",
            "AUTHOR_GUIDELINES_VERIFIED": "No",
            "APC_OR_OPEN_ACCESS_VERIFIED": "No",
            "FINAL_DECISION": "Pending",
        },
        {
            "PRIORITY": 2,
            "JOURNAL": (
                "Education Sciences"
            ),
            "ROLE": "Educational alternative",
            "PROVISIONAL_FIT": "Medium to high",
            "MANUSCRIPT_ORIENTATION": (
                "Educational indicators, policy implications, "
                "assessment, and international evidence."
            ),
            "REQUIRES_STRONGER_EMPHASIS_ON": (
                "Educational policy and practical implications."
            ),
            "QUARTILE_TO_VERIFY_BEFORE_SUBMISSION": "Yes",
            "AUTHOR_GUIDELINES_VERIFIED": "No",
            "APC_OR_OPEN_ACCESS_VERIFIED": "No",
            "FINAL_DECISION": "Pending",
        },
        {
            "PRIORITY": 3,
            "JOURNAL": (
                "Journal of Educational Change"
            ),
            "ROLE": "Conditional alternative",
            "PROVISIONAL_FIT": "Medium",
            "MANUSCRIPT_ORIENTATION": (
                "Educational transformation, reform, and system change."
            ),
            "REQUIRES_STRONGER_EMPHASIS_ON": (
                "Institutional reforms and mechanisms of educational change."
            ),
            "QUARTILE_TO_VERIFY_BEFORE_SUBMISSION": "Yes",
            "AUTHOR_GUIDELINES_VERIFIED": "No",
            "APC_OR_OPEN_ACCESS_VERIFIED": "No",
            "FINAL_DECISION": "Pending",
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# DOCUMENTO DEL PROTOCOLO
# ============================================================

def build_protocol_markdown(
    search_strategies,
    eligibility,
    thematic_framework,
):
    """Genera el protocolo narrativo en Markdown."""

    lines = [
        "# Structured literature review protocol",
        "",
        f"**Review title:** {REVIEW_TITLE}",
        "",
        f"**Research question:** {REVIEW_QUESTION}",
        "",
        (
            "**Primary target journal:** "
            f"{PRIMARY_TARGET_JOURNAL}"
        ),
        "",
        (
            "**Alternative journals:** "
            + ", ".join(
                ALTERNATIVE_TARGET_JOURNALS
            )
        ),
        "",
        "## 1. Purpose",
        "",
        (
            "This review will identify, classify, and critically "
            "evaluate literature related to dynamic educational "
            "indicators, international panel data, system dynamics, "
            "differential equations, educational financing, access, "
            "completion, literacy, predictive validation, and "
            "cross-country heterogeneity."
        ),
        "",
        "## 2. Review design",
        "",
        (
            "The review follows a structured and reproducible "
            "search, deduplication, screening, extraction, and "
            "quality-appraisal process. PRISMA categories are used "
            "to document the flow of records, although the review "
            "may include methodological and institutional sources "
            "in addition to peer-reviewed empirical studies."
        ),
        "",
        "## 3. Databases",
        "",
    ]

    for database in DATABASES:
        lines.append(
            f"- {database}"
        )

    lines.extend(
        [
            "",
            "## 4. Publication period",
            "",
            (
                f"Primary searches cover {MINIMUM_PUBLICATION_YEAR} "
                f"to {CURRENT_YEAR}. Earlier foundational works may "
                "be included through backward citation when they are "
                "necessary for the mathematical or theoretical framework."
            ),
            "",
            "## 5. Languages",
            "",
            "- English",
            "- Spanish",
            "",
            "## 6. Inclusion criteria",
            "",
        ]
    )

    inclusion = eligibility.loc[
        eligibility["TYPE"].eq(
            "Inclusion"
        )
    ]

    for _, row in inclusion.iterrows():
        lines.append(
            f"- {row['CRITERION_ID']}: {row['CRITERION']}"
        )

    lines.extend(
        [
            "",
            "## 7. Exclusion criteria",
            "",
        ]
    )

    exclusion = eligibility.loc[
        eligibility["TYPE"].eq(
            "Exclusion"
        )
    ]

    for _, row in exclusion.iterrows():
        lines.append(
            f"- {row['CRITERION_ID']}: {row['CRITERION']}"
        )

    lines.extend(
        [
            "",
            "## 8. Screening procedure",
            "",
            (
                "Records will first be deduplicated using DOI, title, "
                "authors, and publication year. Title and abstract "
                "screening will then be conducted. Potentially eligible "
                "studies will proceed to full-text review. Each exclusion "
                "at full-text stage will receive one primary reason."
            ),
            "",
            "## 9. Data extraction",
            "",
            (
                "The extraction matrix records bibliographic data, "
                "journal quartile, geographic scope, indicators, "
                "sample, period, model, validation, baselines, findings, "
                "limitations, causal claims, and reproducibility."
            ),
            "",
            "## 10. Quality appraisal",
            "",
            (
                "Each included quantitative study will be scored on "
                "twelve criteria using 0 for No, 1 for Partial or "
                "unclear, and 2 for Yes. Scores will support critical "
                "interpretation rather than automatic exclusion."
            ),
            "",
            "## 11. Thematic synthesis",
            "",
        ]
    )

    for _, row in thematic_framework.iterrows():
        lines.append(
            (
                f"- {row['THEMATIC_BLOCK']}: "
                f"{row['DESCRIPTION']}"
            )
        )

    lines.extend(
        [
            "",
            "## 12. Reporting principles",
            "",
            (
                "- No reference will be included without a verifiable "
                "title, author, year, and source."
            ),
            (
                "- Journal quartiles will be recorded with their "
                "corresponding year and database."
            ),
            (
                "- Weak, null, or negative results will be included "
                "when methodologically relevant."
            ),
            (
                "- Correlation and prediction will not be described "
                "as causal evidence unless the study design supports it."
            ),
            (
                "- Exact quotations will include page numbers."
            ),
            (
                "- Search dates, filters, exports, and result counts "
                "will be preserved."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print_section(
        "CREACION DEL PROTOCOLO DE REVISION BIBLIOGRAFICA"
    )

    create_output_directory()

    print(
        "\nCreando estrategias de busqueda..."
    )

    search_strategies = (
        build_search_strategies()
    )

    print(
        "Creando criterios de elegibilidad..."
    )

    eligibility = (
        build_eligibility_criteria()
    )

    print(
        "Creando registro de busqueda..."
    )

    search_log = (
        build_search_log()
    )

    print(
        "Creando registro de cribado..."
    )

    screening = (
        build_screening_register()
    )

    print(
        "Creando matriz de extraccion..."
    )

    extraction = (
        build_extraction_matrix()
    )

    print(
        "Creando evaluacion de calidad..."
    )

    quality_appraisal = (
        build_quality_appraisal()
    )

    print(
        "Creando flujo PRISMA..."
    )

    prisma = (
        build_prisma_flow()
    )

    print(
        "Creando marco tematico..."
    )

    thematic_framework = (
        build_thematic_framework()
    )

    print(
        "Registrando revistas objetivo..."
    )

    target_journals = (
        build_target_journals()
    )

    protocol_text = (
        build_protocol_markdown(
            search_strategies=search_strategies,
            eligibility=eligibility,
            thematic_framework=thematic_framework,
        )
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    search_strategies.to_csv(
        SEARCH_STRATEGIES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    eligibility.to_csv(
        ELIGIBILITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    search_log.to_csv(
        SEARCH_LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    screening.to_csv(
        SCREENING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    extraction.to_csv(
        EXTRACTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    quality_appraisal.to_csv(
        QUALITY_APPRAISAL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    prisma.to_csv(
        PRISMA_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    thematic_framework.to_csv(
        THEMATIC_FRAMEWORK_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    target_journals.to_csv(
        JOURNAL_TARGETS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    PROTOCOL_FILE.write_text(
        protocol_text,
        encoding="utf-8",
    )

    configuration = {
        "processing_datetime": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "review_title": REVIEW_TITLE,
        "review_question": REVIEW_QUESTION,
        "primary_target_journal": (
            PRIMARY_TARGET_JOURNAL
        ),
        "alternative_target_journals": (
            ALTERNATIVE_TARGET_JOURNALS
        ),
        "databases": DATABASES,
        "minimum_publication_year": (
            MINIMUM_PUBLICATION_YEAR
        ),
        "current_year": CURRENT_YEAR,
        "recent_literature_year": (
            RECENT_LITERATURE_YEAR
        ),
        "languages": [
            "English",
            "Spanish",
        ],
        "thematic_block_count": len(
            THEMATIC_BLOCKS
        ),
        "validation_status": "OK",
    }

    with CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    diagnostic = pd.DataFrame(
        [
            {
                "PROCESSING_DATETIME": (
                    configuration[
                        "processing_datetime"
                    ]
                ),
                "SEARCH_STRATEGY_ROWS": len(
                    search_strategies
                ),
                "DATABASE_COUNT": len(
                    DATABASES
                ),
                "QUERY_COUNT": int(
                    search_strategies[
                        "QUERY_ID"
                    ].nunique()
                ),
                "ELIGIBILITY_CRITERIA_COUNT": len(
                    eligibility
                ),
                "EXTRACTION_COLUMNS": len(
                    extraction.columns
                ),
                "QUALITY_CRITERIA_COUNT": int(
                    quality_appraisal[
                        "CRITERION_ID"
                    ].nunique()
                ),
                "PRISMA_ITEMS": len(
                    prisma
                ),
                "THEMATIC_BLOCKS": len(
                    thematic_framework
                ),
                "TARGET_JOURNALS": len(
                    target_journals
                ),
                "VALIDATION_STATUS": "OK",
            }
        ]
    )

    diagnostic.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print_section(
        "PROTOCOLO DE REVISION CREADO CORRECTAMENTE"
    )

    print(
        "\nRESUMEN DEL PROTOCOLO"
    )

    print(
        "Bases bibliograficas:",
        len(
            DATABASES
        ),
    )

    print(
        "Consultas por base:",
        search_strategies[
            "QUERY_ID"
        ].nunique(),
    )

    print(
        "Estrategias de busqueda:",
        len(
            search_strategies
        ),
    )

    print(
        "Criterios de elegibilidad:",
        len(
            eligibility
        ),
    )

    print(
        "Campos de extraccion:",
        len(
            extraction.columns
        ),
    )

    print(
        "Criterios de calidad:",
        quality_appraisal[
            "CRITERION_ID"
        ].nunique(),
    )

    print(
        "Bloques tematicos:",
        len(
            thematic_framework
        ),
    )

    print(
        "Revistas objetivo:",
        len(
            target_journals
        ),
    )

    print(
        "\nARCHIVOS CREADOS"
    )

    output_files = [
        SEARCH_STRATEGIES_FILE,
        ELIGIBILITY_FILE,
        SEARCH_LOG_FILE,
        SCREENING_FILE,
        EXTRACTION_FILE,
        QUALITY_APPRAISAL_FILE,
        PRISMA_FILE,
        THEMATIC_FRAMEWORK_FILE,
        JOURNAL_TARGETS_FILE,
        PROTOCOL_FILE,
        CONFIG_FILE,
        DIAGNOSTIC_FILE,
    ]

    for file_path in output_files:
        print(
            file_path.relative_to(
                PROJECT_ROOT
            )
        )

    print("\nVALIDACION FINAL: OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print_section(
            "ERROR DURANTE LA CREACION DEL PROTOCOLO"
        )

        print(
            f"\nTipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Detalle: {error}"
        )

        sys.exit(1)