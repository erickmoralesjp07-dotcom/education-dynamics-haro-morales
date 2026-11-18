from pathlib import Path
from datetime import datetime
import shutil


ROOT = Path(__file__).resolve().parents[1]

MANUSCRIPT_FILE = (
    ROOT
    / "manuscript"
    / "seps"
    / "main_manuscript.md"
)

BACKUP_FILE = (
    ROOT
    / "manuscript"
    / "seps"
    / "main_manuscript_before_citations.md"
)

REPORT_FILE = (
    ROOT
    / "manuscript"
    / "seps"
    / "citation_insertion_report.txt"
)

NOW = datetime.now().astimezone().isoformat()


CITATIONS = [
    {
        "label": "multilevel inequality",
        "old": (
            "Educational outcomes are structured at several levels. "
            "Students are nested within schools, regions, and national "
            "systems, while the relevant socioeconomic conditions operate "
            "at household, institutional, territorial, and policy levels."
        ),
        "new": (
            "Educational outcomes are structured at several levels. "
            "Students are nested within schools, regions, and national "
            "systems, while the relevant socioeconomic conditions operate "
            "at household, institutional, territorial, and policy levels "
            "(Bernardi et al., 2026; Yardım et al., 2024)."
        ),
    },
    {
        "label": "PISA inequality",
        "old": (
            "Research based on PISA and other international assessments "
            "shows that inequality may appear both within and between "
            "schools and countries."
        ),
        "new": (
            "Research based on PISA and other international assessments "
            "shows that inequality may appear both within and between "
            "schools and countries (Sempé & Esposito, 2025; Yang, 2026)."
        ),
    },
    {
        "label": "longitudinal attainment",
        "old": (
            "Longitudinal studies suggest that educational trajectories "
            "and later attainment depend on conditions observed years "
            "earlier, but predictive relationships are generally modest "
            "and context-dependent."
        ),
        "new": (
            "Longitudinal studies suggest that educational trajectories "
            "and later attainment depend on conditions observed years "
            "earlier, but predictive relationships are generally modest "
            "and context-dependent (Borgonovi et al., 2021; "
            "Parasnis & Swan, 2020)."
        ),
    },
    {
        "label": "PISA eligibility",
        "old": (
            "International assessments require specialized treatment."
        ),
        "new": (
            "International assessments require specialized treatment, "
            "including attention to sample eligibility and population "
            "coverage (Spaull, 2017)."
        ),
    },
    {
        "label": "PISA measurement validity",
        "old": (
            "Construct validity also matters because survey indicators "
            "may combine the intended concept with response behavior, "
            "self-perception, cultural interpretation, or other "
            "irrelevant variation."
        ),
        "new": (
            "Construct validity also matters because survey indicators "
            "may combine the intended concept with response behavior, "
            "self-perception, cultural interpretation, or other "
            "irrelevant variation (Hansen & Strietholt, 2018)."
        ),
    },
    {
        "label": "attainment measurement",
        "old": (
            "Historical and cross-national attainment data contain "
            "missing observations, grouped categories, imputation, "
            "and measurement error."
        ),
        "new": (
            "Historical and cross-national attainment data contain "
            "missing observations, grouped categories, imputation, "
            "and measurement error (Friedman et al., 2018; "
            "Morrisson & Murtin, 2009)."
        ),
    },
    {
        "label": "attainment validation",
        "old": (
            "Methods for estimating average years of schooling from "
            "binned data show that alternative measurement procedures "
            "can materially affect comparative results."
        ),
        "new": (
            "Methods for estimating average years of schooling from "
            "binned data show that alternative measurement procedures "
            "can materially affect comparative results "
            "(Friedman et al., 2018)."
        ),
    },
    {
        "label": "planning methods",
        "old": (
            "Educational planning studies frequently use panel models, "
            "fixed effects, multilevel regression, efficiency models, "
            "or machine learning."
        ),
        "new": (
            "Educational planning studies frequently use panel models, "
            "fixed effects, multilevel regression, efficiency models, "
            "or machine learning (Krueger & Lindahl, 2000; "
            "Bernardi et al., 2026)."
        ),
    },
    {
        "label": "UNESCO source",
        "old": (
            "The study uses open national data from the UNESCO Institute "
            "for Statistics."
        ),
        "new": (
            "The study uses open national data from the UNESCO Institute "
            "for Statistics (UNESCO Institute for Statistics, 2026a, "
            "2026b)."
        ),
    },
    {
        "label": "Ridge method",
        "old": (
            "Ridge regression was used as the principal model because "
            "the ACLF predictors may be correlated and the sample is limited."
        ),
        "new": (
            "Ridge regression was used as the principal model because "
            "the ACLF predictors may be correlated and the sample is limited "
            "(Hoerl & Kennard, 1970)."
        ),
    },
    {
        "label": "Elastic Net",
        "old": (
            "The benchmark additionally evaluated:"
        ),
        "new": (
            "The benchmark additionally evaluated Elastic Net "
            "(Zou & Hastie, 2005), Random Forest (Breiman, 2001), "
            "and the following alternatives:"
        ),
    },
    {
        "label": "remove duplicated list models",
        "old": (
            "- Elastic Net;\n"
            "- Random Forest;\n"
            "- HistGradientBoosting;"
        ),
        "new": (
            "- HistGradientBoosting;"
        ),
    },
    {
        "label": "Kazakhstan PISA",
        "old": (
            "These conditions make simple causal narratives inappropriate."
        ),
        "new": (
            "These conditions make simple causal narratives inappropriate "
            "(Ospanova et al., 2023; Hansen & Strietholt, 2018)."
        ),
    },
]


def main():
    if not MANUSCRIPT_FILE.exists():
        raise FileNotFoundError(
            MANUSCRIPT_FILE
        )

    text = MANUSCRIPT_FILE.read_text(
        encoding="utf-8",
    )

    if not BACKUP_FILE.exists():
        shutil.copy2(
            MANUSCRIPT_FILE,
            BACKUP_FILE,
        )

    inserted = []
    missing = []

    for citation in CITATIONS:
        old = citation["old"]
        new = citation["new"]

        if old in text:
            text = text.replace(
                old,
                new,
                1,
            )

            inserted.append(
                citation["label"]
            )

        elif new in text:
            inserted.append(
                citation["label"]
                + " [already present]"
            )

        else:
            missing.append(
                citation["label"]
            )

    citation_groups = [
        "(Bernardi et al., 2026; Yardım et al., 2024)",
        "(Sempé & Esposito, 2025; Yang, 2026)",
        "(Borgonovi et al., 2021; Parasnis & Swan, 2020)",
        "(Spaull, 2017)",
        "(Hansen & Strietholt, 2018)",
        "(Friedman et al., 2018; Morrisson & Murtin, 2009)",
        "(Friedman et al., 2018)",
        "(Krueger & Lindahl, 2000; Bernardi et al., 2026)",
        "(UNESCO Institute for Statistics, 2026a, 2026b)",
        "(Hoerl & Kennard, 1970)",
        "(Zou & Hastie, 2005)",
        "(Breiman, 2001)",
        "(Ospanova et al., 2023; Hansen & Strietholt, 2018)",
    ]

    citation_count = sum(
        text.count(group)
        for group in citation_groups
    )

    required_authors = [
        "Bernardi",
        "Borgonovi",
        "Breiman",
        "Friedman",
        "Hansen",
        "Hoerl",
        "Krueger",
        "Morrisson",
        "Ospanova",
        "Parasnis",
        "Sempé",
        "Spaull",
        "UNESCO Institute for Statistics",
        "Yang",
        "Yardım",
        "Zou",
    ]

    missing_authors = [
        author
        for author in required_authors
        if author not in text
    ]

    if "# 14. References" not in text:
        raise ValueError(
            "No se encontró la sección de referencias."
        )

    MANUSCRIPT_FILE.write_text(
        text,
        encoding="utf-8",
    )

    report_lines = [
        "CITATION INSERTION REPORT",
        f"Generated: {NOW}",
        "",
        f"Inserted or present: {len(inserted)}",
        *[
            f"- {label}"
            for label in inserted
        ],
        "",
        f"Missing replacements: {len(missing)}",
        *[
            f"- {label}"
            for label in missing
        ],
        "",
        f"Citation groups detected: {citation_count}",
        (
            "Missing required authors: "
            + (
                ", ".join(missing_authors)
                if missing_authors
                else "None"
            )
        ),
        f"Backup: {BACKUP_FILE}",
        f"Manuscript: {MANUSCRIPT_FILE}",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    valid = (
        len(missing) == 0
        and len(missing_authors) == 0
        and citation_count >= 13
        and BACKUP_FILE.exists()
        and len(text.split()) > 3000
    )

    print(
        "===== CITAS INSERTADAS ====="
    )

    print(
        "REEMPLAZOS:",
        len(inserted),
    )

    print(
        "NO ENCONTRADOS:",
        len(missing),
    )

    for label in missing:
        print(
            " -",
            label,
        )

    print(
        "GRUPOS DE CITAS:",
        citation_count,
    )

    print(
        "AUTORES FALTANTES:",
        (
            ", ".join(missing_authors)
            if missing_authors
            else "NINGUNO"
        ),
    )

    print(
        "PALABRAS:",
        len(text.split()),
    )

    print(
        "COPIA DE SEGURIDAD:",
        BACKUP_FILE,
    )

    print(
        "VALIDACION:",
        "OK"
        if valid
        else "REVISAR",
    )

    if not valid:
        raise ValueError(
            "La inserción de citas requiere revisión."
        )


if __name__ == "__main__":
    main()