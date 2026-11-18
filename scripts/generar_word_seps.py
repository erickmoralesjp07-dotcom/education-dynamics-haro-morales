from pathlib import Path
import csv
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "manuscript" / "seps"

FRONT_FILE = SOURCE_DIR / "front_matter.md"
MANUSCRIPT_FILE = SOURCE_DIR / "complete_manuscript.md"

FIGURE_DIR = SOURCE_DIR / "figures"
TABLE_DIR = SOURCE_DIR / "tables"

OUTPUT_FILE = (
    SOURCE_DIR
    / "Morales_Haro_Educational_Dynamics_SEPS.docx"
)

TITLE = (
    "Predicting Changes in Adult Educational Attainment "
    "from International Education Indicators: "
    "A Regularized Dynamic-System Analysis"
)

AUTHORS = (
    "Erick Jair Morales Portilla¹*; "
    "David Sebastian Haro Perugachi¹"
)

AFFILIATION = (
    "¹ Universidad Técnica del Norte, "
    "Ibarra, Ecuador"
)

EMAILS = (
    "Erick Jair Morales Portilla: "
    "ejmoralesp@utn.edu.ec\n"
    "David Sebastian Haro Perugachi: "
    "dsharop@utn.edu.ec"
)

CORRESPONDING = (
    "* Corresponding author: "
    "Erick Jair Morales Portilla, "
    "ejmoralesp@utn.edu.ec"
)

ORCID_NOTE = (
    "ORCID identifiers: pending registration."
)

FIGURES = [
    {
        "file": (
            "Figure_1_global_predictive_performance.png"
        ),
        "caption": (
            "Figure 1. Ridge predictive performance "
            "relative to the zero-change baseline. "
            "Positive values indicate lower RMSE than "
            "the baseline."
        ),
    },
    {
        "file": (
            "Figure_2_dc_model_benchmark.png"
        ),
        "caption": (
            "Figure 2. Nested model comparison for "
            "changes in adult educational attainment "
            "(dC/dt)."
        ),
    },
    {
        "file": (
            "Figure_3_ecuador_trajectories.png"
        ),
        "caption": (
            "Figure 3. Observed ACLF trajectories "
            "for Ecuador, 2009-2023."
        ),
    },
    {
        "file": (
            "Figure_4_ecuador_observed_predicted_rates.png"
        ),
        "caption": (
            "Figure 4. Observed and predicted ACLF "
            "annual rates for Ecuador."
        ),
    },
]

TABLES = [
    {
        "file": "Table_1_selected_indicators.csv",
        "title": (
            "Table 1. Education indicators selected "
            "for the analytical framework."
        ),
        "max_rows": 20,
    },
    {
        "file": "Table_2_predictive_results.csv",
        "title": (
            "Table 2. Main predictive results relative "
            "to the zero-change reference."
        ),
        "max_rows": 25,
    },
    {
        "file": "Table_3_dc_stability.csv",
        "title": (
            "Table 3. Stability of the predictive "
            "signal for adult educational attainment."
        ),
        "max_rows": 20,
    },
    {
        "file": "Table_4_literature_evidence.csv",
        "title": (
            "Table 4. Final evidence candidates from "
            "the structured literature review."
        ),
        "max_rows": 20,
    },
    {
        "file": "Table_5_final_validation.csv",
        "title": (
            "Table 5. Summary of the final scientific "
            "validation."
        ),
        "max_rows": 20,
    },
]


def set_cell_shading(cell, fill):
    properties = cell._tc.get_or_add_tcPr()

    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)

    properties.append(shading)


def set_cell_margins(
    cell,
    top=60,
    start=60,
    bottom=60,
    end=60,
):
    properties = cell._tc.get_or_add_tcPr()

    margins = properties.first_child_found_in(
        "w:tcMar"
    )

    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)

    for margin_name, margin_value in [
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ]:
        node = margins.find(
            qn(f"w:{margin_name}")
        )

        if node is None:
            node = OxmlElement(
                f"w:{margin_name}"
            )
            margins.append(node)

        node.set(
            qn("w:w"),
            str(margin_value),
        )

        node.set(
            qn("w:type"),
            "dxa",
        )


def set_run_font(
    run,
    name="Arial",
    size=10,
    bold=False,
    italic=False,
    color=None,
):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic

    run._element.get_or_add_rPr()

    run._element.rPr.rFonts.set(
        qn("w:ascii"),
        name,
    )

    run._element.rPr.rFonts.set(
        qn("w:hAnsi"),
        name,
    )

    if color is not None:
        run.font.color.rgb = RGBColor(
            *color
        )


def style_all_runs(
    paragraph,
    size=10,
    bold=False,
    italic=False,
):
    for run in paragraph.runs:
        set_run_font(
            run,
            size=size,
            bold=bold,
            italic=italic,
        )


def clean_markdown(text):
    text = text.replace("**", "")
    text = text.replace("*", "")

    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    return text.strip()


def add_page_number(paragraph):
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.RIGHT
    )

    run = paragraph.add_run()

    begin = OxmlElement("w:fldChar")
    begin.set(
        qn("w:fldCharType"),
        "begin",
    )

    instruction = OxmlElement("w:instrText")
    instruction.set(
        qn("xml:space"),
        "preserve",
    )
    instruction.text = " PAGE "

    end = OxmlElement("w:fldChar")
    end.set(
        qn("w:fldCharType"),
        "end",
    )

    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(end)

    set_run_font(
        run,
        size=9,
    )


def configure_document(document):
    section = document.sections[0]

    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    styles = document.styles

    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)

    normal._element.rPr.rFonts.set(
        qn("w:ascii"),
        "Arial",
    )

    normal._element.rPr.rFonts.set(
        qn("w:hAnsi"),
        "Arial",
    )

    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = (
        WD_ALIGN_PARAGRAPH.JUSTIFY
    )

    for style_name, size in [
        ("Title", 16),
        ("Heading 1", 13),
        ("Heading 2", 11),
        ("Heading 3", 10),
    ]:
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True

        style._element.rPr.rFonts.set(
            qn("w:ascii"),
            "Arial",
        )

        style._element.rPr.rFonts.set(
            qn("w:hAnsi"),
            "Arial",
        )

    footer = section.footer
    add_page_number(
        footer.paragraphs[0]
    )


def add_title_page(document):
    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )
    paragraph.paragraph_format.space_after = (
        Pt(18)
    )

    run = paragraph.add_run(TITLE)
    set_run_font(
        run,
        size=17,
        bold=True,
        color=(31, 78, 121),
    )

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(AUTHORS)
    set_run_font(
        run,
        size=11,
        bold=True,
    )

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(AFFILIATION)
    set_run_font(
        run,
        size=10,
    )

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(EMAILS)
    set_run_font(
        run,
        size=9,
    )

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(CORRESPONDING)
    set_run_font(
        run,
        size=9,
        italic=True,
    )

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(ORCID_NOTE)
    set_run_font(
        run,
        size=9,
        italic=True,
    )

    document.add_paragraph()

    paragraph = document.add_paragraph()
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    run = paragraph.add_run(
        "Research Article\n"
        "Prepared for submission to "
        "Socio-Economic Planning Sciences"
    )

    set_run_font(
        run,
        size=10,
        bold=True,
    )

    document.add_page_break()


def extract_section(
    text,
    heading,
    next_heading=None,
):
    start = text.find(heading)

    if start < 0:
        return ""

    start += len(heading)

    if next_heading:
        end = text.find(
            next_heading,
            start,
        )

        if end < 0:
            end = len(text)
    else:
        end = len(text)

    return text[start:end].strip()


def add_front_matter(document):
    front = FRONT_FILE.read_text(
        encoding="utf-8",
    )

    abstract = extract_section(
        front,
        "## Abstract",
        "## Keywords",
    )

    keywords = extract_section(
        front,
        "## Keywords",
        "## Highlights",
    )

    highlights = extract_section(
        front,
        "## Highlights",
        "## Article type",
    )

    document.add_heading(
        "Abstract",
        level=1,
    )

    paragraph = document.add_paragraph(
        clean_markdown(abstract)
    )

    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.JUSTIFY
    )

    document.add_heading(
        "Keywords",
        level=1,
    )

    paragraph = document.add_paragraph(
        clean_markdown(keywords)
    )

    document.add_heading(
        "Highlights",
        level=1,
    )

    for line in highlights.splitlines():
        line = clean_markdown(line)

        if line.startswith("- "):
            line = line[2:].strip()

        if line:
            paragraph = document.add_paragraph(
                style="List Bullet"
            )

            paragraph.add_run(line)

    document.add_page_break()


def add_inline_markdown(paragraph, text):
    text = clean_markdown(text)

    doi_pattern = re.compile(
        r"(https?://\S+)"
    )

    parts = doi_pattern.split(text)

    for part in parts:
        if not part:
            continue

        run = paragraph.add_run(part)

        if part.startswith("http"):
            run.font.color.rgb = RGBColor(
                0,
                102,
                204,
            )

            run.font.underline = True

        set_run_font(
            run,
            size=10,
        )


def add_markdown_content(document):
    text = MANUSCRIPT_FILE.read_text(
        encoding="utf-8",
    )

    lines = text.splitlines()

    in_equation = False
    equation_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.strip() == "\\[":
            in_equation = True
            equation_lines = []
            continue

        if line.strip() == "\\]":
            in_equation = False

            paragraph = document.add_paragraph()
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER
            )

            equation = " ".join(
                x.strip()
                for x in equation_lines
            )

            run = paragraph.add_run(equation)
            set_run_font(
                run,
                name="Cambria Math",
                size=10,
                italic=True,
            )
            continue

        if in_equation:
            equation_lines.append(line)
            continue

        if not line.strip():
            continue

        if line.startswith("# "):
            title = clean_markdown(
                line[2:]
            )

            if title == "14. References":
                document.add_page_break()

            document.add_heading(
                title,
                level=1,
            )
            continue

        if line.startswith("## "):
            document.add_heading(
                clean_markdown(
                    line[3:]
                ),
                level=2,
            )
            continue

        if line.startswith("### "):
            document.add_heading(
                clean_markdown(
                    line[4:]
                ),
                level=3,
            )
            continue

        if re.match(
            r"^\d+\.\s+",
            line,
        ):
            content = re.sub(
                r"^\d+\.\s+",
                "",
                line,
            )

            paragraph = document.add_paragraph(
                style="List Number"
            )

            add_inline_markdown(
                paragraph,
                content,
            )
            continue

        if line.startswith("- "):
            paragraph = document.add_paragraph(
                style="List Bullet"
            )

            add_inline_markdown(
                paragraph,
                line[2:],
            )
            continue

        paragraph = document.add_paragraph()

        add_inline_markdown(
            paragraph,
            line,
        )

        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.JUSTIFY
        )


def read_csv_rows(path, max_rows):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.reader(file)
        )

    if not rows:
        return []

    return (
        [rows[0]]
        + rows[1:max_rows + 1]
    )


def shorten(value, limit=90):
    value = str(value).strip()

    if len(value) <= limit:
        return value

    return value[:limit - 3] + "..."


def add_tables(document):
    document.add_page_break()

    document.add_heading(
        "Tables",
        level=1,
    )

    for item in TABLES:
        path = TABLE_DIR / item["file"]

        if not path.exists():
            continue

        title = document.add_paragraph()
        title.alignment = (
            WD_ALIGN_PARAGRAPH.LEFT
        )

        run = title.add_run(
            item["title"]
        )

        set_run_font(
            run,
            size=10,
            bold=True,
        )

        rows = read_csv_rows(
            path,
            item["max_rows"],
        )

        if not rows:
            continue

        column_count = len(rows[0])

        table = document.add_table(
            rows=1,
            cols=column_count,
        )

        table.style = "Table Grid"
        table.autofit = True

        header_cells = table.rows[0].cells

        for column_index, value in enumerate(
            rows[0]
        ):
            cell = header_cells[
                column_index
            ]

            cell.text = shorten(
                value,
                55,
            )

            set_cell_shading(
                cell,
                "D9EAF7",
            )

            cell.vertical_alignment = (
                WD_CELL_VERTICAL_ALIGNMENT.CENTER
            )

            set_cell_margins(cell)

            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER
                )

                style_all_runs(
                    paragraph,
                    size=7,
                    bold=True,
                )

        for row_values in rows[1:]:
            row_cells = table.add_row().cells

            for column_index in range(
                column_count
            ):
                value = (
                    row_values[column_index]
                    if column_index
                    < len(row_values)
                    else ""
                )

                row_cells[
                    column_index
                ].text = shorten(
                    value,
                    100,
                )

                row_cells[
                    column_index
                ].vertical_alignment = (
                    WD_CELL_VERTICAL_ALIGNMENT.CENTER
                )

                set_cell_margins(
                    row_cells[
                        column_index
                    ]
                )

                for paragraph in (
                    row_cells[
                        column_index
                    ].paragraphs
                ):
                    paragraph.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT
                    )

                    style_all_runs(
                        paragraph,
                        size=7,
                    )

        note = document.add_paragraph(
            "Source: Authors' calculations "
            "from the project analytical outputs."
        )

        note.paragraph_format.space_after = (
            Pt(12)
        )

        style_all_runs(
            note,
            size=8,
            italic=True,
        )


def add_figures(document):
    document.add_page_break()

    document.add_heading(
        "Figures",
        level=1,
    )

    for item in FIGURES:
        path = FIGURE_DIR / item["file"]

        if not path.exists():
            continue

        paragraph = document.add_paragraph()
        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER
        )

        run = paragraph.add_run()

        run.add_picture(
            str(path),
            width=Inches(6.4),
        )

        caption = document.add_paragraph()
        caption.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER
        )

        run = caption.add_run(
            item["caption"]
        )

        set_run_font(
            run,
            size=9,
            italic=True,
        )

        caption.paragraph_format.space_after = (
            Pt(12)
        )


def add_editorial_note(document):
    document.add_page_break()

    document.add_heading(
        "Editorial information pending confirmation",
        level=1,
    )

    items = [
        (
            "ORCID identifiers for both authors "
            "must be added before submission."
        ),
        (
            "Any additional author must meet "
            "authorship criteria and approve the "
            "final manuscript."
        ),
        (
            "The statement on institutional support "
            "for the article processing charge must "
            "only be added after formal confirmation."
        ),
        (
            "The selected journal format and current "
            "submission requirements must be checked "
            "immediately before submission."
        ),
    ]

    for item in items:
        paragraph = document.add_paragraph(
            style="List Bullet"
        )

        paragraph.add_run(item)


def validate_sources():
    required = [
        FRONT_FILE,
        MANUSCRIPT_FILE,
    ]

    required.extend(
        FIGURE_DIR / item["file"]
        for item in FIGURES
    )

    required.extend(
        TABLE_DIR / item["file"]
        for item in TABLES
    )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def main():
    validate_sources()

    document = Document()

    configure_document(document)
    add_title_page(document)
    add_front_matter(document)
    add_markdown_content(document)
    add_tables(document)
    add_figures(document)
    add_editorial_note(document)

    core_properties = (
        document.core_properties
    )

    core_properties.title = TITLE
    core_properties.subject = (
        "Educational dynamics, planning and "
        "predictive validation"
    )

    core_properties.author = (
        "Erick Jair Morales Portilla; "
        "David Sebastian Haro Perugachi"
    )

    core_properties.keywords = (
        "educational attainment; education planning; "
        "UNESCO UIS; Ridge regression; "
        "temporal validation"
    )

    core_properties.comments = (
        "Preliminary manuscript prepared for "
        "Socio-Economic Planning Sciences."
    )

    document.save(
        OUTPUT_FILE
    )

    file_size = OUTPUT_FILE.stat().st_size

    print(
        "===== WORD PRELIMINAR SEPS ====="
    )

    print(
        "ARCHIVO:",
        OUTPUT_FILE,
    )

    print(
        "BYTES:",
        file_size,
    )

    print(
        "FIGURAS:",
        len(FIGURES),
    )

    print(
        "TABLAS:",
        len(TABLES),
    )

    print(
        "VALIDACION:",
        "OK"
        if file_size > 100000
        else "REVISAR",
    )

    if file_size <= 100000:
        raise ValueError(
            "El Word generado parece incompleto."
        )


if __name__ == "__main__":
    main()