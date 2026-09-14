"""Build, render, and validate the focused Roe/RAR book chapter."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pymupdf
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Chapter_Roe_RAR_focused.md"
DOCX = HERE / "Chapter_Roe_RAR_focused.docx"
PDF = HERE / "Chapter_Roe_RAR_focused.pdf"
REPORT = HERE / "Chapter_Roe_RAR_focused_validation.json"


def set_cell_shading(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    props.append(shading)


def set_cell_width(cell, inches):
    props = cell._tc.get_or_add_tcPr()
    width = props.first_child_found_in("w:tcW")
    if width is None:
        width = OxmlElement("w:tcW")
        props.append(width)
    width.set(qn("w:w"), str(int(inches * 1440)))
    width.set(qn("w:type"), "dxa")


def add_page_field(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    code = OxmlElement("w:instrText")
    code.set(qn("xml:space"), "preserve")
    code.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, code, separate, end])


def remove_table_borders(table):
    props = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ["top", "bottom", "insideH", "left", "right", "insideV"]:
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single" if side in {"top", "bottom"} else "nil")
        edge.set(qn("w:sz"), "6")
        edge.set(qn("w:color"), "505050")
        borders.append(edge)
    props.append(borders)


def style_document():
    doc = Document(DOCX)
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.90)
    section.right_margin = Inches(0.90)
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.78)
    section.header_distance = Inches(0.32)
    section.footer_distance = Inches(0.35)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.3
    normal.paragraph_format.space_after = Pt(4.5)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.widow_control = True

    for style_name in ["Body Text", "First Paragraph", "Compact"]:
        if style_name not in doc.styles:
            continue
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(12)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.line_spacing = 1.3
        style.paragraph_format.space_after = Pt(4.5)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        style.paragraph_format.widow_control = True

    for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attr in ["asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"]:
            fonts.attrib.pop(qn("w:" + attr), None)

    title = doc.paragraphs[0]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    title.paragraph_format.keep_with_next = True
    for run in title.runs:
        run.font.size = Pt(16)
        run.font.bold = True

    # Pandoc emits the author and affiliation as the two paragraphs after title.
    for paragraph in doc.paragraphs[1:3]:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Inches(0)
        paragraph.paragraph_format.space_after = Pt(2)
        for run in paragraph.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(10.5)

    header = section.header.paragraphs[0]
    header.text = "Roe-flux PINNs and residual-based adaptive refinement"
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in header.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(8.5)
        run.font.italic = True
        run.font.color.rgb = RGBColor(80, 80, 80)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(footer)
    for run in footer.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)

    reference_mode = False
    after_heading = False
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        style_name = paragraph.style.name
        if text == "References":
            reference_mode = True
            paragraph.paragraph_format.page_break_before = True
        elif style_name.startswith("Heading"):
            after_heading = True
        elif text:
            if style_name not in {"Image Caption", "Caption"} and not re.match(r"^(Figure|Table) \d+\.", text):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                paragraph.paragraph_format.first_line_indent = Inches(0 if after_heading else 0.20)
            after_heading = False

        if paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Inches(0)
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(5)
            paragraph.paragraph_format.space_after = Pt(1)

        if style_name in {"Image Caption", "Caption"} or re.match(r"^Figure \d+\.", text):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph.paragraph_format.first_line_indent = Inches(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(6)
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9)

        if re.match(r"^Table \d+\.", text):
            paragraph.paragraph_format.first_line_indent = Inches(0)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
                run.font.italic = True

        if paragraph._p.xpath(".//m:oMathPara"):
            paragraph.paragraph_format.first_line_indent = Inches(0)
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.space_after = Pt(4)

        if reference_mode and text and style_name != "Heading 2":
            paragraph.paragraph_format.first_line_indent = Inches(-0.22)
            paragraph.paragraph_format.left_indent = Inches(0.22)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.paragraph_format.keep_together = True
            for run in paragraph.runs:
                run.font.size = Pt(9.5)

    # Keep a lead-in paragraph with the display equation or table that follows it.
    for current, following in zip(doc.paragraphs, doc.paragraphs[1:]):
        if following._p.xpath(".//m:oMathPara"):
            current.paragraph_format.keep_with_next = True

    for shape in doc.inline_shapes:
        if shape.width > Inches(6.60):
            ratio = Inches(6.60) / shape.width
            shape.width = Inches(6.60)
            shape.height = int(shape.height * ratio)

    for index, table in enumerate(doc.tables):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        remove_table_borders(table)
        rows = len(table.rows)
        cols = len(table.columns)
        if cols == 8:
            widths = [0.78, 1.30, 0.55, 0.55, 0.58, 0.90, 0.72, 0.52]
        elif cols == 4:
            widths = [1.85, 1.35, 1.85, 1.35]
        elif cols == 2:
            widths = [1.25, 5.2]
        else:
            widths = [6.7 / cols] * cols
        for row_index, row in enumerate(table.rows):
            row_props = row._tr.get_or_add_trPr()
            no_split = OxmlElement("w:cantSplit")
            row_props.append(no_split)
            if row_index == 0:
                repeat = OxmlElement("w:tblHeader")
                repeat.set(qn("w:val"), "true")
                row_props.append(repeat)
            for cell, width in zip(row.cells, widths):
                set_cell_width(cell, width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if row_index == 0:
                    set_cell_shading(cell, "E8EDF2")
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.first_line_indent = Inches(0)
                    paragraph.paragraph_format.space_before = Pt(2)
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(7.6 if cols == 8 else 9)
                        run.font.bold = row_index == 0

    doc.core_properties.title = (
        "Roe-flux physics-informed neural networks for wet-bed dam-break flow"
    )
    doc.core_properties.subject = (
        "Controlled study of residual formulation and adaptive sampling"
    )
    doc.core_properties.author = "Danial Goodarzi; Abdolmajid Mohammadian"
    doc.save(DOCX)


def render_pdf():
    subprocess.run(
        ["python", str(HERE / "render_chapter.py"), str(DOCX)],
        cwd=HERE.parent,
        check=True,
    )


def find_page(pdf, phrase):
    for index, page in enumerate(pdf):
        if phrase in page.get_text():
            return index + 1
    raise ValueError(f"Heading not found in PDF: {phrase}")


def validate():
    text = SOURCE.read_text(encoding="utf-8")
    doc = Document(DOCX)
    pdf = pymupdf.open(PDF)
    intro_start = find_page(pdf, "1. Introduction")
    method_start = find_page(pdf, "2. Method")
    intro_pages = method_start - intro_start + 1
    captions = [p.text for p in doc.paragraphs if re.match(r"^Figure \d+\.", p.text)]
    report = {
        "source": str(SOURCE),
        "docx": str(DOCX),
        "pdf": str(PDF),
        "pages": len(pdf),
        "introduction_starts_on_page": intro_start,
        "method_starts_on_page": method_start,
        "maximum_pages_touched_by_introduction": intro_pages,
        "words_including_references": len(re.findall(r"\b[\w'-]+\b", text)),
        "figures": len(doc.inline_shapes),
        "figure_captions": len(captions),
        "tables": len(doc.tables),
        "native_math_objects": len(doc._element.xpath(".//m:oMath")),
        "quantitative_source": "runs.tar.gz only",
    }
    if not 20 <= len(pdf) <= 25:
        raise AssertionError(f"Page count is {len(pdf)}, expected 20--25")
    if intro_pages > 5:
        raise AssertionError(f"Introduction touches {intro_pages} pages")
    if len(doc.inline_shapes) != 8 or len(captions) != 8:
        raise AssertionError("Expected eight figures and eight captions")
    if len(doc.tables) != 4:
        raise AssertionError(f"Expected four tables, found {len(doc.tables)}")
    if re.search(r"\b(TODO|TBD|PLACEHOLDER)\b|\{\{", text, re.I):
        raise AssertionError("Draft placeholder remains")
    if any("Error! Reference source not found" in page.get_text() for page in pdf):
        raise AssertionError("Broken Word cross-reference")
    if any(len(page.get_text().split()) < 15 for page in pdf):
        raise AssertionError("Unexpected blank or nearly blank page")
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    subprocess.run(["python", str(HERE / "analyze_focused_archive.py")], cwd=HERE.parent, check=True)
    subprocess.run(
        [
            "pandoc",
            str(SOURCE),
            "--from=markdown+tex_math_dollars",
            "--to=docx",
            "--resource-path=" + str(HERE),
            "-o",
            str(DOCX),
        ],
        cwd=HERE,
        check=True,
    )
    style_document()
    render_pdf()
    validate()


if __name__ == "__main__":
    main()
