from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable
import re

from nlp_project.config import AppConfig
from nlp_project.data import DataValidationError, load_dataset, validate_dataset


def generate_proposal_markdown(config: AppConfig) -> str:
    dataset_description = _dataset_description(config)
    team_lines = _team_lines(config)
    questions = "\n".join(f"- {question}" for question in config.proposal.research_questions)

    return "\n".join(
        [
            f"# {config.project.title}",
            "",
            "## NLP task and domain/application area",
            (
                f"Task: {config.project.task}. "
                f"Domain: {config.project.domain}. "
                f"Application area: {config.project.application_area}."
            ),
            "",
            "## Motivation and problem statement",
            f"{config.proposal.motivation} {config.proposal.problem_statement}",
            "",
            "## Expected final product",
            (
                f"{config.proposal.expected_product} "
                f"It improves: {config.proposal.process_improvement}"
            ),
            "",
            "## Research questions",
            questions,
            "",
            "## Dataset",
            dataset_description,
            "",
            "## Team responsibilities",
            team_lines,
            "",
            "## GitHub repository link",
            config.project.github_repo_url,
            "",
            "## Submission checklist",
            "- [ ] Private GitHub repo with collaborators `drelhaj` and `whistle-hikhi`.",
            "- [ ] Final GitHub link included in this proposal and the PDF.",
            "- [ ] Proposal PDF exported in 12 pt Times-compatible font, submitted to Canvas.",
            "",
        ]
    )


def write_proposal(config: AppConfig, output_path: Path | str | None = None) -> Path:
    path = Path(output_path) if output_path is not None else config.repo_root / "proposal.md"
    if not path.is_absolute():
        path = config.repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_proposal_markdown(config), encoding="utf-8")
    return path


def render_proposal_pdf(
    markdown_path: Path | str,
    pdf_path: Path | str,
) -> Path | None:
    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path)
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        print(f"PDF export skipped: reportlab is unavailable ({exc}).")
        return None

    markdown_text = markdown_path.read_text(encoding="utf-8")
    word_count = len(markdown_text.split())
    if word_count > 700:
        print(
            "Warning: proposal.md is likely too long for one page. "
            "Shorten config text before final PDF submission."
        )

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "ProjectTimes",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=12,
        leading=12.5,
        alignment=TA_LEFT,
        spaceAfter=1,
    )
    title = ParagraphStyle(
        "ProjectTitle",
        parent=base,
        fontName="Times-Bold",
        fontSize=12,
        leading=12.5,
        spaceAfter=2,
    )
    heading = ParagraphStyle(
        "ProjectHeading",
        parent=base,
        fontName="Times-Bold",
        textColor=colors.black,
        spaceBefore=1,
        spaceAfter=1,
    )

    story = []
    for line in markdown_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 1))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_clean_markdown(line[2:]), title))
        elif line.startswith("## "):
            story.append(Paragraph(_clean_markdown(line[3:]), heading))
        elif line.startswith("- [ ] "):
            story.append(Paragraph(f"[ ] {_clean_markdown(line[6:])}", base))
        elif line.startswith("- "):
            story.append(Paragraph(f"&bull; {_clean_markdown(line[2:])}", base))
        else:
            story.append(Paragraph(_clean_markdown(line), base))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    doc.build(story)
    page_count = _count_pdf_pages(pdf_path)
    if page_count > 1:
        print(
            f"Warning: proposal PDF rendered to {page_count} pages. "
            "Shorten config text to meet the one-page submission requirement."
        )
    return pdf_path


def _dataset_description(config: AppConfig) -> str:
    try:
        df = load_dataset(config)
        summary = validate_dataset(df, config)
        class_counts = ", ".join(
            f"{label}: {count}" for label, count in summary.class_counts.items()
        )
        size = f"{summary.rows} rows, {len(summary.columns)} columns"
        columns = ", ".join(summary.columns)
    except DataValidationError as exc:
        size = f"Configured dataset could not be validated yet: {exc}"
        columns = "Update after replacing or fixing the dataset."
        class_counts = "Update after validation."

    challenges = "; ".join(config.data.challenges)
    return (
        f"Source: {config.data.source} {config.data.provenance} "
        f"Size: {size}; labels: {class_counts}. "
        f"Domain: {config.data.domain}. "
        f"Challenges: {challenges}"
    )


def _team_lines(config: AppConfig) -> str:
    lines = []
    for member in config.team.members:
        responsibilities = "; ".join(member.responsibilities)
        lines.append(f"- **{member.name}**: {responsibilities}")
    return "\n".join(lines)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return escape(text)


def _count_pdf_pages(pdf_path: Path) -> int:
    content = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", content))


def required_proposal_sections() -> Iterable[str]:
    return (
        "NLP task and domain/application area",
        "Motivation and problem statement",
        "Expected final product",
        "Research questions",
        "Dataset",
        "Team responsibilities",
        "GitHub repository link",
        "Submission checklist",
    )
