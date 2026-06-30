import structlog

from markdown_pdf import MarkdownPdf, Section

from config.config import FINAL_PDF_FILEPATH

logger = structlog.get_logger()


def download_pdf_report(
    pre_report: str = None,
    clinical_report: str = None,
):
    if pre_report is None and clinical_report is None:
        logger.warning("Aborting PDF generation: No report data provided.")
        return
    # Remove title from Clinical Report
    clinical_report = "\n".join(clinical_report.split("\n")[1:])
    # Generate PDF data with both reports
    pdf = MarkdownPdf()
    pdf.add_section(Section("\n".join([pre_report, "---", clinical_report])))
    # Set PDF metadata
    pdf.meta["title"] = "HeRMeS Pre-Visit Clinical Report"
    pdf.meta["author"] = "Health Report & Monitoring System (HeRMeS)"
    # Save PDF file
    pdf.save(FINAL_PDF_FILEPATH)
    logger.info(f"✅ Full clinical PDF report generated at: {FINAL_PDF_FILEPATH}")
