"""HTML report generation for Connectomix.

This module provides classes for generating HTML reports that summarize
analysis results at participant and group levels.
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# CSS styles for reports
REPORT_CSS = """
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header h1 {
        margin: 0 0 10px 0;
        font-size: 2em;
    }
    .header .subtitle {
        opacity: 0.9;
        font-size: 1.1em;
    }
    .section {
        background: white;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section h2 {
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
        margin-top: 0;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #666;
        font-size: 0.9em;
        margin-top: 5px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th, td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #667eea;
        color: white;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    .image-container {
        text-align: center;
        margin: 20px 0;
    }
    .image-container img {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .image-caption {
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
    .warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .success {
        background-color: #d4edda;
        border: 1px solid #28a745;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9em;
        padding: 20px;
        border-top: 1px solid #ddd;
        margin-top: 30px;
    }
</style>
"""


class HTMLReportGenerator:
    """Generate HTML reports for Connectomix analysis results.
    
    This class builds HTML reports section by section, allowing flexible
    composition of different content types.
    
    Attributes:
        title: Report title.
        subtitle: Report subtitle.
        sections: List of HTML section strings.
    
    Example:
        >>> report = HTMLReportGenerator("Participant Report", "sub-01")
        >>> report.add_header("Analysis Parameters")
        >>> report.add_metrics({"tSNR": 45.2, "Volumes": 200})
        >>> report.add_image(figure, "Connectivity Matrix")
        >>> report.generate("report.html")
    """
    
    def __init__(self, title: str, subtitle: str = ""):
        """Initialize report generator.
        
        Args:
            title: Main report title.
            subtitle: Subtitle (e.g., subject ID, analysis name).
        """
        self.title = title
        self.subtitle = subtitle
        self.sections: List[str] = []
        self._generated = False
    
    def add_header(self, text: str, level: int = 2) -> None:
        """Add a section header.
        
        Args:
            text: Header text.
            level: Header level (2-4).
        """
        level = max(2, min(4, level))
        self.sections.append(f"<h{level}>{text}</h{level}>")
    
    def add_text(self, text: str) -> None:
        """Add paragraph text.
        
        Args:
            text: Text content (can include HTML).
        """
        self.sections.append(f"<p>{text}</p>")
    
    def add_metrics(
        self,
        metrics: Dict[str, Any],
        section_title: Optional[str] = None,
    ) -> None:
        """Add metrics as styled cards.
        
        Args:
            metrics: Dictionary of metric names to values.
            section_title: Optional section header.
        """
        html = ""
        
        if section_title:
            html += f'<div class="section"><h2>{section_title}</h2>'
        
        html += '<div class="metrics-grid">'
        
        for name, value in metrics.items():
            # Format value
            if isinstance(value, float):
                formatted = f"{value:.3f}"
            else:
                formatted = str(value)
            
            html += f'''
            <div class="metric-card">
                <div class="metric-value">{formatted}</div>
                <div class="metric-label">{name}</div>
            </div>
            '''
        
        html += "</div>"
        
        if section_title:
            html += "</div>"
        
        self.sections.append(html)
    
    def add_image(
        self,
        figure: plt.Figure,
        caption: Optional[str] = None,
        width: Optional[int] = None,
    ) -> None:
        """Add matplotlib figure as embedded image.
        
        Args:
            figure: Matplotlib Figure object.
            caption: Image caption.
            width: Image width in pixels (None for auto).
        """
        # Convert figure to base64 PNG
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        
        # Build image HTML
        style = f'width: {width}px;' if width else ""
        html = f'''
        <div class="image-container">
            <img src="data:image/png;base64,{img_data}" style="{style}">
        '''
        
        if caption:
            html += f'<div class="image-caption">{caption}</div>'
        
        html += "</div>"
        
        self.sections.append(html)
    
    def add_image_file(
        self,
        image_path: Path,
        caption: Optional[str] = None,
        width: Optional[int] = None,
    ) -> None:
        """Add image from file path.
        
        Args:
            image_path: Path to image file.
            caption: Image caption.
            width: Image width in pixels.
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return
        
        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        # Read and encode image
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Build HTML
        style = f'width: {width}px;' if width else ""
        html = f'''
        <div class="image-container">
            <img src="data:{mime_type};base64,{img_data}" style="{style}">
        '''
        
        if caption:
            html += f'<div class="image-caption">{caption}</div>'
        
        html += "</div>"
        
        self.sections.append(html)
    
    def add_table(
        self,
        dataframe: pd.DataFrame,
        section_title: Optional[str] = None,
        max_rows: int = 50,
    ) -> None:
        """Add pandas DataFrame as HTML table.
        
        Args:
            dataframe: DataFrame to display.
            section_title: Optional section header.
            max_rows: Maximum rows to display.
        """
        html = ""
        
        if section_title:
            html += f'<div class="section"><h2>{section_title}</h2>'
        
        # Truncate if needed
        if len(dataframe) > max_rows:
            html += f'<p class="warning">Showing first {max_rows} of {len(dataframe)} rows</p>'
            dataframe = dataframe.head(max_rows)
        
        # Convert to HTML
        table_html = dataframe.to_html(
            index=True,
            classes="dataframe",
            border=0,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
        
        html += table_html
        
        if section_title:
            html += "</div>"
        
        self.sections.append(html)
    
    def add_connectivity_matrix(
        self,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        section_title: str = "Connectivity Matrix",
    ) -> None:
        """Add connectivity matrix visualization.
        
        Args:
            matrix: Square connectivity matrix.
            labels: Row/column labels.
            section_title: Section header.
        """
        from .visualization import plot_connectivity_matrix
        
        fig = plot_connectivity_matrix(
            matrix,
            labels=labels,
            title=section_title,
        )
        
        html = f'<div class="section"><h2>{section_title}</h2>'
        
        # Add figure
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.close(fig)
        
        html += f'''
        <div class="image-container">
            <img src="data:image/png;base64,{img_data}">
        </div>
        </div>
        '''
        
        self.sections.append(html)
    
    def add_statistical_summary(
        self,
        stats: Dict[str, Any],
        section_title: str = "Statistical Summary",
    ) -> None:
        """Add statistical analysis summary.
        
        Args:
            stats: Dictionary with statistical results.
            section_title: Section header.
        """
        html = f'<div class="section"><h2>{section_title}</h2>'
        
        # Format stats into table
        html += "<table>"
        html += "<tr><th>Measure</th><th>Value</th></tr>"
        
        for key, value in stats.items():
            # Format value
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            elif isinstance(value, (list, np.ndarray)):
                formatted = f"[{len(value)} items]"
            else:
                formatted = str(value)
            
            html += f"<tr><td>{key}</td><td>{formatted}</td></tr>"
        
        html += "</table></div>"
        
        self.sections.append(html)
    
    def add_warning(self, message: str) -> None:
        """Add warning message box.
        
        Args:
            message: Warning text.
        """
        self.sections.append(f'<div class="warning">⚠️ {message}</div>')
    
    def add_success(self, message: str) -> None:
        """Add success message box.
        
        Args:
            message: Success text.
        """
        self.sections.append(f'<div class="success">✓ {message}</div>')
    
    def add_section_start(self, title: str) -> None:
        """Start a new section container.
        
        Args:
            title: Section title.
        """
        self.sections.append(f'<div class="section"><h2>{title}</h2>')
    
    def add_section_end(self) -> None:
        """End current section container."""
        self.sections.append("</div>")
    
    def add_raw_html(self, html: str) -> None:
        """Add raw HTML content.
        
        Args:
            html: Raw HTML string.
        """
        self.sections.append(html)
    
    def generate(self, output_path: Path) -> Path:
        """Generate final HTML report.
        
        Args:
            output_path: Path to save HTML file.
        
        Returns:
            Path to generated report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build HTML document
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {REPORT_CSS}
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <div class="subtitle">{self.subtitle}</div>
    </div>
    
    {''.join(self.sections)}
    
    <div class="footer">
        Generated by Connectomix v3.0.0 | {timestamp}
    </div>
</body>
</html>
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        self._generated = True
        logger.info(f"Generated HTML report: {output_path}")
        
        return output_path


def generate_participant_report(
    subject_id: str,
    session: Optional[str],
    run_info: Dict[str, Any],
    output_dir: Path,
    figures: Optional[Dict[str, plt.Figure]] = None,
) -> Path:
    """Generate participant-level analysis report.
    
    Creates a summary report for a single participant's connectivity analysis.
    
    Args:
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session: Session label (without 'ses-' prefix).
        run_info: Dictionary with analysis metadata.
        output_dir: Directory to save report.
        figures: Optional dictionary of named figures to include.
    
    Returns:
        Path to generated report.
    """
    session_str = f" | Session: {session}" if session else ""
    subtitle = f"Subject: {subject_id}{session_str}"
    
    report = HTMLReportGenerator("Connectomix Participant Report", subtitle)
    
    # Analysis parameters section
    report.add_section_start("Analysis Parameters")
    
    params = {
        "Method": run_info.get("method", "N/A"),
        "Space": run_info.get("space", "N/A"),
        "Resolution": run_info.get("resolution", "N/A"),
        "N Volumes": run_info.get("n_volumes", "N/A"),
    }
    report.add_metrics(params)
    report.add_section_end()
    
    # Add figures if provided
    if figures:
        for name, fig in figures.items():
            if fig is not None:
                report.add_section_start(name)
                report.add_image(fig, caption=name)
                report.add_section_end()
    
    # Add QC metrics if available
    if "qc_metrics" in run_info:
        report.add_metrics(run_info["qc_metrics"], section_title="Quality Control")
    
    # Processing notes
    if "warnings" in run_info and run_info["warnings"]:
        report.add_section_start("Processing Notes")
        for warning in run_info["warnings"]:
            report.add_warning(warning)
        report.add_section_end()
    
    report.add_success("Analysis completed successfully")
    
    # Save report
    output_path = output_dir / f"sub-{subject_id}"
    if session:
        output_path = output_path / f"ses-{session}"
    output_path = output_path / "report.html"
    
    return report.generate(output_path)


def generate_group_report(
    analysis_name: str,
    run_info: Dict[str, Any],
    output_dir: Path,
    cluster_table: Optional[pd.DataFrame] = None,
    figures: Optional[Dict[str, plt.Figure]] = None,
) -> Path:
    """Generate group-level analysis report.
    
    Creates a summary report for group-level statistical analysis.
    
    Args:
        analysis_name: Name of the group analysis.
        run_info: Dictionary with analysis metadata.
        output_dir: Directory to save report.
        cluster_table: Optional cluster table from thresholding.
        figures: Optional dictionary of named figures to include.
    
    Returns:
        Path to generated report.
    """
    report = HTMLReportGenerator("Connectomix Group Report", analysis_name)
    
    # Analysis overview
    report.add_section_start("Analysis Overview")
    
    overview = {
        "N Subjects": run_info.get("n_subjects", "N/A"),
        "Method": run_info.get("method", "N/A"),
        "Contrast": run_info.get("contrast", "N/A"),
        "Threshold": run_info.get("threshold_method", "N/A"),
    }
    report.add_metrics(overview)
    report.add_section_end()
    
    # Design matrix
    if "design_matrix" in run_info and figures and "design_matrix" in figures:
        report.add_section_start("Design Matrix")
        report.add_image(figures["design_matrix"], caption="Group design matrix")
        report.add_section_end()
    
    # Statistical results
    if cluster_table is not None and not cluster_table.empty:
        report.add_table(
            cluster_table,
            section_title="Significant Clusters",
            max_rows=30,
        )
    else:
        report.add_section_start("Results")
        report.add_warning("No significant clusters found at the specified threshold")
        report.add_section_end()
    
    # Add stat map figures
    if figures:
        for name, fig in figures.items():
            if fig is not None and name != "design_matrix":
                report.add_section_start(name)
                report.add_image(fig, caption=name)
                report.add_section_end()
    
    # Statistical summary
    if "statistics" in run_info:
        report.add_statistical_summary(
            run_info["statistics"],
            section_title="Statistical Summary",
        )
    
    report.add_success("Group analysis completed successfully")
    
    # Save report
    output_path = output_dir / "group" / f"{analysis_name}_report.html"
    
    return report.generate(output_path)
