"""HTML report generation for Connectomix participant-level analyses.

This module provides comprehensive HTML report generation with:
- Analysis parameters and methods documentation
- Connectivity matrix visualizations
- Denoising timecourse plots
- Quality assurance metrics
- Scientific references
- Downloadable figures

Example:
    >>> from connectomix.utils.reports import ParticipantReportGenerator
    >>> report = ParticipantReportGenerator(
    ...     subject_id="01",
    ...     session="1",
    ...     config=config,
    ...     output_dir=Path("/data/output")
    ... )
    >>> report.add_connectivity_matrix(matrix, labels, "schaefer2018_100")
    >>> report.add_denoising_plot(confounds_df, ["trans_x", "rot_y"])
    >>> report.generate()
"""

import base64
import json
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for report generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from connectomix.core.version import __version__

logger = logging.getLogger(__name__)


# ============================================================================
# CSS Styles - Professional, modern, shareable design
# ============================================================================

REPORT_CSS = """
<style>
:root {
    --primary-color: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary-color: #7c3aed;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--gray-800);
    background-color: var(--gray-50);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

/* Navigation */
.nav-bar {
    position: sticky;
    top: 0;
    background: white;
    border-bottom: 1px solid var(--gray-200);
    padding: 15px 20px;
    z-index: 100;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.nav-content {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.nav-brand {
    font-weight: 700;
    font-size: 1.25em;
    color: var(--primary-color);
}

.nav-links {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.nav-links a {
    color: var(--gray-600);
    text-decoration: none;
    font-size: 0.9em;
    transition: color 0.2s;
}

.nav-links a:hover {
    color: var(--primary-color);
}

/* Header */
.header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 40px;
    margin-bottom: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}

.header h1 {
    margin: 0 0 10px 0;
    font-size: 2.5em;
    font-weight: 700;
}

.header .subtitle {
    font-size: 1.2em;
    opacity: 0.9;
}

.header .meta-info {
    margin-top: 20px;
    display: flex;
    gap: 30px;
    flex-wrap: wrap;
    font-size: 0.95em;
}

.header .meta-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Table of Contents */
.toc {
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.toc h2 {
    margin-top: 0;
    color: var(--gray-800);
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 10px;
}

.toc-list {
    list-style: none;
    padding: 0;
    margin: 0;
    columns: 2;
    column-gap: 40px;
}

.toc-list li {
    margin-bottom: 8px;
    break-inside: avoid;
}

.toc-list a {
    color: var(--gray-700);
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 8px;
}

.toc-list a:hover {
    color: var(--primary-color);
}

.toc-number {
    background: var(--gray-100);
    color: var(--gray-600);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: 600;
}

/* Sections */
.section {
    background: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.section h2 {
    color: var(--gray-800);
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 12px;
    margin-top: 0;
    margin-bottom: 25px;
    font-size: 1.5em;
}

.section h3 {
    color: var(--gray-700);
    margin-top: 25px;
    margin-bottom: 15px;
    font-size: 1.2em;
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.metric-card {
    background: var(--gray-50);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    border-left: 4px solid var(--primary-color);
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: var(--primary-color);
}

.metric-label {
    color: var(--gray-600);
    font-size: 0.9em;
    margin-top: 5px;
}

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    border-left: 4px solid var(--primary-color);
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
}

.info-box h4 {
    margin-top: 0;
    color: var(--primary-dark);
}

.info-box p {
    margin-bottom: 0.5em;
}

.info-box code {
    background: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}

/* Parameter Tables */
.param-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 0.95em;
}

.param-table th,
.param-table td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid var(--gray-200);
}

.param-table th {
    background: var(--gray-100);
    font-weight: 600;
    color: var(--gray-700);
}

.param-table tr:hover {
    background: var(--gray-50);
}

.param-table code {
    background: var(--gray-100);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
}

/* Figures */
.figure-container {
    margin: 25px 0;
    text-align: center;
}

.figure-wrapper {
    display: inline-block;
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    position: relative;
}

.figure-wrapper img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
}

.figure-caption {
    color: var(--gray-600);
    font-style: italic;
    margin-top: 12px;
    font-size: 0.95em;
}

.download-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    background: var(--primary-color);
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85em;
    opacity: 0.9;
    transition: opacity 0.2s;
}

.download-btn:hover {
    opacity: 1;
}

/* Code Blocks */
.code-block {
    background: var(--gray-900);
    color: #e5e7eb;
    padding: 20px;
    border-radius: 8px;
    overflow-x: auto;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
    line-height: 1.5;
    margin: 15px 0;
}

.code-block .comment {
    color: #6b7280;
}

.code-block .string {
    color: #10b981;
}

.code-block .keyword {
    color: #8b5cf6;
}

/* Alert Boxes */
.alert {
    padding: 15px 20px;
    border-radius: 8px;
    margin: 15px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.alert-icon {
    font-size: 1.2em;
    flex-shrink: 0;
}

.alert-success {
    background: #d1fae5;
    border: 1px solid var(--success-color);
    color: #065f46;
}

.alert-warning {
    background: #fef3c7;
    border: 1px solid var(--warning-color);
    color: #92400e;
}

.alert-info {
    background: #dbeafe;
    border: 1px solid var(--primary-color);
    color: #1e40af;
}

/* References */
.references {
    font-size: 0.95em;
}

.reference-item {
    margin-bottom: 15px;
    padding-left: 25px;
    position: relative;
}

.reference-item::before {
    content: "•";
    position: absolute;
    left: 8px;
    color: var(--primary-color);
}

.reference-item a {
    color: var(--primary-color);
    text-decoration: none;
}

.reference-item a:hover {
    text-decoration: underline;
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px 20px;
    color: var(--gray-600);
    font-size: 0.9em;
    border-top: 1px solid var(--gray-200);
    margin-top: 40px;
}

.footer a {
    color: var(--primary-color);
    text-decoration: none;
}

/* Tabs for multiple views */
.tabs {
    display: flex;
    border-bottom: 2px solid var(--gray-200);
    margin-bottom: 20px;
}

.tab {
    padding: 12px 20px;
    cursor: pointer;
    border: none;
    background: none;
    font-size: 1em;
    color: var(--gray-600);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
}

.tab:hover {
    color: var(--primary-color);
}

.tab.active {
    color: var(--primary-color);
    border-bottom-color: var(--primary-color);
    font-weight: 600;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

/* Collapsible sections */
.collapsible {
    cursor: pointer;
    padding: 15px;
    width: 100%;
    border: none;
    text-align: left;
    outline: none;
    font-size: 1.1em;
    background: var(--gray-100);
    border-radius: 8px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.collapsible:hover {
    background: var(--gray-200);
}

.collapsible::after {
    content: '▼';
    font-size: 0.8em;
    transition: transform 0.2s;
}

.collapsible.active::after {
    transform: rotate(180deg);
}

.collapsible-content {
    padding: 0 15px;
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    .header {
        padding: 25px;
    }
    
    .header h1 {
        font-size: 1.8em;
    }
    
    .nav-links {
        display: none;
    }
    
    .toc-list {
        columns: 1;
    }
    
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Print styles */
@media print {
    .nav-bar, .download-btn {
        display: none;
    }
    
    .section {
        break-inside: avoid;
    }
    
    body {
        background: white;
    }
}
</style>
"""


# ============================================================================
# JavaScript for interactivity
# ============================================================================

REPORT_JS = """
<script>
// Download figure functionality
function downloadFigure(imgId, filename) {
    const img = document.getElementById(imgId);
    const link = document.createElement('a');
    link.href = img.src;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Collapsible sections
document.addEventListener('DOMContentLoaded', function() {
    const collapsibles = document.querySelectorAll('.collapsible');
    collapsibles.forEach(function(coll) {
        coll.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + 'px';
            }
        });
    });
    
    // Tab functionality
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            const tabGroup = this.parentElement;
            const contentGroup = tabGroup.nextElementSibling;
            
            // Remove active class from all tabs and contents
            tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            contentGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            const targetId = this.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });
    
    // Smooth scroll for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
});
</script>
"""


# ============================================================================
# Scientific References
# ============================================================================

REFERENCES = {
    "fmriprep": {
        "title": "fMRIPrep: a robust preprocessing pipeline for functional MRI",
        "authors": "Esteban O, et al.",
        "journal": "Nature Methods",
        "year": "2019",
        "doi": "10.1038/s41592-018-0235-4",
        "url": "https://doi.org/10.1038/s41592-018-0235-4"
    },
    "nilearn": {
        "title": "Machine learning for neuroimaging with scikit-learn",
        "authors": "Abraham A, et al.",
        "journal": "Frontiers in Neuroinformatics",
        "year": "2014",
        "doi": "10.3389/fninf.2014.00014",
        "url": "https://doi.org/10.3389/fninf.2014.00014"
    },
    "schaefer": {
        "title": "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI",
        "authors": "Schaefer A, et al.",
        "journal": "Cerebral Cortex",
        "year": "2018",
        "doi": "10.1093/cercor/bhx179",
        "url": "https://doi.org/10.1093/cercor/bhx179"
    },
    "aal": {
        "title": "Automated Anatomical Labeling of Activations in SPM Using a Macroscopic Anatomical Parcellation of the MNI MRI Single-Subject Brain",
        "authors": "Tzourio-Mazoyer N, et al.",
        "journal": "NeuroImage",
        "year": "2002",
        "doi": "10.1006/nimg.2001.0978",
        "url": "https://doi.org/10.1006/nimg.2001.0978"
    },
    "connectivity": {
        "title": "Functional connectivity in the motor cortex of resting human brain using echo-planar MRI",
        "authors": "Biswal B, et al.",
        "journal": "Magnetic Resonance in Medicine",
        "year": "1995",
        "doi": "10.1002/mrm.1910340409",
        "url": "https://doi.org/10.1002/mrm.1910340409"
    },
    "denoising": {
        "title": "A Component Based Noise Correction Method (CompCor) for BOLD and Perfusion Based fMRI",
        "authors": "Behzadi Y, et al.",
        "journal": "NeuroImage",
        "year": "2007",
        "doi": "10.1016/j.neuroimage.2007.04.042",
        "url": "https://doi.org/10.1016/j.neuroimage.2007.04.042"
    }
}


# ============================================================================
# Report Generator Class
# ============================================================================

class ParticipantReportGenerator:
    """Generate comprehensive HTML reports for participant-level analyses.
    
    This class builds professional, navigable HTML reports with:
    - Analysis parameters and configuration
    - Connectivity matrix visualizations
    - Denoising quality assessment
    - Scientific references
    - Downloadable figures
    
    Attributes:
        subject_id: BIDS subject ID (can include full label like 'sub-01_ses-1_task-rest')
        config: ParticipantConfig instance
        output_dir: Directory for saving reports and figures
    """
    
    def __init__(
        self,
        subject_id: str,
        config: Any,  # ParticipantConfig
        output_dir: Path,
        confounds_df: Optional[pd.DataFrame] = None,
        selected_confounds: Optional[List[str]] = None,
        connectivity_matrix: Optional[np.ndarray] = None,
        roi_names: Optional[List[str]] = None,
        connectivity_paths: Optional[List[Path]] = None,
        logger: Optional[logging.Logger] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        space: Optional[str] = None,
        desc: Optional[str] = None,
        label: Optional[str] = None,
        censoring_summary: Optional[Dict[str, Any]] = None,
    ):
        """Initialize report generator.
        
        Args:
            subject_id: Subject ID or full BIDS label
            config: ParticipantConfig with analysis parameters
            output_dir: Output directory for reports
            confounds_df: Confounds DataFrame from fMRIPrep
            selected_confounds: List of confound columns used
            connectivity_matrix: Connectivity matrix (for ROI methods)
            roi_names: ROI labels for matrix methods
            connectivity_paths: List of output paths from connectivity
            logger: Logger instance
            session: Session ID (without 'ses-' prefix)
            task: Task name
            run: Run number
            space: Space name
            desc: Description entity for filename (e.g., atlas name, method)
            label: Custom label entity for filename
            censoring_summary: Summary of temporal censoring applied
        """
        self.subject_id = subject_id
        self.session = session
        self.config = config
        self.output_dir = Path(output_dir)
        self.task = task
        self.run = run
        self.space = space
        self.desc = desc
        self.label = label
        self.connectivity_paths = connectivity_paths or []
        self._logger = logger or logging.getLogger(__name__)
        
        # Figures directory for saving plots
        self.figures_dir: Optional[Path] = None
        
        # Storage for report content
        self.sections: List[str] = []
        self.figures: Dict[str, Tuple[plt.Figure, str]] = {}  # id -> (figure, caption)
        self.toc_items: List[Tuple[str, str]] = []  # (id, title)
        self._figure_counter = 0
        
        # QA metrics
        self.qa_metrics: Dict[str, Any] = {}
        
        # Denoising info
        self.confounds_used: List[str] = selected_confounds or []
        self.confounds_df: Optional[pd.DataFrame] = confounds_df
        
        # Connectivity results
        self.connectivity_matrices: List[Tuple[np.ndarray, List[str], str]] = []
        
        # Add connectivity matrix if provided
        if connectivity_matrix is not None and roi_names is not None:
            atlas_name = config.atlas if hasattr(config, 'atlas') and config.atlas else "connectivity"
            self.add_connectivity_matrix(connectivity_matrix, roi_names, atlas_name)
        
        # Command/config info
        self.command_line: Optional[str] = None
        self.config_dict: Optional[Dict] = None
        
        # Temporal censoring summary
        self.censoring_summary: Optional[Dict[str, Any]] = censoring_summary
        
        self._logger.debug(f"Initialized report generator for {subject_id}")
    
    def _get_unique_figure_id(self) -> str:
        """Get unique figure ID."""
        self._figure_counter += 1
        return f"fig_{self._figure_counter}"
    
    def _figure_to_base64(self, fig: plt.Figure, dpi: int = 150) -> str:
        """Convert matplotlib figure to base64 PNG."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return img_data
    
    def _save_figure_to_disk(self, fig: plt.Figure, filename: str, dpi: int = 150) -> Optional[Path]:
        """Save figure to the figures directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Filename for the saved figure (without path)
            dpi: Resolution for saving
            
        Returns:
            Path to saved figure, or None if figures_dir not set
        """
        if self.figures_dir is None:
            return None
        
        try:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            fig_path = self.figures_dir / filename
            fig.savefig(fig_path, format='png', dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._logger.debug(f"  Saved figure: {fig_path}")
            return fig_path
        except Exception as e:
            self._logger.warning(f"Could not save figure to disk: {e}")
            return None
    
    def set_command_line(self, command: str) -> None:
        """Store command line used to run analysis."""
        self.command_line = command
    
    def set_config_dict(self, config_dict: Dict) -> None:
        """Store configuration dictionary."""
        self.config_dict = config_dict
    
    def add_qa_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add quality assurance metrics."""
        self.qa_metrics.update(metrics)
    
    def add_connectivity_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        name: str,
    ) -> None:
        """Add connectivity matrix for visualization.
        
        Args:
            matrix: Connectivity matrix (N x N)
            labels: ROI labels
            name: Atlas/analysis name
        """
        self.connectivity_matrices.append((matrix, labels, name))
    
    def add_denoising_info(
        self,
        confounds_df: pd.DataFrame,
        confounds_used: List[str],
    ) -> None:
        """Add denoising information.
        
        Args:
            confounds_df: Full confounds DataFrame from fMRIPrep
            confounds_used: List of confound columns used in denoising
        """
        self.confounds_df = confounds_df
        self.confounds_used = confounds_used
    
    def _build_header(self) -> str:
        """Build report header section."""
        # subject_id may already include full BIDS label like 'sub-01_ses-1_task-rest'
        # or just the ID like '01'
        if self.subject_id.startswith('sub-'):
            # Already formatted, use as-is
            display_label = self.subject_id.replace('_', ' | ')
        else:
            # Build from parts
            session_str = f"ses-{self.session}" if self.session else ""
            task_str = f"task-{self.task}" if self.task else ""
            
            title_parts = [f"sub-{self.subject_id}"]
            if session_str:
                title_parts.append(session_str)
            if task_str:
                title_parts.append(task_str)
            display_label = ' | '.join(title_parts)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html = f'''
        <div class="header">
            <h1>🧠 Connectomix Participant Report</h1>
            <div class="subtitle">{display_label}</div>
            <div class="meta-info">
                <div class="meta-item">📅 {timestamp}</div>
                <div class="meta-item">🔬 Method: {self.config.method}</div>
                <div class="meta-item">📊 Connectomix v{__version__}</div>
            </div>
        </div>
        '''
        return html
    
    def _build_toc(self) -> str:
        """Build table of contents."""
        if not self.toc_items:
            return ""
        
        items_html = ""
        for i, (section_id, title) in enumerate(self.toc_items, 1):
            items_html += f'''
                <li>
                    <a href="#{section_id}">
                        <span class="toc-number">{i}</span>
                        {title}
                    </a>
                </li>
            '''
        
        return f'''
        <div class="toc">
            <h2>📋 Table of Contents</h2>
            <ul class="toc-list">
                {items_html}
            </ul>
        </div>
        '''
    
    def _build_overview_section(self) -> str:
        """Build analysis overview section."""
        self.toc_items.append(("overview", "Analysis Overview"))
        
        method_descriptions = {
            "roiToRoi": "ROI-to-ROI correlation analysis using atlas parcellation",
            "seedToVoxel": "Seed-based correlation mapping",
            "roiToVoxel": "ROI-based whole-brain correlation",
            "seedToSeed": "Seed-to-seed correlation matrix"
        }
        
        method_desc = method_descriptions.get(self.config.method, self.config.method)
        
        html = f'''
        <div class="section" id="overview">
            <h2>📊 Analysis Overview</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.config.method}</div>
                    <div class="metric-label">Analysis Method</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{getattr(self.config, 'atlas', 'N/A')}</div>
                    <div class="metric-label">Atlas</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.config.high_pass:.3f}</div>
                    <div class="metric-label">High-pass (Hz)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.config.low_pass:.3f}</div>
                    <div class="metric-label">Low-pass (Hz)</div>
                </div>
            </div>
            
            <h3>Method Description</h3>
            <p>{method_desc}</p>
            
            <div class="alert alert-info">
                <span class="alert-icon">ℹ️</span>
                <div>This analysis computes functional connectivity from preprocessed 
                fMRI data using confound regression and temporal filtering.</div>
            </div>
        </div>
        '''
        return html
    
    def _build_parameters_section(self) -> str:
        """Build analysis parameters section."""
        self.toc_items.append(("parameters", "Analysis Parameters"))
        
        # Preprocessing parameters
        preproc_params = [
            ("High-pass filter", f"{self.config.high_pass} Hz"),
            ("Low-pass filter", f"{self.config.low_pass} Hz"),
            ("Confounds", ", ".join(self.config.confounds[:5]) + ("..." if len(self.config.confounds) > 5 else "")),
            ("Standardize", "True"),
            ("Detrend", "True"),
        ]
        
        preproc_rows = ""
        for name, value in preproc_params:
            preproc_rows += f"<tr><td>{name}</td><td><code>{value}</code></td></tr>"
        
        # Method-specific parameters
        method_params = []
        if self.config.method in ["roiToRoi", "roiToVoxel"]:
            method_params.append(("Atlas", getattr(self.config, 'atlas', 'N/A')))
        if self.config.method in ["seedToVoxel", "seedToSeed"]:
            method_params.append(("Seeds file", str(getattr(self.config, 'seeds_file', 'N/A'))))
            method_params.append(("Sphere radius", f"{getattr(self.config, 'radius', 5.0)} mm"))
        
        method_rows = ""
        for name, value in method_params:
            method_rows += f"<tr><td>{name}</td><td><code>{value}</code></td></tr>"
        
        html = f'''
        <div class="section" id="parameters">
            <h2>⚙️ Analysis Parameters</h2>
            
            <h3>Preprocessing Parameters</h3>
            <table class="param-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                {preproc_rows}
            </table>
            
            <h3>Method-Specific Parameters</h3>
            <table class="param-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                {method_rows}
            </table>
        </div>
        '''
        return html
    
    def _build_confounds_section(self) -> str:
        """Build confounds/denoising section."""
        self.toc_items.append(("denoising", "Denoising"))
        
        confounds_list = "</li><li>".join(self.config.confounds)
        n_confounds = len(self.config.confounds)
        
        html = f'''
        <div class="section" id="denoising">
            <h2>🧹 Denoising Strategy</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{n_confounds}</div>
                    <div class="metric-label">Confound Regressors</div>
                </div>
            </div>
            
            <h3>Confounds Used</h3>
            <ul>
                <li>{confounds_list}</li>
            </ul>
            
            <p>These confounds were regressed from the BOLD signal before computing 
            connectivity. The signal was also bandpass filtered between 
            {self.config.high_pass} and {self.config.low_pass} Hz.</p>
        '''
        
        # Add denoising plot if we have confounds data
        if self.confounds_df is not None and len(self.confounds_used) > 0:
            fig = self._create_confounds_plot()
            if fig is not None:
                fig_id = self._get_unique_figure_id()
                img_data = self._figure_to_base64(fig)
                
                # Save figure to disk
                self._save_figure_to_disk(fig, 'confounds_timeseries.png')
                
                plt.close(fig)
                
                html += f'''
                <h3>Confound Time Series</h3>
                <div class="figure-container">
                    <div class="figure-wrapper">
                        <img id="{fig_id}" src="data:image/png;base64,{img_data}">
                        <button class="download-btn" onclick="downloadFigure('{fig_id}', 'confounds_timeseries.png')">
                            ⬇️ Download
                        </button>
                    </div>
                    <div class="figure-caption">
                        Figure: Time series of confound regressors used for denoising. 
                        Values are z-scored for visualization.
                    </div>
                </div>
                '''
            
            # Add confounds correlation matrix
            corr_fig = self._create_confounds_correlation_plot()
            if corr_fig is not None:
                corr_fig_id = self._get_unique_figure_id()
                corr_img_data = self._figure_to_base64(corr_fig)
                
                # Save figure to disk
                self._save_figure_to_disk(corr_fig, 'confounds_correlation.png')
                
                plt.close(corr_fig)
                
                html += f'''
                <h3>Confound Inter-Correlations</h3>
                <div class="figure-container">
                    <div class="figure-wrapper">
                        <img id="{corr_fig_id}" src="data:image/png;base64,{corr_img_data}">
                        <button class="download-btn" onclick="downloadFigure('{corr_fig_id}', 'confounds_correlation.png')">
                            ⬇️ Download
                        </button>
                    </div>
                    <div class="figure-caption">
                        Figure: Pearson correlation matrix between confound regressors. 
                        High correlations between confounds may indicate redundancy in the denoising strategy.
                    </div>
                </div>
                '''
        
        html += "</div>"
        return html
    
    def _create_confounds_plot(self) -> Optional[plt.Figure]:
        """Create confounds time series plot."""
        try:
            # Select confounds that exist in the dataframe
            available = [c for c in self.confounds_used if c in self.confounds_df.columns]
            if not available:
                return None
            
            # Limit to first 12 confounds for readability
            available = available[:12]
            
            fig, axes = plt.subplots(len(available), 1, figsize=(12, 2 * len(available)), 
                                      sharex=True)
            if len(available) == 1:
                axes = [axes]
            
            for i, confound in enumerate(available):
                data = self.confounds_df[confound].values
                # Z-score for visualization
                data = (data - np.nanmean(data)) / (np.nanstd(data) + 1e-10)
                
                axes[i].plot(data, color='#2563eb', linewidth=0.8)
                axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
                axes[i].set_ylabel(confound, fontsize=9)
                axes[i].set_xlim(0, len(data))
                axes[i].tick_params(labelsize=8)
                
                # Remove top/right spines
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
            
            axes[-1].set_xlabel('Volume', fontsize=10)
            fig.suptitle('Confound Regressors (z-scored)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.warning(f"Could not create confounds plot: {e}")
            return None
    
    def _create_confounds_correlation_plot(self) -> Optional[plt.Figure]:
        """Create correlation matrix plot between confounds."""
        try:
            # Select confounds that exist in the dataframe
            available = [c for c in self.confounds_used if c in self.confounds_df.columns]
            if len(available) < 2:
                return None
            
            # Limit to first 20 confounds for readability
            available = available[:20]
            
            # Extract data and compute correlation matrix
            confounds_data = self.confounds_df[available].dropna()
            if len(confounds_data) < 10:  # Need enough data points
                return None
            
            corr_matrix = confounds_data.corr().values
            
            # Determine figure size based on number of confounds
            n_conf = len(available)
            base_size = max(6, min(12, n_conf * 0.5))
            fig, ax = plt.subplots(figsize=(base_size, base_size))
            
            # Plot heatmap
            vmax = 1.0
            vmin = -1.0
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
            cbar.ax.tick_params(labelsize=9)
            
            # Add labels
            ax.set_xticks(range(n_conf))
            ax.set_yticks(range(n_conf))
            
            # Shorten labels if too long
            short_labels = [l[:15] + '...' if len(l) > 15 else l for l in available]
            ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
            ax.set_yticklabels(short_labels, fontsize=8)
            
            ax.set_title('Confound Inter-Correlation Matrix', fontsize=12, fontweight='bold', pad=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.warning(f"Could not create confounds correlation plot: {e}")
            return None
    
    def _build_censoring_section(self) -> str:
        """Build temporal censoring section."""
        if self.censoring_summary is None or not self.censoring_summary.get('enabled', False):
            return ""
        
        self.toc_items.append(("censoring", "Temporal Censoring"))
        
        summary = self.censoring_summary
        n_original = summary.get('n_original', 0)
        n_retained = summary.get('n_retained', 0)
        n_censored = summary.get('n_censored', 0)
        fraction = summary.get('fraction_retained', 1.0)
        reason_counts = summary.get('reason_counts', {})
        
        # Determine retention status color
        if fraction < 0.5:
            status_class = "badge-error"
            status_text = "Low Retention"
        elif fraction < 0.7:
            status_class = "badge-warning"
            status_text = "Moderate Retention"
        else:
            status_class = "badge-success"
            status_text = "Good Retention"
        
        # Check if this is condition-based censoring
        conditions = summary.get('conditions', {})
        has_conditions = len(conditions) > 0
        
        # Description text varies based on censoring type
        if has_conditions:
            description = '''Temporal censoring was applied to select specific task conditions. 
            Connectivity was computed separately for each condition using only the timepoints 
            belonging to that condition.'''
            retained_label = "Volumes Used (union of conditions)"
        else:
            description = '''Temporal censoring removes specific timepoints (volumes) from the fMRI data 
            before connectivity analysis. This is useful for excluding high-motion frames 
            or initial equilibration volumes.'''
            retained_label = "Retained Volumes"
        
        html = f'''
        <div class="section" id="censoring">
            <h2>⏱️ Temporal Censoring</h2>
            
            <p>{description}</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{n_original}</div>
                    <div class="metric-label">Original Volumes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_retained}</div>
                    <div class="metric-label">{retained_label}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_censored}</div>
                    <div class="metric-label">Excluded Volumes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fraction:.1%}</div>
                    <div class="metric-label"><span class="{status_class}">{status_text}</span></div>
                </div>
            </div>
        '''
        
        # Censoring reasons breakdown (only if there are reasons from global censoring)
        if reason_counts:
            html += '''
            <h3>Censoring Breakdown</h3>
            <table>
                <thead>
                    <tr>
                        <th>Reason</th>
                        <th>Volumes Censored</th>
                    </tr>
                </thead>
                <tbody>
            '''
            
            reason_labels = {
                'initial_drop': 'Initial volume drop (dummy scans)',
                'custom_mask': 'Custom censoring mask',
            }
            
            for reason, count in sorted(reason_counts.items()):
                # Format reason for display
                if reason.startswith('motion_fd>'):
                    fd_thresh = reason.replace('motion_fd>', '')
                    label = f'Motion censoring (FD > {fd_thresh}mm)'
                else:
                    label = reason_labels.get(reason, reason)
                
                html += f'''
                    <tr>
                        <td>{label}</td>
                        <td>{count}</td>
                    </tr>
                '''
            
            html += '''
                </tbody>
            </table>
            '''
        
        # Condition-specific breakdown
        conditions = summary.get('conditions', {})
        if conditions:
            html += '''
            <h3>Condition-Specific Volumes</h3>
            <p>Connectivity was computed separately for each condition using only 
            the timepoints belonging to that condition:</p>
            <table>
                <thead>
                    <tr>
                        <th>Condition</th>
                        <th>Volumes</th>
                        <th>Fraction of Total</th>
                    </tr>
                </thead>
                <tbody>
            '''
            
            for cond_name, cond_info in sorted(conditions.items()):
                cond_vols = cond_info.get('n_volumes', 0)
                cond_frac = cond_info.get('fraction', 0)
                
                html += f'''
                    <tr>
                        <td><strong>{cond_name}</strong></td>
                        <td>{cond_vols}</td>
                        <td>{cond_frac:.1%}</td>
                    </tr>
                '''
            
            html += '''
                </tbody>
            </table>
            '''
        
        # Create censoring visualization
        censor_fig = self._create_censoring_plot()
        if censor_fig is not None:
            fig_id = self._get_unique_figure_id()
            img_data = self._figure_to_base64(censor_fig)
            self._save_figure_to_disk(censor_fig, 'temporal_censoring.png')
            plt.close(censor_fig)
            
            html += f'''
            <h3>Censoring Mask Visualization</h3>
            <div class="figure-container">
                <div class="figure-wrapper">
                    <img id="{fig_id}" src="data:image/png;base64,{img_data}">
                    <button class="download-btn" onclick="downloadFigure('{fig_id}', 'temporal_censoring.png')">
                        ⬇️ Download
                    </button>
                </div>
                <div class="figure-caption">
                    Figure: Temporal censoring mask. Green = retained volumes, Red = censored volumes.
                </div>
            </div>
            '''
        
        html += "</div>"
        return html
    
    def _create_censoring_plot(self) -> Optional[plt.Figure]:
        """Create temporal censoring visualization."""
        if self.censoring_summary is None:
            return None
        
        try:
            mask = self.censoring_summary.get('mask', [])
            if not mask:
                return None
            
            mask = np.array(mask)
            n_volumes = len(mask)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 2))
            
            # Create image representation of mask
            # Green for retained, red for censored
            colors = np.zeros((1, n_volumes, 3))
            colors[0, mask, :] = [0.2, 0.8, 0.2]   # Green for retained
            colors[0, ~mask, :] = [0.9, 0.2, 0.2]  # Red for censored
            
            ax.imshow(colors, aspect='auto', extent=[0, n_volumes, 0, 1])
            ax.set_xlabel('Volume', fontsize=10)
            ax.set_yticks([])
            ax.set_xlim(0, n_volumes)
            ax.set_title('Temporal Censoring Mask', fontsize=12, fontweight='bold')
            
            # Add text summary
            n_retained = np.sum(mask)
            ax.text(n_volumes * 0.98, 0.5, f'{n_retained}/{n_volumes} retained',
                   ha='right', va='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.warning(f"Could not create censoring plot: {e}")
            return None
    
    def _build_connectivity_section(self) -> str:
        """Build connectivity results section."""
        if not self.connectivity_matrices:
            return ""
        
        self.toc_items.append(("connectivity", "Connectivity Results"))
        
        # Find data file path for this connectivity result
        data_file_path = ""
        if self.connectivity_paths:
            # Find .npy files (matrix files)
            matrix_files = [p for p in self.connectivity_paths if p.suffix == '.npy']
            if matrix_files:
                data_file_path = str(matrix_files[0])
        
        # Correlation method explanation
        correlation_explanation = """
            <div class="info-box">
                <h4>📐 Correlation Method</h4>
                <p>The connectivity matrix displays <strong>Pearson correlation coefficients</strong> 
                between the time series of each pair of regions. For each pair of regions (i, j), 
                the correlation r<sub>ij</sub> is computed as:</p>
                <p style="text-align: center; font-family: monospace; margin: 10px 0;">
                    r<sub>ij</sub> = Σ[(x<sub>i</sub> - μ<sub>i</sub>)(x<sub>j</sub> - μ<sub>j</sub>)] / (σ<sub>i</sub> × σ<sub>j</sub> × n)
                </p>
                <p>where x<sub>i</sub> and x<sub>j</sub> are the time series, μ and σ are their means 
                and standard deviations, and n is the number of time points. Values range from -1 
                (perfect anti-correlation) to +1 (perfect correlation). The diagonal is set to zero 
                as self-correlations are not meaningful for connectivity analysis.</p>
            </div>
        """
        
        html = f'''
        <div class="section" id="connectivity">
            <h2>🔗 Connectivity Results</h2>
            
            {correlation_explanation}
        '''
        
        for i, (matrix, labels, name) in enumerate(self.connectivity_matrices):
            fig = self._create_connectivity_plot(matrix, labels, name)
            if fig is not None:
                fig_id = self._get_unique_figure_id()
                img_data = self._figure_to_base64(fig, dpi=150)
                
                # Save figure to disk
                fig_filename = f"connectivity_{name}.png"
                self._save_figure_to_disk(fig, fig_filename, dpi=150)
                
                plt.close(fig)
                
                # Compute summary statistics
                upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                mean_conn = np.mean(upper_tri)
                std_conn = np.std(upper_tri)
                max_conn = np.max(upper_tri)
                min_conn = np.min(upper_tri)
                
                # Get specific data file for this matrix if multiple
                current_data_file = ""
                if i < len(self.connectivity_paths):
                    matrix_files = [p for p in self.connectivity_paths if p.suffix == '.npy']
                    if i < len(matrix_files):
                        current_data_file = str(matrix_files[i])
                elif data_file_path:
                    current_data_file = data_file_path
                
                html += f'''
                <h3>{name}</h3>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{matrix.shape[0]}</div>
                        <div class="metric-label">ROIs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{mean_conn:.3f}</div>
                        <div class="metric-label">Mean Correlation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{std_conn:.3f}</div>
                        <div class="metric-label">Std Correlation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">[{min_conn:.2f}, {max_conn:.2f}]</div>
                        <div class="metric-label">Range</div>
                    </div>
                </div>
                
                <div class="figure-container">
                    <div class="figure-wrapper">
                        <img id="{fig_id}" src="data:image/png;base64,{img_data}">
                        <button class="download-btn" onclick="downloadFigure('{fig_id}', 'connectivity_{name}.png')">
                            ⬇️ Download
                        </button>
                    </div>
                    <div class="figure-caption">
                        Figure: Pearson correlation matrix showing pairwise connectivity between 
                        {matrix.shape[0]} regions from the {name} atlas.
                        <br><strong>Data file:</strong> <code>{current_data_file}</code>
                    </div>
                </div>
                '''
        
        html += "</div>"
        return html
    
    def _create_connectivity_plot(
        self,
        matrix: np.ndarray,
        labels: List[str],
        name: str
    ) -> Optional[plt.Figure]:
        """Create connectivity matrix plot using nilearn-style visualization."""
        try:
            n_regions = matrix.shape[0]
            
            # Determine figure size based on matrix size
            base_size = min(12, max(8, n_regions / 10))
            fig, ax = plt.subplots(figsize=(base_size, base_size))
            
            # Use nilearn-style diverging colormap
            vmax = np.max(np.abs(matrix[~np.eye(n_regions, dtype=bool)]))
            vmin = -vmax
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
            cbar.ax.tick_params(labelsize=9)
            
            # Add labels for smaller matrices
            if n_regions <= 50 and labels:
                ax.set_xticks(range(n_regions))
                ax.set_yticks(range(n_regions))
                ax.set_xticklabels(labels, rotation=90, fontsize=7)
                ax.set_yticklabels(labels, fontsize=7)
            else:
                ax.set_xlabel(f'Regions (n={n_regions})', fontsize=11)
                ax.set_ylabel(f'Regions (n={n_regions})', fontsize=11)
            
            ax.set_title(f'{name} Connectivity Matrix', fontsize=13, fontweight='bold', pad=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.warning(f"Could not create connectivity plot: {e}")
            return None
    
    def _build_qa_section(self) -> str:
        """Build quality assurance section."""
        if not self.qa_metrics:
            return ""
        
        self.toc_items.append(("qa", "Quality Assurance"))
        
        metrics_html = ""
        for name, value in self.qa_metrics.items():
            if isinstance(value, float):
                formatted = f"{value:.3f}"
            else:
                formatted = str(value)
            
            metrics_html += f'''
            <div class="metric-card">
                <div class="metric-value">{formatted}</div>
                <div class="metric-label">{name}</div>
            </div>
            '''
        
        html = f'''
        <div class="section" id="qa">
            <h2>✅ Quality Assurance</h2>
            
            <div class="metrics-grid">
                {metrics_html}
            </div>
            
            <div class="alert alert-success">
                <span class="alert-icon">✓</span>
                <div>Quality metrics are within acceptable ranges for functional connectivity analysis.</div>
            </div>
        </div>
        '''
        return html
    
    def _build_command_section(self) -> str:
        """Build command line / configuration section."""
        self.toc_items.append(("reproducibility", "Reproducibility"))
        
        html = '''
        <div class="section" id="reproducibility">
            <h2>🔄 Reproducibility</h2>
            
            <p>The following information can be used to reproduce this analysis.</p>
        '''
        
        if self.command_line:
            html += f'''
            <h3>Command Line</h3>
            <div class="code-block">
{self.command_line}
            </div>
            '''
        
        if self.config_dict:
            config_json = json.dumps(self.config_dict, indent=2, default=str)
            html += f'''
            <h3>Configuration</h3>
            <button class="collapsible">View Full Configuration</button>
            <div class="collapsible-content">
                <div class="code-block">
{config_json}
                </div>
            </div>
            '''
        
        # Software versions
        html += f'''
        <h3>Software Versions</h3>
        <table class="param-table">
            <tr><th>Software</th><th>Version</th></tr>
            <tr><td>Connectomix</td><td><code>{__version__}</code></td></tr>
            <tr><td>Python</td><td><code>{sys.version.split()[0]}</code></td></tr>
            <tr><td>NumPy</td><td><code>{np.__version__}</code></td></tr>
            <tr><td>Pandas</td><td><code>{pd.__version__}</code></td></tr>
        </table>
        </div>
        '''
        return html
    
    def _build_references_section(self) -> str:
        """Build scientific references section."""
        self.toc_items.append(("references", "References"))
        
        # Select relevant references based on analysis
        refs_to_include = ["fmriprep", "nilearn", "connectivity", "denoising"]
        
        if hasattr(self.config, 'atlas'):
            if "schaefer" in self.config.atlas.lower():
                refs_to_include.append("schaefer")
            elif "aal" in self.config.atlas.lower():
                refs_to_include.append("aal")
        
        refs_html = ""
        for ref_key in refs_to_include:
            if ref_key in REFERENCES:
                ref = REFERENCES[ref_key]
                refs_html += f'''
                <div class="reference-item">
                    <strong>{ref["authors"]}</strong> ({ref["year"]}). 
                    {ref["title"]}. <em>{ref["journal"]}</em>. 
                    <a href="{ref["url"]}" target="_blank">DOI: {ref["doi"]}</a>
                </div>
                '''
        
        html = f'''
        <div class="section" id="references">
            <h2>📚 References</h2>
            
            <p>If you use Connectomix in your research, please cite the following:</p>
            
            <div class="references">
                {refs_html}
            </div>
            
            <h3>Software Repository</h3>
            <p>
                Connectomix is open-source software available at:<br>
                <a href="https://github.com/ln2t/connectomix" target="_blank">
                    https://github.com/ln2t/connectomix
                </a>
            </p>
        </div>
        '''
        return html
    
    def _build_nav_bar(self) -> str:
        """Build navigation bar."""
        links = ""
        for section_id, title in self.toc_items:
            links += f'<a href="#{section_id}">{title}</a>'
        
        return f'''
        <nav class="nav-bar">
            <div class="nav-content">
                <div class="nav-brand">🧠 Connectomix Report</div>
                <div class="nav-links">
                    {links}
                </div>
            </div>
        </nav>
        '''
    
    def generate(self) -> Path:
        """Generate the complete HTML report.
        
        Returns:
            Path to generated report file.
        """
        self._logger.info(f"Generating participant report for {self.subject_id}")
        
        # Determine output paths first so we can set up figures directory
        # subject_id could be: '01', 'sub-01', 'sub-01_ses-1_task-rest', etc.
        if self.subject_id.startswith('sub-'):
            # Parse BIDS entities from subject_id
            parts = self.subject_id.split('_')
            sub_part = parts[0]  # sub-XX
            ses_part = None
            for p in parts[1:]:
                if p.startswith('ses-'):
                    ses_part = p
            
            # Build output path
            output_path = self.output_dir / sub_part
            if ses_part:
                output_path = output_path / ses_part
        else:
            # Simple subject ID without BIDS formatting
            output_path = self.output_dir / f"sub-{self.subject_id}"
            if self.session:
                output_path = output_path / f"ses-{self.session}"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up figures directory
        self.figures_dir = output_path / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"  Figures directory: {self.figures_dir}")
        
        # Build all sections (this will save figures to figures_dir)
        sections_html = ""
        sections_html += self._build_overview_section()
        sections_html += self._build_parameters_section()
        sections_html += self._build_confounds_section()
        sections_html += self._build_censoring_section()
        sections_html += self._build_connectivity_section()
        sections_html += self._build_qa_section()
        sections_html += self._build_command_section()
        sections_html += self._build_references_section()
        
        # Build navigation and TOC
        nav_html = self._build_nav_bar()
        toc_html = self._build_toc()
        header_html = self._build_header()
        
        # Build footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_html = f'''
        <div class="footer">
            <p>Generated by <strong>Connectomix v{__version__}</strong></p>
            <p>{timestamp}</p>
            <p>
                <a href="https://github.com/ln2t/connectomix" target="_blank">GitHub</a> | 
                <a href="https://github.com/ln2t/connectomix/issues" target="_blank">Report Issues</a>
            </p>
        </div>
        '''
        
        # Build title from subject_id
        title_label = self.subject_id if self.subject_id.startswith('sub-') else f"sub-{self.subject_id}"
        
        # Assemble full HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Connectomix Report - {title_label}</title>
    {REPORT_CSS}
</head>
<body>
    {nav_html}
    
    <div class="container">
        {header_html}
        {toc_html}
        {sections_html}
        {footer_html}
    </div>
    
    {REPORT_JS}
</body>
</html>
'''
        
        # Build report filename
        if self.subject_id.startswith('sub-'):
            filename = self.subject_id
            if self.label:
                filename += f"_label-{self.label}"
            if self.desc:
                filename += f"_desc-{self.desc}"
            filename += "_report.html"
            report_path = output_path / filename
        else:
            filename_parts = [f"sub-{self.subject_id}"]
            if self.session:
                filename_parts.append(f"ses-{self.session}")
            if self.task:
                filename_parts.append(f"task-{self.task}")
            if self.label:
                filename_parts.append(f"label-{self.label}")
            if self.desc:
                filename_parts.append(f"desc-{self.desc}")
            filename_parts.append("report.html")
            report_path = output_path / "_".join(filename_parts)
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self._logger.info(f"Saved participant report: {report_path}")
        self._logger.info(f"Figures saved to: {self.figures_dir}")
        
        return report_path


# ============================================================================
# Convenience function for backward compatibility
# ============================================================================

def generate_participant_report(
    subject_id: str,
    session: Optional[str],
    run_info: Dict[str, Any],
    output_dir: Path,
    figures: Optional[Dict[str, plt.Figure]] = None,
) -> Path:
    """Generate participant-level analysis report (legacy interface).
    
    For new code, use ParticipantReportGenerator directly.
    """
    from connectomix.config.defaults import ParticipantConfig
    
    # Create a minimal config
    config = ParticipantConfig()
    
    report = ParticipantReportGenerator(
        subject_id=subject_id,
        session=session,
        config=config,
        output_dir=output_dir,
    )
    
    return report.generate()
