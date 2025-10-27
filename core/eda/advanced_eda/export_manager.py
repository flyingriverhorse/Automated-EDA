"""Export Manager for Advanced EDA.

Handles exporting EDA sessions to various formats including Jupyter notebooks,
HTML reports, and other analysis artifacts.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages export of EDA sessions to various formats."""
    
    def __init__(self, export_dir: Path = None):
        """Initialize the export manager.
        
        Args:
            export_dir: Directory to store exported files
        """
        self.export_dir = export_dir or Path(__file__).parent.parent.parent.parent / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    async def export_to_notebook(
        self,
        session_data: Dict[str, Any],
        user_id: str,
        export_format: str = "jupyter",
        include_outputs: bool = True
    ) -> Dict[str, Any]:
        """Export session data to a Jupyter notebook.
        
        Args:
            session_data: Session data to export
            user_id: User ID for the export
            export_format: Format to export (jupyter, html, pdf)
            include_outputs: Whether to include cell outputs
            
        Returns:
            Dict with export results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_id = session_data.get('dataset_id', 'unknown')
            filename = f"eda_session_{dataset_id}_{timestamp}.ipynb"
            
            # Create basic notebook structure
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Add header cell
            header_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# EDA Session Export\n\n",
                    f"**Dataset:** {dataset_id}\n",
                    f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                    f"**User:** {user_id}\n\n"
                ]
            }
            notebook["cells"].append(header_cell)
            
            # Add data loading cell
            data_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n\n",
                    f"# Load dataset {dataset_id}\n",
                    "# df = pd.read_csv('your_data_file.csv')  # Update with actual file path\n",
                    "print('Dataset loaded successfully')\n"
                ]
            }
            notebook["cells"].append(data_cell)
            
            # Add cells from session data
            cells = session_data.get('cells', [])
            for cell_data in cells:
                cell = {
                    "cell_type": cell_data.get("type", "code"),
                    "execution_count": cell_data.get("execution_count"),
                    "metadata": {},
                    "source": cell_data.get("source", [])
                }
                
                if include_outputs and cell_data.get("outputs"):
                    cell["outputs"] = cell_data["outputs"]
                else:
                    cell["outputs"] = []
                    
                notebook["cells"].append(cell)
            
            # Save notebook
            export_path = self.export_dir / filename
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "file_path": str(export_path),
                "filename": filename,
                "export_type": export_format
            }
            
        except Exception as e:
            logger.error(f"Notebook export error: {e}")
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }
    
    async def export_to_html(
        self,
        session_data: Dict[str, Any],
        user_id: str,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """Export session data to HTML report.
        
        Args:
            session_data: Session data to export
            user_id: User ID for the export
            include_plots: Whether to include plots in HTML
            
        Returns:
            Dict with export results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_id = session_data.get('dataset_id', 'unknown')
            filename = f"eda_report_{dataset_id}_{timestamp}.html"
            
            # Create basic HTML structure
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>EDA Report - {dataset_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #007bff; }}
                    .code {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }}
                    .output {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin-top: 10px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>EDA Session Report</h1>
                    <p><strong>Dataset:</strong> {dataset_id}</p>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>User:</strong> {user_id}</p>
                </div>
            """
            
            # Add analysis sections
            analyses = session_data.get('analyses', [])
            for i, analysis in enumerate(analyses, 1):
                html_content += f"""
                <div class="section">
                    <h2>Analysis {i}: {analysis.get('title', 'Untitled')}</h2>
                    <p>{analysis.get('description', 'No description available')}</p>
                    <div class="code">{analysis.get('code', 'No code available')}</div>
                    <div class="output">{analysis.get('output', 'No output available')}</div>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            export_path = self.export_dir / filename
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                "success": True,
                "file_path": str(export_path),
                "filename": filename,
                "export_type": "html"
            }
            
        except Exception as e:
            logger.error(f"HTML export error: {e}")
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }
