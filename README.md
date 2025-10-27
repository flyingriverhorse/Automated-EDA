# Advanced EDA — Developer Documentation

## Overview

The Advanced EDA (Exploratory Data Analysis) system is a modular, extensible framework for automated data analysis. It provides a comprehensive suite of statistical analyses, visualizations, and data quality checks that can be dynamically composed and executed based on your dataset characteristics and analysis needs.

### Key Goals

- **Automation**: Automatically generate relevant analyses based on dataset properties
- **Modularity**: Each analysis component is self-contained and independently testable
- **Extensibility**: Easy to add new analysis types without modifying core logic
- **Code Generation**: Produce executable Python code for reproducibility
- **API-First**: RESTful API for integration with any frontend or data pipeline

## Architecture

The system is built around a component-based architecture:

1. **Components**: Individual analysis units (e.g., correlation analysis, outlier detection)
2. **Registry**: Central catalog of all available components and their capabilities
3. **Code Generators**: Transform component specifications into executable Python code
4. **Service Layer**: Orchestrates component selection and execution
5. **API Layer**: FastAPI routes exposing functionality via HTTP

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn

### Installation

```bash
# Install dependencies
pip install -r requirements.txt 

# (Optional) Install development dependencies
You can use: uv
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or with custom configuration
uvicorn main:app --reload --port 8080
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Basic Usage

```bash
# Example: Analyze a CSV file
curl -X POST "http://localhost:8000/api/eda/analyze" \
  -F "file=@your_data.csv" \
  -F "analysis_types=correlation_analysis,outlier_detection"

# Get list of available components
curl "http://localhost:8000/api/eda/components"
```

## Documentation Structure

### Core Documentation

- **`services.AdvancedEDAService.md`** — Main service orchestration methods
- **`generators.GranularAnalysisCodeGenerators.md`** — Code generation utilities
- **`routes.endpoints.md`** — FastAPI endpoint reference
- **`components.contract.md`** — Component interface specification and registry

### High-Level Guides

- **`GRANULAR_EDA_COMPONENTS_GUIDE.md`** — How to add new components and extend the system

## Component Library

Per-component READMEs are located in `docs/advanced_eda/components/`:

### Data Quality & Structure
- `dataset_shape_analysis.md` — Dataset dimensions and memory usage
- `data_range_validation.md` — Value range checks and constraints
- `data_types_validation.md` — Schema validation and type inference
- `missing_value_analysis.md` — Missing data patterns and statistics
- `duplicate_detection.md` — Duplicate row identification

### Univariate Analysis (Numeric)
- `summary_statistics.md` — Mean, median, std dev, quartiles
- `distribution_plots.md` — Histograms and density plots
- `skewness_analysis.md` — Distribution symmetry measures
- `normality_test.md` — Shapiro-Wilk and other normality tests

### Univariate Analysis (Categorical)
- `categorical_frequency_analysis.md` — Value counts and proportions
- `categorical_visualization.md` — Bar charts and pie charts

### Bivariate/Multivariate Analysis
- `correlation_analysis.md` — Pearson, Spearman correlation matrices
- `scatter_plot_analysis.md` — Pairwise relationship visualization
- `cross_tabulation_analysis.md` — Contingency tables and chi-square tests

### Outlier & Anomaly Detection
- `iqr_outlier_detection.md` — Interquartile range method
- `zscore_outlier_detection.md` — Z-score based detection
- `visual_outlier_inspection.md` — Box plots and visual tools

### Time-Series Exploration
- `temporal_trend_analysis.md` — Trend decomposition and analysis
- `seasonality_detection.md` — Seasonal pattern identification

### Relationship Exploration
- `multicollinearity_analysis.md` — VIF and correlation diagnostics
- `pca_dimensionality_reduction.md` — Principal component analysis

## Adding New Components

See `GRANULAR_EDA_COMPONENTS_GUIDE.md` for detailed instructions. Quick overview:

1. Create component class implementing the required interface
2. Register component in the component registry
3. Add code generation logic in `GranularAnalysisCodeGenerators`
4. Update component documentation
5. Add tests

## API Endpoints

Key endpoints (see `routes.endpoints.md` for complete reference):

- `POST /api/eda/analyze` — Run analysis on uploaded dataset
- `GET /api/eda/components` — List all available components
- `POST /api/eda/generate-code` — Generate executable analysis code
- `GET /api/eda/health` — Health check endpoint

## Development

```bash
# Run tests
pytest tests/

# Run with hot reload for development
uvicorn main:app --reload

# Format code
black .

# Lint
flake8 .
```

## Contributing

When adding new components or features:

1. Follow the component contract specification
2. Add comprehensive documentation
3. Include unit tests
4. Update relevant README files
5. Ensure backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.