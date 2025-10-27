# Granular EDA Components — Developer Guide

This guide explains how the Advanced EDA’s granular component system works, how to add new components safely, what to validate, how to test end-to-end, and the issues we ran into (with root causes and fixes) so we don’t repeat them.


## Overview

The Advanced EDA module was refactored from “monolithic analyses” into small, composable granular components. Each component is a single, focused analysis (e.g., correlation_analysis, summary_statistics, missing_value_analysis). Components declare their own metadata, validate compatibility against the dataset, and generate executable Python code. The backend stitches multiple components together into one runnable script and executes it against the uploaded dataset.


## Architecture map

- Granular Components (one class per analysis)
  - Location: `core/eda/advanced_eda/granular_components/**`
  - Contract methods in each component:
    - `get_metadata() -> Dict[str, Any]`
    - `validate_data_compatibility(data_preview) -> bool`
    - `generate_code(data_preview) -> str`
- Component Registry
  - Location: `core/eda/advanced_eda/granular_components/__init__.py`
  - Exposes:
    - `get_all_granular_components()` → { id: { component: Class, category: str, metadata: dict } }
    - `get_components_by_category()` → { category: [{ id, component, metadata }, ...] }
- Code Generators
  - Location: `core/eda/advanced_eda/granular_generators.py`
  - Class: `GranularAnalysisCodeGenerators`
    - Builds display names and metadata
    - Validates compatibility via component’s `validate_data_compatibility`
    - Generates code for one or multiple analysis IDs
- Service layer
  - Location: `core/eda/advanced_eda/services.py`
  - Class: `AdvancedEDAService`
    - `run_granular_analysis(source_id, analysis_ids)` → generate code + execute
    - `generate_analysis_code(analysis_ids, source_id)` → generate only
    - `validate_analysis_compatibility(analysis_id, source_id)` → checks data
    - `execute_code(source_id, code, context)` → runs code (subprocess) with dataset loaded as `df`
    - `_get_full_dataset_context(source_id)` → builds `data_preview` for compatibility decisions
- Routes (FastAPI)
  - Location: `core/eda/advanced_eda/routes.py`
  - Endpoints (prefix `/advanced-eda`):
    - `GET /components/available` → grouped component list
    - `GET /components/{component_id}/info` → metadata for one component
    - `GET /components/recommendations?source_id=...` → recommended component IDs
    - `POST /components/generate-code` → code only
    - `POST /components/run-analysis` → code + execute and return outputs
    - Plus notebook/custom code endpoints (LLM and custom) for completeness


## Component contract (must-have methods)

Implement these in every component class:

- get_metadata() → Dict[str, Any]
  - Required keys we use:
    - `name` (machine id, usually matches registry key)
    - `display_name` (human-readable)
    - `description`
    - `category` (free-form; high-level grouping)
    - `complexity` (e.g., basic|intermediate|advanced)
    - `required_data_types` (e.g., ["numeric"])
    - `estimated_runtime` (string)
    - `icon` (emoji/name)
    - `tags` (list)

- validate_data_compatibility(data_preview) → bool
  - Receives a dictionary created by `_get_full_dataset_context` with keys like:
    - `columns` (List[str])
    - `sample_data` (List[Dict[str, Any]]; first few records)
    - `dtypes` (Dict[str, str])
    - `numeric_columns`, `categorical_columns`, `object_columns`, `datetime_columns`, `boolean_columns`
    - `shape`, `total_rows`, `file_type`, `file_path`
  - Return `True` if the analysis can run on this dataset; `False` if not.
  - Be tolerant — if preview is missing for any reason, default to `True` unless the analysis would definitely fail.

- generate_code(data_preview) → str
  - Return valid, runnable Python code. Assume a pandas DataFrame named `df` is already defined (the service injects it).
  - Perform your prints/plots inside this code. Prefer printing human-friendly headers and summaries so the output is visible in the notebook UI.
  - Keep imports light. The service already includes pandas/numpy/matplotlib/seaborn. If you need extra libs, print a warning if missing rather than crash.


## Adding a new component (step-by-step)

1) Pick a category folder (or create one) under `core/eda/advanced_eda/granular_components`.
   - Examples: `numeric/`, `categorical/`, `bivariate/`, `outliers/`, `time_series/`, `relationships/`, `data_quality/`.

2) Create a new file with a single class for your component.
   - Example skeleton:

   ```python
   # core/eda/advanced_eda/granular_components/numeric/my_new_metric.py
   from typing import Dict, Any

   class MyNewMetricAnalysis:
       def get_metadata(self) -> Dict[str, Any]:
           return {
               "name": "my_new_metric",
               "display_name": "My New Metric",
               "description": "Computes ...",
               "category": "univariate",
               "complexity": "basic",
               "required_data_types": ["numeric"],
               "estimated_runtime": "1-3 seconds",
               "icon": "📐",
               "tags": ["numeric", "statistics"],
           }

       def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
           if not data_preview:
               return True
           return len(data_preview.get("numeric_columns", [])) > 0

       def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
           return '''
   print("=== MY NEW METRIC ===")
   numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
   if not numeric_cols:
       print("No numeric columns; skipping.")
   else:
       # Your analysis here
       print("Columns:", numeric_cols)
           '''
   ```

3) Register the component ID in `core/eda/advanced_eda/granular_components/__init__.py` inside `get_all_granular_components()`:

   ```python
   from .numeric.my_new_metric import MyNewMetricAnalysis

   def get_all_granular_components():
       return {
           # ...existing...
           'my_new_metric': {
               'component': MyNewMetricAnalysis,
               'category': 'Univariate Analysis (Numeric)',
               'metadata': MyNewMetricAnalysis().get_metadata(),
           },
       }
   ```

4) Verify display metadata:
   - The generator will build display names from `metadata.display_name` or fall back to `id` → title case. Keep `display_name` present for clarity.

5) Test end-to-end (see Testing section). If the component should be recommended automatically, update the heuristic in `granular_generators.get_analysis_recommendations()` as needed.


## Service and generator method references (what they check/do)

- `AdvancedEDAService._get_full_dataset_context(source_id)`
  - Loads file path from ingestion service, inspects the file (csv/xlsx/json), builds a preview dict with:
    - columns, sample_data (first 5 rows), dtypes, total_rows, shape
    - per-type column lists: numeric/categorical/object/datetime/boolean
  - This is the input to `validate_data_compatibility()` and hints for `generate_code()`.

- `AdvancedEDAService.validate_analysis_compatibility(analysis_id, source_id)`
  - Builds preview via `_get_full_dataset_context` then calls the component’s `validate_data_compatibility`.
  - Returns True if compatible or if preview could not be built (fail-open, to avoid blocking good cases).

- `GranularAnalysisCodeGenerators.validate_analysis_compatibility(analysis_id, data_preview)`
  - Instantiates the component and calls `validate_data_compatibility` if present.

- `GranularAnalysisCodeGenerators.generate_analysis_code(analysis_id, data_preview)`
  - Instantiates the component and calls `generate_code`.

- `GranularAnalysisCodeGenerators.generate_multiple_analyses_code(analysis_ids, data_preview)`
  - Prefixes a friendly header, then for each component:
    - Adds a printed header: Analysis i/N: Display Name
    - Appends the component’s `generate_code()`
    - Appends a printed footer: completed!

- `AdvancedEDAService.generate_analysis_code(analysis_ids, source_id)`
  - Gets preview, calls generator for multiple IDs. Returns the code string and IDs.

- `AdvancedEDAService.run_granular_analysis(source_id, analysis_ids)`
  - Validates requested IDs exist and are compatible.
  - Generates the combined code and executes it.
  - Returns success flag, output streams, and execution metadata.

- `AdvancedEDAService.execute_code(source_id, code, context)`
  - Locates the uploaded dataset file by `source_id` in `uploads/data/` and loads it into `df`.
  - Wraps/executes code in a subprocess with UTF‑8 and a 60s timeout.
  - Contexts:
    - `custom_notebook`: wraps user code in try/except, auto-prints last expression result.
    - other (granular): runs code as-is (assumes components generate valid code).


## Endpoints quick reference

Prefix: `/advanced-eda`

- `GET /components/available` → grouped list of components with display name, description, tags.
- `GET /components/{id}/info` → full metadata for one component.
- `GET /components/recommendations?source_id=...` → suggested analysis IDs based on dataset characteristics.
- `POST /components/generate-code` (body: `{ source_id, analysis_ids: [...] }`) → returns combined Python code string.
- `POST /components/run-analysis` (body: `{ source_id, analysis_ids: [...] }`) → executes generated code and returns stdout/stderr.


## Testing a new component

1) Upload a dataset and capture the `source_id` used in uploads (files live under `uploads/data/{source_id}_...`).
2) Call `GET /advanced-eda/components/available` and verify your component appears in the right category with the expected display name.
3) Call `GET /advanced-eda/components/{id}/info` to see metadata.
4) Call `GET /advanced-eda/components/recommendations?source_id=...` and confirm behavior (optional).
5) Call `POST /advanced-eda/components/generate-code` with your component ID to inspect generated code.
6) Call `POST /advanced-eda/components/run-analysis` and verify:
   - stdout contains your headers and insights
   - no exceptions in stderr
   - runtime is reasonable


## Common pitfalls and how we fixed them

- “No valid analyses selected”
  - Causes:
    - Component IDs in the frontend didn’t match registry IDs.
    - Component not registered in `granular_components/__init__.py`.
    - Incompatibility checks returned False for all requested analyses.
  - Fix:
    - Align IDs across frontend and backend. Double-check `get_all_granular_components()` keys.
    - Add robust logging and return debug info (requested vs available).

- “Analysis completed successfully (no output)”
  - Causes:
    - Generated code didn’t print anything visible; some components returned only plots or silent operations.
    - For custom code, the final expression wasn’t printed.
  - Fix:
    - Ensure each component’s `generate_code()` prints structured headers and summaries.
    - For `custom_notebook`, auto-print the last expression when it’s not a statement.

- Missing data source (404 / not found)
  - Cause:
    - `source_id` didn’t exist in uploads or DB (e.g., `0ffdde64` vs actual `98b8d51c`).
  - Fix:
    - Improved error handling on both backend and frontend.
    - Added diagnostics to look for files under `uploads/data/` and return actionable messages.

- Indentation and string formatting errors during execution
  - Cause:
    - Wrapping generated/user code incorrectly (indent level mismatches, mixed f-strings with braces inside format).
  - Fix:
    - For granular code, execute as-is without an extra wrapper.
    - For custom code, indent consistently inside a try/except and auto-handle last expression.
    - Use `.format(...)` where needed to avoid f-string escaping issues in generator strings.

- Frontend/Backend ID mismatches
  - Cause:
    - UI checkbox IDs didn’t match backend registry IDs.
  - Fix:
    - Centralize IDs from backend and populate UI dynamically; avoid hardcoding duplicated copies if possible.

- Endpoint path confusion
  - Cause:
    - Mixing legacy endpoint paths with new granular endpoints.
  - Fix:
    - Consolidated under `/advanced-eda/components/*` and documented the contract.


## Pre-merge checklist for new components

- [ ] Component file created in appropriate folder
- [ ] Methods implemented: get_metadata, validate_data_compatibility, generate_code
- [ ] Registered in `granular_components/__init__.py` with stable ID
- [ ] Displays human-friendly name; description is clear
- [ ] Prints useful headers and summaries
- [ ] Compatibility check is not overly restrictive
- [ ] `POST /components/generate-code` returns expected code
- [ ] `POST /components/run-analysis` runs and shows output
- [ ] No hard dependency on extra packages, or prints a friendly warning if missing


## Notes on data preview and performance

- `_get_full_dataset_context` reads small samples for structure and estimates total rows. For CSV, row count uses a line count minus header, which is fast. For Excel/JSON, counts are more expensive.
- Execution runs in a subprocess with UTF‑8 handling and a default 60s timeout. Heavy/long-running analyses should be avoided or gated behind explicit user action.


## Example: Where existing components live

- Data Quality & Structure: duplicate_detection, missing_value_analysis, data_types_validation, dataset_shape_analysis, data_range_validation
- Univariate (Numeric): summary_statistics, skewness_analysis, normality_test, distribution_plots
- Univariate (Categorical): categorical_frequency_analysis, categorical_visualization
- Bivariate/Multivariate: correlation_analysis, scatter_plot_analysis, cross_tabulation_analysis
- Outliers: iqr_outlier_detection, zscore_outlier_detection, visual_outlier_inspection
- Time-Series: temporal_trend_analysis, seasonality_detection
- Relationships: multicollinearity_analysis, pca_dimensionality_reduction


## FAQ

- Q: Do I need to import pandas/numpy/matplotlib in my component code?
  - A: Not strictly; the service pre-imports these. It’s safe to assume `import pandas as pd`, `import numpy as np`, `import matplotlib.pyplot as plt`, and `import seaborn as sns` are available. If you include them, it won’t harm.

- Q: How does plotting work?
  - A: Plots rendered by matplotlib display in notebook contexts that capture outputs. Prefer printing textual insights too so users see value even without figure capture.

- Q: What if my component needs external packages?
  - A: Prefer standard libs. If unavoidable, detect and print a friendly message if the import fails rather than raising.


## Appendix: Minimal component template

```python
from typing import Dict, Any

class MyComponent:
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "my_component",
            "display_name": "My Component",
            "description": "One-liner explaining what this does.",
            "category": "univariate",
            "complexity": "basic",
            "required_data_types": [],
            "estimated_runtime": "<1s",
            "icon": "📊",
            "tags": ["example"],
        }

    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        # Be permissive unless truly incompatible
        return True

    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        return '''
print("=== MY COMPONENT ===")
print("df shape:", df.shape)
        '''
```


## Credits and changes

Refactor and stability work included: granular component registry, per-component code generation, robust execution with UTF‑8, improved frontend/back-end ID alignment, better error messages (missing data sources), and fixes for indentation/string formatting in code execution.

## Related docs

- Advanced EDA docs index: `docs/advanced_eda/README.md`
- Service methods reference: `docs/advanced_eda/services.AdvancedEDAService.md`
- Generators methods reference: `docs/advanced_eda/generators.GranularAnalysisCodeGenerators.md`
- Endpoints reference: `docs/advanced_eda/routes.endpoints.md`
- Component contract: `docs/advanced_eda/components.contract.md`
