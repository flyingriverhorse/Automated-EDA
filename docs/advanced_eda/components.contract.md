# Granular Component Contract and Registry

## Component class requirements

Each component should implement:

- get_metadata() -> Dict[str, Any]
  - Keys: name (id), display_name, description, category, complexity, required_data_types, estimated_runtime, icon, tags.

- validate_data_compatibility(data_preview: Dict[str, Any]) -> bool
  - Receives preview from `_get_full_dataset_context` with keys like columns, sample_data, dtypes, numeric_columns, categorical_columns, object_columns, datetime_columns, boolean_columns, total_rows, shape, file_type, file_path.
  - Return True if the component can run on this dataset.

- generate_code(data_preview: Dict[str, Any] | None) -> str
  - Returns runnable Python code. Assume `df` exists. Print headers and summaries for visibility. Keep imports minimal.

## Registry

- File: `core/eda/advanced_eda/granular_components/__init__.py`
- Function: `get_all_granular_components()` returns mapping:
  - id → { component: Class, category: str, metadata: dict }
- Function: `get_components_by_category()` groups by category for UI.

## Example skeleton

```python
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

    def generate_code(self, data_preview: Dict[str, Any] | None = None) -> str:
        return '''
print("=== MY NEW METRIC ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    print("No numeric columns; skipping.")
else:
    print("Columns:", numeric_cols)
        '''
```

## Notes

- Be permissive in `validate_data_compatibility` unless the analysis would clearly fail.
- Prefer textual output in addition to plots.
- Avoid extra package dependencies; if needed, detect and print a friendly message instead of crashing.
