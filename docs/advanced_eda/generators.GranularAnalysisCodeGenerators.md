# GranularAnalysisCodeGenerators — Methods Reference

Source: `core/eda/advanced_eda/granular_generators.py`

## Methods

- get_available_analyses() -> List[str]
  - Returns all registered component IDs.

- get_analysis_metadata(analysis_id: str) -> Dict[str, Any]
  - Returns component metadata (or {} if unknown).

- get_analysis_display_name(analysis_id: str) -> str
  - Display name from metadata.display_name or ID in Title Case.

- validate_analysis_compatibility(analysis_id: str, data_preview: Dict[str, Any] | None) -> bool
  - Instantiates the component and calls `validate_data_compatibility` if present. Defaults to True on errors.

- generate_analysis_code(analysis_id: str, data_preview: Dict[str, Any] | None) -> str
  - Instantiates the component and returns its `generate_code()` string. Returns an error comment if unknown.

- generate_multiple_analyses_code(analysis_ids: List[str], data_preview: Dict[str, Any] | None) -> str
  - Builds a readable script with headers for each analysis, appends each component’s code, and a footer. Assumes `df` exists.

- get_grouped_analyses() -> Dict[str, List[Dict[str, Any]]]
  - Groups components by category with display name, description, tags, etc. for UI.

- get_analysis_recommendations(data_preview: Dict[str, Any]) -> List[str]
  - Heuristic suggestions based on column types. Always includes data-quality basics; adds numeric/categorical/time-series/relationship items as applicable. Returns up to 15.

- get_component_info(analysis_id: str) -> Dict[str, Any]
  - Returns id, name, description, category, complexity, estimated_runtime, tags, icon, component_class.
