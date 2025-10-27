# `core/eda/advanced_eda`

Advanced Exploratory Data Analysis stack: domain detection, granular analysis components, caching/resource management, notebook export tooling, and the FastAPI routes that power the "Advanced EDA" UI.

## Top-level modules

| Module | What it does | Key collaborators |
| --- | --- | --- |
| `analysis_results.py` | Dataclasses for metrics/insights/tables/charts, plus helpers to make results JSON-safe (base64 figures, pandas/NumPy coercion). | Produced by granular runtime engines, consumed by `AdvancedEDAService` responses and export templates. |
| `cache_manager.py` | Async-friendly in-memory cache with TTL + LRU eviction. | Instantiated by `DomainAnalyzer` and `AdvancedEDAService`, exercised in `test/test_resource_and_cache_managers.py`. |
| `data_manager.py` | `EDADataManager` that selects loading strategies (full, chunked, lazy via Dask), tracks memory usage, and caches dataset metadata per user/session. | Used by `AdvancedEDAService.prepare_for_analysis`, `SessionManager`, and preprocessing flows; relies on SQLAlchemy async sessions and `log_data_action`. |
| `domain_analyzer.py` | Heuristic-driven domain classifier + workflow executor; maps dataset traits to recommended analyses (classification, time-series, marketing, etc.) and coordinates cached runs. | Calls into granular runtime components, uses `CacheManager` and `ResourceMonitor`; invoked by `AdvancedEDAService` endpoints and automation tasks. |
| `export_manager.py` | Creates notebook/HTML exports from saved sessions, managing the `exports/` directory layout. | Triggered by `AdvancedEDAService.export_*` endpoints; outputs are subsequently downloaded via `routes.py`. |
| `granular_generators.py` | Registry + code generator for granular analysis components; converts selections into runnable Python templates. | Referenced by `AdvancedEDAService.generate_analysis_code` and notebook exports; depends on `granular_components`. |
| `granular_runtime.py` | Shim that re-exports `granular_runtime/__init__.py` to maintain backwards compatibility with the legacy import path. | Allows `from core.eda.advanced_eda.granular_runtime import ...` to resolve to the modular runtime package. |
| `llm_notebook_engine.py` | Placeholder (LLM engine removed) kept for compatibility with routes/tests that import the module. | Provides a no-op surface so older code paths fail gracefully. |
| `notebook_templates.py` | Large template catalog + selection logic (driven by Kaggle analysis) for pre-built notebooks, including domain/complexity classifiers. | Used by `AdvancedEDAService` and `/advanced-eda` routes when generating template-driven sessions; logs activity via `log_data_action`. |
| `resource_monitor.py` | Concurrency and system guard: checks CPU/memory headroom, tracks in-flight operations, records execution history. | Injected into `DomainAnalyzer` and `AdvancedEDAService`; surfaced in tests and observability dashboards. |
| `routes.py` | FastAPI router for the Advanced EDA web experience: UI pages, dataset info, session persistence, granular analysis endpoints, sandboxed code execution, exports. | Depends on `AdvancedEDAService`, `EDAService`, security sandbox (`core.eda.security`), FastAPI dependencies, and template engine. |
| `services.py` | Core `AdvancedEDAService` orchestrating data preparation, caching, column insights, granular analysis runtime, session/export integration, and sandboxed code execution. | Coordinates `EDADataManager`, `CacheManager`, `ResourceMonitor`, granular runtime modules, preprocessing state store, `SessionManager`, `ExportManager`, and the persistent sandbox. |
| `session_manager.py` | Persists per-user analysis sessions (checkpointing to disk, TTL cleanup, export history) and exposes async lifecycle management. | Created by `AdvancedEDAService` for session-based APIs; stores artefacts under `data/eda_sessions`. |
| `__init__.py` | Re-exports granular component registry helpers for convenient imports elsewhere. | Used throughout advanced service/tests to list available analyses. |

## Subpackages & configuration

| Folder | Contents | Notes |
| --- | --- | --- |
| `config/` | `eda_config.yaml` with feature flags and default thresholds for advanced workflows. | Loaded by `AdvancedEDAService` to tune behaviour. |
| `granular_components/` | Domain-specific analysis building blocks grouped by topic (`data_quality/`, `numeric/`, `categorical/`, `time_series/`, `text/`, `marketing/`, etc.). Each module exposes a class with `run`, metadata, and optional compatibility checks. | Registered via `granular_components/__init__.py`; consumed by `granular_generators` and granular runtime executors. |
| `granular_runtime/` | Execution engine for granular components: preprocessing utilities, per-domain runners (e.g., `numeric.py`, `time_series.py`), and state stores (e.g., `preprocessing.py` with `preprocessing_state_store`). | Called by `AdvancedEDAService.run_granular_analysis` and domain analyzers; integrates with preprocessing cache and result dataclasses. |
| `granular_runtime/preprocessing.py` | Defines `PreprocessingOptions`, `PreprocessingReport`, and in-memory state store used to reuse preprocessing decisions across cache hits. | Referenced in services/tests for cache assertions. |

## Execution flow in practice

1. **Routing layer**: `/advanced-eda/...` endpoints in `routes.py` validate auth, marshal request bodies, and fetch shared dependencies (DB session, persistent sandbox, template engine).
2. **Service orchestration**: `AdvancedEDAService` prepares data via `EDADataManager`, checks `ResourceMonitor`, builds cache keys, and either returns a cached payload (`CacheManager`) or executes fresh granular analyses.
3. **Granular execution**: The service hands off to the granular runtime (via `granular_runtime` and component classes) to generate `AnalysisResult` objects, which are serialized for API responses and exports.
4. **Sessions & exports**: Interactive flows store state through `SessionManager`, while notebook/HTML exports go through `ExportManager` and reuse template/code generators.
5. **Security**: Any user-provided Python (LLM-generated code, notebook snippets) routes through the persistent sandbox in `core.eda.security`, ensuring filesystem/network restrictions remain enforced.

## Related tests

- `test/test_resource_and_cache_managers.py` – cache + resource monitor behaviour, cache hits for column insights and granular analyses.
- `test/test_granular_runtime.py`, `test/test_granular_components.py` – coverage for runtime/component integration.
- `test/test_complete_flow.py`, `test/test_advanced_bypasses.py`, `test/test_security_vulnerabilities.py` – end-to-end verification of routes, sandboxing, and security boundaries.

Use this README as the authoritative map when extending the advanced EDA stack or onboarding new contributors.
