# `core/eda`

Exploratory Data Analysis (EDA) entry point for the platform. This package exposes the lightweight EDA API/service layer, text analytics helpers, and the hardened sandbox utilities that power both the basic and advanced EDA experiences.

## Directory map

| Path | What it contains | Key dependencies | Consumed by |
| --- | --- | --- | --- |
| `__init__.py` | Lazy exports for the public interface (`EDAService`, `eda_router`) so tests can import `core.eda` without pulling heavy dependencies if pandas/FastAPI are unavailable. | `core.eda.services`, `core.eda.routes` | Application bootstrap (`main.py`, `run_fastapi.py`), unit tests. |
| `routes.py` | FastAPI router that wires `/eda/api/...` endpoints for preview and quality reports. Performs dependency injection via the data-ingestion layer and instantiates `EDAService` per request. | `fastapi`, `core.data_ingestion.dependencies`, `EDAService`, `core.database.models.User` | Mounted in the global API router (see `main.py`), hit by frontend dashboards, end-to-end tests such as `test/test_complete_flow.py`. |
| `services.py` | Implements `EDAService` with file discovery, sampling, basic statistics, and quality reporting (nulls, dtype summaries, text profiling). | `pandas`, `numpy`, `core.data_ingestion.service.DataIngestionService`, `core.data_ingestion.serialization.JSONSafeSerializer`, `log_data_action`, `core.eda.text_analysis.get_text_insights` | `routes.py`, advanced EDA fallbacks, analytics tests (`test/test_analysis_fixes.py`, `test/test_pandas_specific.py`). |
| `text_analysis.py` | Detailed heuristics for categorising object columns, computing text quality metrics, and generating EDA recommendations for NLP-leaning data. | `pandas`, Python `re`/`string` | `EDAService.quality_report`, advanced text workflows, tests covering text runtime (`test/test_text_runtime.py`). |
| `security/` | Hardened execution sandboxes (`persistent_worker.py`, `persistent_sandbox.py`, `code_sandbox.py`, etc.) plus rate-limiting helpers. Guards notebook/code execution for both base and advanced EDA. A dedicated `README.md` in this folder documents the subsystems. | `ast`, `multiprocessing`, `psutil`, shared logging utilities | Used by `core.eda.advanced_eda.routes` for sandboxed notebook code execution, by monitoring tests (`test/test_persistent_sandbox.py`, `test/test_security_vulnerabilities.py`). |
| `advanced_eda/` | Full advanced analysis stack: domain-aware analyzers, granular runtime, caching/monitoring, session & export managers, FastAPI routes. | See `core/eda/advanced_eda/README.md` | Advanced EDA UI, automation pipelines, large test matrix (`test/test_granular_runtime.py`, `test/test_advanced_bypasses.py`). |

## How the pieces connect

1. **HTTP request → router**: Frontend calls `/eda/api/...`; `routes.py` validates access (via `require_data_access`) and creates an `EDAService` tied to the caller's async DB session.
2. **Service orchestration**: `EDAService` coordinates file discovery with `DataIngestionService`, loads samples into pandas, and enriches responses using `text_analysis` for object columns.
3. **Cross-module flow**: Advanced EDA builds on these primitives—`advanced_eda/routes.py` still depends on `EDAService` for simple previews and on the security sandbox for executing user-provided snippets safely.
4. **Testing & observability**: Modules log actions via `log_data_action`, feeding monitoring dashboards; pytest suites under `test/` assert both the base service responses and sandbox protections remain intact.

When extending the package, prefer adding new domain-specific logic inside `advanced_eda/` while keeping `services.py` focused on the lightweight preview/quality-report use cases.
