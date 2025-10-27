# Restoring Presidio-Based PII Detection

This project currently ships with a lightweight regex-only PII detector inside `core/eda/text_analysis.py`. If you want to bring back the richer Microsoft Presidio + spaCy integration later, follow the steps below.

## 1. Reintroduce the Dependencies

Update both dependency manifests so the packages reinstall on the next `uv sync`.

- Add the following entries back to `[project.dependencies]` inside `pyproject.toml`:
  - `presidio-analyzer<=2.2.360`
  - `spacy>=3.7.4`
- Mirror the same lines in `requirements-fastapi.txt` under the *NLP analytics* section if you keep that file in sync with the FastAPI deployment environment.

After editing the manifests, run:

```powershell
uv sync
```

This will repopulate the lockfile and install the packages into the uv-managed virtual environment.

## 2. Install the spaCy Model

Presidio’s default spaCy NLP engine requires a language model. Use uv to download and cache it inside the environment:

```powershell
uv run python -m spacy download en_core_web_sm
```

Alternatively, you can pin the wheel directly:

```powershell
uv pip install "en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
```

## 3. Restore the Presidio Hooks in Code

If you retained the git history, you can recover the Presidio-aware implementation by checking out the earlier version of `core/eda/text_analysis.py` (and any helper functions) from the commit where Presidio was last active. The relevant helpers were named `_get_presidio_analyzer` and `detect_presidio_entities`, and `detect_text_patterns` merged their results with the regex findings.

Summary of changes to reapply:

- Reintroduce `_get_presidio_analyzer` and `detect_presidio_entities` in `core/eda/text_analysis.py`.
- Update `detect_text_patterns` to call both the regex detector and the Presidio entity detector, combining the results.
- Ensure `core/eda/services.py` keeps the logic that interprets Presidio pattern types as PII (the `_pattern_type_is_pii` helper already works for both regex and Presidio outputs).

## 4. Validate the Integration

1. Restart your FastAPI app (`uv run uvicorn run_fastapi:app --reload`).
2. Trigger an endpoint that runs the EDA text analysis (for example, the quality report or column insights route).
3. Confirm that the logs show Presidio loading without download errors and that the API response now includes Presidio-detected entities.
4. Run the focused pytest slice to guard against regressions:

```powershell
uv run --env PYTHONPATH=. pytest -k text_analysis
```

## 5. Operational Notes

- spaCy model downloads require outbound network access. If your deployment target cannot reach GitHub, package and ship the model wheel alongside your application.
- Presidio is Apache 2.0 licensed; review licensing requirements if you embed it in a commercial product.
- Keep an eye on memory usage. Presidio’s NLP engine increases startup time and RAM footprint compared to the regex-only fallback.

With these steps, you can toggle between the lightweight regex detector and the richer Presidio-based solution whenever needed.
