from __future__ import annotations

"""Dataset preprocessing utilities for the granular runtime pipeline."""

import copy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


@dataclass
class PreprocessingOptions:
    """User-configurable preprocessing directives."""

    drop_missing_columns_threshold: Optional[float] = None
    drop_missing_rows_threshold: Optional[float] = None
    apply_drop_duplicates: bool = False
    explicit_drop_columns: Optional[List[str]] = None
    suggested_drop_columns: Optional[List[str]] = None
    imputation_strategy: Optional[str] = None
    imputation_fill_value: Optional[Any] = None
    imputation_neighbors: int = 5

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "PreprocessingOptions":
        """Create options from a loosely structured payload (typically JSON)."""

        if not payload or not isinstance(payload, dict):
            return cls()

        def _extract_threshold(section: Dict[str, Any]) -> Optional[float]:
            if not section or not isinstance(section, dict):
                return None
            if not section.get("enabled"):
                return None
            threshold = section.get("threshold")
            if threshold is None:
                return None
            try:
                value = float(threshold)
            except (TypeError, ValueError):
                return None
            return float(np.clip(value, 0.0, 100.0))

        drop_cols_section = payload.get("drop_missing_columns", {})
        drop_rows_section = payload.get("drop_missing_rows", {})
        imputation_section = payload.get("imputation", {})

        def _extract_columns(section: Any) -> Optional[List[str]]:
            if not section:
                return None

            if isinstance(section, dict):
                raw_columns = section.get("columns") or section.get("values") or section.get("list")
            else:
                raw_columns = section

            if raw_columns is None:
                return None

            if isinstance(raw_columns, (str, bytes)):
                raw_iterable: Iterable[Any] = [raw_columns]
            elif isinstance(raw_columns, Iterable):
                raw_iterable = raw_columns
            else:
                return None

            normalized: List[str] = []
            for value in raw_iterable:
                if value is None:
                    continue
                name = str(value).strip()
                if not name or name in normalized:
                    continue
                normalized.append(name)

            return normalized or None

        strategy = imputation_section.get("strategy") if isinstance(imputation_section, dict) else None
        if strategy:
            strategy = str(strategy).strip().lower()
            if strategy in {"", "none"}:
                strategy = None

        neighbors = 5
        if isinstance(imputation_section, dict) and imputation_section.get("neighbors") is not None:
            try:
                neighbors_val = int(imputation_section.get("neighbors"))
                if neighbors_val > 0:
                    neighbors = neighbors_val
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        fill_value = None
        if isinstance(imputation_section, dict):
            fill_value = imputation_section.get("fill_value")

        explicit_drop_columns = _extract_columns(
            payload.get("explicit_drop_columns")
            or payload.get("drop_columns")
            or payload.get("drop_specific_columns")
        )

        suggested_drop_columns = _extract_columns(
            payload.get("suggested_drop_columns")
            or payload.get("drop_column_suggestions")
        )

        return cls(
            drop_missing_columns_threshold=_extract_threshold(drop_cols_section),
            drop_missing_rows_threshold=_extract_threshold(drop_rows_section),
            apply_drop_duplicates=bool(payload.get("drop_duplicates", False)),
            explicit_drop_columns=explicit_drop_columns,
            suggested_drop_columns=suggested_drop_columns,
            imputation_strategy=strategy,
            imputation_fill_value=fill_value,
            imputation_neighbors=neighbors,
        )

    @property
    def has_operations(self) -> bool:
        return any(
            [
                self.drop_missing_columns_threshold is not None,
                self.drop_missing_rows_threshold is not None,
                self.apply_drop_duplicates,
                bool(self.explicit_drop_columns),
                bool(self.suggested_drop_columns),
                bool(self.imputation_strategy),
            ]
        )

    def to_payload(self) -> Dict[str, Any]:
        """Convert options back into a JSON-friendly payload structure."""

        return {
            "drop_missing_columns": {
                "enabled": self.drop_missing_columns_threshold is not None,
                "threshold": self.drop_missing_columns_threshold,
            },
            "drop_missing_rows": {
                "enabled": self.drop_missing_rows_threshold is not None,
                "threshold": self.drop_missing_rows_threshold,
            },
            "drop_columns": {
                "enabled": bool(self.explicit_drop_columns),
                "columns": list(self.explicit_drop_columns or []),
            },
            "suggested_drop_columns": {
                "enabled": bool(self.suggested_drop_columns),
                "columns": list(self.suggested_drop_columns or []),
            },
            "drop_duplicates": self.apply_drop_duplicates,
            "imputation": {
                "strategy": self.imputation_strategy or "none",
                "fill_value": self.imputation_fill_value,
                "neighbors": self.imputation_neighbors,
            },
        }


@dataclass
class PreprocessingReport:
    """Structured summary of preprocessing outcomes."""

    options: Dict[str, Any]
    applied: bool
    applied_operations: List[str] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    dropped_columns: List[str] = field(default_factory=list)
    manual_dropped_columns: List[str] = field(default_factory=list)
    suggested_dropped_columns: List[str] = field(default_factory=list)
    missing_requested_columns: List[str] = field(default_factory=list)
    column_drop_recommendations: List[str] = field(default_factory=list)
    dropped_rows: int = 0
    duplicate_rows_removed: int = 0
    imputation_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure ints are native Python ints for JSON serialization
        data["dropped_rows"] = int(self.dropped_rows)
        data["duplicate_rows_removed"] = int(self.duplicate_rows_removed)
        data["original_shape"] = [int(self.original_shape[0]), int(self.original_shape[1])]
        data["final_shape"] = [int(self.final_shape[0]), int(self.final_shape[1])]
        return data


@dataclass
class PreprocessingState:
    """Cached state for incremental preprocessing workflows."""

    dataframe: pd.DataFrame
    report: PreprocessingReport
    options: Optional[PreprocessingOptions]
    updated_at: datetime


class PreprocessingStateStore:
    """In-memory cache for managing preprocessed dataset snapshots per source."""

    def __init__(self) -> None:
        self._store: Dict[str, PreprocessingState] = {}

    def has_state(self, source_id: Optional[str]) -> bool:
        if not source_id:
            return False
        return source_id in self._store

    def get_dataframe(self, source_id: Optional[str]) -> Optional[pd.DataFrame]:
        if not source_id or source_id not in self._store:
            return None
        return self._store[source_id].dataframe.copy()

    def get_report(self, source_id: Optional[str]) -> Optional[PreprocessingReport]:
        if not source_id or source_id not in self._store:
            return None
        return copy.deepcopy(self._store[source_id].report)

    def get_options(self, source_id: Optional[str]) -> Optional[PreprocessingOptions]:
        if not source_id or source_id not in self._store:
            return None
        options = self._store[source_id].options
        return copy.deepcopy(options) if options is not None else None

    def set_state(
        self,
        source_id: str,
        dataframe: pd.DataFrame,
        options: Optional[PreprocessingOptions],
        report: PreprocessingReport,
    ) -> None:
        self._store[source_id] = PreprocessingState(
            dataframe=dataframe.copy(),
            report=copy.deepcopy(report),
            options=copy.deepcopy(options) if options is not None else None,
            updated_at=datetime.now(UTC),
        )

    def clear(self, source_id: Optional[str]) -> None:
        if not source_id:
            return
        self._store.pop(source_id, None)

    def snapshot(self, source_id: Optional[str]) -> Optional[PreprocessingState]:
        if not source_id or source_id not in self._store:
            return None
        state = self._store[source_id]
        return PreprocessingState(
            dataframe=state.dataframe.copy(),
            report=copy.deepcopy(state.report),
            options=copy.deepcopy(state.options) if state.options is not None else None,
            updated_at=state.updated_at,
        )


preprocessing_state_store = PreprocessingStateStore()


def run_preprocessing_with_state(
    source_id: Optional[str],
    df: pd.DataFrame,
    options: Optional[PreprocessingOptions],
    *,
    base: str = "auto",
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """Apply preprocessing while maintaining incremental state for the source.

    Args:
        source_id: Identifier for the dataset being processed.
        df: The dataframe to preprocess (used when no cached state exists or when resetting).
        options: The preprocessing options to apply.
        base: Strategy for selecting the baseline dataframe. Supported values:
            - "auto": use cached dataframe when available, otherwise the provided df.
            - "current": force reuse of cached dataframe when available.
            - "original"/"reset": clear cache and start from provided df.

    Returns:
        Tuple of (processed dataframe, preprocessing report).
    """

    if options is None:
        options = PreprocessingOptions()

    normalized_base = (base or "auto").lower()
    if normalized_base not in {"auto", "current", "original", "reset"}:
        normalized_base = "auto"

    if normalized_base == "auto":
        normalized_base = "current" if preprocessing_state_store.has_state(source_id) else "original"

    if normalized_base in {"original", "reset"}:
        preprocessing_state_store.clear(source_id)
        baseline_df = df
    elif normalized_base == "current":
        cached_df = preprocessing_state_store.get_dataframe(source_id)
        baseline_df = cached_df if cached_df is not None else df
    else:  # Fallback safety
        baseline_df = df

    processed_df, report = apply_preprocessing(baseline_df, options)

    if source_id:
        preprocessing_state_store.set_state(source_id, processed_df, options, report)

    return processed_df, report


def _drop_missing_columns(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    missing_pct = df.isna().mean() * 100
    columns_to_drop = missing_pct[missing_pct >= threshold].index.tolist()
    if not columns_to_drop:
        return df, []
    return df.drop(columns=columns_to_drop), columns_to_drop


def _drop_missing_rows(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, int]:
    row_missing_pct = df.isna().mean(axis=1) * 100
    rows_to_drop = row_missing_pct[row_missing_pct >= threshold].index
    if rows_to_drop.empty:
        return df, 0
    return df.drop(index=rows_to_drop), int(len(rows_to_drop))


def _recommend_columns_to_drop(
    df: pd.DataFrame,
    options: PreprocessingOptions,
) -> List[str]:
    if df.empty:
        return []

    threshold = options.drop_missing_columns_threshold
    if threshold is None:
        threshold = 40.0 # Default recommendation threshold
    threshold = float(np.clip(threshold, 0.0, 100.0))

    missing_pct = df.isna().mean() * 100
    suggestions = set(missing_pct[missing_pct >= threshold].index.tolist())

    # Also flag near-constant columns as potential drop candidates
    for column in df.columns:
        series = df[column]
        try:
            unique_count = series.nunique(dropna=True)
        except Exception:  # pragma: no cover - defensive
            unique_count = len(pd.unique(series.dropna()))
        if unique_count <= 1:
            suggestions.add(column)

    if not suggestions:
        return []

    # Preserve deterministic ordering for reporting
    ordered = [col for col in df.columns if col in suggestions]
    return ordered


def _apply_simple_imputer(
    df: pd.DataFrame,
    columns: Iterable[str],
    strategy: str,
    fill_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    imputer_kwargs: Dict[str, Any] = {"strategy": strategy}
    if strategy == "constant":
        imputer_kwargs["fill_value"] = fill_value

    imputer = SimpleImputer(**imputer_kwargs)
    target_cols = list(columns)
    if not target_cols:
        return df, {}

    transformed = imputer.fit_transform(df[target_cols])
    df.loc[:, target_cols] = transformed

    details: Dict[str, Any] = {"strategy": strategy, "columns": target_cols}
    if strategy == "constant":
        details["fill_value"] = imputer_kwargs["fill_value"]
    return df, details


def _apply_knn_imputer(df: pd.DataFrame, columns: Iterable[str], neighbors: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    target_cols = list(columns)
    if not target_cols:
        return df, {}

    imputer = KNNImputer(n_neighbors=max(1, neighbors))
    transformed = imputer.fit_transform(df[target_cols])
    df.loc[:, target_cols] = transformed

    return df, {"strategy": "knn", "columns": target_cols, "neighbors": int(max(1, neighbors))}


def apply_preprocessing(
    df: pd.DataFrame,
    options: Optional[PreprocessingOptions] = None,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """Apply configured preprocessing operations to a dataframe."""

    if df is None:
        raise ValueError("Dataframe must not be None")

    working_df = df.copy()
    original_shape = (int(df.shape[0]), int(df.shape[1]))

    if options is None:
        options = PreprocessingOptions()

    applied_operations: List[str] = []
    dropped_columns: List[str] = []
    manual_dropped_columns: List[str] = []
    suggested_dropped_columns: List[str] = []
    missing_requested_columns: List[str] = []
    dropped_rows = 0
    duplicate_rows_removed = 0
    imputation_details: Optional[Dict[str, Any]] = None

    column_drop_recommendations = _recommend_columns_to_drop(df, options)

    # Drop duplicates first to avoid unnecessary preprocessing work
    if options.apply_drop_duplicates:
        before = len(working_df)
        working_df = working_df.drop_duplicates()
        duplicate_rows_removed = int(before - len(working_df))
        if duplicate_rows_removed > 0:
            applied_operations.append("drop_duplicates")

    if options.explicit_drop_columns:
        requested: List[Any] = []
        for col in options.explicit_drop_columns:
            if col is None:
                continue
            normalized = col.strip() if isinstance(col, str) else col
            if isinstance(normalized, str) and not normalized:
                continue
            if normalized in requested:
                continue
            requested.append(normalized)
        existing = [col for col in requested if col in working_df.columns]
        missing = sorted(set(requested) - set(existing))
        if existing:
            working_df = working_df.drop(columns=existing)
            manual_dropped_columns.extend(existing)
            dropped_columns.extend(existing)
            applied_operations.append("drop_explicit_columns")
        if missing:
            missing_requested_columns.extend(missing)

    if options.suggested_drop_columns:
        suggested: List[Any] = []
        for col in options.suggested_drop_columns:
            if col is None:
                continue
            normalized = col.strip() if isinstance(col, str) else col
            if isinstance(normalized, str) and not normalized:
                continue
            if normalized in suggested:
                continue
            suggested.append(normalized)
        existing = [col for col in suggested if col in working_df.columns]
        missing = sorted(set(suggested) - set(existing))
        if existing:
            working_df = working_df.drop(columns=existing)
            suggested_dropped_columns.extend(existing)
            dropped_columns.extend(existing)
            applied_operations.append("drop_suggested_columns")
        if missing:
            missing_requested_columns.extend(missing)

    # Drop columns exceeding missing threshold
    if options.drop_missing_columns_threshold is not None:
        threshold = float(np.clip(options.drop_missing_columns_threshold, 0.0, 100.0))
        working_df, threshold_dropped_columns = _drop_missing_columns(working_df, threshold)
        if threshold_dropped_columns:
            dropped_columns.extend(threshold_dropped_columns)
            applied_operations.append("drop_missing_columns")
    else:
        threshold_dropped_columns = []

    # Drop rows exceeding missing threshold
    if options.drop_missing_rows_threshold is not None:
        threshold = float(np.clip(options.drop_missing_rows_threshold, 0.0, 100.0))
        working_df, dropped_rows = _drop_missing_rows(working_df, threshold)
        if dropped_rows:
            applied_operations.append("drop_missing_rows")

    # Imputation strategies
    if options.imputation_strategy:
        strategy = options.imputation_strategy.lower()
        numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = working_df.select_dtypes(exclude=[np.number]).columns.tolist()

        if strategy in {"mean", "median"}:
            working_df, imputation_details = _apply_simple_imputer(working_df, numeric_columns, strategy)
        elif strategy == "most_frequent":
            details_numeric = {}
            details_categorical = {}
            if numeric_columns:
                working_df, details_numeric = _apply_simple_imputer(working_df, numeric_columns, "most_frequent")
            if categorical_columns:
                working_df, details_categorical = _apply_simple_imputer(working_df, categorical_columns, "most_frequent")
            imputation_details = {
                "strategy": "most_frequent",
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
            }
            if details_numeric:
                imputation_details["numeric_imputer"] = details_numeric
            if details_categorical:
                imputation_details["categorical_imputer"] = details_categorical
        elif strategy == "constant":
            fill_value = options.imputation_fill_value
            if fill_value is None or (isinstance(fill_value, str) and fill_value.strip() == ""):
                # Provide sensible defaults based on column types
                fill_value = 0
            working_df = working_df.fillna(fill_value)
            imputation_details = {
                "strategy": "constant",
                "fill_value": fill_value,
            }
        elif strategy == "knn":
            working_df, imputation_details = _apply_knn_imputer(working_df, numeric_columns, options.imputation_neighbors)
        else:  # pragma: no cover - defensive
            imputation_details = {"strategy": strategy, "warning": "Unknown strategy ignored"}

        if imputation_details:
            applied_operations.append("imputation")

    final_shape = (int(working_df.shape[0]), int(working_df.shape[1]))

    # Deduplicate while preserving original drop order
    if dropped_columns:
        seen = set()
        dropped_columns = [col for col in dropped_columns if not (col in seen or seen.add(col))]
    if manual_dropped_columns:
        seen_manual = set()
        manual_dropped_columns = [col for col in manual_dropped_columns if not (col in seen_manual or seen_manual.add(col))]
    if suggested_dropped_columns:
        seen_suggested = set()
        suggested_dropped_columns = [
            col for col in suggested_dropped_columns if not (col in seen_suggested or seen_suggested.add(col))
        ]
    if missing_requested_columns:
        missing_requested_columns = sorted(set(missing_requested_columns), key=lambda value: str(value))

    report = PreprocessingReport(
        options=options.to_payload(),
        applied=options.has_operations,
        applied_operations=applied_operations,
        original_shape=original_shape,
        final_shape=final_shape,
        dropped_columns=dropped_columns,
        manual_dropped_columns=manual_dropped_columns,
        suggested_dropped_columns=suggested_dropped_columns,
        missing_requested_columns=missing_requested_columns,
        column_drop_recommendations=column_drop_recommendations,
        dropped_rows=dropped_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        imputation_details=imputation_details,
    )

    return working_df, report


__all__ = [
    "PreprocessingOptions",
    "PreprocessingReport",
    "PreprocessingState",
    "PreprocessingStateStore",
    "preprocessing_state_store",
    "run_preprocessing_with_state",
    "apply_preprocessing",
]
