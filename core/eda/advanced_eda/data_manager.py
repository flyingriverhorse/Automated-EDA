"""
EDA Data Manager - Central data management for EDA operations
Handles data loading strategies, caching, and access control
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

from config import get_settings
from sqlalchemy.ext.asyncio import AsyncSession

from core.utils.logging_utils import log_data_action


class DataLoadStrategy:
    """Enum-like class for data loading strategies"""
    FULL_MEMORY = "full_memory"
    CHUNKED = "chunked"
    LAZY = "lazy"
    SAMPLED = "sampled"


class EDADataManager:
    """
    Centralized data management for EDA operations.
    
    Features:
    - Smart loading based on file size
    - Data caching with TTL
    - Session-based data isolation
    - Memory management
    - Data versioning
    """
    
    # Size thresholds in bytes
    SMALL_DATA_THRESHOLD = 100_000_000  # 100MB
    MEDIUM_DATA_THRESHOLD = 500_000_000  # 500MB
    
    # Cache settings
    CACHE_TTL_HOURS = 4
    MAX_CACHE_SIZE_GB = 2

    # Interactive preview limits
    MAX_INTERACTIVE_SAMPLE_ROWS = 1_000_000
    
    def __init__(self, session: AsyncSession, cache_dir: Path = None):
        """
        Initialize EDA Data Manager
        
        Args:
            session: Async database session
            cache_dir: Directory for data cache
        """
        self.session = session
        try:
            self.settings = get_settings()
        except Exception:  # pragma: no cover - defensive fallback
            self.settings = None
        self.cache_dir = cache_dir or Path("data/eda_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active datasets
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Track data access patterns
        self.access_log: Dict[str, List[datetime]] = {}
        
        # Session registry
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Resource tracking
        self.current_memory_usage = 0
        self.max_memory_usage = 2 * 1024**3  # 2GB default limit

    def _apply_global_sample_limit(self, sample_size: int) -> int:
        """Apply globally configured sample limit when explicitly set."""
        if sample_size is None:
            return sample_size

        if self.settings and hasattr(self.settings, "is_field_set") and self.settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT"):
            limit = self.settings.EDA_GLOBAL_SAMPLE_LIMIT
            if limit is not None:
                return min(sample_size, max(limit, 0))
        return sample_size
        
    async def prepare_for_eda(
        self, 
        source_id: str, 
        user_id: str,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare data for EDA with intelligent loading strategy.
        
        Args:
            source_id: Data source identifier
            user_id: User identifier for session tracking
            force_reload: Force reload from disk
            
        Returns:
            Dictionary with data info and loading strategy
        """
        cache_key = f"{source_id}_{user_id}"
        
        # Check memory cache first
        if not force_reload and cache_key in self.memory_cache:
            cache_entry = self.memory_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                log_data_action("EDA_CACHE_HIT", details=f"source_id: {source_id}")
                self._update_access_log(cache_key)
                return cache_entry["data_info"]
        
        # Load from source
        file_path = await self._get_source_path(source_id)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Data source {source_id} not found")
        
        file_size = file_path.stat().st_size
        strategy = self._determine_load_strategy(file_size)
        
        log_data_action(
            "EDA_DATA_LOAD",
            details=f"source_id: {source_id}, size: {file_size}, strategy: {strategy}"
        )
        
        # Load data based on strategy
        data_info = await self._load_with_strategy(
            file_path, 
            strategy, 
            source_id
        )
        
        # Update cache
        self.memory_cache[cache_key] = {
            "data_info": data_info,
            "timestamp": datetime.now().isoformat(),
            "access_count": 1
        }
        
        self._update_access_log(cache_key)
        
        # Clean old cache entries if needed
        await self._manage_cache_size()
        
        return data_info
    
    def _determine_load_strategy(self, file_size: int) -> str:
        """Determine optimal loading strategy based on file size."""
        if file_size < self.SMALL_DATA_THRESHOLD:
            return DataLoadStrategy.FULL_MEMORY
        elif file_size < self.MEDIUM_DATA_THRESHOLD:
            return DataLoadStrategy.CHUNKED
        else:
            return DataLoadStrategy.LAZY
    
    async def _load_with_strategy(
        self, 
        file_path: Path, 
        strategy: str,
        source_id: str
    ) -> Dict[str, Any]:
        """Load data using specified strategy."""
        
        file_ext = file_path.suffix.lower()
        sample_cap = self._apply_global_sample_limit(self.MAX_INTERACTIVE_SAMPLE_ROWS)
        
        if strategy == DataLoadStrategy.FULL_MEMORY:
            df = await self._load_full_dataset(file_path)
            return {
                "strategy": strategy,
                "data": df,
                "metadata": self._extract_metadata(df),
                "file_path": str(file_path),
                "source_id": source_id,
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
        elif strategy == DataLoadStrategy.CHUNKED:
            # Load sample for preview, setup chunked iterator for processing #interactive
            sample_df = await self._load_sample(
                file_path,
                sample_size=sample_cap
            )
            chunk_iterator = self._create_chunk_iterator(file_path)
            
            return {
                "strategy": strategy,
                "data": sample_df,  # Sample for immediate use
                "chunk_iterator": chunk_iterator,
                "metadata": self._extract_metadata(sample_df),
                "file_path": str(file_path),
                "source_id": source_id,
                "estimated_shape": self._estimate_shape(file_path),
                "chunk_size": sample_cap
            }
            
        else:  # LAZY strategy
            # Use Dask for lazy loading
            dask_df = self._setup_dask_dataframe(file_path)
            head_rows = sample_cap if sample_cap is not None else self.MAX_INTERACTIVE_SAMPLE_ROWS
            sample_df = dask_df.head(head_rows)
            
            return {
                "strategy": strategy,
                "data": sample_df,  # Small sample
                "dask_df": dask_df,  # Lazy dataframe
                "metadata": self._extract_metadata(sample_df),
                "file_path": str(file_path),
                "source_id": source_id,
                "estimated_shape": (len(dask_df), len(dask_df.columns)),
                "partitions": dask_df.npartitions
            }
    
    async def _load_full_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load complete dataset into memory."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Try UTF-8 first, fallback to other encodings
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file_path, encoding='latin1')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, encoding='cp1252')
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    async def _load_sample(
        self, 
        file_path: Path, 
        sample_size: int = MAX_INTERACTIVE_SAMPLE_ROWS
    ) -> pd.DataFrame:
        """Load sample of dataset."""
        sample_size = self._apply_global_sample_limit(sample_size)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Read sample rows with encoding handling
            try:
                return pd.read_csv(file_path, nrows=sample_size, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file_path, nrows=sample_size, encoding='latin1')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, nrows=sample_size, encoding='cp1252')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return df.head(sample_size)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            return df.head(sample_size)
        else:
            return await self._load_full_dataset(file_path)
    
    def _create_chunk_iterator(
    self, 
    file_path: Path, 
    chunk_size: int = MAX_INTERACTIVE_SAMPLE_ROWS
    ):
        """Create iterator for chunked reading."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Try different encodings for chunked reading
            try:
                return pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file_path, chunksize=chunk_size, encoding='latin1')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, chunksize=chunk_size, encoding='cp1252')
        else:
            # For non-CSV, return None (will fall back to full load)
            return None
    
    def _setup_dask_dataframe(self, file_path: Path) -> dd.DataFrame:
        """Setup Dask dataframe for lazy loading."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Try different encodings for Dask
            try:
                return dd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return dd.read_csv(file_path, encoding='latin1')
                except UnicodeDecodeError:
                    return dd.read_csv(file_path, encoding='cp1252')
        elif file_ext == '.parquet':
            return dd.read_parquet(file_path)
        else:
            # Fall back to pandas and convert
            try:
                df = pd.read_csv(file_path, encoding='utf-8') if file_ext == '.csv' else pd.read_excel(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1') if file_ext == '.csv' else pd.read_excel(file_path)
            return dd.from_pandas(df, npartitions=4)
    
    def _extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive metadata from dataframe."""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _estimate_shape(self, file_path: Path) -> Tuple[int, int]:
        """Estimate shape without loading full file."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # Count lines for row estimate with encoding handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for line in f) - 1
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin1') as f:
                        row_count = sum(1 for line in f) - 1
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='cp1252') as f:
                        row_count = sum(1 for line in f) - 1
            
            # Read first row for column count
            try:
                df_sample = pd.read_csv(file_path, nrows=1, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_sample = pd.read_csv(file_path, nrows=1, encoding='latin1')
                except UnicodeDecodeError:
                    df_sample = pd.read_csv(file_path, nrows=1, encoding='cp1252')
            col_count = len(df_sample.columns)
            
            return (row_count, col_count)
        else:
            # For other formats, need to load to get shape
            return (0, 0)
    
    async def _get_source_path(self, source_id: str) -> Optional[Path]:
        """Get file path for data source."""
        try:
            # First, try to get the data source from the database
            from core.database.models import DataSource
            from sqlalchemy import select
            
            stmt = select(DataSource).where(DataSource.source_id == source_id)
            result = await self.session.execute(stmt)
            data_source = result.scalar_one_or_none()
            
            if data_source:
                # Check if there's a file_path in config
                config = data_source.config or {}
                if 'file_path' in config:
                    file_path = Path(config['file_path'])
                    if file_path.exists():
                        return file_path
            
            # Fallback: Look for files in uploads directory
            uploads_dir = Path("uploads/data")
            if uploads_dir.exists():
                for file_path in uploads_dir.iterdir():
                    if file_path.is_file() and file_path.name.startswith(f"{source_id}_"):
                        return file_path
            
            # Additional fallback: Check uploads root
            uploads_root = Path("uploads")
            if uploads_root.exists():
                for file_path in uploads_root.rglob("*"):
                    if file_path.is_file() and source_id in file_path.name:
                        return file_path
            
            return None
            
        except Exception as e:
            log_data_action("SOURCE_PATH_ERROR", details=f"source_id:{source_id},error:{str(e)}")
            # Still try file system fallback
            uploads_dir = Path("uploads/data") 
            if uploads_dir.exists():
                for file_path in uploads_dir.iterdir():
                    if file_path.is_file() and source_id in file_path.name:
                        return file_path
            return None
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False
        
        age = datetime.now() - cache_entry["timestamp"]
        return age < timedelta(hours=self.CACHE_TTL_HOURS)
    
    def _update_access_log(self, cache_key: str):
        """Update access log for cache management."""
        if cache_key not in self.access_log:
            self.access_log[cache_key] = []
        
        self.access_log[cache_key].append(datetime.now())
        
        # Keep only last 100 accesses
        self.access_log[cache_key] = self.access_log[cache_key][-100:]
    
    async def _manage_cache_size(self):
        """Manage cache size using LRU eviction."""
        total_size = sum(
            entry.get("data_info", {}).get("memory_usage", 0)
            for entry in self.memory_cache.values()
        )
        
        max_size_bytes = self.MAX_CACHE_SIZE_GB * 1024**3
        
        if total_size > max_size_bytes:
            # Sort by last access time
            sorted_keys = sorted(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].get("timestamp", datetime.min)
            )
            
            # Remove oldest entries until under limit
            while total_size > max_size_bytes and sorted_keys:
                key_to_remove = sorted_keys.pop(0)
                removed_entry = self.memory_cache.pop(key_to_remove, None)
                
                if removed_entry:
                    removed_size = removed_entry.get("data_info", {}).get("memory_usage", 0)
                    total_size -= removed_size
                    
                    log_data_action(
                        "EDA_CACHE_EVICT",
                        details=f"Evicted {key_to_remove}, freed {removed_size} bytes"
                    )
    
    async def register_session(
        self, 
        source_id: str, 
        user_id: str,
        session_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new EDA session.
        
        Args:
            source_id: Data source identifier
            user_id: User identifier
            session_config: Optional session configuration
            
        Returns:
            Session key
        """
        session_key = f"{user_id}_{source_id}_{datetime.now().timestamp()}"
        session_hash = hashlib.md5(session_key.encode()).hexdigest()[:8]
        session_key = f"{user_id}_{source_id}_{session_hash}"
        
        # Prepare data
        data_info = await self.prepare_for_eda(source_id, user_id)
        
        # Register session
        self.active_sessions[session_key] = {
            "source_id": source_id,
            "user_id": user_id,
            "data_info": data_info,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "config": session_config or {},
            "state": "active",
            "checkpoints": []
        }
        
        log_data_action(
            "EDA_SESSION_CREATE",
            details=f"session_key: {session_key}, source_id: {source_id}"
        )
        
        return session_key
    
    def get_session_data(self, session_key: str) -> Optional[pd.DataFrame]:
        """
        Get data for active session.
        
        Args:
            session_key: Session identifier
            
        Returns:
            DataFrame or None if session not found
        """
        session = self.active_sessions.get(session_key)
        
        if not session:
            return None
        
        # Update last accessed
        session["last_accessed"] = datetime.now()
        
        data_info = session.get("data_info", {})
        return data_info.get("data")
    
    def get_session_info(self, session_key: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self.active_sessions.get(session_key)
    
    async def save_checkpoint(
        self, 
        session_key: str, 
        checkpoint_data: Dict[str, Any]
    ) -> bool:
        """
        Save session checkpoint for recovery.
        
        Args:
            session_key: Session identifier
            checkpoint_data: Data to checkpoint
            
        Returns:
            Success status
        """
        session = self.active_sessions.get(session_key)
        
        if not session:
            return False
        
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "data": checkpoint_data
        }
        
        session["checkpoints"].append(checkpoint)
        
        # Keep only last 10 checkpoints
        session["checkpoints"] = session["checkpoints"][-10:]
        
        # Also save to disk for persistence
        checkpoint_file = self.cache_dir / f"checkpoint_{session_key}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(
                    checkpoint_data,
                    f,
                    default=str  # Handle datetime serialization
                )
            
            log_data_action(
                "EDA_CHECKPOINT_SAVE",
                details=f"session_key: {session_key}"
            )
            
            return True
            
        except Exception as e:
            log_data_action(
                "EDA_CHECKPOINT_ERROR",
                success=False,
                details=str(e)
            )
            return False
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        expired_keys = []
        
        for key, session in self.active_sessions.items():
            age = now - session.get("last_accessed", now)
            
            if age > timedelta(hours=self.CACHE_TTL_HOURS):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_sessions[key]
            
            # Clean up checkpoint files
            checkpoint_file = self.cache_dir / f"checkpoint_{key}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            log_data_action(
                "EDA_SESSION_CLEANUP",
                details=f"Removed expired session: {key}"
            )
        
        return len(expired_keys)

    async def load_data_preview(
        self, 
        source_id: str,
        user_id: str,
        sample_size: int = 100,
        mode: str = "head"
    ) -> Dict[str, Any]:
        """
        Load a preview of the data for EDA operations.
        
        Args:
            source_id: Data source identifier
            user_id: User identifier
            sample_size: Number of rows to preview
            mode: Preview mode ("head", "tail", "sample")
            
        Returns:
            Dictionary with preview data and metadata
        """
        try:
            sample_size = self._apply_global_sample_limit(sample_size)
            # Get source path
            file_path = await self._get_source_path(source_id)
            if not file_path:
                raise FileNotFoundError(f"Data source {source_id} not found")

            def _load_preview() -> Dict[str, Any]:
                # Load preview data based on mode
                if mode == "head":
                    df_local = pd.read_csv(file_path, nrows=sample_size)
                elif mode == "tail":
                    # For tail, we need to count rows first
                    total_rows = sum(1 for _ in open(file_path)) - 1  # subtract header
                    skip_rows = max(0, total_rows - sample_size)
                    df_local = pd.read_csv(file_path, skiprows=skip_rows)
                else:  # sample mode
                    # Load full data and sample
                    full_df = pd.read_csv(file_path)
                    if len(full_df) > sample_size:
                        df_local = full_df.sample(n=sample_size)
                    else:
                        df_local = full_df

                # Convert to serializable format
                threshold_setting = (
                    getattr(self.settings, "EDA_PREVIEW_FULL_DATA_THRESHOLD", 50_000)
                    if self.settings
                    else 50_000
                )
                threshold = float("inf") if threshold_setting is None else threshold_setting
                data_rows_local = (
                    df_local.values.tolist()
                    if sample_size >= threshold
                    else df_local.head(sample_size).values.tolist()
                )

                preview_data_local = {
                    "columns": df_local.columns.tolist(),
                    "data": data_rows_local,
                    "dtypes": df_local.dtypes.astype(str).to_dict(),
                    "shape": df_local.shape,
                }

                metadata_local = self._extract_metadata(df_local)

                return preview_data_local, metadata_local, len(df_local)

            preview_data, metadata, row_count = await asyncio.to_thread(_load_preview)

            log_data_action("EDA_PREVIEW", details=f"source:{source_id},rows:{row_count}")

            return {
                "data": preview_data,
                "metadata": metadata,
                "source_id": source_id,
                "mode": mode,
                "sample_size": sample_size,
            }

        except Exception as e:
            log_data_action("EDA_PREVIEW_ERROR", details=f"source:{source_id},error:{str(e)}")
            raise