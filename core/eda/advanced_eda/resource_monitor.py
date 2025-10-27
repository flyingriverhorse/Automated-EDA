"""Resource monitoring utilities for advanced EDA workflows."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Deque, Dict, Optional

try:  # pragma: no cover - psutil is an optional runtime dependency
	import psutil  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
	psutil = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class OperationRecord:
	"""Historic record describing an executed analysis step."""

	name: str
	duration: float
	timestamp: float


class ResourceMonitor:
	"""Lightweight runtime monitor for domain analysis executions.

	The monitor exposes two core responsibilities:

	1. ``check_available_resources`` – an awaitable guard used prior to launching
	   a new workflow to make sure global system limits have not been exceeded.
	2. ``track_operation`` – a context manager that records execution metrics and
	   keeps an accurate count of concurrent analysis steps.

	Both are intentionally lightweight so they can run in async contexts without
	introducing noticeable overhead.
	"""

	def __init__(
		self,
		*,
		max_cpu_percent: float = 85.0,
		max_memory_percent: float = 85.0,
		max_concurrent_operations: Optional[int] = 6,
		max_operation_seconds: float = 120.0,
		history_size: int = 256,
		cpu_sample_interval: float = 0.05,
	) -> None:
		self.max_cpu_percent = max_cpu_percent
		self.max_memory_percent = max_memory_percent
		self.max_concurrent_operations = max_concurrent_operations
		self.max_operation_seconds = max_operation_seconds
		self.cpu_sample_interval = cpu_sample_interval

		self._active_operations = 0
		self._records: Deque[OperationRecord] = deque(maxlen=history_size)
		self._sync_lock = Lock()
		self._async_lock = asyncio.Lock()

		# Prime psutil's CPU sampler so the first real measurement is accurate.
		if psutil is not None:  # pragma: no cover - trivial branch
			try:
				psutil.cpu_percent(interval=None)
			except Exception:  # pragma: no cover - defensive
				logger.debug("Unable to prime psutil CPU sampler", exc_info=True)

	async def check_available_resources(self) -> bool:
		"""Return ``True`` when system and concurrency limits allow new work.

		This method is intentionally conservative: if any limit is exceeded we
		return ``False`` so callers can short-circuit before heavy processing.
		"""

		async with self._async_lock:
			with self._sync_lock:
				active_ops = self._active_operations

			if (
				self.max_concurrent_operations is not None
				and active_ops >= self.max_concurrent_operations
			):
				logger.debug(
					"Resource monitor denying execution: %s active >= limit %s",
					active_ops,
					self.max_concurrent_operations,
				)
				return False

			if psutil is None:
				return True

			try:
				cpu_usage = psutil.cpu_percent(interval=self.cpu_sample_interval)
				if cpu_usage > self.max_cpu_percent:
					logger.debug(
						"Resource monitor denying execution due to CPU %.2f%% > %.2f%%",
						cpu_usage,
						self.max_cpu_percent,
					)
					return False

				memory_usage = psutil.virtual_memory().percent
				if memory_usage > self.max_memory_percent:
					logger.debug(
						"Resource monitor denying execution due to memory %.2f%% > %.2f%%",
						memory_usage,
						self.max_memory_percent,
					)
					return False

			except Exception as exc:  # pragma: no cover - defensive
				logger.warning("Resource monitor unable to sample system stats", exc_info=exc)
				return True

			return True

	@contextmanager
	def track_operation(self, name: str):
		"""Context manager that tracks an in-flight operation.

		Counts concurrent operations, logs slow spans, and records execution
		metrics for later inspection.
		"""

		start_time = time.perf_counter()
		timestamp = time.time()

		with self._sync_lock:
			self._active_operations += 1
			active_now = self._active_operations

		logger.debug("Resource monitor starting '%s' (active=%s)", name, active_now)

		try:
			yield
		finally:
			duration = time.perf_counter() - start_time
			with self._sync_lock:
				self._active_operations = max(0, self._active_operations - 1)
				self._records.append(OperationRecord(name=name, duration=duration, timestamp=timestamp))
				active_after = self._active_operations

			if duration > self.max_operation_seconds:
				logger.warning(
					"Resource monitor detected slow operation '%s': %.2fs",
					name,
					duration,
				)

			logger.debug("Resource monitor finished '%s' (active=%s)", name, active_after)

	async def get_stats(self) -> Dict[str, Any]:
		"""Return a snapshot of recent monitoring metrics."""

		async with self._async_lock:
			with self._sync_lock:
				history = list(self._records)
				active_ops = self._active_operations

		return {
			"active_operations": active_ops,
			"history": [record.__dict__ for record in history],
			"max_cpu_percent": self.max_cpu_percent,
			"max_memory_percent": self.max_memory_percent,
			"max_concurrent_operations": self.max_concurrent_operations,
			"max_operation_seconds": self.max_operation_seconds,
		}
