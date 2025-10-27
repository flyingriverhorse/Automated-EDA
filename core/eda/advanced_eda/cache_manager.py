"""Simple async-aware cache manager for domain analysis workflows."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
	"""Container for cached values with optional TTL metadata."""

	value: Any
	expires_at: Optional[float]


class CacheManager:
	"""Tiny in-memory cache with coroutine-friendly APIs.

	The implementation intentionally keeps the feature set focused: an LRU
	eviction policy, optional TTL handling, and lightweight async methods that
	can be awaited within FastAPI endpoints without blocking the loop.
	"""

	def __init__(self, *, max_size: int = 256) -> None:
		self.max_size = max_size
		self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
		self._lock = asyncio.Lock()

	async def get(self, key: str) -> Any:
		"""Return a cached value if present and not expired."""

		async with self._lock:
			self._purge_expired_locked()

			entry = self._entries.get(key)
			if entry is None:
				return None

			# Maintain LRU ordering by moving the key to the end
			self._entries.move_to_end(key)
			return entry.value

	async def set(self, key: str, value: Any, *, ttl: Optional[int] = None) -> None:
		"""Store a value in the cache, optionally expiring after ``ttl`` seconds."""

		expires_at = (time.time() + ttl) if ttl else None

		async with self._lock:
			self._purge_expired_locked()
			self._entries[key] = CacheEntry(value=value, expires_at=expires_at)
			self._entries.move_to_end(key)

			while len(self._entries) > self.max_size:
				self._entries.popitem(last=False)

	async def delete(self, key: str) -> None:
		"""Remove a cached entry if it exists."""

		async with self._lock:
			self._entries.pop(key, None)

	async def clear(self) -> None:
		"""Clear the entire cache."""

		async with self._lock:
			self._entries.clear()

	async def stats(self) -> Dict[str, Any]:
		"""Return diagnostic cache statistics."""

		async with self._lock:
			self._purge_expired_locked()
			return {
				"size": len(self._entries),
				"max_size": self.max_size,
				"keys": list(self._entries.keys()),
			}

	def _purge_expired_locked(self) -> None:
		"""Remove any expired entries (caller must hold ``_lock``)."""

		now = time.time()
		expired_keys = [
			key for key, entry in self._entries.items() if entry.expires_at and entry.expires_at <= now
		]
		for key in expired_keys:
			self._entries.pop(key, None)
