"""
Rate Limiter for EDA Code Execution

Provides per-user rate limiting to prevent abuse and ensure fair resource usage
across multiple users.
"""

import time
from typing import Any, Dict, Optional
from collections import defaultdict, deque
import logging

from config import get_settings

logger = logging.getLogger(__name__)


class ExecutionRateLimiter:
    """Rate limiter for code execution with per-user tracking"""

    # Default limits
    DEFAULT_MAX_EXECUTIONS_PER_MINUTE = 20
    DEFAULT_MAX_CONCURRENT_EXECUTIONS = 1
    DEFAULT_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        max_executions_per_minute: Optional[int] = DEFAULT_MAX_EXECUTIONS_PER_MINUTE,
        max_concurrent_executions: Optional[int] = DEFAULT_MAX_CONCURRENT_EXECUTIONS,
        cleanup_interval_seconds: Optional[int] = DEFAULT_CLEANUP_INTERVAL_SECONDS,
    ):
        self.max_executions_per_minute = self._normalize_limit(max_executions_per_minute)
        self.max_concurrent_executions = self._normalize_limit(max_concurrent_executions)
        self.cleanup_interval_seconds = (
            cleanup_interval_seconds
            if cleanup_interval_seconds and cleanup_interval_seconds > 0
            else self.DEFAULT_CLEANUP_INTERVAL_SECONDS
        )

        # Per-user execution tracking
        self.user_executions: Dict[str, deque] = defaultdict(deque)
        self.user_concurrent: Dict[str, int] = defaultdict(int)

        # Last cleanup time
        self.last_cleanup = time.time()

        exec_desc = (
            f"{self.max_executions_per_minute} exec/min"
            if self.max_executions_per_minute is not None
            else "unlimited exec/min"
        )
        concurrent_desc = (
            f"{self.max_concurrent_executions} concurrent per user"
            if self.max_concurrent_executions is not None
            else "unlimited concurrent per user"
        )

        logger.info(
            "Rate limiter initialized: %s, %s, cleanup every %ss",
            exec_desc,
            concurrent_desc,
            self.cleanup_interval_seconds,
        )
    
    @staticmethod
    def _normalize_limit(limit: Optional[int]) -> Optional[int]:
        if limit is None:
            return None
        if limit <= 0:
            return None
        return limit

    def _cleanup_old_executions(self):
        """Remove execution records older than 1 minute"""
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago
        
        for user_id in list(self.user_executions.keys()):
            user_queue = self.user_executions[user_id]
            
            # Remove old executions
            while user_queue and user_queue[0] < cutoff_time:
                user_queue.popleft()
            
            # Clean up empty queues
            if not user_queue:
                del self.user_executions[user_id]
        
        self.last_cleanup = current_time
        logger.debug("Cleaned up old execution records")
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed"""
        return time.time() - self.last_cleanup > self.cleanup_interval_seconds
    
    def check_rate_limit(self, user_id: str) -> Dict[str, Any]:
        """Check if user can execute code based on rate limits"""
        if self._should_cleanup():
            self._cleanup_old_executions()
        
        current_time = time.time()
        
        # Check concurrent executions
        concurrent_count = self.user_concurrent.get(user_id, 0)
        if (
            self.max_concurrent_executions is not None
            and concurrent_count >= self.max_concurrent_executions
        ):
            return {
                "allowed": False,
                "reason": "concurrent_limit",
                "message": f"Too many concurrent executions ({concurrent_count}/{self.max_concurrent_executions})",
                "retry_after": 30  # seconds
            }
        
        # Check executions per minute
        user_queue = self.user_executions[user_id]
        minute_ago = current_time - 60
        
        # Count executions in the last minute
        recent_executions = sum(1 for exec_time in user_queue if exec_time > minute_ago)

        if (
            self.max_executions_per_minute is not None
            and recent_executions >= self.max_executions_per_minute
        ):
            # Calculate when the oldest execution will be outside the 1-minute window
            if user_queue:
                oldest_in_window = next((t for t in user_queue if t > minute_ago), None)
                retry_after = int(oldest_in_window + 60 - current_time) if oldest_in_window else 60
            else:
                retry_after = 60
                
            return {
                "allowed": False,
                "reason": "rate_limit",
                "message": f"Rate limit exceeded ({recent_executions}/{self.max_executions_per_minute} per minute)",
                "retry_after": retry_after
            }

        remaining = (
            self.max_executions_per_minute - recent_executions
            if self.max_executions_per_minute is not None
            else None
        )

        return {
            "allowed": True,
            "remaining_executions": remaining,
            "concurrent_executions": concurrent_count
        }
    
    def record_execution_start(self, user_id: str):
        """Record the start of a code execution"""
        current_time = time.time()
        
        # Add to execution history
        self.user_executions[user_id].append(current_time)
        
        # Increment concurrent counter
        self.user_concurrent[user_id] = self.user_concurrent.get(user_id, 0) + 1
        
        logger.debug(f"User {user_id} started execution (concurrent: {self.user_concurrent[user_id]})")
    
    def record_execution_end(self, user_id: str):
        """Record the end of a code execution"""
        # Decrement concurrent counter
        if user_id in self.user_concurrent:
            self.user_concurrent[user_id] = max(0, self.user_concurrent[user_id] - 1)
            if self.user_concurrent[user_id] == 0:
                del self.user_concurrent[user_id]
        
        logger.debug(f"User {user_id} finished execution (concurrent: {self.user_concurrent.get(user_id, 0)})")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get execution statistics for a user"""
        if self._should_cleanup():
            self._cleanup_old_executions()
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        user_queue = self.user_executions.get(user_id, deque())
        recent_executions = sum(1 for exec_time in user_queue if exec_time > minute_ago)
        concurrent_executions = self.user_concurrent.get(user_id, 0)

        remaining = None
        if self.max_executions_per_minute is not None:
            remaining = max(0, self.max_executions_per_minute - recent_executions)
        
        return {
            "user_id": user_id,
            "executions_last_minute": recent_executions,
            "remaining_executions": remaining,
            "concurrent_executions": concurrent_executions,
            "max_executions_per_minute": self.max_executions_per_minute,
            "max_concurrent_executions": self.max_concurrent_executions,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global execution statistics"""
        if self._should_cleanup():
            self._cleanup_old_executions()
        
        total_users = len(self.user_executions)
        total_concurrent = sum(self.user_concurrent.values())
        
        return {
            "total_active_users": total_users,
            "total_concurrent_executions": total_concurrent,
            "max_executions_per_minute": self.max_executions_per_minute,
            "max_concurrent_executions": self.max_concurrent_executions,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
        }


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> ExecutionRateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = ExecutionRateLimiter(
            max_executions_per_minute=settings.EDA_CUSTOM_EXECUTIONS_PER_MINUTE,
            max_concurrent_executions=settings.EDA_CUSTOM_MAX_CONCURRENT_EXECUTIONS,
            cleanup_interval_seconds=settings.EDA_CUSTOM_RATE_CLEANUP_INTERVAL_SECONDS,
        )
    return _rate_limiter


def reset_rate_limiter():
    """Reset global rate limiter (useful for testing)"""
    global _rate_limiter
    _rate_limiter = None