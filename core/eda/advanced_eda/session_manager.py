"""
Session Manager for EDA operations
Handles user sessions, state persistence, and recovery
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.utils.logging_utils import log_data_action


class EDASession:
    """Represents a single EDA session"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        source_id: str,
        data_info: Dict[str, Any]
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.source_id = source_id
        self.data_info = data_info
        
        # Session state
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.is_active = True
        
        # Analysis state
        self.completed_analyses = []
        self.current_analysis = None
        self.analysis_history = []
        
        # Notebook state
        self.notebook_cells = []
        self.variables = {}
        
        # Export history
        self.exports = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "is_active": self.is_active,
            "completed_analyses": self.completed_analyses,
            "current_analysis": self.current_analysis,
            "analysis_history": self.analysis_history,
            "notebook_cells": self.notebook_cells,
            "exports": self.exports
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], data_info: Dict[str, Any]) -> 'EDASession':
        """Create session from dictionary"""
        session = cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            source_id=data["source_id"],
            data_info=data_info
        )
        
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_accessed = datetime.fromisoformat(data["last_accessed"])
        session.is_active = data["is_active"]
        session.completed_analyses = data["completed_analyses"]
        session.current_analysis = data["current_analysis"]
        session.analysis_history = data["analysis_history"]
        session.notebook_cells = data["notebook_cells"]
        session.exports = data["exports"]
        
        return session


class SessionManager:
    """
    Manages EDA sessions with persistence and recovery
    """
    
    SESSION_TTL_HOURS = 4
    CHECKPOINT_INTERVAL_MINUTES = 10
    MAX_SESSIONS_PER_USER = 5
    
    def __init__(
        self,
        db_session: AsyncSession,
        session_dir: Path = None
    ):
        self.db_session = db_session
        self.session_dir = session_dir or Path("data/eda_sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Active sessions in memory
        self.active_sessions: Dict[str, EDASession] = {}
        
        # User session mapping
        self.user_sessions: Dict[str, List[str]] = {}
        
        # Background tasks
        self.checkpoint_task = None
        self.cleanup_task = None
        
    async def start(self):
        """Start background tasks"""
        self.checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop background tasks"""
        if self.checkpoint_task:
            self.checkpoint_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        # Save all active sessions
        await self._save_all_sessions()
    
    async def create_session(
        self,
        user_id: str,
        source_id: str,
        data_info: Dict[str, Any]
    ) -> str:
        """
        Create new EDA session
        
        Args:
            user_id: User identifier
            source_id: Data source identifier
            data_info: Data information from data manager
            
        Returns:
            Session ID
        """
        # Check user session limit
        user_sessions = self.user_sessions.get(user_id, [])
        if len(user_sessions) >= self.MAX_SESSIONS_PER_USER:
            # Remove oldest session
            oldest_session_id = user_sessions[0]
            await self.close_session(oldest_session_id)
        
        # Generate session ID
        session_id = f"{user_id}_{source_id}_{uuid4().hex[:8]}"
        
        # Create session
        session = EDASession(
            session_id=session_id,
            user_id=user_id,
            source_id=source_id,
            data_info=data_info
        )
        
        # Register session
        self.active_sessions[session_id] = session
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Save initial state
        await self._save_session(session)
        
        log_data_action(
            "SESSION_CREATE",
            details=f"session_id: {session_id}, user_id: {user_id}"
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[EDASession]:
        """
        Get active session
        
        Args:
            session_id: Session identifier
            
        Returns:
            EDASession or None
        """
        session = self.active_sessions.get(session_id)
        
        if session:
            session.last_accessed = datetime.now()
            return session
        
        # Try to restore from disk
        return await self._restore_session(session_id)
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update session state
        
        Args:
            session_id: Session identifier
            updates: Updates to apply
            
        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        
        if not session:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_accessed = datetime.now()
        
        return True
    
    async def add_analysis_result(
        self,
        session_id: str,
        analysis_type: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Add analysis result to session
        
        Args:
            session_id: Session identifier
            analysis_type: Type of analysis
            result: Analysis result
            
        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        
        if not session:
            return False
        
        analysis_entry = {
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        session.completed_analyses.append(analysis_type)
        session.analysis_history.append(analysis_entry)
        
        return True
    
    async def add_notebook_cell(
        self,
        session_id: str,
        cell_type: str,
        content: str,
        output: Optional[Any] = None
    ) -> bool:
        """
        Add notebook cell to session
        
        Args:
            session_id: Session identifier
            cell_type: Type of cell (code/markdown)
            content: Cell content
            output: Cell output
            
        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        
        if not session:
            return False
        
        cell = {
            "id": uuid4().hex,
            "type": cell_type,
            "content": content,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        
        session.notebook_cells.append(cell)
        
        return True
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close and save session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        session = self.active_sessions.get(session_id)
        
        if not session:
            return False
        
        session.is_active = False
        
        # Save final state
        await self._save_session(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Remove from user sessions
        if session.user_id in self.user_sessions:
            self.user_sessions[session.user_id].remove(session_id)
        
        log_data_action(
            "SESSION_CLOSE",
            details=f"session_id: {session_id}"
        )
        
        return True
    
    async def get_user_sessions(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for user
        
        Args:
            user_id: User identifier
            include_inactive: Include inactive sessions
            
        Returns:
            List of session summaries
        """
        sessions = []
        
        # Active sessions
        for session_id in self.user_sessions.get(user_id, []):
            session = self.active_sessions.get(session_id)
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "source_id": session.source_id,
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "is_active": True,
                    "analyses_count": len(session.completed_analyses)
                })
        
        # Inactive sessions from disk
        if include_inactive:
            user_session_files = self.session_dir.glob(f"{user_id}_*.json")
            for file_path in user_session_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if not data.get("is_active", True):
                            sessions.append({
                                "session_id": data["session_id"],
                                "source_id": data["source_id"],
                                "created_at": data["created_at"],
                                "last_accessed": data["last_accessed"],
                                "is_active": False,
                                "analyses_count": len(data.get("completed_analyses", []))
                            })
                except Exception:
                    continue
        
        return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)
    
    async def _save_session(self, session: EDASession):
        """Save session to disk"""
        file_path = self.session_dir / f"{session.session_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            log_data_action(
                "SESSION_SAVE_ERROR",
                success=False,
                details=str(e)
            )
    
    async def _restore_session(self, session_id: str) -> Optional[EDASession]:
        """Restore session from disk"""
        file_path = self.session_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Note: data_info needs to be reloaded from data manager
            # This is a simplified version
            session = EDASession.from_dict(data, {})
            
            # Re-register if still valid
            age = datetime.now() - session.last_accessed
            if age < timedelta(hours=self.SESSION_TTL_HOURS):
                self.active_sessions[session_id] = session
                
                if session.user_id not in self.user_sessions:
                    self.user_sessions[session.user_id] = []
                self.user_sessions[session.user_id].append(session_id)
                
                log_data_action(
                    "SESSION_RESTORE",
                    details=f"session_id: {session_id}"
                )
                
                return session
            
        except Exception as e:
            log_data_action(
                "SESSION_RESTORE_ERROR",
                success=False,
                details=str(e)
            )
        
        return None
    
    async def _save_all_sessions(self):
        """Save all active sessions"""
        for session in self.active_sessions.values():
            await self._save_session(session)
    
    async def _checkpoint_loop(self):
        """Background task for periodic checkpoints"""
        while True:
            try:
                await asyncio.sleep(self.CHECKPOINT_INTERVAL_MINUTES * 60)
                await self._save_all_sessions()
                
                log_data_action(
                    "SESSION_CHECKPOINT",
                    details=f"Saved {len(self.active_sessions)} sessions"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_data_action(
                    "SESSION_CHECKPOINT_ERROR",
                    success=False,
                    details=str(e)
                )
    
    async def _cleanup_loop(self):
        """Background task for cleaning expired sessions"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    age = now - session.last_accessed
                    if age > timedelta(hours=self.SESSION_TTL_HOURS):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.close_session(session_id)
                
                if expired_sessions:
                    log_data_action(
                        "SESSION_CLEANUP",
                        details=f"Closed {len(expired_sessions)} expired sessions"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_data_action(
                    "SESSION_CLEANUP_ERROR",
                    success=False,
                    details=str(e)
                )