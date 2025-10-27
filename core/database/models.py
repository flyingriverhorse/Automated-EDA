"""
Database Models for FastAPI

SQLAlchemy models that mirror the existing Flask database structure.
These models are compatible with the existing database schema.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from .engine import Base


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """
    User model - mirrors the Flask user table structure.
    Compatible with existing Flask-Login users.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # User profile information
    full_name = Column(String(200), nullable=True)
    
    # User status
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Authentication tracking
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # Relationship to DataSource model
    data_sources = relationship("DataSource", back_populates="creator")
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    def to_dict(self):
        """Convert model to dictionary (excluding password)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "is_verified": self.is_verified,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "login_count": self.login_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DataSource(Base, TimestampMixin):
    """
    Data Source model for data ingestion connections.
    Compatible with existing data_sources table.
    """
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(50), unique=True, nullable=True, index=True)  # UUID string identifier for file naming
    name = Column(String(100), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # 'snowflake', 'postgres', 'api', etc.
    
    # Connection configuration (stored as JSON)
    config = Column(JSON, nullable=False)
    
    # Credentials (encrypted in production)
    credentials = Column(JSON, nullable=True)
    
    # Status and metadata
    is_active = Column(Boolean, default=True, nullable=False)
    last_tested = Column(DateTime, nullable=True)
    test_status = Column(String(20), default="untested", nullable=False)  # 'success', 'failed', 'untested'
    
    # User who created this source
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)  # Foreign key to users.id
    
    # Relationship to User model
    creator = relationship("User", back_populates="data_sources")
    
    # Description and documentation
    description = Column(Text, nullable=True)
    
    # Additional source metadata (stored as JSON)
    source_metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<DataSource {self.name} ({self.type})>"
    
    def to_dict(self):
        """Convert model to dictionary (excluding sensitive credentials)."""
        # Safely get creator username
        try:
            creator_name = self.creator.username if self.creator else "Unknown"
        except:
            creator_name = "Unknown"
            
        return {
            "id": self.id,
            "source_id": self.source_id,  # UUID string identifier
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "metadata": self.source_metadata,  # Include source metadata (renamed for API compatibility)
            "is_active": self.is_active,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "test_status": self.test_status,
            "created_by": creator_name,
            "created_by_id": self.created_by,  # Keep the ID as well for internal use
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @property
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission. Simplified for this model."""
        # For now, return basic permission logic
        # In a real app, this would check against user roles/permissions
        return True  # Placeholder
    
    @property
    def is_admin(self) -> bool:
        """Check if creator has admin privileges."""
        # This would be determined by looking up the user
        return False  # Placeholder


# Unused tables removed: DataIngestionJob, SystemLog
# These tables were not used anywhere in the application and caused schema differences


@asynccontextmanager
async def get_database_session(
    engine: Optional[AsyncEngine] = None,
    *,
    expire_on_commit: bool = False,
) -> AsyncIterator[AsyncSession]:
    """Provide an async SQLAlchemy session scoped to the given engine.

    Args:
        engine: Optional preconfigured AsyncEngine. Falls back to the global engine
            initialized via :func:`core.database.engine.init_db` when not provided.
        expire_on_commit: Whether attributes should be expired after commit. Matches
            SQLAlchemy's ``expire_on_commit`` flag for session factories.

    Yields:
        AsyncSession: A managed session with automatic commit/rollback semantics.
    """

    from .engine import get_engine  # Local import to avoid circular dependency

    resolved_engine = engine or get_engine()
    session_factory = async_sessionmaker(
        bind=resolved_engine,
        class_=AsyncSession,
        expire_on_commit=expire_on_commit,
    )

    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()