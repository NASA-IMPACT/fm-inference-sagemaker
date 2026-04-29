import os
import enum
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Enum,
    JSON,
    CheckConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://fm_user:fm_password@localhost:5432/fm_finetuning"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=30,
    pool_recycle=3600,
)

Base = declarative_base()

_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


class InferenceStatus(str, enum.Enum):
    queued = "queued"
    download = "download"
    pre_inference = "pre_inference"
    inference = "inference"
    post_inference = "post_inference"
    complete = "complete"
    failed = "failed"


class Inference(Base):
    __tablename__ = "inferences"
    __table_args__ = (
        CheckConstraint(
            "user_email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
            name="valid_email_format",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    query = Column(JSON, nullable=True)
    status = Column(
        Enum(InferenceStatus), nullable=False, default=InferenceStatus.queued
    )
    error_stage = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    user_email = Column(String(254), nullable=False)
    results = Column(JSON, nullable=True)
