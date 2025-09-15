import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, DateTime, Enum, ForeignKey, CheckConstraint, Table, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()

class SourceType(str, enum.Enum):
    huggingface = "huggingface"
    s3 = "s3"

inference_finetuned_model = Table(
    "inference_finetuned_model",
    Base.metadata,
    Column("inference_id", UUID(as_uuid=True), ForeignKey("inferences.id"), primary_key=True),
    Column("finetuned_model_id", UUID(as_uuid=True), ForeignKey("finetuned_models.id"), primary_key=True)
)

class FinetunedModel(Base):
    __tablename__ = "finetuned_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    source_type = Column(Enum(SourceType), nullable=False)
    source_details = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    data_config = Column(JSON, nullable=True)

class Inference(Base):
    __tablename__ = "inferences"
    __table_args__ = (
        CheckConstraint(
            "user_email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
            name='valid_email_format'
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    query = Column(JSON, nullable=True) # contains bbox, date, or date range
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    user_email = Column(String(254), nullable=False, unique=True)
    finetuned_models = relationship(
        "FinetunedModel",
        secondary=inference_finetuned_model,
        backref="inferences"
    )

    results = Column(JSON, nullable=True)  # Store results as JSON

class PreloadedEvent(Base):
    __tablename__ = "preloaded_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String, nullable=False)
    event_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    inference_id = Column(UUID(as_uuid=True), ForeignKey("inferences.id"), nullable=True)
    inference = relationship("Inference", backref="preloaded_events")
