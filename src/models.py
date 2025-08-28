import enum
import uuid

from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Enum, ForeignKey, Boolean, Integer, JSON
)
from sqlalchemy import event, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()

class SourceType(str, enum.Enum):
    huggingface = "huggingface"
    s3 = "s3"

class FinetunedModel(Base):
    __tablename__ = "finetuned_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    source_type = Column(Enum(SourceType), nullable=False)
    source_details = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    data_config = Column(JSON, nullable=True)

inference_finetuned_model = Table(
    "inference_finetuned_model",
    Base.metadata,
    Column("inference_id", UUID(as_uuid=True), ForeignKey("jobs.id"), primary_key=True),
    Column("finetuned_model_id", UUID(as_uuid=True), ForeignKey("finetuned_models.id"), primary_key=True)
)

class Inference(Base):
    __tablename__ = "inferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    query = Column(JSON, nullable=True) # contains bbox, date, or date range
    result_s3_path = Column(String, nullable=True)
    result_geojson = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finetuned_models = relationship(
        "FinetunedModel",
        secondary=inference_finetuned_model,
        backref="inferences"
    )

class PreloadedEvent:
    __tablename__ = "preloaded_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String, nullable=False)
    event_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    inference_id = Column(UUID(as_uuid=True), ForeignKey("inferences.id"), nullable=True)
    inference = relationship("Inference", backref="preloaded_events")


# will be handy later if we add status tracking to inferences
def set_timestamps(mapper, connection, target):
    # Set started_at when status changes to running
    if hasattr(target, 'status') and target.status == JobStatus.running and not target.started_at:
        target.started_at = datetime.utcnow()
    # Set completed_at when status changes to completed or failed
    if hasattr(target, 'status') and target.status in [JobStatus.completed, JobStatus.failed] and not target.completed_at:
        target.completed_at = datetime.utcnow()
    # Set failed_at when status changes to failed
    if hasattr(target, 'status') and target.status == JobStatus.failed and not getattr(target, 'failed_at', None):
        target.failed_at = datetime.utcnow()
    # Set deleted_at when is_deleted is True
    if hasattr(target, 'is_deleted') and target.is_deleted and not target.deleted_at:
        target.deleted_at = datetime.utcnow()

# Listen for before_insert and before_update events on Job

event.listen(Inference, 'before_insert', set_timestamps)
event.listen(Inference, 'before_update', set_timestamps)
