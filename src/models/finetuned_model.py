import enum

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone

class SourceType(str, enum.Enum):
    huggingface = "huggingface"
    s3 = "s3"

class FinetunedModelBase(BaseModel):
    id: UUID
    name: str
    source_type: SourceType
    source_details: dict
    created_at: datetime
    data_config: Optional[dict]

    class Config:
        from_attributes = True
        # Ensure all datetime fields are timezone-aware and serialized in UTC
        json_encoders = {
            datetime: lambda v: v.isoformat() if v.tzinfo else v.replace(tzinfo=timezone.utc).isoformat()
        }

class FinetunedModelRead(FinetunedModelBase):
    @validator('created_at', pre=True)
    def validate_created_at(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            # If datetime is naive, assume it's UTC
            return value.replace(tzinfo=timezone.utc)
        return value

class FinetunedModelUpdate(FinetunedModelBase):
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    name: Optional[str] = None
    source_type: Optional[SourceType] = SourceType.s3
