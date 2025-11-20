from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone
import enum

class PreloadedEventBase(BaseModel):
    id: UUID
    event_name: str
    event_details: Optional[dict]
    created_at: datetime
    inference_id: Optional[UUID]

    class Config:
        from_attributes = True
        # Ensure all datetime fields are timezone-aware and serialized in UTC
        json_encoders = {
            datetime: lambda v: v.isoformat() if v.tzinfo else v.replace(tzinfo=timezone.utc).isoformat()
        }

class PreloadedEventRead(PreloadedEventBase):
    inference: Optional["InferenceRead"]

    @validator('created_at', pre=True)
    def validate_created_at(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            # If datetime is naive, assume it's UTC
            return value.replace(tzinfo=timezone.utc)
        return value

class PreloadedEventUpdate(PreloadedEventBase):
    event_name: Optional[str] = ''
    event_details: Optional[dict] = {}
    inference_id: Optional[UUID]
    created_at: Optional[datetime] = None
    id: Optional[UUID] = None
