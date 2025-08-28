from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import enum

class PreloadedEventBase(BaseModel):
    id: UUID
    event_name: str
    event_details: Optional[dict]
    created_at: datetime
    inference_id: Optional[UUID]

    class Config:
        from_attributes = True

class PreloadedEventRead(PreloadedEventBase):
    inference: Optional["InferenceRead"]

class PreloadedEventUpdate(BaseModel):
    event_name: Optional[str]
    event_details: Optional[dict]
    inference_id: Optional[UUID]
