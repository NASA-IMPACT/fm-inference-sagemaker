import json
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone

class InferenceBase(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    query: Optional[dict] = {}
    results: Optional[dict] = {}
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        # Ensure all datetime fields are timezone-aware and serialized in UTC
        json_encoders = {
            datetime: lambda v: v.isoformat() if v.tzinfo else v.replace(tzinfo=timezone.utc).isoformat()
        }


class InferenceRead(InferenceBase):
    finetuned_models: Optional[List["FinetunedModelRead"]]
    # preloaded_events: Optional[List["PreloadedEventRead"]]

    @validator('created_at', pre=True)
    def validate_created_at(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            # If datetime is naive, assume it's UTC
            return value.replace(tzinfo=timezone.utc)
        return value

    @validator('query', pre=True)
    def validate_query(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for query")
        return value


class InferenceUpdate(InferenceBase):
    query: dict
    finetuned_model_ids: Optional[List[UUID]] = []

    @validator('query', pre=True)
    def validate_query(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for query")
        return value
