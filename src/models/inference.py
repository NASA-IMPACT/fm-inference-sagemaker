import json
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class InferenceBase(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    query: Optional[dict] = {}
    results: Optional[dict] = {}
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class InferenceRead(InferenceBase):
    finetuned_models: Optional[List["FinetunedModelRead"]]
    # preloaded_events: Optional[List["PreloadedEventRead"]]

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
