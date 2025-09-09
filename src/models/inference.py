from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class InferenceBase(BaseModel):
    id: Optional[UUID]
    name: str
    query: Optional[dict]
    results: Optional[dict]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class InferenceRead(InferenceBase):
    finetuned_models: Optional[List["FinetunedModelRead"]]
    preloaded_events: Optional[List["PreloadedEventRead"]]


class InferenceUpdate(BaseModel):
    name: Optional[str] = None
    query: dict
    results: Optional[dict] = {}
    finetuned_model_ids: Optional[List[UUID]] = []
