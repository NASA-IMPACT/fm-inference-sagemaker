from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class InferenceBase(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    query: Optional[dict] = {}
    results: Optional[dict]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class InferenceRead(InferenceBase):
    finetuned_models: Optional[List["FinetunedModelRead"]]
    preloaded_events: Optional[List["PreloadedEventRead"]]


class InferenceUpdate(InferenceBase):
    query: dict
    finetuned_model_ids: Optional[List[UUID]] = []
