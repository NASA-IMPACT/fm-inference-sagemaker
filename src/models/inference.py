from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class InferenceBase(BaseModel):
    id: Optional[UUID]
    name: str
    query: Optional[dict]
    result_s3_path: Optional[str]
    result_geojson: Optional[List[dict]]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class InferenceRead(InferenceBase):
    finetuned_models: Optional[List["FinetunedModelRead"]]
    preloaded_events: Optional[List["PreloadedEventRead"]]


class InferenceUpdate(BaseModel):
    name: Optional[str]
    query: dict
    result_s3_path: Optional[List[str]]
    result_geojson: Optional[List[dict]]
    # finetuned_model_ids: Optional[List[UUID]]
