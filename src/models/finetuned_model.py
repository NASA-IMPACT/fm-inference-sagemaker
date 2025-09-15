import enum

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class SourceType(str, enum.Enum):
    huggingface = "huggingface"
    s3 = "s3"

class FinetunedModelBase(BaseModel):
    name: str
    source_type: SourceType
    source_details: Optional[dict] = None
    data_config: Optional[dict] = None

class FinetunedModelCreate(FinetunedModelBase):
    """Model for creating new finetuned models (POST requests)"""
    pass

class FinetunedModelRead(FinetunedModelBase):
    """Model for reading finetuned models (GET responses)"""
    id: UUID
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class FinetunedModelUpdate(BaseModel):
    """Model for updating existing finetuned models (PUT/PATCH requests)"""
    name: Optional[str] = None
    source_type: Optional[SourceType] = None
    source_details: Optional[dict] = None
    data_config: Optional[dict] = None

# If you want to handle bulk creation (list of models)
class FinetunedModelBulkCreate(BaseModel):
    models: list[FinetunedModelCreate]

