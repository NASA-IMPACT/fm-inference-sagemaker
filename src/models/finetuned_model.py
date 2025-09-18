import enum

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime

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

class FinetunedModelRead(FinetunedModelBase):
    pass

class FinetunedModelUpdate(FinetunedModelBase):
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    name: Optional[str] = None
    source_type: Optional[SourceType] = SourceType.s3

