import time
from typing import Dict, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta

from ...db.database import get_db
from ...db.models import FinetunedModel, Inference, PreloadedEvent
from ...lib.utils import get_api_key
from ...models.finetuned_model import FinetunedModelRead
from ...models.inference import InferenceRead
from ...models.preloaded_event import PreloadedEventRead, PreloadedEventUpdate

router = APIRouter(prefix="/v1/preloaded_events", tags=["preloaded_events"], dependencies=[Depends(get_api_key)])

@router.get("/", response_model=List[PreloadedEventRead], status_code=status.HTTP_200_OK)
def get_preloaded_events(db: Session = Depends(get_db)):
    """Get all preloaded events."""
    try:
        # get all preloaded events
        preloaded_events = db.query(
            PreloadedEvent
        ).all()

        return preloaded_events
    except Exception as e:
        return {"error": str(e)}

@router.get("/{preloaded_events_id}", response_model=PreloadedEventRead, status_code=status.HTTP_200_OK)
def get_preloaded_events(preloaded_events_id: str, db: Session = Depends(get_db)):
    """Get a specific Preloaded Event by ID."""
    try:
        preloaded_event = db.query(PreloadedEvent).filter(PreloadedEvent.id == preloaded_events_id).first()
        if not preloaded_event:
            return {"error": "Preloaded Event not found"}
        return preloaded_event
    except Exception as e:
        return {"error": str(e)}

@router.get("/{preloaded_event_id}/models", response_model=List[FinetunedModelRead], status_code=status.HTTP_200_OK)
def get_models_preloaded_events(preloaded_event_id: str, db: Session = Depends(get_db)):
    """Get preloaded events associated with a specific model."""
    try:
        models = db.query(FinetunedModel).join(
            FinetunedModel.inferences
        ).join(
            Inference.preloaded_events
        ).filter(
            PreloadedEvent.id == preloaded_event_id
        ).all()
        return models
    except Exception as e:
        return {"error": str(e)}

@router.post("/", response_model=PreloadedEventRead, status_code=status.HTTP_201_CREATED)
def create_model(preloaded_event: PreloadedEventUpdate, db: Session = Depends(get_db)):
    """Create a new finetuned model."""
    try:
        preloaded_event = PreloadedEvent(
            name=preloaded_event.name,
            details=preloaded_event.details,
            inference_id=preloaded_event.inference_id
        )
        db.add(preloaded_event)
        db.commit()
        db.refresh(preloaded_event)
        return db_model
    except Exception as e:
        return {"error": str(e)}

@router.delete("/{preloaded_event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preloaded_event(preloaded_event_id: str, db: Session = Depends(get_db)):
    """Delete a finetuned model by ID."""
    # TODO: soft delete and handle related objects
    try:
        preloaded_event = db.query(PreloadedEvent).filter(PreloadedEvent.id == preloaded_event_id).first()
        if not preloaded_event:
            return {"error": "Preloaded Event not found"}
        db.delete(preloaded_event)
        db.commit()
        return {"message": "Preloaded Event deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
