import time
from typing import Dict, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta

from ...db.database import get_db
from ...db.models import FinetunedModel, Inference, PreloadedEvent
from ...models.finetuned_model import FinetunedModelRead
from ...models.inference import InferenceRead, InferenceUpdate
from ...models.preloaded_event import PreloadedEventRead

router = APIRouter(prefix="/v1/inferences", tags=["inferences"])

@router.get("/", response_model=List[InferenceRead], status_code=status.HTTP_200_OK)
def get_models(db: Session = Depends(get_db)):
    """Get all inferences."""
    try:
        # get all inferences
        inferences = db.query(
            Inference
        ).all()

        return inferences
    except Exception as e:
        return {"error": str(e)}

@router.get("/{inference_id}", response_model=InferenceRead, status_code=status.HTTP_200_OK)
def get_inference(inference_id: str, db: Session = Depends(get_db)):
    """Get a specific inference by ID."""
    try:
        inference = db.query(Inference).filter(Inference.id == inference_id).first()
        if not inference:
            return {"error": "inference not found"}
        return inference
    except Exception as e:
        return {"error": str(e)}

@router.get("/{inference_id}/preloaded_events", response_model=List[PreloadedEventRead], status_code=status.HTTP_200_OK)
def get_inference_preloaded_events(inference_id: str, db: Session = Depends(get_db)):
    """Get preloaded events associated with a specific model."""
    try:
        events = db.query(PreloadedEvent).filter(
            PreloadedEvent.inference_id == inference_id
        ).all()
        return events
    except Exception as e:
        return {"error": str(e)}

@router.post("/", response_model=InferenceRead, status_code=status.HTTP_201_CREATED)
def create_model(inference: InferenceUpdate, db: Session = Depends(get_db)):
    """Create a new finetuned model."""
    try:
        inference_name = inference.name if inference.name else time.strftime("inference_%Y%m%d_%H%M%S")
        inference = Inference(
            name=inference.name,
            query=inference.query
        )
        db.add(inference)
        db.commit()
        db.refresh(inference)
        # get inference from specific finetuned models and update the inference object
        return db_model
    except Exception as e:
        return {"error": str(e)}

@router.delete("/{inference_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inference(inference_id: str, db: Session = Depends(get_db)):
    """Delete a finetuned model by ID."""
    # TODO: soft delete and handle related objects
    try:
        inference = db.query(Inference).filter(Inference.id == inference_id).first()
        if not inference:
            return {"error": "Inference not found"}
        db.delete(inference)
        db.commit()
        return {"message": "Inference deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
