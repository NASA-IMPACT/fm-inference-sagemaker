from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ...db.database import get_db
from ...db.models import BaseFM, FinetunedModel, Inference, PreloadedEvent
from ...models.finetuned_model import FinetunedModelRead
from ...models.preloaded_event import PreloadedEventRead, PreloadedEventUpdate

router = APIRouter(prefix="/v1/preloaded_events", tags=["preloaded_events"])


@router.get("", response_model=list[PreloadedEventRead], status_code=status.HTTP_200_OK)
def get_preloaded_events(base_fm: BaseFM = None, db: Session = Depends(get_db)):
    """Get all preloaded events."""
    try:
        # get all preloaded events
        preloaded_events = db.query(PreloadedEvent)

        if base_fm:
            preloaded_events = (
                preloaded_events.join(PreloadedEvent.inference)
                .join(Inference.finetuned_models)
                .filter(Inference.finetuned_models.any(base_fm=base_fm))
            )
        preloaded_events = preloaded_events.all()
        return preloaded_events
    except Exception as e:
        return {"error": str(e)}


@router.get(
    "/{preloaded_events_id}",
    response_model=PreloadedEventRead,
    status_code=status.HTTP_200_OK,
)
def get_preloaded_event(preloaded_events_id: str, db: Session = Depends(get_db)):
    """Get a specific Preloaded Event by ID."""
    try:
        preloaded_event = (
            db.query(PreloadedEvent)
            .filter(PreloadedEvent.id == preloaded_events_id)
            .first()
        )
        if not preloaded_event:
            return {"error": "Preloaded Event not found"}
        return preloaded_event
    except Exception as e:
        return {"error": str(e)}


@router.get(
    "/{preloaded_event_id}/models",
    response_model=list[FinetunedModelRead],
    status_code=status.HTTP_200_OK,
)
def get_models_preloaded_events(preloaded_event_id: str, db: Session = Depends(get_db)):
    """Get preloaded events associated with a specific model."""
    try:
        models = (
            db.query(FinetunedModel)
            .join(FinetunedModel.inferences)
            .join(Inference.preloaded_events)
            .filter(PreloadedEvent.id == preloaded_event_id)
            .all()
        )
        return models
    except Exception as e:
        return {"error": str(e)}


@router.post("", response_model=PreloadedEventRead, status_code=status.HTTP_201_CREATED)
def create_preloaded_event(
    preloaded_event: PreloadedEventUpdate, db: Session = Depends(get_db)
):
    """Create a new preloaded event."""
    try:
        preloaded_event = PreloadedEvent(
            event_name=preloaded_event.event_name,
            event_details=preloaded_event.event_details,
            inference_id=preloaded_event.inference_id,
        )
        db.add(preloaded_event)
        db.commit()
        db.refresh(preloaded_event)
        return preloaded_event
    except Exception as e:
        return {"error": str(e)}


@router.delete("/{preloaded_event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preloaded_event(preloaded_event_id: str, db: Session = Depends(get_db)):
    """Delete a preloaded event by ID."""
    # TODO: soft delete and handle related objects
    try:
        preloaded_event = (
            db.query(PreloadedEvent)
            .filter(PreloadedEvent.id == preloaded_event_id)
            .first()
        )
        if not preloaded_event:
            return {"error": "Preloaded Event not found"}
        db.delete(preloaded_event)
        db.commit()
        return {"message": "Preloaded Event deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
