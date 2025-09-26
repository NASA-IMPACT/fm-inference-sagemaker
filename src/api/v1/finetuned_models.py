from typing import Dict, List, Callable, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...db.database import get_db
from ...db.models import FinetunedModel, Inference, PreloadedEvent
from ...lib.utils import get_api_key
from ...models.finetuned_model import FinetunedModelRead, FinetunedModelUpdate
from ...models.inference import InferenceRead
from ...models.preloaded_event import PreloadedEventRead

def create_models_router(auth_dependency: Callable) -> APIRouter:
    router = APIRouter(prefix="/v1/models", tags=["models"])

    @router.get("", response_model=List[FinetunedModelRead], status_code=status.HTTP_200_OK)
    def get_models(_: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Get all finetuned models."""
        try:
            # Total jobs by status
            models = db.query(
                FinetunedModel
            ).all()
            print('----------------')
            print(len(models))
            return models
        except Exception as e:
            return {"error": str(e)}


    @router.post("", status_code=status.HTTP_201_CREATED)
    def create_model(model: FinetunedModelUpdate,claims: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Create a new finetuned model."""
        user_groups = claims.get("groups") or claims.get("cognito:groups", [])
        if model.name not in user_groups + ["admin"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"You do not have permission to create {model.name} model")
        try:
            db_model = FinetunedModel(
                name=model.name,
                source_type=model.source_type,
                source_details=model.source_details,
                data_config=model.data_config
            )
            db.add(db_model)
            db.commit()
            db.refresh(db_model)
            return {"message": f"Model {model.name} was created successfully!"}
        except Exception as e:
            return {"error": str(e)}





    @router.get("/{model_id}", response_model=FinetunedModelRead, status_code=status.HTTP_200_OK)
    def get_model(model_id: str, _: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Get a specific model by ID."""
        try:
            model = db.query(FinetunedModel).filter(FinetunedModel.id == model_id).first()
            if not model:
                return {"error": "Model not found"}
            return model
        except Exception as e:
            return {"error": str(e)}

    @router.get("/{model_id}/inferences", response_model=List[InferenceRead], status_code=status.HTTP_200_OK)
    def get_model_inferences(model_id: str, _: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Get inferences associated with a specific model."""
        try:
            inferences = db.query(Inference).join(
                Inference.finetuned_models
            ).filter(
                FinetunedModel.id == model_id
            ).all()
            return inferences
        except Exception as e:
            return {"error": str(e)}

    @router.get("/{model_id}/preloaded_events", response_model=List[PreloadedEventRead], status_code=status.HTTP_200_OK)
    def get_model_preloaded_events(model_id: str, _: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Get preloaded events associated with a specific model."""
        try:
            events = db.query(PreloadedEvent).join(
                PreloadedEvent.inference
            ).join(
                Inference.finetuned_models
            ).filter(
                FinetunedModel.id == model_id
            ).all()
            return events
        except Exception as e:
            return {"error": str(e)}



    @router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_model(model_id: str,claims: Dict[str, Any] = Depends(auth_dependency), db: Session = Depends(get_db)):
        """Delete a finetuned model by ID."""
        try:
            model = db.query(FinetunedModel).filter(FinetunedModel.id == model_id).first()
            user_groups = claims.get("groups") or claims.get("cognito:groups", [])
            if not model:
                return {"error": "Model not found"}
            # Only admins can delete models
            if user_groups != ["admin"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"You do not have permission to delete {model.name} model")
            db.delete(model)
            db.commit()
            return {"message": "Model deleted successfully"}
        except Exception as e:
            return {"error": str(e)}
    return router
