import time
import requests

from typing import Dict, List
from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta

from ...db.database import get_db
from ...db.models import FinetunedModel, Inference, PreloadedEvent
from ...lib.downloader import Downloader
from ...lib.utils import get_api_key
from ...models.finetuned_model import FinetunedModelRead
from ...models.inference import InferenceRead, InferenceUpdate
from ...models.preloaded_event import PreloadedEventRead

router = APIRouter(prefix="/v1/inferences", tags=["inferences"])

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/", response_model=List[InferenceRead], status_code=status.HTTP_200_OK)
def get_models(db: Session = Depends(get_db)):
    """Get all inferences."""
    try:
        inferences = db.query(Inference).all()
        return inferences
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/{inference_id}", response_model=InferenceRead, status_code=status.HTTP_200_OK)
def get_inference(inference_id: str, db: Session = Depends(get_db)):
    """Get a specific inference by ID."""
    try:
        inference = db.query(Inference).filter(Inference.id == inference_id).first()
        if not inference:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inference not found")
        return inference
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.get("/{inference_id}/preloaded_events", response_model=List[PreloadedEventRead], status_code=status.HTTP_200_OK)
def get_inference_preloaded_events(inference_id: str, db: Session = Depends(get_db)):
    """Get preloaded events associated with a specific model."""
    try:
        events = db.query(PreloadedEvent).filter(
            PreloadedEvent.inference_id == inference_id
        ).all()
        return events
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=List[InferenceRead])
def create_model(inference: InferenceUpdate, db: Session = Depends(get_db)):
    """Create a new finetuned model."""
    try:
    # TODO: handle db session properly
    # handle large requests properly with background tasks
    # send back a job id and let the client poll for status/results
        inference_name = inference.name if inference.name else time.strftime("inference_%Y%m%d_%H%M%S")
        finetuned_models = db.query(FinetunedModel).filter(FinetunedModel.id.in_(inference.finetuned_model_ids)).all()
        if not finetuned_models or len(finetuned_models) != len(inference.finetuned_model_ids):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more finetuned models not found")
        inference.finetuned_models = finetuned_models

        # better to upload merged_file to s3 and pass the s3 path to the inference pipeline
        # for now, we will just pass the local file path
        # call specific model inference pipeline with the file name/path here.
        # create a dict with model names and their inference results
        results = {}
        for finetuned_model in finetuned_models:
            # print(f"Running inference for model: {model.name} on data: {merged_file}")
            model_id = str(finetuned_model.source_details.get('model_id'))
            downloader = Downloader(
                inference.query['date'],
                inference.query['bounding_box'],
                model.data_config['sources']
            )
            merged_file = downloader.find_and_prepare_data()
            # download extra data if needed here
            # also calculate any indices if needed here
            # pass these extra files to the inference pipeline as needed
            url = f"http://{model_id}-service:8080/api/v1/invocations"
            response = requests.post(url, json={
                'filename': merged_file,
                'scale': finetuend_model.data_config.get('scaled', False),
                'model_id': model_id,
                'qa_flags': finetuned_model.data_config.get('qa_flags', ['cloud', 'shadow', 'adjacent_cloud']),
                'bounding_box': inference.query['bounding_box'],
                'date': inference.query['date']
            })
            results[model_id] = response.json()[model_id]

            floods = results[model_id]
            inference.results[model_id] = {
                "geojson": floods['geojson'],
                "s3_link": floods['s3_link']
            }

        db.add(inference)
        db.commit()
        db.refresh(inference)

        return inference
    except Exception as e:
        return {"error": str(e)}

@router.delete("/{inference_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inference(inference_id: str, db: Session = Depends(get_db)):
    """Delete a finetuned model by ID."""
    # TODO: soft delete and handle related objects
    try:
        inference = db.query(Inference).filter(Inference.id == inference_id).first()
        if not inference:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inference not found")
        db.delete(inference)
        db.commit()
        return {"message": "Inference deleted successfully"}
    except Exception as e:
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
