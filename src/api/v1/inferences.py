import time
import requests

from typing import Dict, List
from fastapi import APIRouter, Depends, Query, status
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

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_model(inference: InferenceUpdate): #, db: Session = Depends(get_db)):
    """Create a new finetuned model."""
    try:
        # inference_name = inference.name if inference.name else
        inference.name = time.strftime("inference_%Y%m%d_%H%M%S")
        # inference = Inference(
        #     name=time.strftime("inference_%Y%m%d_%H%M%S"),
        #     query=inference.query
        # )
        # finetuned_models = db.query(FinetunedModel).filter(FinetunedModel.id.in_(inference.finetuned_model_ids)).all()
        # if not finetuned_models or len(finetuned_models) != len(inference.finetuned_model_ids):
        #     return {"error": "One or more finetuned models not found"}
        # inference.finetuned_models = finetuned_models

        # better to upload merged_file to s3 and pass the s3 path to the inference pipeline
        # for now, we will just pass the local file path
        # call specific model inference pipeline with the file name/path here.
        # create a dict with model names and their inference results
        results = {}
        for model_id in ['floods']:
            # print(f"Running inference for model: {model.name} on data: {merged_file}")
            downloader = Downloader(inference.query['date'], inference.query['bounding_box'], layers=['HLSS30', 'HLSL30'])#model.data_config['sources'])
            print('Downloading files')
            merged_files = downloader.find_and_prepare_data()
            print(f'Downloaded and merged file at: {merged_file}')
            url = f"http://{model_id}-service:8080/api/v1/invocations"
            print(f'Calling model endpoint at: {url}')
            # Todo why it is a list?
            merged_file = merged_files[0]
            response = requests.post(url, json={'filename': merged_file, 'scaled': True}) #model.data_config['scaled']})
            print(f'Model response: {response.status_code}, {response.text}')
            results['floods'] = response.json()
            floods = results['floods']
            inference.result_geojson = [floods['geojson']]
            inference.result_s3_path = floods['s3_path']
            # build model pipeline url here
            # call the model endpoint with the merged_file
            # update results dict with the inference results
        # store the s3 results and geojson in inference object


        # db.add(inference)
        # db.commit()
        # db.refresh(inference)

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
            return {"error": "Inference not found"}
        db.delete(inference)
        db.commit()
        return {"message": "Inference deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
