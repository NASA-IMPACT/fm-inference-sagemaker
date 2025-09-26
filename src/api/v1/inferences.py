import time
import requests

from typing import Dict, List, Callable, Any
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from ...db.database import get_db
from ...db.models import FinetunedModel, Inference, PreloadedEvent
from ...lib.downloader import Downloader
from ...models.inference import InferenceRead, InferenceUpdate
from ...models.preloaded_event import PreloadedEventRead


def create_inference_router(auth_dependency: Callable) -> APIRouter:
    router = APIRouter(prefix="/v1/inferences", tags=["inferences"])

    @router.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow()}

    @router.get("", response_model=List[InferenceRead], status_code=status.HTTP_200_OK)
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

    @router.post("", status_code=status.HTTP_201_CREATED, response_model=InferenceRead)
    def create_inference(inference: InferenceUpdate,
                        claims: Dict[str, Any] = Depends(auth_dependency),
                        db: Session = Depends(get_db)
                        ):
        """Create a new finetuned model."""
        # try:
        # TODO: handle db session properly
        # handle large requests properly with background tasks
        # send back a job id and let the client poll for status/results
        user_groups = claims.get("groups") or claims.get("cognito:groups", [])
        user_email = claims.get("email")
        inference_name = inference.name if inference.name else time.strftime("inference_%Y%m%d_%H%M%S")
        finetuned_models = db.query(FinetunedModel).filter(FinetunedModel.id.in_(inference.finetuned_model_ids)).all()
        if not finetuned_models or len(finetuned_models) != len(inference.finetuned_model_ids):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more finetuned models not found")

        # better to upload merged_file to s3 and pass the s3 path to the inference pipeline
        # for now, we will just pass the local file path
        # call specific model inference pipeline with the file name/path here.
        # Check key validity before running anything
        for finetuned_model in finetuned_models:
            model_id = str(finetuned_model.source_details.get('model_id'))
            if model_id not in user_groups:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Key provided is not authorized to run inference on {model_id}")
        results = {}
        for finetuned_model in finetuned_models:
            # print(f"Running inference for model: {model.name} on data: {merged_file}")
            model_id = str(finetuned_model.source_details.get('model_id'))
            timeseries = finetuned_model.source_details.get('timeseries', False)
            downloader = Downloader(
                inference.query['dates'],
                inference.query['bounding_box'],
                finetuned_model.data_config['sources'],
                timeseries=timeseries
            )
            prepared_data = downloader.find_and_prepare_data()
            for date, merged_file in prepared_data.items():
                if '.tif' not in merged_file:
                    results[model_id] = results.get(model_id, {})
                    results[model_id][date] = results[model_id].get(date, {})
                    results[model_id][date] = {}
                    continue
                port = finetuned_model.source_details['port']
                # download extra data if needed here
                # also calculate any indices if needed here
                # pass these extra files to the inference pipeline as needed
                url = f"http://{model_id.replace('_', '-')}-service:{port}/api/v1/invocations"
                response = requests.post(url, json={
                    'filename': merged_file,
                    'scale': finetuned_model.data_config.get('scaled', False),
                    'model_id': model_id,
                    'qa_flags': finetuned_model.data_config.get('qa_flags', ['cloud', 'shadow', 'adjacent_cloud']),
                    'bounding_box': inference.query['bounding_box'],
                    'date': date,
                    'timeseries': timeseries
                })
                results[model_id] = results.get(model_id, {})
                results[model_id][date] = results[model_id].get(date, {})
                results[model_id][date] = response.json()[model_id]

                infered_results = results[model_id][date]
                inference.results = inference.results if inference.results else {}
                inference.results[model_id] = inference.results.get(model_id, {})
                inference.results[model_id][date] = {
                    # "qa_geojson": infered_results['qa_geojson'],
                    "s3_link": infered_results['s3_link'],
                    "stats": infered_results['stats']
                }

        # Convert Pydantic model to ORM model before adding to DB
        inference_details = inference.dict()
        inference_details['name'] = inference_name
        inference_details['finetuned_models'] = finetuned_models
        inference_details['user_email'] = user_email
        del(inference_details['finetuned_model_ids'])
        inference_orm = Inference(**inference_details)
        db.add(inference_orm)
        db.commit()
        db.refresh(inference_orm)

        return inference_orm

    @router.delete("/{inference_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_inference(inference_id: str,
                        claims: Dict[str, Any] = Depends(auth_dependency),
                        db: Session = Depends(get_db)):
        """Delete a finetuned model by ID."""
        # TODO: soft delete and handle related objects
        try:
            inference = db.query(Inference).filter(Inference.id == inference_id).first()
            if not inference:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inference not found")
            user_email = claims.get("email")
            if user_email != inference.user_email:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Action not allowed")
            db.delete(inference)
            db.commit()
            return {"message": "Inference deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return router
