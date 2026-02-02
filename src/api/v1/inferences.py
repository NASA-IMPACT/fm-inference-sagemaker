import time
import requests

from typing import Dict, List, Callable, Any
from fastapi import APIRouter, Depends, status, HTTPException, Response
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from ...db.database import get_db
from ...db.models import BaseFM, FinetunedModel, Inference, PreloadedEvent
from ...lib.downloader import Downloader
from ...lib.pagination import PaginationHelper
from ...models.inference import InferenceRead, InferenceUpdate
from ...models.preloaded_event import PreloadedEventRead


def create_inference_router(auth_dependency: Callable) -> APIRouter:
    router = APIRouter(prefix="/v1/inferences", tags=["inferences"])

    @router.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

    @router.get("", response_model=List[InferenceRead], status_code=status.HTTP_200_OK)
    def get_inferences(
        response: Response,
        base_fm: BaseFM = None,
        skip: int = 0,
        limit: int = 10,
        claims: Dict[str, Any] = Depends(auth_dependency),
        db: Session = Depends(get_db)
    ):
        """Get paginated inferences filtered by the logged-in user."""
        try:
            user_email = claims.get("email")
            if not user_email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User email not found"
                )

            # Validate pagination parameters
            skip, limit = PaginationHelper.get_pagination_params(skip, limit)

            # Query inferences filtered by user email, ordered by creation date
            query = db.query(Inference).filter(
                Inference.user_email == user_email
            )

            if base_fm:
                query = query.filter(
                    Inference.finetuned_models.any(base_fm=base_fm)
                )

            # Apply pagination and set response headers
            inferences = PaginationHelper.paginate(
                query=query.order_by(Inference.created_at.desc()),
                response=response,
                base_url="/v1/inferences",
                skip=skip,
                limit=limit
            )

            return inferences
        except HTTPException:
            raise
        except Exception as e:
            print(e)
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
        t0 = time.time()
        user_groups = claims.get("groups") or claims.get("cognito:groups", [])
        user_email = claims.get("email")
        inference_name = inference.name if inference.name else datetime.now(timezone.utc).strftime("inference_%Y%m%d_%H%M%S")
        finetuned_models = db.query(FinetunedModel).filter(FinetunedModel.id.in_(inference.finetuned_model_ids)).all()
        if not finetuned_models or len(finetuned_models) != len(inference.finetuned_model_ids):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more finetuned models not found")

        if finetuned_models[0].name == 'Surya':
            finetuned_model = finetuned_models[0]
            model_id = str(finetuned_model.source_details.get('model_id'))
            port = finetuned_model.source_details['port']
            url = f"http://{model_id.replace('_', '-')}-service:{port}/api/v1/invocations"
            t = time.time()
            response = requests.post(url, json={
                'num_frames': inference.query['num_frames'],
                'cadence_in_minutes': inference.query['cadence_in_minutes'],
                'selected_datetime': inference.query['selected_datetime']
            })
            inference.results[model_id] = response.json()
        else:
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
                    t = time.time()
                    response = requests.post(url, json={
                        'filename': merged_file,
                        'scale': finetuned_model.data_config.get('scaled', False),
                        'model_id': model_id,
                        'qa_flags': finetuned_model.data_config.get('qa_flags', ['cloud', 'shadow', 'adjacent_cloud']),
                        'bounding_box': inference.query['bounding_box'],
                        'date': date,
                        'timeseries': timeseries
                    })
                    print(f"Inference service Took to respond {time.time() - t:.2f} seconds")
                    results[model_id] = results.get(model_id, {})
                    results[model_id][date] = results[model_id].get(date, {})
                    results[model_id][date] = response.json()[model_id]

                    infered_results = results[model_id][date]
                    inference.results = inference.results or {}
                    inference.results[model_id] = inference.results.get(model_id, {})
                    result_entry = {
                        "s3_link": infered_results['s3_link'],
                        "stats": infered_results['stats'],
                    }
                    result_entry['qa_links'] = infered_results.get("qa_links", {})
                    result_entry["postprocess_links"] = infered_results.get("postprocess_links", {})
                    inference.results[model_id][date] = result_entry

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
        print(f"End to End took {time.time() - t0:.2f} seconds")
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
