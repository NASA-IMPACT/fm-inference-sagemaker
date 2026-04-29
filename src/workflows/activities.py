from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

from sqlalchemy.orm.attributes import flag_modified
from temporalio import activity

from ..db.database import get_db
from ..db.models import Inference, InferenceStatus
from ..lib.inference_utils import prepare_inference_data


@contextmanager
def db_session():
    """Provide a transactional database session scope."""
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


@activity.defn(name="download_and_prepare")
def download_and_prepare(params: Dict[str, Any]) -> Dict[str, Any]:
    inference_id = params["inference_id"]
    query = params["query"]
    model_configs = params["model_configs"]

    activity.logger.info(
        "download_and_prepare started for inference_id=%s, %d models",
        inference_id,
        len(model_configs),
    )

    activity.heartbeat("updating_status")

    with db_session() as db:
        inference = db.query(Inference).filter(Inference.id == inference_id).first()
        if inference:
            inference.status = InferenceStatus.download
            db.add(inference)
            db.commit()

    activity.heartbeat("downloading_data")

    invocations, empty_results = prepare_inference_data(
        query=query,
        model_configs=model_configs,
        inference_id=inference_id,
    )

    activity.heartbeat("updating_download_db_status")

    if empty_results:
        with db_session() as db:
            inference = db.query(Inference).filter(Inference.id == inference_id).first()
            if inference:
                results = inference.results or {}
                for model_id, dates in empty_results.items():
                    results.setdefault(model_id, {})
                    for date, result in dates.items():
                        results[model_id][date] = result
                inference.results = results
                flag_modified(inference, "results")
                db.add(inference)
                db.commit()

    activity.logger.info(
        "download_and_prepare completed for inference_id=%s, %d invocations",
        inference_id,
        len(invocations),
    )

    return {"invocations": invocations}
