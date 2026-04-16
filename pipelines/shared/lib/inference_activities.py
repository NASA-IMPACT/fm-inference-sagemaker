from __future__ import annotations
import time
from typing import Any, Dict
import httpx
import logging
from temporalio import activity
from pydantic import BaseModel
from typing import Optional
from lib.db import get_db, Inference, InferenceStatus  # type: ignore
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class InvocationData(BaseModel):
    filename: str
    scale: Optional[bool] = False
    model_id: str
    bounding_box: list[float]
    date: Optional[str] = None
    qa_flags: Optional[list[str]] = (["cloud", "shadow", "adjacent_cloud"],)
    timeseries: Optional[bool] = False


_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=600.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http_client


async def _run_model_via_http(
    invocation_model: InvocationData, inference_id: str | None = None
) -> Dict[str, Any]:
    model_id = invocation_model.model_id
    url = "http://localhost:8001/api/v1/invocations"

    payload = {
        "filename": invocation_model.filename,
        "scale": invocation_model.scale,
        "model_id": model_id,
        "bounding_box": invocation_model.bounding_box,
        "date": invocation_model.date,
        "qa_flags": invocation_model.qa_flags,
        "timeseries": bool(invocation_model.timeseries),
    }

    logger.info("Calling model API at %s for inference_id=%s", url, inference_id)
    client = await get_http_client()
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def _merge_results_into_inference(
    inference: Inference, invocation_model: InvocationData, result: Dict[str, Any]
) -> None:
    existing_results = inference.results or {}
    if isinstance(result, dict):
        for model_id, model_result in result.items():
            date_key = invocation_model.date or "unknown"
            model_bucket = existing_results.get(model_id, {})
            model_bucket[date_key] = {
                "qa_tif": model_result.get("qa_link"),
                "s3_link": model_result.get("s3_link"),
                "stats": model_result.get("stats"),
                "postprocess_links": model_result.get("postprocess_links", {}),
            }
            existing_results[model_id] = model_bucket
    inference.results = existing_results


@activity.defn(name="run_inference")
async def run_inference(params: Dict[str, Any]) -> Dict[str, Any]:
    inference_id = params["inference_id"]
    invocation = params["invocation"]
    activity.logger.info(
        "run_inference started for inference_id=%s, model_id=%s",
        inference_id,
        invocation.get("model_id"),
    )

    invocation_model = InvocationData(**invocation)

    with get_db() as db:
        inference: Inference | None = (
            db.query(Inference).filter(Inference.id == inference_id).first()
        )
        if inference is None:
            logger.warning(
                "Inference %s not found in DB; running model but not updating DB",
                inference_id,
            )
            return await _run_model_via_http(invocation_model)

        inference.status = InferenceStatus.inference
        db.add(inference)
        db.commit()

        try:
            t0 = time.time()
            result = await _run_model_via_http(invocation_model, inference_id)
            activity.logger.info(
                "Model inference completed for %s in %.2f seconds",
                inference_id,
                time.time() - t0,
            )
        except Exception:
            inference.status = InferenceStatus.failed
            inference.error_stage = "inference"
            inference.error_message = "Model inference failed"
            db.add(inference)
            db.commit()
            activity.logger.error(
                "run_inference failed for inference_id=%s", inference_id
            )
            raise

        await _merge_results_into_inference(inference, invocation_model, result)
        inference.status = InferenceStatus.complete
        db.add(inference)
        db.commit()

        activity.logger.info(
            f"WORKER INDEX: {os.environ.get('POD_INDEX', 0)}\n"
            f"Inference {inference_id}\n"
            f"updated in DB with status={inference.status.value}"
        )

        return result
