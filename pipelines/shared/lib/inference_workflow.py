from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict
import logging

from temporalio import workflow
from temporalio.common import RetryPolicy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


@workflow.defn(name="InferenceWorkflow")
class InferenceWorkflow:
    @workflow.run
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        inference_id = params["inference_id"]
        invocation = params["invocation"]

        logger.info(
            "InferenceWorkflow started for inference_id=%s, model_id=%s",
            inference_id,
            invocation.get("model_id"),
        )

        # Call the activity by name so the workflow sandbox does not import
        # heavy dependencies like boto3 that live in the activity module.
        result = await workflow.execute_activity(
            "run_inference",
            {"inference_id": inference_id, "invocation": invocation},
            start_to_close_timeout=timedelta(days=2),
            retry_policy=RetryPolicy(
                maximum_attempts=2,
                initial_interval=timedelta(minutes=1),
                maximum_interval=timedelta(minutes=3),
                backoff_coefficient=2.0,
            ),
        )

        logger.info(
            "InferenceWorkflow completed for inference_id=%s, model_id=%s",
            inference_id,
            invocation.get("model_id"),
        )
        return result
