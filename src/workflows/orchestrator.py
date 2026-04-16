from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError


@workflow.defn(name="InferenceOrchestrator")
class InferenceOrchestrator:
    """
    Orchestrator workflow that:
    1. Downloads and prepares data via activity (runs on predictor pod)
    2. Spawns child InferenceWorkflow for each model/date combination
       (runs on pipeline pods)
    """

    @workflow.run
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        inference_id = params["inference_id"]
        model_configs = params["model_configs"]
        # FIX 1: Safely handle missing 'query' — this was likely causing the
        # KeyError that crashed every workflow task attempt.
        query = params.get("query")
        if query is None:
            workflow.logger.error(
                "Missing 'query' in params for inference_id=%s. Received keys: %s",
                inference_id,
                list(params.keys()),
            )
            return {
                "inference_id": inference_id,
                "error": "Missing required param: 'query'",
            }

        workflow.logger.info(
            "InferenceOrchestrator started for inference_id=%s, %d models",
            inference_id,
            len(model_configs),
        )

        # Step 1: Download and prepare data (runs on predictor pod)
        prepared_data = await workflow.execute_activity(
            "download_and_prepare",
            {
                "inference_id": inference_id,
                "model_configs": model_configs,
                "query": query,
            },
            start_to_close_timeout=timedelta(days=2),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=5),
                maximum_interval=timedelta(seconds=60),
                backoff_coefficient=2.0,
            ),
        )

        # Step 2: Start child workflows for each model/date (runs on pipeline pods)
        workflow_handles = []
        for item in prepared_data["invocations"]:
            model_id = item["model_id"]
            date = item["date"]
            workflow_id = f"inference-{inference_id}-{model_id}-{date}"
            task_queue = f"{model_id}-tq"

            workflow_input = {
                "inference_id": inference_id,
                "invocation": item,
            }

            workflow.logger.info(
                "Starting child InferenceWorkflow %s on queue %s",
                workflow_id,
                task_queue,
            )

            handle = await workflow.start_child_workflow(
                "InferenceWorkflow",
                workflow_input,
                id=workflow_id,
                task_queue=task_queue,
                execution_timeout=timedelta(days=2),
            )
            workflow_handles.append(handle)

        # Step 3: Wait for all child workflows concurrently
        async def _await_child(handle) -> Dict[str, Any]:
            try:
                return await handle
            except Exception as e:
                workflow.logger.error("Child workflow failed: %s", e)
                return {"error": str(e)}

        results = list(
            await asyncio.gather(*[_await_child(h) for h in workflow_handles])
        )

        # Fail the parent if ANY child failed
        errors = [r for r in results if "error" in r]
        if errors:
            raise ApplicationError(
                f"{len(errors)} child workflow(s) failed: "
                + "; ".join(r["error"] for r in errors),
            )

        workflow.logger.info(
            "InferenceOrchestrator completed for inference_id=%s, %d results",
            inference_id,
            len(results),
        )

        return {"inference_id": inference_id, "results": results}
