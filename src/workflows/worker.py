import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from .orchestrator import InferenceOrchestrator
from .activities import download_and_prepare

logger = logging.getLogger(__name__)

TEMPORAL_SERVER_URL = os.getenv(
    "TEMPORAL_SERVER_URL", "temporal-frontend.temporal:7233"
)
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")
ORCHESTRATOR_TASK_QUEUE = os.getenv("ORCHESTRATOR_TASK_QUEUE", "orchestrator-tq")
ORCHESTRATOR_CONCURRENT_ACTIVITIES = int(
    os.getenv("ORCHESTRATOR_CONCURRENT_ACTIVITIES", "4")
)


def _on_worker_done(task: asyncio.Task) -> None:
    """Callback to log unexpected worker crashes."""
    if task.cancelled():
        logger.info("Worker task was cancelled")
        return
    exc = task.exception()
    if exc:
        logger.error("Worker crashed", exc_info=exc)


async def run_orchestrator_worker() -> None:
    """
    Run the Temporal worker for InferenceOrchestrator workflow
    and download_and_prepare activity.

    This worker runs on the predictor pod (main app) and handles:
    - InferenceOrchestrator workflow execution
    - download_and_prepare activity (data download/preparation)
    """
    logger.info(
        "Starting Orchestrator Worker\n"
        f"  task_queue={ORCHESTRATOR_TASK_QUEUE}\n"
        f"  namespace={TEMPORAL_NAMESPACE}\n"
        f"  temporal_host={TEMPORAL_SERVER_URL}\n"
        f"  concurrent_activities={ORCHESTRATOR_CONCURRENT_ACTIVITIES}"
    )

    client = await Client.connect(
        TEMPORAL_SERVER_URL,
        namespace=TEMPORAL_NAMESPACE,
    )
    logger.info("Connected to Temporal server")

    activity_executor = ThreadPoolExecutor(
        max_workers=ORCHESTRATOR_CONCURRENT_ACTIVITIES
    )

    # Configure sandbox to allow modules that use datetime.now at import time
    sandbox_runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "src.db.models",
            "src.db.database",
            "src.lib.downloader",
            "src.lib.inference_utils",
        )
    )

    worker = Worker(
        client,
        task_queue=ORCHESTRATOR_TASK_QUEUE,
        workflows=[InferenceOrchestrator],
        activities=[download_and_prepare],
        activity_executor=activity_executor,
        max_concurrent_activities=ORCHESTRATOR_CONCURRENT_ACTIVITIES,
        workflow_runner=sandbox_runner,
    )

    logger.info("Orchestrator worker initialized, waiting for workflows...")

    try:
        await worker.run()
    finally:
        activity_executor.shutdown(wait=True)
        logger.info("Activity executor shut down")


async def start_worker_background() -> asyncio.Task:
    """
    Start the orchestrator worker as a background task.
    Returns the task handle for lifecycle management.
    """
    task = asyncio.create_task(run_orchestrator_worker())
    task.add_done_callback(_on_worker_done)
    return task


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        asyncio.run(run_orchestrator_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    except Exception:
        logger.exception("Orchestrator worker crashed")
