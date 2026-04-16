import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

# from predictor import ensure_model_loaded
from lib.inference_workflow import InferenceWorkflow
from lib.inference_activities import run_inference

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_inference_semaphore: asyncio.Semaphore | None = None


async def main() -> None:
    temporal_host = os.getenv("TEMPORAL_SERVER_URL", "temporal-frontend.temporal:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    usecase = os.environ.get("USECASE")
    task_queue = f"{usecase}-tq"
    concurrent_tasks = int(os.getenv("TEMPORAL_ACTIVITY_TASKS_COUNT", 1))

    global _inference_semaphore
    _inference_semaphore = asyncio.Semaphore(concurrent_tasks)

    logger.info(
        f"Starting Inference Worker for usecase={usecase})\n"
        f"task_queue={task_queue}\n"
        f"namespace={namespace}\n"
        f"temporal_host={temporal_host}\n"
        f"concurrent_tasks={concurrent_tasks}"
    )

    try:
        client = await Client.connect(
            temporal_host,
            namespace=namespace,
        )
        logger.info("✓ Connected to Temporal server")

        shared_executor = ThreadPoolExecutor(max_workers=concurrent_tasks)

        worker_instance = Worker(
            client,
            task_queue=task_queue,
            workflows=[InferenceWorkflow],
            activities=[run_inference],
            activity_executor=shared_executor,
            max_concurrent_activities=concurrent_tasks,
        )

        logger.info("✓ Worker initialized, waiting for workflows...")

        await worker_instance.run()

    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        logging.exception("Worker terminated with an unexpected exception")
        raise
