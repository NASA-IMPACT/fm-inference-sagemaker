from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..lib.downloader import Downloader

logger = logging.getLogger(__name__)


def prepare_inference_data(
    query: Dict[str, Any],
    model_configs: List[Dict[str, Any]],
    inference_id: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Download and prepare data for inference.

    Args:
        query: Contains 'dates' and 'bounding_box'
        model_configs: List of dicts with 'model_id', 'timeseries', 'data_config'
        inference_id: Optional inference ID for logging

    Returns:
        List of invocation dicts ready for model inference, each containing:
        - filename, scale, model_id, qa_flags, bounding_box, date, timeseries

        Also returns empty results for dates without .tif data.
    """
    invocations: List[Dict[str, Any]] = []
    empty_results: Dict[str, Dict[str, Dict]] = {}

    for model_config in model_configs:
        model_id = model_config["model_id"]
        timeseries = model_config.get("timeseries", False)
        data_config = model_config["data_config"]

        logger.info(
            "Preparing data for inference_id=%s, model_id=%s, timeseries=%s",
            inference_id,
            model_id,
            timeseries,
        )

        downloader = Downloader(
            query["dates"],
            query["bounding_box"],
            data_config["sources"],
            timeseries=timeseries,
        )
        prepared_data = downloader.find_and_prepare_data()

        logger.info(
            "Downloader produced %d items for inference_id=%s, model_id=%s",
            len(prepared_data),
            inference_id,
            model_id,
        )

        for date, merged_file in prepared_data.items():
            if ".tif" not in merged_file:
                # Track empty results for dates without data
                if model_id not in empty_results:
                    empty_results[model_id] = {}
                empty_results[model_id][date] = {}
                logger.info(
                    "No .tif for inference_id=%s, model_id=%s, date=%s",
                    inference_id,
                    model_id,
                    date,
                )
                continue

            invocation = {
                "filename": merged_file,
                "scale": data_config.get("scaled", False),
                "model_id": model_id,
                "qa_flags": data_config.get(
                    "qa_flags", ["cloud", "shadow", "adjacent_cloud"]
                ),
                "bounding_box": query["bounding_box"],
                "date": date,
                "timeseries": timeseries,
            }
            invocations.append(invocation)

    return invocations, empty_results


def build_model_configs(finetuned_models) -> List[Dict[str, Any]]:
    """
    Build model config dicts from FinetunedModel ORM objects.
    Used to pass model info to activities/workflows without ORM dependencies.
    """
    return [
        {
            "model_id": str(m.source_details.get("model_id")),
            "timeseries": m.source_details.get("timeseries", False),
            "data_config": m.data_config,
            "port": m.source_details.get("port"),
            "name": m.name,
        }
        for m in finetuned_models
    ]
