import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import rasterio
import torch

from lib.infer import Infer
from lib.dem_downloader import DEMDownloader
from terratorch.tasks import SemanticSegmentationTask


logger = logging.getLogger(__name__)  # flood inference module


def _flood_postprocess_worker(
    bbox,
    date,
    prediction,
    image,
    source_width,
    source_height,
    use_smart_aerosol=True,
):
    """
    Runs DEM-based postprocessing in a *separate process*.

    This is basically the old FloodInfer.postprocess logic, but without `self`,
    so it can be executed safely in a subprocess.
    """
    try:
        dem_downloader = DEMDownloader(bbox, date)

        # 1. Download / reuse DEM tiles
        dem_files = dem_downloader.download_dem_tiles()

        # Early return if no DEM tiles available
        if not dem_files:
            logger.warning(
                "No DEM tiles available for bbox %s, returning original prediction",
                bbox,
            )
            return prediction

        # 2. Merge & clip DEM to match requested width/height
        dem_file = dem_downloader.merge_and_clip_dems(
            dem_files,
            width=source_width,
            height=source_height,
        )

        # Check if DEM merge/clip failed
        if not dem_file:
            logger.warning(
                "Failed to merge/clip DEM for bbox %s, returning original prediction",
                bbox,
            )
            return prediction

        # 3. Apply all DEM-driven postprocessing (terrain shadows, aerosol, vegetation)
        final_prediction = dem_downloader.apply_all_postprocessing(
            flood_detection_file=prediction,
            hls_file=image,
            dem_file=dem_file,
            use_smart_aerosol=use_smart_aerosol,
        )

        # Check if postprocessing succeeded
        if not final_prediction:
            logger.warning(
                "Postprocessing failed for %s, returning original prediction",
                prediction,
            )
            return prediction

        return final_prediction

    except Exception as exc:
        logger.exception("Error in flood DEM postprocess worker: %s", exc)
        # On any error, just fall back to the original prediction file
        return prediction


class FloodInfer(Infer):
    def __init__(self, config, checkpoint, max_queue_size=4, num_streams=2):
        super().__init__(config, checkpoint, max_queue_size, num_streams)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        if self.model:
            return

        # Indices for prithvi_eo_v2_600
        indices = [7, 15, 23, 31]

        model_args = {
            # Backbone
            "backbone": "prithvi_eo_v2_600",
            "backbone_pretrained": True,
            "backbone_num_frames": 1,
            "backbone_img_size": 512,
            "backbone_bands": [
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
            ],
            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": indices,
                },
                {
                    "name": "ReshapeTokensToImage",
                },
                {
                    "name": "LearnedInterpolateToPyramidal",
                },
            ],
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            # Head
            "head_dropout": 0.1,
            "num_classes": 2,
        }

        self.model = SemanticSegmentationTask.load_from_checkpoint(
            self.checkpoint_filename,
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
        )

        self.model = self.model.eval()
        self.model.to(self.device)

    def postprocess(
        self,
        bbox,
        date,
        prediction,
        image,
        source_width=None,
        source_height=None,
    ):
        """
        Run DEM-based postprocessing in a *disposable subprocess*.

        Signature and external behavior stay the same as before:
        it takes bbox, date, prediction path, image path (+ optional width/height)
        and returns the final corrected prediction path.
        """
        # Determine width/height in the main process if not provided
        if source_width is not None and source_height is not None:
            width, height = source_width, source_height
        else:
            with rasterio.open(image) as src:
                width, height = src.width, src.height

        try:
            # Use 'spawn' context for safety in long-lived services
            ctx = multiprocessing.get_context("spawn")

            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
                future = executor.submit(
                    _flood_postprocess_worker,
                    bbox,
                    date,
                    prediction,
                    image,
                    width,
                    height,
                    True,  # use_smart_aerosol
                )
                result = future.result()  

        except Exception as exc:
            self.logger.exception(
                "DEM postprocess failed in subprocess for %s: %s",
                prediction,
                exc,
            )
            return prediction

        # Fallback if worker returned a falsy value (e.g., None or "")
        if not result:
            self.logger.warning(
                "DEM postprocess returned falsy result for %s; using original prediction",
                prediction,
            )
            return prediction

        if isinstance(result, dict):
            final_prediction = result.get("final") or prediction
            # Attach artifacts to this instance for upstream consumers
            self.postprocess_artifacts = result.get("artifacts", {})
            return final_prediction

        self.postprocess_artifacts = {}
        return result
