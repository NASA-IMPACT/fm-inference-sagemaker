import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import rasterio
import torch

from lib.infer import Infer
from lib.dem_downloader import DEMDownloader
from terratorch.tasks import SemanticSegmentationTask

# GPU postprocessing - enabled by default, can be disabled via env var
USE_GPU_POSTPROCESS = os.environ.get("USE_GPU_POSTPROCESS", "true").lower() == "true"

if USE_GPU_POSTPROCESS:
    try:
        from lib.dem_downloader_gpu import DEMDownloaderGPU
    except ImportError:
        USE_GPU_POSTPROCESS = False


logger = logging.getLogger(__name__)


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
        _t0 = time.time()
        dem_downloader = DEMDownloader(bbox, date)

        # 1. Download / reuse DEM tiles
        dem_files = dem_downloader.download_dem_tiles()
        print(f"[flood_postproc] DEM download:      {time.time() - _t0:.3f}s")

        # Early return if no DEM tiles available
        if not dem_files:
            logger.warning(
                "No DEM tiles available for bbox %s, returning original prediction",
                bbox,
            )
            return prediction

        # 2. Merge & clip DEM to match requested width/height
        _t1 = time.time()
        dem_file = dem_downloader.merge_and_clip_dems(
            dem_files,
            width=source_width,
            height=source_height,
        )
        print(f"[flood_postproc] DEM merge/clip:    {time.time() - _t1:.3f}s")

        # Check if DEM merge/clip failed
        if not dem_file:
            logger.warning(
                "Failed to merge/clip DEM for bbox %s, returning original prediction",
                bbox,
            )
            return prediction

        # 3. Apply all DEM-driven postprocessing (terrain shadows, aerosol, vegetation)
        _t2 = time.time()
        final_prediction = dem_downloader.apply_all_postprocessing(
            flood_detection_file=prediction,
            hls_file=image,
            dem_file=dem_file,
            use_smart_aerosol=use_smart_aerosol,
        )
        print(f"[flood_postproc] apply_all_postprocessing: {time.time() - _t2:.3f}s")
        print(f"[flood_postproc] total worker time: {time.time() - _t0:.3f}s")

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
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)
        self.logger = logging.getLogger(__name__)
        # Persistent pool — worker process is started once and reused across calls,
        # avoiding the 1-3s "spawn" overhead on every postprocess invocation.
        self._postprocess_executor = self._make_executor()

    def _make_executor(self):
        ctx = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(max_workers=1, mp_context=ctx)

    def load_model(self):
        if self.model:
            return

        # Indices for prithvi_eo_v2_600
        indices = [7, 15, 23, 31]

        model_args = {
            # Backbone
            "backbone": "prithvi_eo_v2_600",
            "backbone_pretrained": False,
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
        if self.use_cuda:
            # Use "default" mode to avoid CUDA graph conflicts with DataLoader workers
            self.model = torch.compile(self.model, mode="reduce-overhead")

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
        Run DEM-based postprocessing.

        If GPU postprocessing is enabled (USE_GPU_POSTPROCESS=true), runs in
        the main process using GPU acceleration. Otherwise falls back to
        CPU-based subprocess execution.

        Args:
            bbox: Bounding box (west, south, east, north)
            date: Date string
            prediction: Path to flood detection GeoTIFF
            image: Path to HLS mosaic GeoTIFF
            source_width: Image width (avoids file re-read if provided)
            source_height: Image height (avoids file re-read if provided)

        Returns:
            Path to postprocessed prediction GeoTIFF
        """
        if USE_GPU_POSTPROCESS and self.use_cuda:
            return self._postprocess_gpu(
                bbox, date, prediction, image, source_width, source_height
            )
        else:
            return self._postprocess_subprocess(
                bbox, date, prediction, image, source_width, source_height
            )

    def _postprocess_gpu(
        self,
        bbox,
        date,
        prediction,
        image,
        source_width=None,
        source_height=None,
    ):
        """
        GPU-accelerated postprocessing in main process.

        Runs DEM-based postprocessing using the existing GPU context
        for accelerated slope, SLIA, and mask calculations.
        """
        _t0 = time.time()

        # Get dimensions if not provided
        if source_width is None or source_height is None:
            with rasterio.open(image) as src:
                source_width, source_height = src.width, src.height

        # Initialize GPU-accelerated DEM processor (uses same device as model)
        dem_downloader = DEMDownloaderGPU(bbox, date, device=self.device)

        # 1. Download DEM tiles (cached if already exists)
        _t1 = time.time()
        dem_files = dem_downloader.download_dem_tiles()
        print(f"[flood_postproc_gpu] DEM download:      {time.time() - _t1:.3f}s")

        if not dem_files:
            self.logger.warning(
                "No DEM tiles available for bbox %s, returning original prediction",
                bbox,
            )
            self.postprocess_artifacts = {}
            return prediction

        # 2. Merge & clip DEM to match prediction dimensions
        _t2 = time.time()
        dem_file = dem_downloader.merge_and_clip_dems(
            dem_files,
            width=source_width,
            height=source_height,
        )
        print(f"[flood_postproc_gpu] DEM merge/clip:    {time.time() - _t2:.3f}s")

        if not dem_file:
            self.logger.warning(
                "Failed to merge/clip DEM for bbox %s, returning original prediction",
                bbox,
            )
            self.postprocess_artifacts = {}
            return prediction

        # 3. GPU-accelerated postprocessing
        _t3 = time.time()
        try:
            result = dem_downloader.apply_all_postprocessing(
                flood_detection_file=prediction,
                hls_file=image,
                dem_file=dem_file,
                use_smart_aerosol=True,
            )
        except Exception as exc:
            self.logger.exception(
                "GPU postprocessing failed for %s: %s, falling back to subprocess",
                prediction,
                exc,
            )
            # Fall back to subprocess on GPU failure
            return self._postprocess_subprocess(
                bbox, date, prediction, image, source_width, source_height
            )

        print(
            f"[flood_postproc_gpu] apply_all_postprocessing: {time.time() - _t3:.3f}s"
        )
        print(f"[flood_postproc_gpu] total time: {time.time() - _t0:.3f}s")

        # Handle result
        if not result:
            self.logger.warning(
                "GPU postprocessing returned empty result for %s, using original",
                prediction,
            )
            self.postprocess_artifacts = {}
            return prediction

        if isinstance(result, dict):
            self.postprocess_artifacts = result.get("artifacts", {})
            return result.get("final") or prediction

        self.postprocess_artifacts = {}
        return result

    def _postprocess_subprocess(
        self,
        bbox,
        date,
        prediction,
        image,
        source_width=None,
        source_height=None,
    ):
        """
        CPU-based postprocessing in subprocess (fallback).

        Runs DEM-based postprocessing in a separate process for isolation.
        """
        # Determine width/height in the main process if not provided
        if source_width is not None and source_height is not None:
            width, height = source_width, source_height
        else:
            with rasterio.open(image) as src:
                width, height = src.width, src.height

        try:
            future = self._postprocess_executor.submit(
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
            # Worker may have crashed — recreate the pool so the next call works
            try:
                self._postprocess_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._postprocess_executor = self._make_executor()
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
