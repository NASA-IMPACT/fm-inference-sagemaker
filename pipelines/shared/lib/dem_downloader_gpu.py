"""
GPU-accelerated DEM downloader and postprocessing.

This module provides GPU-accelerated versions of the postprocessing operations
using PyTorch. The download and file I/O operations remain on CPU, while
compute-intensive operations (slope calculation, SLIA, resampling, masks)
are executed on GPU.

Usage:
    from lib.dem_downloader_gpu import DEMDownloaderGPU

    # Use as drop-in replacement for DEMDownloader
    downloader = DEMDownloaderGPU(bbox, date_str)
    dem_files = downloader.download_dem_tiles()
    dem_file = downloader.merge_and_clip_dems(dem_files, width, height)
    result = downloader.apply_all_postprocessing(flood_file, hls_file, dem_file)
"""

import os
import time
import logging
import requests
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.env import Env
from osgeo import gdal

gdal.UseExceptions()

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", "/root/.cache/")
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"

# Enhanced GDAL configuration for rasterio operations
GDAL_CONFIG = {
    "GDAL_CACHEMAX": 8192,
    "GDAL_WARP_MEMORY_LIMIT": 1073741824,
    "CPL_VSIL_CURL_CACHE_SIZE": 200000000,
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": True,
    "GDAL_HTTP_MULTIPLEX": True,
    "GDAL_HTTP_VERSION": 2,
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "GDAL_TIFF_INTERNAL_MASK": True,
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "GDAL_MAX_DATASET_POOL_SIZE": 450,
    "GDAL_TIFF_OVR_BLOCKSIZE": 256,
    "GDAL_TIFF_DIRECT_IO": True,
    "CPL_CURL_VERBOSE": False,
    "GDAL_HTTP_TIMEOUT": 60,
    "GDAL_HTTP_CONNECTTIMEOUT": 30,
    "CPL_LOG": "/tmp/gdal.log",
    "CPL_DEBUG": False,
}

COMPRESSION_PROFILE = {
    "compress": "DEFLATE",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "NUM_THREADS": "ALL_CPUS",
    "BIGTIFF": "IF_SAFER",
}


def get_gdal_env():
    """Returns a configured GDAL environment for optimal rasterio operations."""
    return Env(**GDAL_CONFIG)


class GPUPostProcessor:
    """
    GPU-accelerated postprocessing operations using PyTorch.

    All compute-intensive operations (sobel, resampling, mask operations)
    are executed on GPU. Data is transferred to GPU once at the start
    and back to CPU once at the end to minimize transfer overhead.
    """

    # Sobel kernels for gradient calculation (3x3)
    SOBEL_X = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    SOBEL_Y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    def __init__(self, device: str = None):
        """
        Initialize GPU postprocessor.

        Args:
            device: CUDA device to use. If None, auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_gpu = self.device != "cpu"

        # Move sobel kernels to device
        self.sobel_x = self.SOBEL_X.to(self.device)
        self.sobel_y = self.SOBEL_Y.to(self.device)

        self.logger = logging.getLogger(__name__)

        if self.use_gpu:
            self.logger.info(
                f"GPUPostProcessor initialized on {torch.cuda.get_device_name()}"
            )
        else:
            self.logger.warning(
                "GPUPostProcessor running on CPU - no CUDA device available"
            )

    def _to_tensor(
        self, array: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Convert numpy array to GPU tensor."""
        tensor = torch.from_numpy(array.astype(np.float32))
        if dtype != torch.float32:
            tensor = tensor.to(dtype)
        return tensor.to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array."""
        return tensor.cpu().numpy()

    def resample(
        self, source: torch.Tensor, target_shape: tuple, mode: str = "bilinear"
    ) -> torch.Tensor:
        """
        GPU-accelerated bilinear resampling.

        Replaces rasterio.warp.reproject for same-CRS resampling.

        Args:
            source: Input tensor (H, W) or (C, H, W)
            target_shape: Target (H, W) dimensions
            mode: Interpolation mode ('bilinear', 'nearest')

        Returns:
            Resampled tensor matching target_shape
        """
        if source.shape[-2:] == target_shape:
            return source

        # Add batch and channel dims if needed
        ndim = source.ndim
        if ndim == 2:
            source = source.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif ndim == 3:
            source = source.unsqueeze(0)  # (1, C, H, W)

        resampled = F.interpolate(
            source,
            size=target_shape,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )

        # Remove added dims
        if ndim == 2:
            resampled = resampled.squeeze(0).squeeze(0)
        elif ndim == 3:
            resampled = resampled.squeeze(0)

        return resampled

    def calculate_slope(
        self,
        dem_data: torch.Tensor,
        pixel_size_x: float,
        pixel_size_y: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated slope calculation using Sobel filters.

        Args:
            dem_data: DEM elevation tensor (H, W)
            pixel_size_x: Pixel size in meters (X direction)
            pixel_size_y: Pixel size in meters (Y direction)

        Returns:
            Tuple of (slope_degrees, dz_dx, dz_dy)
        """
        # Add batch and channel dims for conv2d
        dem_4d = dem_data.unsqueeze(0).unsqueeze(0)

        # Apply Sobel filters
        dz_dx = F.conv2d(dem_4d, self.sobel_x, padding=1).squeeze() / (8 * pixel_size_x)
        dz_dy = F.conv2d(dem_4d, self.sobel_y, padding=1).squeeze() / (8 * pixel_size_y)

        # Calculate slope in degrees
        slope_rad = torch.atan(torch.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = torch.rad2deg(slope_rad)

        return slope_deg, dz_dx, dz_dy

    def calculate_solar_incidence_angle(
        self,
        dz_dx: torch.Tensor,
        dz_dy: torch.Tensor,
        sza_data: torch.Tensor,
        saa_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        GPU-accelerated solar incidence angle (SLIA) calculation.

        Args:
            dz_dx: X gradient from slope calculation
            dz_dy: Y gradient from slope calculation
            sza_data: Solar zenith angle (degrees)
            saa_data: Solar azimuth angle (degrees)

        Returns:
            Solar incidence angle in degrees
        """
        # Surface normal vectors from terrain gradients
        nx = -dz_dx
        ny = -dz_dy
        nz = torch.ones_like(dz_dx)

        # Normalize
        norm = torch.sqrt(nx**2 + ny**2 + nz**2)
        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        # Solar vector from angles
        zenith_rad = torch.deg2rad(sza_data)
        azimuth_rad = torch.deg2rad(saa_data)

        sx = torch.sin(zenith_rad) * torch.sin(azimuth_rad)
        sy = torch.sin(zenith_rad) * torch.cos(azimuth_rad)
        sz = torch.cos(zenith_rad)

        # Incidence angle = acos(dot product of normal and solar vectors)
        cos_incidence = torch.clamp(sx * nx + sy * ny + sz * nz, -1.0, 1.0)
        incidence_angle = torch.rad2deg(torch.acos(cos_incidence))

        return incidence_angle

    def apply_terrain_shadow_mask(
        self,
        flood_data: torch.Tensor,
        slia_data: torch.Tensor,
        slope_data: torch.Tensor,
        slia_threshold: float = 85.0,
        slope_threshold: float = 15.0,
    ) -> torch.Tensor:
        """
        Apply terrain shadow mask on GPU.

        Pixels with high solar incidence angle AND steep slope are likely
        terrain shadows misclassified as water.

        Args:
            flood_data: Flood detection mask (0/1)
            slia_data: Solar incidence angle in degrees
            slope_data: Slope in degrees
            slia_threshold: SLIA threshold (default 85 degrees)
            slope_threshold: Slope threshold (default 15 degrees)

        Returns:
            Modified flood_data with shadows removed
        """
        shadow_mask = (slia_data > slia_threshold) & (slope_data > slope_threshold)
        flood_data = torch.where(
            (flood_data == 1) & shadow_mask, torch.zeros_like(flood_data), flood_data
        )
        return flood_data

    def apply_aerosol_mask(
        self,
        flood_data: torch.Tensor,
        green_data: torch.Tensor,
        nir_data: torch.Tensor,
        fmask_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply smart aerosol filter on GPU.

        High aerosol pixels with negative NDWI and low NIR are likely
        false positives from atmospheric interference.

        Args:
            flood_data: Flood detection mask (0/1)
            green_data: Green band values
            nir_data: NIR band values
            fmask_data: Fmask QA band (bits 6-7 contain aerosol level)

        Returns:
            Modified flood_data with aerosol artifacts removed
        """
        # Extract aerosol level (bits 6-7)
        aerosol_level = (fmask_data.int() >> 6) & 3

        # Calculate NDWI
        denom = green_data + nir_data
        ndwi = torch.where(
            denom != 0, (green_data - nir_data) / denom, torch.zeros_like(green_data)
        )

        # Apply mask: high aerosol + negative NDWI + low NIR = false positive
        aerosol_mask = (
            (flood_data == 1) & (aerosol_level >= 2) & (ndwi < 0) & (nir_data < 500)
        )

        flood_data = torch.where(aerosol_mask, torch.zeros_like(flood_data), flood_data)
        return flood_data

    def calculate_ndvi(
        self,
        red_data: torch.Tensor,
        nir_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate NDVI on GPU.

        Args:
            red_data: Red band values
            nir_data: NIR band values

        Returns:
            Tuple of (ndvi, valid_mask)
        """
        denom = red_data + nir_data
        valid_mask = denom != 0

        ndvi = torch.where(
            valid_mask, (nir_data - red_data) / denom, torch.zeros_like(red_data)
        )

        return ndvi, valid_mask

    def apply_vegetation_mask(
        self,
        flood_data: torch.Tensor,
        ndvi: torch.Tensor,
        ndvi_threshold: float = 0.7,
    ) -> torch.Tensor:
        """
        Apply vegetation filter on GPU.

        High NDVI pixels are vegetation, not water.

        Args:
            flood_data: Flood detection mask (0/1)
            ndvi: NDVI values
            ndvi_threshold: Threshold above which pixels are vegetation

        Returns:
            Modified flood_data with vegetation removed
        """
        veg_mask = (ndvi > ndvi_threshold) & (flood_data == 1)
        flood_data = torch.where(veg_mask, torch.zeros_like(flood_data), flood_data)
        return flood_data


class DEMDownloaderGPU:
    """
    GPU-accelerated DEM downloader and postprocessor.

    Drop-in replacement for DEMDownloader with GPU-accelerated postprocessing.
    Download and file I/O operations remain on CPU.
    """

    def __init__(self, bbox, date_str, device: str = None):
        """
        Initialize DEMDownloaderGPU.

        Args:
            bbox: Bounding box (west, south, east, north)
            date_str: Date string for the analysis
            device: CUDA device. If None, auto-detects.
        """
        self.output_dir = DOWNLOAD_FOLDER
        self.bbox = bbox
        self.date_str = date_str
        self.dem_dir = os.path.join(self.output_dir, "dem_tiles")
        os.makedirs(self.dem_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize GPU processor
        self.gpu = GPUPostProcessor(device=device)

    def _get_tile_name(self, lat, lon):
        """Generate Copernicus DEM tile name from lat/lon."""
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{lat_prefix}{abs(lat):02d}_00_{lon_prefix}{abs(lon):03d}_00_DEM"

    def _download_tile(self, url, output_file):
        """Download a single DEM tile."""
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_file
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None

    def download_dem_tiles(self, max_workers=10):
        """
        Download DEM tiles covering the bounding box.

        Args:
            max_workers: Maximum concurrent downloads

        Returns:
            List of downloaded tile paths, or None if no tiles found
        """
        west, south, east, north = self.bbox
        min_lon, max_lon = int(np.floor(west)), int(np.floor(east))
        min_lat, max_lat = int(np.floor(south)), int(np.floor(north))

        urls_to_download = []
        for lat in range(min_lat, max_lat + 1):
            for lon in range(min_lon, max_lon + 1):
                tile_name = self._get_tile_name(lat, lon)
                output_file = os.path.join(self.dem_dir, f"{tile_name}.tif")
                if not os.path.exists(output_file):
                    url = URL.format(tile_name=tile_name)
                    urls_to_download.append((url, output_file))

        dem_tiles = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._download_tile, url, path): path
                for url, path in urls_to_download
            }
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    dem_tiles.append(result)

        # Include already-downloaded tiles
        for lat in range(min_lat, max_lat + 1):
            for lon in range(min_lon, max_lon + 1):
                tile_name = self._get_tile_name(lat, lon)
                output_file = os.path.join(self.dem_dir, f"{tile_name}.tif")
                if os.path.exists(output_file) and output_file not in dem_tiles:
                    dem_tiles.append(output_file)

        if not dem_tiles:
            self.logger.warning("No DEM tiles found or downloaded.")
            return None
        return dem_tiles

    def merge_and_clip_dems(self, dem_tiles, width=None, height=None):
        """
        Merge and clip DEM tiles to bounding box.

        Uses GDAL VRT for efficient mosaic without data copy.

        Args:
            dem_tiles: List of DEM tile paths
            width: Optional output width (for resampling)
            height: Optional output height (for resampling)

        Returns:
            Path to clipped DEM file, or None on failure
        """
        try:
            west, south, east, north = self.bbox
            base_filename = f"{west}_{south}_{east}_{north}".replace(".", "_").replace(
                "-", "m"
            )
            filename = f"{self.dem_dir}/{base_filename}_dem_clipped.tif"

            if os.path.exists(filename):
                return filename

            # Build VRT mosaic (instant, no data copy)
            vrt_path = os.path.join(self.output_dir, f"{base_filename}_dem_mosaic.vrt")
            gdal.BuildVRT(vrt_path, dem_tiles)

            if not os.path.exists(vrt_path):
                self.logger.error("Failed to create VRT mosaic")
                return None

            # Clip with GDAL Warp
            wkt_poly = f"POLYGON(({west} {south},{west} {north},{east} {north},{east} {south},{west} {south}))"

            warp_kwargs = {
                "dstSRS": "EPSG:4326",
                "format": "GTiff",
                "cutlineWKT": wkt_poly,
                "cropToCutline": True,
                "dstNodata": -9999,
                "resampleAlg": "bilinear",
                "multithread": True,
                "warpMemoryLimit": 1073741824,
                "creationOptions": [
                    "TILED=YES",
                    "BLOCKXSIZE=256",
                    "BLOCKYSIZE=256",
                    "NUM_THREADS=ALL_CPUS",
                    "BIGTIFF=IF_SAFER",
                    "COMPRESS=DEFLATE",
                ],
            }

            if width and height:
                warp_kwargs["width"] = width
                warp_kwargs["height"] = height

            warp_options = gdal.WarpOptions(**warp_kwargs)
            gdal.Warp(filename, vrt_path, options=warp_options)

            # Cleanup VRT
            try:
                os.remove(vrt_path)
            except OSError:
                pass

            if not os.path.exists(filename):
                self.logger.error("Failed to create clipped DEM")
                return None

            return filename

        except Exception as e:
            self.logger.error(f"Error merging and clipping DEMs: {e}")
            return None

    @staticmethod
    def save_slope_file(slope_deg, dem_src_name, dem_profile):
        """Save slope array to file."""
        slope_path = f"{DOWNLOAD_FOLDER}/slopes"
        os.makedirs(slope_path, exist_ok=True)
        slope_file = os.path.join(
            slope_path,
            os.path.basename(dem_src_name).replace(
                "_dem_clipped.tif", "_slope_degrees.tif"
            ),
        )

        profile = dem_profile.copy()
        profile.update(dtype=rasterio.float32, **COMPRESSION_PROFILE)

        with get_gdal_env():
            with rasterio.open(slope_file, "w", **profile) as dst:
                dst.write(slope_deg.astype(np.float32), 1)

        return slope_file

    @staticmethod
    def save_slia_file(slia_deg, dem_src_name, dem_profile):
        """Save SLIA array to file."""
        output_dir = os.path.join(DOWNLOAD_FOLDER, "solar_incidence")
        os.makedirs(output_dir, exist_ok=True)

        slia_file = os.path.join(
            output_dir,
            os.path.basename(dem_src_name).replace(
                "_dem_clipped.tif", "_slia_degrees.tif"
            ),
        )
        profile = dem_profile.copy()
        profile.update(**COMPRESSION_PROFILE)

        with get_gdal_env():
            with rasterio.open(slia_file, "w", **profile) as dst:
                dst.write(slia_deg.astype(np.float32), 1)
        return slia_file

    @staticmethod
    def save_ndvi_file(ndvi, valid_pixels, dem_filename, flood_profile):
        """Save NDVI array to file."""
        ndvi_file = dem_filename.replace("_dem_clipped.tif", "_ndvi.tif")
        ndvi_profile = flood_profile.copy()
        ndvi_profile.update(dtype="float32", nodata=-999, **COMPRESSION_PROFILE)

        ndvi_out = ndvi.copy()
        ndvi_out[~valid_pixels] = -999

        with get_gdal_env():
            with rasterio.open(ndvi_file, "w", **ndvi_profile) as dst:
                dst.write(ndvi_out, 1)
        return ndvi_file

    def apply_all_postprocessing(
        self,
        flood_detection_file,
        hls_file,
        dem_file,
        use_smart_aerosol=True,
    ):
        """
        GPU-accelerated postprocessing pipeline.

        Performs all postprocessing on GPU:
        1. Calculate slope from DEM
        2. Calculate solar incidence angle (SLIA)
        3. Resample arrays to common grid (if needed)
        4. Apply terrain shadow mask
        5. Apply aerosol mask (optional)
        6. Calculate NDVI and apply vegetation mask

        Args:
            flood_detection_file: Path to flood detection GeoTIFF
            hls_file: Path to HLS mosaic GeoTIFF
            dem_file: Path to clipped DEM GeoTIFF
            use_smart_aerosol: Whether to apply aerosol filtering

        Returns:
            Dict with 'final' (corrected prediction path) and 'artifacts' (diagnostic files)
        """
        postproc_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(postproc_dir, exist_ok=True)

        _t0 = time.time()

        # ============================================================
        # STAGE 1: Read all data from disk (CPU)
        # ============================================================
        with get_gdal_env():
            with (
                rasterio.open(dem_file) as dem_src,
                rasterio.open(hls_file) as hls_src,
                rasterio.open(flood_detection_file) as flood_src,
            ):
                # DEM data
                dem_data = dem_src.read(1)
                dem_transform = dem_src.transform
                dem_bounds = dem_src.bounds
                dem_src_name = dem_src.name
                dem_profile = dem_src.profile.copy()

                # Flood prediction
                flood_data = flood_src.read(1).copy()
                flood_profile = flood_src.profile.copy()
                flood_shape = flood_data.shape

                # HLS bands
                green_data = hls_src.read(2).astype(np.float32)
                red_data = hls_src.read(3).astype(np.float32)
                nir_data = hls_src.read(4).astype(np.float32)
                fmask_data = hls_src.read(7).astype(np.float32)
                sza_data = hls_src.read(8).astype(np.float32)
                saa_data = hls_src.read(9).astype(np.float32)

        print(f"  [postproc-gpu] file reads:       {time.time() - _t0:.3f}s")

        # ============================================================
        # STAGE 2: Transfer to GPU and compute (GPU)
        # ============================================================
        _t1 = time.time()

        # Calculate pixel sizes for slope computation
        lat_center = (dem_bounds.bottom + dem_bounds.top) / 2
        pixel_size_x = abs(dem_transform[0]) * 111000 * np.cos(np.radians(lat_center))
        pixel_size_y = abs(dem_transform[4]) * 111000

        # Transfer DEM to GPU
        dem_gpu = self.gpu._to_tensor(dem_data)

        # Calculate slope on GPU
        slope_deg, dz_dx, dz_dy = self.gpu.calculate_slope(
            dem_gpu, pixel_size_x, pixel_size_y
        )

        # Transfer and resample SZA/SAA to DEM grid if needed
        sza_gpu = self.gpu._to_tensor(sza_data)
        saa_gpu = self.gpu._to_tensor(saa_data)

        if sza_data.shape != dem_data.shape:
            sza_gpu = self.gpu.resample(sza_gpu, dem_data.shape)
            saa_gpu = self.gpu.resample(saa_gpu, dem_data.shape)

        # Calculate SLIA on GPU
        slia_deg = self.gpu.calculate_solar_incidence_angle(
            dz_dx, dz_dy, sza_gpu, saa_gpu
        )

        print(f"  [postproc-gpu] slope+SLIA calc:  {time.time() - _t1:.3f}s")

        # ============================================================
        # STAGE 3: Resample all arrays to flood grid (GPU)
        # ============================================================
        _t2 = time.time()

        # Resample slope and SLIA to flood grid if needed
        if slope_deg.shape != flood_shape:
            slope_deg = self.gpu.resample(slope_deg, flood_shape)
            slia_deg = self.gpu.resample(slia_deg, flood_shape)

        # Transfer HLS bands to GPU and resample if needed
        green_gpu = self.gpu._to_tensor(green_data)
        red_gpu = self.gpu._to_tensor(red_data)
        nir_gpu = self.gpu._to_tensor(nir_data)
        fmask_gpu = self.gpu._to_tensor(fmask_data)

        if green_data.shape != flood_shape:
            green_gpu = self.gpu.resample(green_gpu, flood_shape)
            red_gpu = self.gpu.resample(red_gpu, flood_shape)
            nir_gpu = self.gpu.resample(nir_gpu, flood_shape)
            fmask_gpu = self.gpu.resample(fmask_gpu, flood_shape, mode="nearest")

        print(f"  [postproc-gpu] resample:         {time.time() - _t2:.3f}s")

        # ============================================================
        # STAGE 4: Apply all masks (GPU)
        # ============================================================
        _t3 = time.time()

        # Transfer flood data to GPU
        flood_gpu = self.gpu._to_tensor(flood_data)

        # Apply terrain shadow mask
        flood_gpu = self.gpu.apply_terrain_shadow_mask(flood_gpu, slia_deg, slope_deg)

        # Apply aerosol mask
        if use_smart_aerosol:
            flood_gpu = self.gpu.apply_aerosol_mask(
                flood_gpu, green_gpu, nir_gpu, fmask_gpu
            )

        # Calculate NDVI and apply vegetation mask
        ndvi_gpu, valid_ndvi_gpu = self.gpu.calculate_ndvi(red_gpu, nir_gpu)
        flood_gpu = self.gpu.apply_vegetation_mask(flood_gpu, ndvi_gpu)

        print(f"  [postproc-gpu] mask operations:  {time.time() - _t3:.3f}s")

        # ============================================================
        # STAGE 5: Transfer back to CPU (single transfer)
        # ============================================================
        _t4 = time.time()

        flood_data = self.gpu._to_numpy(flood_gpu)
        slope_deg_np = self.gpu._to_numpy(slope_deg)
        slia_deg_np = self.gpu._to_numpy(slia_deg)
        ndvi_np = self.gpu._to_numpy(ndvi_gpu)
        valid_ndvi_np = self.gpu._to_numpy(valid_ndvi_gpu).astype(bool)

        # Free GPU memory
        del flood_gpu, slope_deg, slia_deg, ndvi_gpu, valid_ndvi_gpu
        del dem_gpu, dz_dx, dz_dy, sza_gpu, saa_gpu
        del green_gpu, red_gpu, nir_gpu, fmask_gpu
        torch.cuda.empty_cache()

        print(f"  [postproc-gpu] gpu->cpu transfer: {time.time() - _t4:.3f}s")

        # ============================================================
        # STAGE 6: Write output files in parallel (CPU/Disk)
        # ============================================================
        final_corrected = os.path.join(
            postproc_dir,
            os.path.basename(flood_detection_file).replace(
                ".tif", "_final_corrected.tif"
            ),
        )
        flood_profile.update(**COMPRESSION_PROFILE)

        def _write_final():
            with rasterio.open(final_corrected, "w", **flood_profile) as dst:
                dst.write(flood_data, 1)
            # Explicit sync to ensure file is fully written before returning
            # This prevents race conditions with subsequent reads
            import os as _os

            try:
                fd = _os.open(final_corrected, _os.O_RDONLY)
                _os.fsync(fd)
                _os.close(fd)
            except OSError:
                pass
            return final_corrected

        def _write_slope():
            return self.save_slope_file(slope_deg_np, dem_src_name, dem_profile)

        def _write_slia():
            return self.save_slia_file(slia_deg_np, dem_src_name, dem_profile)

        def _write_ndvi():
            return self.save_ndvi_file(ndvi_np, valid_ndvi_np, dem_file, flood_profile)

        _t5 = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_final = pool.submit(_write_final)
            f_slope = pool.submit(_write_slope)
            f_slia = pool.submit(_write_slia)
            f_ndvi = pool.submit(_write_ndvi)
            final_corrected = f_final.result()
            slope_file = f_slope.result()
            slia_file = f_slia.result()
            ndvi_file = f_ndvi.result()

        print(f"  [postproc-gpu] parallel writes:  {time.time() - _t5:.3f}s")
        print(f"  [postproc-gpu] total:            {time.time() - _t0:.3f}s")

        artifacts = {
            "dem": dem_file,
            "slope": slope_file,
            "solar_incidence": slia_file,
            "ndvi": ndvi_file,
        }

        return {
            "final": final_corrected,
            "artifacts": artifacts,
        }
