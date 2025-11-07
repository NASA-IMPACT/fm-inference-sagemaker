import os
import requests
import numpy as np
import rasterio
from rasterio.env import Env
import geopandas as gpd
import logging
import shutil

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel
from shapely.geometry import box
from osgeo import gdal

gdal.UseExceptions()

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", '/root/.cache/')
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"

# Enhanced GDAL configuration for rasterio operations
GDAL_CONFIG = {
    # Memory and caching - increased for better performance
    'GDAL_CACHEMAX': 8192,  # MB of cache for GDAL operations (increased from 512)
    'GDAL_WARP_MEMORY_LIMIT': 1073741824,  # 1GB warp memory limit
    'CPL_VSIL_CURL_CACHE_SIZE': 200000000,  # 200MB for network file cache
    'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': True,  # Optimize HTTP range requests
    'GDAL_HTTP_MULTIPLEX': True,  # Enable HTTP/2 multiplexing
    'GDAL_HTTP_VERSION': 2,  # Use HTTP/2 for better performance

    # Disable auxiliary files that slow down operations
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',  # Avoid scanning directories
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.TIF,.tiff,.TIFF',  # Only look for TIF files
    'GDAL_TIFF_INTERNAL_MASK': True,  # Use internal mask for better performance

    # Performance optimizations
    'GDAL_NUM_THREADS': 'ALL_CPUS',  # Use all available CPU cores
    'GDAL_MAX_DATASET_POOL_SIZE': 450,  # Connection pool for datasets
    'GDAL_TIFF_OVR_BLOCKSIZE': 256,  # Optimize overview block size
    'GDAL_TIFF_DIRECT_IO': True,  # Enable direct I/O for better performance

    # Network optimizations
    'CPL_CURL_VERBOSE': False,  # Reduce logging overhead
    'GDAL_HTTP_TIMEOUT': 60,  # Timeout for HTTP requests
    'GDAL_HTTP_CONNECTTIMEOUT': 30,  # Connection timeout

    # Error handling
    'CPL_LOG': '/tmp/gdal.log',  # Log GDAL errors
    'CPL_DEBUG': False,  # Disable debug logging for performance
}

# Common profile updates for compressed tiled output - optimized
COMPRESSION_PROFILE = {
    'compress': 'DEFLATE',
    'tiled': True,
    'blockxsize': 256,  # Reduced from 512 for better memory efficiency
    'blockysize': 256,  # Reduced from 512 for better memory efficiency
    'NUM_THREADS': 'ALL_CPUS',  # Multi-threaded compression
    'BIGTIFF': 'IF_SAFER',  # Auto-detect BigTIFF need
}


def get_gdal_env():
    """Returns a configured GDAL environment for optimal rasterio operations."""
    return Env(**GDAL_CONFIG)

class DEMDownloader:
    def __init__(self, bbox, date_str):
        self.output_dir = DOWNLOAD_FOLDER
        self.bbox = bbox
        self.date_str = date_str
        self.dem_dir = os.path.join(self.output_dir, "dem_tiles")
        os.makedirs(self.dem_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_tile_name(self, lat, lon):
        lat_prefix = 'N' if lat >= 0 else 'S'
        lon_prefix = 'E' if lon >= 0 else 'W'
        return f"Copernicus_DSM_COG_10_{lat_prefix}{abs(lat):02d}_00_{lon_prefix}{abs(lon):03d}_00_DEM"

    def _download_tile(self, url, output_file):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_file
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None

    def download_dem_tiles(self, max_workers=10):
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
            future_to_url = {executor.submit(self._download_tile, url, path): path for url, path in urls_to_download}
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    dem_tiles.append(result)

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
        Optimized merge and clip using VRT and GDAL for better performance.
        Uses GDAL's native operations instead of rasterio for speed.
        """
        try:
            west, south, east, north = self.bbox
            base_filename = f"{west}_{south}_{east}_{north}".replace('.', '_').replace('-', 'm')
            filename = f"{self.dem_dir}/{base_filename}_dem_clipped.tif"
            if os.path.exists(filename):
                return filename

            # Step 1: Build VRT mosaic (no data copy, instant)
            vrt_path = os.path.join(self.output_dir, f"{base_filename}_dem_mosaic.vrt")
            gdal.BuildVRT(vrt_path, dem_tiles)

            if not os.path.exists(vrt_path):
                self.logger.error("Failed to create VRT mosaic")
                return None

            # Step 2: Clip and optionally resample using GDAL Warp (single operation)
            wkt_poly = f"POLYGON(({west} {south},{west} {north},{east} {north},{east} {south},{west} {south}))"

            warp_options = gdal.WarpOptions(
                dstSRS='EPSG:4326',
                format='GTiff',
                cutlineWKT=wkt_poly,
                cropToCutline=True,
                dstNodata=-9999,
                resampleAlg='bilinear',
                multithread=True,
                warpMemoryLimit=1073741824,  # 1GB
                creationOptions=[
                    'TILED=YES',
                    'BLOCKXSIZE=256',
                    'BLOCKYSIZE=256',
                    'NUM_THREADS=ALL_CPUS',
                    'BIGTIFF=IF_SAFER',
                    'COMPRESS=DEFLATE',
                ],
            )

            # Add width/height if specified
            if width and height:
                warp_options = gdal.WarpOptions(
                    dstSRS='EPSG:4326',
                    format='GTiff',
                    cutlineWKT=wkt_poly,
                    cropToCutline=True,
                    dstNodata=-9999,
                    resampleAlg='bilinear',
                    multithread=True,
                    warpMemoryLimit=1073741824,
                    width=width,
                    height=height,
                    creationOptions=[
                        'TILED=YES',
                        'BLOCKXSIZE=256',
                        'BLOCKYSIZE=256',
                        'NUM_THREADS=ALL_CPUS',
                        'BIGTIFF=IF_SAFER',
                        'COMPRESS=DEFLATE',
                    ],
                )

            gdal.Warp(filename, vrt_path, options=warp_options)

            # Clean up temporary VRT
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
    def calculate_slope(dem_src):
        dem = dem_src.read(1)
        transform = dem_src.transform
        lat_center = (dem_src.bounds.bottom + dem_src.bounds.top) / 2
        pixel_size_x = abs(transform[0]) * 111000 * np.cos(np.radians(lat_center))
        pixel_size_y = abs(transform[4]) * 111000

        # Vectorized slope calculation
        dz_dx = sobel(dem, axis=1) / (8 * pixel_size_x)
        dz_dy = sobel(dem, axis=0) / (8 * pixel_size_y)

        # In-place operations to reduce memory allocations
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        slope_path = f"{DOWNLOAD_FOLDER}/slopes"
        os.makedirs(slope_path, exist_ok=True)
        slope_file = os.path.join(slope_path, os.path.basename(dem_src.name).replace('_dem_clipped.tif', '_slope_degrees.tif'))

        profile = dem_src.profile.copy()
        profile.update(dtype=rasterio.float32, **COMPRESSION_PROFILE)

        with get_gdal_env():
            with rasterio.open(slope_file, 'w', **profile) as dst:
                dst.write(slope_deg.astype(rasterio.float32), 1)

        return slope_file

    @staticmethod
    def calculate_solar_incidence_angle_from_bands(dem_src, hls_src):
        output_dir = os.path.join(DOWNLOAD_FOLDER, "solar_incidence")
        os.makedirs(output_dir, exist_ok=True)

        dem = dem_src.read(1)
        dem_transform = dem_src.transform
        dem_profile = dem_src.profile.copy()

        lat_center = (dem_src.bounds.bottom + dem_src.bounds.top) / 2
        pixel_size_x = abs(dem_transform[0]) * 111000 * np.cos(np.radians(lat_center))
        pixel_size_y = abs(dem_transform[4]) * 111000

        # Vectorized sobel operations
        dz_dx = sobel(dem, axis=1) / (8 * pixel_size_x)
        dz_dy = sobel(dem, axis=0) / (8 * pixel_size_y)

        # In-place normal vector calculations
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(dem)

        # Vectorized normalization
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        sza_data = hls_src.read(8)
        saa_data = hls_src.read(9)

        # Only create resampled arrays if needed
        if sza_data.shape != dem.shape:
            sza_resampled = np.zeros_like(dem, dtype=np.float32)
            saa_resampled = np.zeros_like(dem, dtype=np.float32)

            reproject(
                source=sza_data,
                destination=sza_resampled,
                src_transform=hls_src.transform,
                src_crs=hls_src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_src.crs,
                resampling=Resampling.bilinear
            )
            reproject(
                source=saa_data,
                destination=saa_resampled,
                src_transform=hls_src.transform,
                src_crs=hls_src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_src.crs,
                resampling=Resampling.bilinear
            )
            sza_data, saa_data = sza_resampled, saa_resampled

        # Vectorized trigonometric operations
        zenith_rad = np.radians(sza_data)
        azimuth_rad = np.radians(saa_data)

        sx = np.sin(zenith_rad) * np.sin(azimuth_rad)
        sy = np.sin(zenith_rad) * np.cos(azimuth_rad)
        sz = np.cos(zenith_rad)

        # Vectorized incidence angle calculation
        cos_incidence = np.clip(sx * nx + sy * ny + sz * nz, -1, 1)
        incidence_angle = np.degrees(np.arccos(cos_incidence))

        slia_file = os.path.join(output_dir, os.path.basename(dem_src.name).replace('_dem_clipped.tif', '_slia_degrees.tif'))
        dem_profile.update(**COMPRESSION_PROFILE)
        with get_gdal_env():
            with rasterio.open(slia_file, 'w', **dem_profile) as dst:
                dst.write(incidence_angle.astype(np.float32), 1)
        return slia_file

    @staticmethod
    def extract_aerosol_level_from_fmask(hls_src, dem_filename, output_dir):
        fmask_data = hls_src.read(7)
        profile = hls_src.profile.copy()

        aerosol_level = (fmask_data >> 6) & 3
        aerosol_file = os.path.join(output_dir, os.path.basename(dem_filename).replace('_dem_clipped.tif', '_aerosol_level.tif'))

        with rasterio.open(aerosol_file, 'w', **profile) as dst:
            dst.write(aerosol_level.astype(np.uint8), 1)
        return aerosol_file, aerosol_level

    @staticmethod
    def postprocess_terrain_shadows(flood_src, slia_file, slope_file, slia_threshold=85, slope_threshold=15):
        """
        Optimized terrain shadow processing with vectorized operations.
        """
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        flood_transform = flood_src.transform

        with rasterio.open(slia_file) as slia_src, rasterio.open(slope_file) as slope_src:
            slia_data = slia_src.read(1)
            slope_data = slope_src.read(1)

            if slia_data.shape != flood_data.shape:
                # Optimized: Single combined reprojection
                slia_resampled = np.zeros_like(flood_data, dtype=np.float32)
                slope_resampled = np.zeros_like(flood_data, dtype=np.float32)

                reproject(
                    source=rasterio.band(slia_src, 1),
                    destination=slia_resampled,
                    src_transform=slia_src.transform,
                    src_crs=slia_src.crs,
                    dst_transform=flood_transform,
                    dst_crs=flood_src.crs,
                    resampling=Resampling.bilinear
                )

                reproject(
                    source=rasterio.band(slope_src, 1),
                    destination=slope_resampled,
                    src_transform=slope_src.transform,
                    src_crs=slope_src.crs,
                    dst_transform=flood_transform,
                    dst_crs=flood_src.crs,
                    resampling=Resampling.bilinear
                )
                slia_data, slope_data = slia_resampled, slope_resampled

        # Vectorized mask creation
        shadow_mask = (slia_data > slia_threshold) & (slope_data > slope_threshold)
        flood_corrected = flood_data.copy()
        pixels_corrected = np.sum((flood_data == 1) & shadow_mask)
        flood_corrected[shadow_mask & (flood_data == 1)] = 0

        output_file = os.path.join(output_dir, os.path.basename(flood_src.name).replace('.tif', '_flood_terrain_corrected.tif'))
        flood_profile.update(**COMPRESSION_PROFILE)
        with get_gdal_env():
            with rasterio.open(output_file, 'w', **flood_profile) as dst:
                dst.write(flood_corrected, 1)
        return output_file, shadow_mask, pixels_corrected

    @staticmethod
    def postprocess_smart_aerosol_filter(flood_src, hls_src):
        """
        Optimized aerosol filter with vectorized operations and efficient reprojection.
        """
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        nir_data = hls_src.read(4).astype(np.float32)
        green_data = hls_src.read(2).astype(np.float32)
        fmask_data = hls_src.read(7).astype(np.uint32)

        if nir_data.shape != flood_data.shape:
            flood_height, flood_width = flood_data.shape

            # Optimized: Stack bands for single reprojection operation
            hls_bands_stacked = np.stack([green_data, nir_data, fmask_data.astype(np.float32)], axis=0)
            reprojected_bands = np.zeros((3, flood_height, flood_width), dtype=np.float32)

            reproject(
                source=hls_bands_stacked,
                destination=reprojected_bands,
                src_transform=hls_src.transform,
                src_crs=hls_src.crs,
                dst_transform=flood_src.transform,
                dst_crs=flood_src.crs,
                resampling=Resampling.bilinear
            )

            green_data = reprojected_bands[0]
            nir_data = reprojected_bands[1]
            fmask_data = reprojected_bands[2].astype(np.uint32)

        # Vectorized aerosol level extraction
        aerosol_level = (fmask_data >> 6) & 3

        # Vectorized NDWI calculation with in-place operations
        ndwi = np.zeros_like(green_data, dtype=np.float32)
        valid = (green_data + nir_data) != 0
        np.divide(green_data - nir_data, green_data + nir_data, out=ndwi, where=valid)

        # Vectorized mask creation
        aerosol_mask = (
            (flood_data == 1) &
            (aerosol_level >= 2) &
            (ndwi < 0) &
            (nir_data < 500)
        )

        flood_corrected = flood_data.copy()
        pixels_corrected = np.sum(aerosol_mask)
        flood_corrected[aerosol_mask] = 0

        output_file = os.path.join(output_dir, os.path.basename(flood_src.name).replace('.tif', '_flood_aerosol_corrected.tif'))
        flood_profile.update(**COMPRESSION_PROFILE)
        with get_gdal_env():
            with rasterio.open(output_file, 'w', **flood_profile) as dst:
                dst.write(flood_corrected, 1)
        return output_file, aerosol_mask, pixels_corrected

    @staticmethod
    def postprocess_vegetation_filter(flood_src, hls_src, dem_filename, ndvi_threshold=0.7):
        """
        Optimized vegetation filter with vectorized NDVI calculation.
        """
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        red_data = hls_src.read(3).astype(np.float32)
        nir_data = hls_src.read(4).astype(np.float32)

        # Vectorized NDVI calculation with in-place operations
        ndvi = np.zeros_like(red_data, dtype=np.float32)
        valid_pixels = (red_data + nir_data) != 0
        np.divide(nir_data - red_data, nir_data + red_data, out=ndvi, where=valid_pixels)

        # Vectorized mask creation
        veg_mask = (ndvi > ndvi_threshold) & (flood_data == 1)
        flood_corrected = flood_data.copy()
        pixels_corrected = np.sum(veg_mask)
        flood_corrected[veg_mask] = 0

        output_file = os.path.join(output_dir, os.path.basename(flood_src.name).replace('.tif', '_flood_veg_corrected.tif'))
        flood_profile.update(**COMPRESSION_PROFILE)
        with get_gdal_env():
            with rasterio.open(output_file, 'w', **flood_profile) as dst:
                dst.write(flood_corrected, 1)

        # NDVI filename should be the same as dem filename but with _ndvi suffix
        ndvi_file = dem_filename.replace('_dem_clipped.tif', '_ndvi.tif')
        ndvi_profile = flood_profile.copy()
        ndvi_profile.update(dtype='float32', nodata=-999, **COMPRESSION_PROFILE)
        with get_gdal_env():
            with rasterio.open(ndvi_file, 'w', **ndvi_profile) as dst:
                ndvi[~valid_pixels] = -999
                dst.write(ndvi, 1)
        return output_file, veg_mask, pixels_corrected, ndvi_file

    @staticmethod
    def apply_all_postprocessing(flood_detection_file, hls_file, dem_file, use_smart_aerosol=True):
        postproc_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(postproc_dir, exist_ok=True)

        with get_gdal_env():
            with rasterio.open(dem_file) as dem_src, \
                 rasterio.open(hls_file) as hls_src, \
                 rasterio.open(flood_detection_file) as flood_src:
                # Calculate slope
                slope_file = DEMDownloader.calculate_slope(dem_src)

                # Calculate SLIA
                slia_file = DEMDownloader.calculate_solar_incidence_angle_from_bands(dem_src, hls_src)

                # 1. Terrain shadow masking
                terrain_output, shadow_mask, terrain_pixels = DEMDownloader.postprocess_terrain_shadows(
                    flood_src, slia_file, slope_file
                )

                # 2. Smart aerosol filter (optional)
                with rasterio.open(terrain_output, 'r') as terrain_out:
                    if use_smart_aerosol:
                        aerosol_output, aerosol_mask, aerosol_pixels = DEMDownloader.postprocess_smart_aerosol_filter(
                            terrain_out, hls_src
                        )
                    else:
                        print("\nSkipping aerosol filter...")
                        aerosol_output = terrain_output
                        aerosol_mask = np.zeros_like(shadow_mask)
                        aerosol_pixels = 0

                # 3. Vegetation filter
                with rasterio.open(aerosol_output, 'r') as aerosol_src:
                    veg_output, veg_mask, veg_pixels, ndvi_file = DEMDownloader.postprocess_vegetation_filter(
                        aerosol_src, hls_src, dem_file
                    )

                # Final corrected file
                final_corrected = os.path.join(postproc_dir, flood_detection_file.replace('.tif', '_final_corrected.tif'))
                shutil.copy(veg_output, final_corrected)

        return final_corrected
