import os
import time
import requests
import numpy as np
import rasterio
from rasterio.env import Env
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel
from osgeo import gdal

gdal.UseExceptions()

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", "/root/.cache/")
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"

# Enhanced GDAL configuration for rasterio operations
GDAL_CONFIG = {
    # Memory and caching - increased for better performance
    "GDAL_CACHEMAX": 8192,  # MB of cache for GDAL operations (increased from 512)
    "GDAL_WARP_MEMORY_LIMIT": 1073741824,  # 1GB warp memory limit
    "CPL_VSIL_CURL_CACHE_SIZE": 200000000,  # 200MB for network file cache
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": True,  # Optimize HTTP range requests
    "GDAL_HTTP_MULTIPLEX": True,  # Enable HTTP/2 multiplexing
    "GDAL_HTTP_VERSION": 2,  # Use HTTP/2 for better performance
    # Disable auxiliary files that slow down operations
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # Avoid scanning directories
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",  # Only look for TIF files
    "GDAL_TIFF_INTERNAL_MASK": True,  # Use internal mask for better performance
    # Performance optimizations
    "GDAL_NUM_THREADS": "ALL_CPUS",  # Use all available CPU cores
    "GDAL_MAX_DATASET_POOL_SIZE": 450,  # Connection pool for datasets
    "GDAL_TIFF_OVR_BLOCKSIZE": 256,  # Optimize overview block size
    "GDAL_TIFF_DIRECT_IO": True,  # Enable direct I/O for better performance
    # Network optimizations
    "CPL_CURL_VERBOSE": False,  # Reduce logging overhead
    "GDAL_HTTP_TIMEOUT": 60,  # Timeout for HTTP requests
    "GDAL_HTTP_CONNECTTIMEOUT": 30,  # Connection timeout
    # Error handling
    "CPL_LOG": "/tmp/gdal.log",  # Log GDAL errors
    "CPL_DEBUG": False,  # Disable debug logging for performance
}

# Common profile updates for compressed tiled output - optimized
COMPRESSION_PROFILE = {
    "compress": "DEFLATE",
    "tiled": True,
    "blockxsize": 256,  # Reduced from 512 for better memory efficiency
    "blockysize": 256,  # Reduced from 512 for better memory efficiency
    "NUM_THREADS": "ALL_CPUS",  # Multi-threaded compression
    "BIGTIFF": "IF_SAFER",  # Auto-detect BigTIFF need
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
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{lat_prefix}{abs(lat):02d}_00_{lon_prefix}{abs(lon):03d}_00_DEM"

    def _download_tile(self, url, output_file):
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
            base_filename = f"{west}_{south}_{east}_{north}".replace(".", "_").replace(
                "-", "m"
            )
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
                dstSRS="EPSG:4326",
                format="GTiff",
                cutlineWKT=wkt_poly,
                cropToCutline=True,
                dstNodata=-9999,
                resampleAlg="bilinear",
                multithread=True,
                warpMemoryLimit=1073741824,  # 1GB
                creationOptions=[
                    "TILED=YES",
                    "BLOCKXSIZE=256",
                    "BLOCKYSIZE=256",
                    "NUM_THREADS=ALL_CPUS",
                    "BIGTIFF=IF_SAFER",
                    "COMPRESS=DEFLATE",
                ],
            )

            # Add width/height if specified
            if width and height:
                warp_options = gdal.WarpOptions(
                    dstSRS="EPSG:4326",
                    format="GTiff",
                    cutlineWKT=wkt_poly,
                    cropToCutline=True,
                    dstNodata=-9999,
                    resampleAlg="bilinear",
                    multithread=True,
                    warpMemoryLimit=1073741824,
                    width=width,
                    height=height,
                    creationOptions=[
                        "TILED=YES",
                        "BLOCKXSIZE=256",
                        "BLOCKYSIZE=256",
                        "NUM_THREADS=ALL_CPUS",
                        "BIGTIFF=IF_SAFER",
                        "COMPRESS=DEFLATE",
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
    def calculate_slope(dem_data, dem_transform, dem_bounds):
        """
        Calculate slope from DEM array. Returns slope in degrees as numpy array.
        """
        lat_center = (dem_bounds.bottom + dem_bounds.top) / 2
        pixel_size_x = abs(dem_transform[0]) * 111000 * np.cos(np.radians(lat_center))
        pixel_size_y = abs(dem_transform[4]) * 111000

        # Vectorized slope calculation
        dz_dx = sobel(dem_data, axis=1) / (8 * pixel_size_x)
        dz_dy = sobel(dem_data, axis=0) / (8 * pixel_size_y)

        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        return slope_deg, dz_dx, dz_dy

    @staticmethod
    def save_slope_file(slope_deg, dem_src_name, dem_profile):
        """Save slope array to file for user visualization."""
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
                dst.write(slope_deg.astype(rasterio.float32), 1)

        return slope_file

    @staticmethod
    def calculate_solar_incidence_angle(
        dz_dx,
        dz_dy,
        dem_data,
        sza_data,
        saa_data,
        dem_transform,
        dem_crs,
        hls_transform,
        hls_crs,
    ):
        """
        Calculate solar incidence angle from pre-computed gradients. Returns SLIA in degrees.
        """
        # Normal vector calculations
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(dem_data)

        # Vectorized normalization
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        # Only create resampled arrays if needed
        if sza_data.shape != dem_data.shape:
            sza_resampled = np.zeros_like(dem_data, dtype=np.float32)
            saa_resampled = np.zeros_like(dem_data, dtype=np.float32)

            reproject(
                source=sza_data,
                destination=sza_resampled,
                src_transform=hls_transform,
                src_crs=hls_crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=saa_data,
                destination=saa_resampled,
                src_transform=hls_transform,
                src_crs=hls_crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
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
        return incidence_angle

    @staticmethod
    def save_slia_file(slia_deg, dem_src_name, dem_profile):
        """Save SLIA array to file for user visualization."""
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
    def extract_aerosol_level(fmask_data):
        """Extract aerosol level from fmask data. Returns aerosol level array."""
        return (fmask_data >> 6) & 3

    @staticmethod
    def apply_terrain_shadow_mask(
        flood_data, slia_data, slope_data, slia_threshold=85, slope_threshold=15
    ):
        """
        Apply terrain shadow mask to flood data in-place.
        Returns the shadow mask for diagnostics.
        """
        shadow_mask = (slia_data > slia_threshold) & (slope_data > slope_threshold)
        flood_data[(flood_data == 1) & shadow_mask] = 0
        return shadow_mask

    @staticmethod
    def apply_aerosol_mask(flood_data, green_data, nir_data, fmask_data):
        """
        Apply smart aerosol filter to flood data in-place.
        Returns the aerosol mask for diagnostics.
        """
        aerosol_level = (fmask_data >> 6) & 3

        # NDWI calculation
        ndwi = np.zeros_like(green_data, dtype=np.float32)
        valid = (green_data + nir_data) != 0
        np.divide(green_data - nir_data, green_data + nir_data, out=ndwi, where=valid)

        aerosol_mask = (
            (flood_data == 1) & (aerosol_level >= 2) & (ndwi < 0) & (nir_data < 500)
        )
        flood_data[aerosol_mask] = 0
        return aerosol_mask

    @staticmethod
    def calculate_ndvi(red_data, nir_data):
        """Calculate NDVI from red and NIR bands. Returns (ndvi, valid_mask)."""
        ndvi = np.zeros_like(red_data, dtype=np.float32)
        valid_pixels = (red_data + nir_data) != 0
        np.divide(
            nir_data - red_data, nir_data + red_data, out=ndvi, where=valid_pixels
        )
        return ndvi, valid_pixels

    @staticmethod
    def apply_vegetation_mask(flood_data, ndvi, ndvi_threshold=0.7):
        """
        Apply vegetation filter to flood data in-place.
        Returns the vegetation mask for diagnostics.
        """
        veg_mask = (ndvi > ndvi_threshold) & (flood_data == 1)
        flood_data[veg_mask] = 0
        return veg_mask

    @staticmethod
    def save_ndvi_file(ndvi, valid_pixels, dem_filename, flood_profile):
        """Save NDVI array to file for user visualization."""
        ndvi_file = dem_filename.replace("_dem_clipped.tif", "_ndvi.tif")
        ndvi_profile = flood_profile.copy()
        ndvi_profile.update(dtype="float32", nodata=-999, **COMPRESSION_PROFILE)

        ndvi_out = ndvi.copy()
        ndvi_out[~valid_pixels] = -999

        with get_gdal_env():
            with rasterio.open(ndvi_file, "w", **ndvi_profile) as dst:
                dst.write(ndvi_out, 1)
        return ndvi_file

    @staticmethod
    def apply_all_postprocessing(
        flood_detection_file, hls_file, dem_file, use_smart_aerosol=True
    ):
        """
        Postprocessing pipeline that operates in-memory.
        """
        postproc_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(postproc_dir, exist_ok=True)

        _t0 = time.time()
        with get_gdal_env():
            with (
                rasterio.open(dem_file) as dem_src,
                rasterio.open(hls_file) as hls_src,
                rasterio.open(flood_detection_file) as flood_src,
            ):
                # Read all required data once
                dem_data = dem_src.read(1)
                dem_transform = dem_src.transform
                dem_bounds = dem_src.bounds
                dem_crs = dem_src.crs
                dem_src_name = dem_src.name
                dem_profile = dem_src.profile.copy()

                flood_data = flood_src.read(1).copy()  # Copy since we modify in-place
                flood_profile = flood_src.profile.copy()
                flood_transform = flood_src.transform
                flood_crs = flood_src.crs

                # HLS bands
                green_data = hls_src.read(2).astype(np.float32)
                red_data = hls_src.read(3).astype(np.float32)
                nir_data = hls_src.read(4).astype(np.float32)
                fmask_data = hls_src.read(7).astype(np.uint32)
                sza_data = hls_src.read(8)
                saa_data = hls_src.read(9)
                hls_transform = hls_src.transform
                hls_crs = hls_src.crs
                print(f"  [postproc] file reads:       {time.time() - _t0:.3f}s")

                _t1 = time.time()
                slope_deg, dz_dx, dz_dy = DEMDownloader.calculate_slope(
                    dem_data, dem_transform, dem_bounds
                )

                slia_deg = DEMDownloader.calculate_solar_incidence_angle(
                    dz_dx,
                    dz_dy,
                    dem_data,
                    sza_data,
                    saa_data,
                    dem_transform,
                    dem_crs,
                    hls_transform,
                    hls_crs,
                )
                print(f"  [postproc] slope+SLIA calc:  {time.time() - _t1:.3f}s")

                _t2 = time.time()
                if slope_deg.shape != flood_data.shape:
                    # Stack slope+SLIA into a single 2-band reproject call instead of two
                    stacked_dem_bands = np.stack(
                        [slope_deg.astype(np.float32), slia_deg.astype(np.float32)],
                        axis=0,
                    )
                    resampled_dem_bands = np.zeros(
                        (2, flood_data.shape[0], flood_data.shape[1]), dtype=np.float32
                    )
                    reproject(
                        source=stacked_dem_bands,
                        destination=resampled_dem_bands,
                        src_transform=dem_transform,
                        src_crs=dem_crs,
                        dst_transform=flood_transform,
                        dst_crs=flood_crs,
                        resampling=Resampling.bilinear,
                    )
                    slope_deg, slia_deg = resampled_dem_bands[0], resampled_dem_bands[1]

                if green_data.shape != flood_data.shape:
                    flood_height, flood_width = flood_data.shape
                    hls_bands_stacked = np.stack(
                        [green_data, red_data, nir_data, fmask_data.astype(np.float32)],
                        axis=0,
                    )
                    reprojected_bands = np.zeros(
                        (4, flood_height, flood_width), dtype=np.float32
                    )

                    reproject(
                        source=hls_bands_stacked,
                        destination=reprojected_bands,
                        src_transform=hls_transform,
                        src_crs=hls_crs,
                        dst_transform=flood_transform,
                        dst_crs=flood_crs,
                        resampling=Resampling.bilinear,
                    )
                    green_data = reprojected_bands[0]
                    red_data = reprojected_bands[1]
                    nir_data = reprojected_bands[2]
                    fmask_data = reprojected_bands[3].astype(np.uint32)
                print(f"  [postproc] reproject:        {time.time() - _t2:.3f}s")

                _t3 = time.time()
                # Terrain shadow mask
                DEMDownloader.apply_terrain_shadow_mask(flood_data, slia_deg, slope_deg)

                # Smart aerosol filter
                if use_smart_aerosol:
                    DEMDownloader.apply_aerosol_mask(
                        flood_data, green_data, nir_data, fmask_data
                    )

                # NDVI calculation and vegetation filter
                ndvi, valid_ndvi = DEMDownloader.calculate_ndvi(red_data, nir_data)
                DEMDownloader.apply_vegetation_mask(flood_data, ndvi)
                print(f"  [postproc] mask operations:  {time.time() - _t3:.3f}s")

            # All rasterio handles are closed. Write the final prediction and the three
            # diagnostic artifacts in parallel — they are independent and each involves
            # DEFLATE compression, so parallelizing cuts wall time to ~1x instead of ~3x.
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
                return final_corrected

            def _write_slope():
                return DEMDownloader.save_slope_file(
                    slope_deg, dem_src_name, dem_profile
                )

            def _write_slia():
                return DEMDownloader.save_slia_file(slia_deg, dem_src_name, dem_profile)

            def _write_ndvi():
                return DEMDownloader.save_ndvi_file(
                    ndvi, valid_ndvi, dem_file, flood_profile
                )

            _t4 = time.time()
            with ThreadPoolExecutor(max_workers=4) as pool:
                f_final = pool.submit(_write_final)
                f_slope = pool.submit(_write_slope)
                f_slia = pool.submit(_write_slia)
                f_ndvi = pool.submit(_write_ndvi)
                final_corrected = f_final.result()
                slope_file = f_slope.result()
                slia_file = f_slia.result()
                ndvi_file = f_ndvi.result()
            print(f"  [postproc] parallel writes:  {time.time() - _t4:.3f}s")
            print(f"  [postproc] total:            {time.time() - _t0:.3f}s")

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
