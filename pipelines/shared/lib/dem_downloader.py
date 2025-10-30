import os
import requests
import numpy as np
import rasterio
import geopandas as gpd
import logging
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from scipy.ndimage import sobel
from shapely.geometry import box

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", '/root/.cache/')
URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"

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

    def download_dem_tiles(self, max_workers=5):
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
        try:
            west, south, east, north = self.bbox
            base_filename = f"{west}_{south}_{east}_{north}".replace('.', '_').replace('-', 'm')
            filename = f"{self.dem_dir}/{base_filename}_dem_clipped.tif"
            if os.path.exists(filename):
                return filename

            with rasterio.open(dem_tiles[0]) as src:
                out_meta = src.meta.copy()

            mosaic, out_transform = merge([rasterio.open(f) for f in dem_tiles])

            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
            })

            merged_dem_path = os.path.join(self.output_dir, f"{base_filename}_dem_merged.tif")
            with rasterio.open(merged_dem_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            clip_geom = gpd.GeoDataFrame({'geometry': [box(west, south, east, north)]}, crs='EPSG:4326')

            with rasterio.open(merged_dem_path) as src:
                out_image, out_transform = mask(src, clip_geom.geometry, crop=True)
                out_meta = src.meta.copy()

            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            })

            with rasterio.open(filename, "w", **out_meta) as dest:
                if width and height:
                    print(f"Reprojecting DEM to width: {width}, height: {height}")
                    reprojected_image = np.empty((height, width), dtype=out_image.dtype)
                    reproject(
                        source=out_image,
                        destination=reprojected_image,
                        src_transform=out_transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
                    dest.write(reprojected_image, 1)
                else:
                    dest.write(out_image)

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

        dz_dx = sobel(dem, axis=1) / (8 * pixel_size_x)
        dz_dy = sobel(dem, axis=0) / (8 * pixel_size_y)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        slope_path = f"{DOWNLOAD_FOLDER}/slopes"
        os.makedirs(slope_path, exist_ok=True)
        slope_file = os.path.join(slope_path, os.path.basename(dem_src.name).replace('_dem_clipped.tif', '_slope_degrees.tif'))
        profile = dem_src.profile.copy()
        profile.update(dtype=rasterio.float32)

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

        dz_dx = sobel(dem, axis=1) / (8 * pixel_size_x)
        dz_dy = sobel(dem, axis=0) / (8 * pixel_size_y)

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(dem)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        sza_data = hls_src.read(8)
        saa_data = hls_src.read(9)
        sza_resampled = np.zeros_like(sza_data, dtype=np.float32)
        saa_resampled = np.zeros_like(saa_data, dtype=np.float32)

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

        zenith_rad = np.radians(sza_data)
        azimuth_rad = np.radians(saa_data)

        sx = np.sin(zenith_rad) * np.sin(azimuth_rad)
        sy = np.sin(zenith_rad) * np.cos(azimuth_rad)
        sz = np.cos(zenith_rad)

        cos_incidence = np.clip(sx * nx + sy * ny + sz * nz, -1, 1)
        incidence_angle = np.degrees(np.arccos(cos_incidence))

        slia_file = os.path.join(output_dir, os.path.basename(dem_src.name).replace('_dem_clipped.tif', '_slia_degrees.tif'))
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
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        flood_transform = flood_src.transform

        with rasterio.open(slia_file) as slia_src, rasterio.open(slope_file) as slope_src:
            slia_data = slia_src.read(1)
            slope_data = slope_src.read(1)

        if slia_data.shape != flood_data.shape:
            slia_resampled = np.zeros_like(flood_data, dtype=np.float32)
            slope_resampled = np.zeros_like(flood_data, dtype=np.float32)

            with rasterio.open(slia_file) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=slia_resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=flood_transform,
                    dst_crs=flood_src.crs,
                    resampling=Resampling.bilinear
                )

            with rasterio.open(slope_file) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=slope_resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=flood_transform,
                    dst_crs=flood_src.crs,
                    resampling=Resampling.bilinear
                )
            slia_data, slope_data = slia_resampled, slope_resampled
        shadow_mask = (slia_data > slia_threshold) & (slope_data > slope_threshold)
        flood_corrected = flood_data.copy()
        pixels_corrected = np.sum((flood_data == 1) & shadow_mask)
        flood_corrected[shadow_mask & (flood_data == 1)] = 0

        output_file = os.path.join(output_dir, os.path.basename(flood_src.name).replace('.tif', '_flood_terrain_corrected.tif'))
        with rasterio.open(output_file, 'w', **flood_profile) as dst:
            dst.write(flood_corrected, 1)
        return output_file, shadow_mask, pixels_corrected

    @staticmethod
    def postprocess_smart_aerosol_filter(flood_src, hls_src):
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        nir_data = hls_src.read(4).astype(np.float32)
        green_data = hls_src.read(2).astype(np.float32)
        fmask_data = hls_src.read(7).astype(np.uint32)

        if nir_data.shape != flood_data.shape:
            print("Reprojecting HLS bands to match flood detection shape...", flood_src.name, hls_src.name)
            print("Shapes before reprojection:", nir_data.shape, green_data.shape, fmask_data.shape, flood_data.shape)
            print("Flood bounds:", flood_src.bounds)
            print("HLS bounds:", hls_src.bounds)
            print("Flood transform:", flood_src.transform)
            print("HLS transform:", hls_src.transform)

            # Use flood_data dimensions explicitly
            flood_height, flood_width = flood_data.shape
            print("Target reprojection shape:", flood_height, flood_width)

            try:
                # Stack all three bands for simultaneous reprojection
                # This ensures they all get exactly the same dimensions
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

                # Extract the reprojected bands
                green_data = reprojected_bands[0]
                nir_data = reprojected_bands[1]
                fmask_data = reprojected_bands[2].astype(np.uint32)

                print("Shapes after reprojection:", nir_data.shape, green_data.shape, fmask_data.shape)

                # Validate that reprojection worked correctly
                if (nir_data.shape != flood_data.shape or
                    green_data.shape != flood_data.shape or
                    fmask_data.shape != flood_data.shape):
                    raise ValueError(f"Reprojection failed: expected shape {flood_data.shape}, "
                                   f"got NIR: {nir_data.shape}, Green: {green_data.shape}, "
                                   f"FMask: {fmask_data.shape}")

            except Exception as e:
                print(f"Reprojection error: {e}")
                # Fallback: resize arrays if reprojection fails
                from scipy.ndimage import zoom

                height_ratio = flood_data.shape[0] / nir_data.shape[0]
                width_ratio = flood_data.shape[1] / nir_data.shape[1]

                print(f"Falling back to resizing with ratios: height={height_ratio:.4f}, width={width_ratio:.4f}")

                nir_data = zoom(nir_data.astype(np.float32), (height_ratio, width_ratio), order=1)
                green_data = zoom(green_data.astype(np.float32), (height_ratio, width_ratio), order=1)
                fmask_data = zoom(fmask_data.astype(np.uint32), (height_ratio, width_ratio), order=0)

                # Ensure exact shape match by cropping or padding if needed
                if nir_data.shape[0] != flood_data.shape[0] or nir_data.shape[1] != flood_data.shape[1]:
                    print(f"Adjusting final shapes from {nir_data.shape} to {flood_data.shape}")

                    # Create arrays of exact target size
                    nir_final = np.zeros(flood_data.shape, dtype=np.float32)
                    green_final = np.zeros(flood_data.shape, dtype=np.float32)
                    fmask_final = np.zeros(flood_data.shape, dtype=np.uint32)

                    # Copy data, handling potential size differences
                    h_end = min(nir_data.shape[0], flood_data.shape[0])
                    w_end = min(nir_data.shape[1], flood_data.shape[1])

                    nir_final[:h_end, :w_end] = nir_data[:h_end, :w_end]
                    green_final[:h_end, :w_end] = green_data[:h_end, :w_end]
                    fmask_final[:h_end, :w_end] = fmask_data[:h_end, :w_end]

                    nir_data = nir_final
                    green_data = green_final
                    fmask_data = fmask_final

                print(f"Final shapes after fallback: NIR: {nir_data.shape}, Green: {green_data.shape}, FMask: {fmask_data.shape}")
        else:
            print("Shapes already match, no reprojection needed:", nir_data.shape, flood_data.shape)

        aerosol_level = (fmask_data >> 6) & 3
        ndwi = np.zeros_like(green_data, dtype=np.float32)
        valid = (green_data + nir_data) != 0
        ndwi[valid] = (green_data[valid] - nir_data[valid]) / (green_data[valid] + nir_data[valid])

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
        with rasterio.open(output_file, 'w', **flood_profile) as dst:
            dst.write(flood_corrected, 1)
        return output_file, aerosol_mask, pixels_corrected

    @staticmethod
    def postprocess_vegetation_filter(flood_src, hls_src, dem_filename, ndvi_threshold=0.7):
        output_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(output_dir, exist_ok=True)
        flood_data = flood_src.read(1)
        flood_profile = flood_src.profile.copy()
        red_data = hls_src.read(3).astype(np.float32)
        nir_data = hls_src.read(4).astype(np.float32)

        ndvi = np.zeros_like(red_data, dtype=np.float32)
        valid_pixels = (red_data + nir_data) != 0
        ndvi[valid_pixels] = (nir_data[valid_pixels] - red_data[valid_pixels]) / (nir_data[valid_pixels] + red_data[valid_pixels])

        veg_mask = (ndvi > ndvi_threshold) & (flood_data == 1)
        flood_corrected = flood_data.copy()
        pixels_corrected = np.sum(veg_mask)
        flood_corrected[veg_mask] = 0

        output_file = os.path.join(output_dir, os.path.basename(flood_src.name).replace('.tif', '_flood_veg_corrected.tif'))
        with rasterio.open(output_file, 'w', **flood_profile) as dst:
            dst.write(flood_corrected, 1)

        # NDVI filename should be the same as dem filename but with _ndvi suffix
        ndvi_file = dem_filename.replace('_dem_clipped.tif', '_ndvi.tif')
        ndvi_profile = flood_profile.copy()
        ndvi_profile.update({'dtype': 'float32', 'nodata': -999})
        with rasterio.open(ndvi_file, 'w', **ndvi_profile) as dst:
            ndvi[~valid_pixels] = -999
            dst.write(ndvi, 1)
        return output_file, veg_mask, pixels_corrected, ndvi_file

    @staticmethod
    def apply_all_postprocessing(flood_detection_file, hls_file, dem_file, use_smart_aerosol=True):
        postproc_dir = f"{DOWNLOAD_FOLDER}/predictions/flood_detection/postprocessed"
        os.makedirs(postproc_dir, exist_ok=True)

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
