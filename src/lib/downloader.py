import datetime
import multiprocessing
import earthaccess
import geopandas as gpd
import hashlib
import json
import morecantile
import numpy as np
import os
import rasterio
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed



from earthaccess import search_data, download
from pyproj import Transformer

from shapely.geometry import box

from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window
from rasterio.mask import mask


BANDS = {
    "HLSL30": ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask", "SAA", "SZA"],
    "HLSS30": ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask", "SAA", "SZA"],
}

LAYERS = {
    'HLS': ['HLSS30', 'HLSL30'],
    'MERRA2': ['M2T1NXSLV', 'M2T1NXLND']
}

PROJECTION = "WebMercatorQuad"
TMS = morecantile.tms.get(PROJECTION)
ZOOM_LEVEL = 12
DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", '/root/.cache/')


WIDTH, HEIGHT = (512, 512)
DELTA = 90

class Downloader:
    def __init__(self, dates, bbox, layers=LAYERS['HLS'], timeseries=False, process_workers=None, thread_workers=10):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.dates = self.prepare_dates(dates)
        self.layers = layers
        self.bbox = bbox
        self.timeseries = timeseries
        self.links = []
        self.process_workers = process_workers or min(len(self.dates), multiprocessing.cpu_count())
        self.thread_workers = thread_workers

    @staticmethod
    def generate_digest(date, bbox):
        """
        Create a digest (hash) based on the combination of date and bounding box.
        Returns:
            str: Hex digest string
        """
        # Use date and bbox as string
        key = f"{date}|{','.join(map(str, bbox))}"
        digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return digest

    def prepare_start_end_date(self, date):
        return (f"{date}T00:00:00Z", f"{date}T23:59:59Z")

    def prepare_dates(self, dates):
        date_list = []
        if ':' in dates:
            start_date, end_date = [_date.strip() for _date in dates.split(':')]
            # validate date format
            date_list.append(start_date)
            end_date_str = start_date
            try:
                while(end_date_str != end_date):
                    end_date_str = (datetime.datetime.strptime(end_date_str, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    date_list.append(end_date_str)
            except ValueError:
                raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        elif ',' in dates:
            date_list = dates.split(',')
            for date in date_list:
                try:
                    datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        else:
            date_list = [dates]
            try:
                datetime.datetime.strptime(dates, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        return date_list

    def login(self):
        self.auth = earthaccess.login(strategy="environment")

    def download_band(self, band, filename):
        # download one band
        pass

    def mkdir(self, foldername):
        if not (os.path.exists(foldername)):
            os.makedirs(foldername)

    def download_bands(self, links):
        filenames = earthaccess.download(links, local_path=DOWNLOAD_FOLDER, threads=16)
        return filenames

    def generate_tiles(self, file_name, shape=(512,512), batch_size=1, overlap=0, scale=False):
        """
        Generate tiles of given shape from the input file.
        Yields (tile_array, window, tile_index) for each tile.
        Args:
            file_name: path to the multi-band raster file
            shape: (height, width) of each tile
        """
        height, width = shape
        with rasterio.open(file_name) as src:
            step_y = height - overlap
            step_x = width - overlap
            nrows = max(1, (src.height - overlap) // step_y)
            ncols = max(1, (src.width - overlap) // step_x)
            batch = []
            for i in range(nrows):
                for j in range(ncols):
                    row_off = i * step_y
                    col_off = j * step_x
                    window = Window(col_off, row_off, width, height)
                    # Calculate actual window shape
                    win_height = min(height, src.height - row_off)
                    win_width = min(width, src.width - col_off)
                    tile = src.read(window=Window(col_off, row_off, win_width, win_height))
                    # Zero pad if needed
                    if win_height < height or win_width < width:
                        pad_shape = (tile.shape[0], height, width)
                        padded = np.zeros(pad_shape, dtype=tile.dtype)
                        if scaled:
                            tile = tile / 10000.0
                            tile = np.clip(tile, 0, 1)
                        padded[:, :win_height, :win_width] = tile
                        tile = padded
                    # Prepare metadata for memory file
                    meta = src.meta.copy()
                    meta.update({
                        "height": height,
                        "width": width
                    })
                    # Calculate correct transform for padded window
                    base_transform = src.window_transform(Window(col_off, row_off, win_width, win_height))
                    # Assign transform for the window (same for padded and non-padded)
                    meta["transform"] = base_transform
                    memfile = MemoryFile()
                    with memfile.open(**meta) as dst:
                        dst.write(tile)
                    batch.append((memfile, (win_height, win_width), window, meta["transform"], (i, j)))
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch

    def merge_bands(self, filenames, date, uuid):
        """
        Merge input files into a single 7-band TIFF, cropped to the bbox.
        Args:
            filenames: list of file paths for each band
            output_name: output file name
        """
        output_name = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(date, self.bbox)}-{uuid}.tif"
        if os.path.exists(output_name):
            print(f"File {output_name} already exists. Skipping merge.")
            return output_name

        # Open all band files and stack as separate bands
        srcs = [rasterio.open(f) for f in filenames]
        # Assume all files have same shape, transform, and CRS
        arrays = [src.read(1) for src in srcs]
        stacked = np.stack(arrays, axis=0)
        out_meta = srcs[0].meta.copy()
        transform = srcs[0].transform

        out_meta.update({
            "driver": "GTiff",
            "count": len(filenames),
            'compress': 'lzw',  # Use a lossless compression
            'tiled': True,  # Required for COG,
            'blockxsize': 512,
            'blockysize': 512,
            'dtype': 'float32',
            'nodata': -9999
        })

        with rasterio.open(output_name, "w", **out_meta) as dst:
            dst.write(stacked)
        # Close all sources
        for s in srcs:
            s.close()
        return output_name

    def reproject_to_crs(self, src_file, dst_file, target_crs):
        """
        Reproject a raster file to a target CRS.
        Args:
            src_file: Source file path
            dst_file: Destination file path
            target_crs: Target CRS to reproject to
        """
        with rasterio.open(src_file) as src:
            # Calculate the transform and dimensions for the target CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            # Create the destination profile
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Reproject and save
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )

    def crop_to_bbox(self, filename):
        """
        Crop the input file to the bounding box and resample to 512x512.
        Args:
            filename: path to the input raster file
        Returns:
            str: path to the cropped and resampled file
        """
        output_name = f"{DOWNLOAD_FOLDER.rstrip('/')}/{filename.split('/')[-1].replace('.tif', '_cropped.tif')}"
        if os.path.exists(output_name):
            print(f"File {output_name} already exists. Skipping crop.")
            return output_name

        with rasterio.open(filename) as src:
            # Create bbox geometry in WGS84
            minx, miny, maxx, maxy = self.bbox
            bbox_geom = box(minx, miny, maxx, maxy)
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=CRS.from_epsg(4326))

            out_image, out_transform = mask(src, bbox_gdf.geometry, crop=True)

            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_name, "w", **out_meta) as dst:
                for i in range(1, src.count + 1):
                    dst.write(out_image[i - 1], i)

        return output_name

    def save_cog(self, mosaic, transform, filename, crs):
        """
        Reproject raster to EPSG:4326 and save as a file.
        Args:
            mosaic (np.ndarray): The raster data.
            transform (affine.Affine): The rasterio transform.
            filename (str): The output filename.
        """
        src_profile = {
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': transform,
            'count': mosaic.shape[0],
            'dtype': mosaic.dtype,
            'crs': crs,
            'nodata': -9999,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        }
        dst_crs = CRS.from_epsg(4326)

        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src:
                src.write(mosaic)

                # Calculate the optimal transform and dimensions for the destination
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

                # Create the destination profile
                dst_profile = src.profile.copy()
                dst_profile.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'nodata': src.nodata
                })

                # Write the reprojected data to the destination file
                with rasterio.open(filename, 'w', **dst_profile) as dst:
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=rasterio.band(dst, list(range(1, dst.count + 1))),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_nodata=dst.nodata,
                        resampling=Resampling.bilinear
                    )
        return filename

    def prepare_date_range(self, date, delta=DELTA):
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        start_time = date_obj + datetime.timedelta(days=delta)
        start_date = datetime.datetime.strftime(start_time, '%Y-%m-%d')
        return self.prepare_start_end_date(start_date)

    def prepare_merged_file(self, date_range, bbox, layers, empty=False, current_merged_file=None, cloud_cover=(0, 100)):
        # prepare merged file for given date range and bbox
        date = date_range[0].split('T')[0]
        output_filename = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(date, bbox)}_merged.tif"
        if os.path.exists(output_filename.replace('.tif', '_cropped.tif')):
            return output_filename.replace('.tif', '_cropped.tif')

        merged_files = []
        target_crs = None

        for layer in layers:
            try:
                granules = search_data(
                    short_name=layer,
                    temporal=date_range,
                    bounding_box=tuple(map(float, bbox)),
                    cloud_hosted=True,
                    cloud_cover=cloud_cover,
                    count=1000
                )
            except Exception as e:
                print(f"[{date}] Granule search exception for layer {layer}: {e}")
                granules = []
            if not granules:
                continue
            for granule in granules:
                all_bands_available = False
                links = [link
                    for band in BANDS[layer] for link in granule.data_links(access='external')
                    if f".{band}." in link
                ]
                if all(band in ' '.join(links) for band in BANDS[layer]):
                    filenames = self.download_bands(links)
                    merged_file = self.merge_bands(filenames, date, granule.uuid)

                    # Set target CRS from the first file
                    if target_crs is None:
                        with rasterio.open(merged_file) as src:
                            target_crs = src.crs

                    # Check if the file has the same CRS as target
                    with rasterio.open(merged_file) as src:
                        if src.crs != target_crs:
                            # Reproject to target CRS
                            reprojected_file = merged_file.replace('.tif', '_reprojected.tif')
                            self.reproject_to_crs(merged_file, reprojected_file, target_crs)
                            merged_files.append(reprojected_file)
                        else:
                            merged_files.append(merged_file)

        if not merged_files:
            if empty:
                # create empty file that has the same shapes and crs as current_merged_file
                if current_merged_file:
                    cropped_file = output_filename.replace('.tif', '_cropped.tif')
                    with rasterio.open(current_merged_file) as src:
                        meta = src.meta.copy()
                        meta.update({
                            "driver": "GTiff",
                            "count": src.count,
                            'compress': 'lzw',  # Use a lossless compression
                            'tiled': True,  # Required for COG,
                            'blockxsize': 512,
                            'blockysize': 512,
                            'dtype': 'float32',
                            'nodata': -9999
                        })
                        empty_data = np.zeros_like(src.read())
                        with rasterio.open(cropped_file, "w", **meta) as dst:
                            dst.write(empty_data)
                    return cropped_file
                else:
                    print("No current merged file provided for empty output.")
            return ''

        mosaic, transform = merge(merged_files, method='first')
        merged_file = self.save_cog(mosaic, transform, output_filename, target_crs)
        cropped_file = self.crop_to_bbox(merged_file)
        return cropped_file

    def find_first_available_file(self, base_date, buffer, delta=DELTA, direction=1, cloud_cover=(0, 200)):
        """Find the earliest (chronologically closest) available merged file going backward (-1) or forward (+1).
        direction: -1 for pre, +1 for post.
        Returns path or '' if nothing found within window.
        """
        min_delta = (delta - buffer) * direction
        max_delta = delta * direction
        if direction < 0:
            min_delta, max_delta = max_delta, min_delta
        base_dt = datetime.datetime.strptime(base_date, '%Y-%m-%d')
        print(f"Searching for {'pre' if direction==-1 else 'post'} date from {base_date}, range: {min_delta} to {max_delta}")
        for step in range(min_delta, max_delta + 1):
            candidate_dt = base_dt + datetime.timedelta(days=step)
            candidate_str = candidate_dt.strftime('%Y-%m-%d')
            date_range = self.prepare_start_end_date(candidate_str)
            path = self.prepare_merged_file(date_range, self.bbox, self.layers, cloud_cover=cloud_cover)
            if path:  # non-empty string means data found
                print('downloaded:', candidate_str, path, step)
                return path
        return ''

    def find_and_prepare_data(self):
        prepared_data = {}
        for date in self.dates:
            if self.timeseries:
                # First get current date file; if not present skip entirely
                current_date_range = self.prepare_start_end_date(date)
                current_cropped_file = self.prepare_merged_file(current_date_range, self.bbox, self.layers)
                if not current_cropped_file:
                    # Skip this date entirely as per requirement
                    continue
                # Find pre and post within window (earliest match)
                pre_cropped_file = self.find_first_available_file(date, buffer=15, delta=DELTA, direction=-1, cloud_cover=(0, 20))
                post_cropped_file = self.find_first_available_file(date, buffer=15, delta=DELTA, direction=1, cloud_cover=(0, 20))
                # If none found, create zero (empty) only then
                if not pre_cropped_file:
                    pre_cropped_file = self.prepare_merged_file(current_date_range, self.bbox, self.layers, empty=True, current_merged_file=current_cropped_file, cloud_cover=(0, 20))
                if not post_cropped_file:
                    post_cropped_file = self.prepare_merged_file(current_date_range, self.bbox, self.layers, empty=True, current_merged_file=current_cropped_file, cloud_cover=(0, 20))
                timeseries_files = [pre_cropped_file, current_cropped_file, post_cropped_file]
                # Build stack
                stacked_arrays = []
                for file in timeseries_files:
                    with rasterio.open(file) as src:
                        stacked_arrays.append(src.read())
                mosaic = np.concatenate(stacked_arrays, axis=0)
                output_filename = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(date, self.bbox)}_timeseries_merged.tif"
                with rasterio.open(current_cropped_file) as src_ref:
                    transform = src_ref.transform
                    crs = src_ref.crs
                merged_file = self.save_cog(mosaic, transform, output_filename, crs)
                cropped_file = self.crop_to_bbox(merged_file)
            else:
                cropped_file = self.prepare_merged_file(self.prepare_start_end_date(date), self.bbox, self.layers)
            prepared_data[date] = cropped_file
        return prepared_data
