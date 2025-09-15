import datetime
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
    def __init__(self, dates, bbox, layers=LAYERS['HLS'], timeseries=False):
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
            start_date, end_date = dates.split(':')
            # validate date format
            end_date_str = start_date.strip()
            try:
                while(end_date_str != end_date.strip()):
                    end_date_str = (datetime.datetime.strptime(end_date_str, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    if end_date_str != end_date.strip():
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

    def prepare_merged_file(self, date_range, bbox, layers):
        # prepare merged file for given date range and bbox
        date = date_range[0].split('T')[0]
        output_filename = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(date, bbox)}_merged.tif"
        if os.path.exists(output_filename.replace('.tif', '_cropped.tif')):
            return output_filename

        merged_files = []

        for layer in layers:
            granules = search_data(
                short_name=layer,
                temporal=date_range,
                bounding_box=tuple(map(float, bbox)),
                cloud_hosted=True,
                count=1000
            )
            for granule in granules:
                all_bands_available = False
                links = [link
                    for band in BANDS[layer] for link in granule.data_links(access='external')
                    if f".{band}." in link
                ]
                if all(band in ' '.join(links) for band in BANDS[layer]):
                    filenames = self.download_bands(links)
                    merged_file = self.merge_bands(filenames, date, granule.uuid)
                    merged_files.append(merged_file)
        mosaic, transform = merge(merged_files, method='first')
        with rasterio.open(merged_files[0], 'r') as src:
            crs = src.crs
        merged_file = self.save_cog(mosaic, transform, output_filename, crs)
        cropped_file = self.crop_to_bbox(merged_file)
        return cropped_file

    def find_and_prepare_data(self):
        prepared_data = {}
        for date in self.dates:
            if self.timeseries:
                output_filename = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(date, self.bbox)}_timeseries_merged.tif"
                if os.path.exists(output_filename.replace('.tif', '_cropped.tif')):
                    return output_filename
                timeseries_files = []

                pre_date_range = self.prepare_date_range(date, delta=-DELTA)
                post_date_range = self.prepare_date_range(date, delta=DELTA)
                current_date_range = self.prepare_date_range(date, delta=DELTA)

                pre_cropped_file = self.prepare_merged_file(pre_date_range, self.bbox, self.layers)
                current_cropped_file = self.prepare_merged_file(current_date_range, self.bbox, self.layers)
                post_cropped_file = self.prepare_merged_file(post_date_range, self.bbox, self.layers)

                timeseries_files = [pre_cropped_file, current_cropped_file, post_cropped_file]
                print(pre_cropped_file, current_cropped_file, post_cropped_file)
                with rasterio.open(pre_cropped_file) as src, rasterio.open(current_cropped_file) as src2, rasterio.open(post_cropped_file) as src3:
                    print('shapes:', src.shape, src2.shape, src3.shape)
                stacked_arrays = []
                for file in timeseries_files:
                    with rasterio.open(file) as src:
                        stacked_arrays.append(src.read())
                mosaic = np.concatenate(stacked_arrays, axis=0)
                with rasterio.open(timeseries_files[0]) as src:
                    transform = src.transform
                    crs = src.crs
                merged_file = self.save_cog(mosaic, transform, output_filename, crs)
                cropped_file = self.crop_to_bbox(merged_file)
            else:
                cropped_file = self.prepare_merged_file(self.prepare_start_end_date(date), self.bbox, self.layers)
            prepared_data[date] = cropped_file
        return prepared_data
