import earthaccess
import hashlib
import json
import morecantile
import numpy as np
import os
import rasterio
import requests
import geopandas as gpd
from shapely.geometry import box
import time

from earthaccess import search_data, download
from pyproj import Transformer

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window


BANDS = {
    "HLSL30": ["B02", "B03", "B04", "B08", "B11", "B12", "Fmask", "SAA", "SZA"],
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

class Downloader:
    def __init__(self, date, bbox, layers=LAYERS['HLS']):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.date = date
        self.date_range = (f"{date}T00:00:00Z", f"{date}T23:59:59Z")
        self.layers = layers
        self.bbox = bbox
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

    def merge_bands(self, filenames):
        """
        Merge input files into a single 7-band TIFF, cropped to the bbox.
        Args:
            filenames: list of file paths for each band
            output_name: output file name
        """

        output_name = f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(self.date, self.bbox)}.tif"
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
            "height": stacked.shape[1],
            "width": stacked.shape[2],
            "transform": transform,
            "count": len(filenames),
            "dtype": 'float32'
        })
        # Reproject the stacked array to EPSG:4326
        dst_crs = 'EPSG:4326'
        dst_transform, dst_width, dst_height = calculate_default_transform(
            srcs[0].crs, dst_crs, stacked.shape[2], stacked.shape[1], *srcs[0].bounds
        )

        reprojected = np.empty((len(filenames), dst_height, dst_width), dtype=np.float32)

        reproject(
            source=stacked.astype(np.float32),
            destination=reprojected,
            src_transform=transform,
            src_crs=srcs[0].crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        out_meta.update({
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "crs": dst_crs,
            "count": len(filenames)
        })

        with rasterio.open(output_name, "w", **out_meta) as dst:
            dst.write(reprojected)
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
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs='EPSG:4326')

            # Reproject bbox to raster CRS
            bbox_reproj = bbox_gdf.to_crs(src.crs)
            bounds = bbox_reproj.total_bounds
            minx_t, miny_t, maxx_t, maxy_t = bounds

            # Create window from bounds
            window = from_bounds(minx_t, miny_t, maxx_t, maxy_t, src.transform)

            # Read window
            data = src.read(window=window)
            window_transform = rasterio.windows.transform(window, src.transform)
            print(f"Cropped data shape: {data.shape}")
            # Resample to 512x512
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, data.shape[2], data.shape[1],
                minx_t, miny_t, maxx_t, maxy_t,
                dst_width=WIDTH, dst_height=HEIGHT
            )

            out_meta = src.meta.copy()
            out_meta.update({
                "height": height,
                "width": width,
                "transform": transform
            })

            with rasterio.open(output_name, "w", **out_meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=data[i - 1],
                        destination=rasterio.band(dst, i),
                        src_transform=window_transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )

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
            'height': mosaic.shape[0],
            'width': mosaic.shape[1],
            'transform': transform,
            'count': mosaic.shape[0],
            'dtype': mosaic.dtype,
            'crs': crs # Assuming default CRS if not provided
        }
        dst_crs = 'EPSG:4326'

        with MemoryFile() as memfile:
            with memfile.open(**src_profile) as src:
                for band in range(mosaic.shape[0]):
                    src.write(mosaic[band], band + 1)

                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )

                dst_profile = src.profile.copy()
                dst_profile.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })

            with MemoryFile() as dst_memfile, memfile.open() as src:
                with dst_memfile.open(**dst_profile) as dst:
                    for band in range(1, src.count + 1):
                        reproject(
                            source=src.read(band),
                            destination=rasterio.band(dst, band),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )
                print(f"Saving reprojected file to {filename}", flush=True)
                print(f"Reprojected raster profile: {dst.profile}", flush=True)
                with dst_memfile.open() as reprojected_raster:
                    with rasterio.open(filename, 'w', **reprojected_raster.profile) as out_raster:
                        out_raster.write(reprojected_raster.read())
        return filename

    def find_and_prepare_data(self):
        # TODO:
        # will also need to update to have timeseries support as needed.
        merged_files = []
        all_granules = []
        for layer in self.layers:
            granules = search_data(
                short_name=layer,
                temporal=self.date_range,
                bounding_box=tuple(map(float, self.bbox)),
                cloud_hosted=True,
                count=1000
            )
            all_granules.extend(granules)
        for granule in all_granules:
            all_bands_available = False
            links = [link
                for band in BANDS[layer] for link in granule.data_links(access='external')
                if f".{band}." in link
            ]
            if all(band in ' '.join(links) for band in BANDS[layer]):
                filenames = self.download_bands(links)
                merged_file = self.merge_bands(filenames)
                merged_files.append(merged_file)
        # stitch together multiple merged files
        mosaic, transform = merge(merged_files, method='first')
        with rasterio.open(merged_files[0], 'r') as src:
            crs = src.crs
        merged_file = self.save_cog(mosaic, transform, f"{DOWNLOAD_FOLDER.rstrip('/')}/{Downloader.generate_digest(self.date, self.bbox)}_merged.tif", crs)
        cropped_file = self.crop_to_bbox(merged_file)

        return cropped_file
