import earthaccess
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

from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window



BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "Fmask"]
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
    def __init__(self, date, bbox, layer='HLS'):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.date = date
        self.date_range = (f"{date}T00:00:00Z", f"{date}T23:59:59Z")
        self.layers = LAYERS[layer]
        self.bbox = bbox
        self.links = []

    def generate_digest(self, date, bbox):
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
        filenames = earthaccess.download(links, local_path=DATA_DOWNLOAD_DIR, threads=16)
        return filenames

    def generate_tiles(self, file_name, shape=(512,512), batch_size=1, overlap=0):
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

        output_name = f"{self.generate_digest(self.date, self.bbox)}.tif"
        if os.path.exists(output_name):
            print(f"File {output_name} already exists. Skipping merge.")
            return output_name

        # Open all band files and stack as separate bands
        srcs = [rasterio.open(f) for f in filenames]
        # Assume all files have same shape, transform, and CRS
        arrays = [src.read(1) for src in srcs]
        stacked = np.stack(arrays, axis=0) / 10000.0
        stacked = np.clip(stacked, 0, 1)
        out_meta = srcs[0].meta.copy()
        transform = srcs[0].transform
        out_meta.update({
            "height": stacked.shape[1],
            "width": stacked.shape[2],
            "transform": transform,
            "count": len(filenames),
            "dtype": 'float32'
        })
        # Crop to bbox before writing
        # Convert bbox from WGS84 to data CRS
        src_crs = 'EPSG:4326'  # WGS84
        dst_crs = out_meta['crs'] if 'crs' in out_meta else srcs[0].crs
        minx, miny, maxx, maxy = self.bbox

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        minx_t, miny_t = transformer.transform(minx, miny)
        maxx_t, maxy_t = transformer.transform(maxx, maxy)
        # Use the transform from the merged files (mosaic)
        window = from_bounds(minx_t, miny_t, maxx_t, maxy_t, transform)
        # Read window from stacked array
        # Calculate window indices
        row_start, row_stop = int(window.row_off), int(window.row_off + window.height)
        col_start, col_stop = int(window.col_off), int(window.col_off + window.width)
        cropped = stacked[:, row_start:row_stop, col_start:col_stop]
        # Update metadata for cropped output
        out_meta.update({
            "height": cropped.shape[1],
            "width": cropped.shape[2],
            "transform": rasterio.windows.transform(window, transform)
        })
        # Write cropped mosaic
        with rasterio.open(output_name, "w", **out_meta) as dst:
            for index in range(cropped.shape[0]):
                dst.write(cropped[index], index + 1)
        # Close all sources
        for s in srcs:
            s.close()
        return output_name

    def find_links(self):
        granules = search_data(
            short_name=self.layers,
            temporal=self.date_range,
            bounding_box=self.bbox,
            cloud_hosted=True,
            count=1000
        )
        merged_files = []
        for granule in granules:
            all_bands_available = False
            links = [link
                for band in BANDS for link in granule.data_links(access='external')
                   if f".{band}." in link
            ]
            if all(band in ' '.join(links) for band in BANDS):
                filenames = self.download_bands(links)
                merged_files.append(self.merge_bands(filenames))
        return merged_files


# Update download process: (one branch)
# 1. Update downloader to use earthaccess, and multi threads for faster download
# 2. Use already existing RDS to add new database for this usecase
# 3. setup sqlalchemy and alembic for migration (along with fast api)
# 4. Update current inference process to use the new downloader
#     1. each inference request is registered in database
#     2. once downloaded, the list of files are then kept track of in redis with a timeout of 30 days.
#     3. Once timedout, those files will be removed.
#     4. inference is stored in the database (s3 link along with the geojson*) *contemplating if geojson is needed


# start_time = time.time()
# earthaccess.login()

# sub_selected = []
# for link in links:
#         if any(b in link for b in BANDS):
#             sub_selected.append(link)

# filtered_granules = [g for g in granules if any(b in g.data_links()[0] for b in BANDS)]
# os.makedirs(LOCAL_DIR, exist_ok=True)
# for g in granules:
#     try:
#         download([g], LOCAL_DIR)
#     except Exception as e:
#         print(f"Download error: {e}")
# band_files = {b: [] for b in BANDS}
# for f in os.listdir(LOCAL_DIR):
#     for b in BANDS:
#         if f.endswith(f"{b}.tif"):
#             band_files[b].append(os.path.join(LOCAL_DIR, f))
# mosaics = []
# out_meta = None
# for b in BANDS:
#     srcs = [rasterio.open(f) for f in band_files[b]]
#     mosaic, transform = merge(srcs, method='first')
#     mosaics.append(mosaic[0])
#     if out_meta is None:
#         out_meta = srcs[0].meta.copy()
#         out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform, "count": len(BANDS)})
#     for s in srcs:
#         s.close()
# stacked = np.stack(mosaics, axis=0) / 10000.0
# stacked = np.clip(stacked, 0, 1)
# with rasterio.open(OUTPUT_NAME, "w", **out_meta) as dst:
#     dst.write(stacked)
# with rasterio.open(OUTPUT_NAME) as src:
#     row_off = (src.height - CHUNK_SIZE) // 2
#     col_off = (src.width - CHUNK_SIZE) // 2
#     window = Window(col_off, row_off, CHUNK_SIZE, CHUNK_SIZE)
#     data = src.read(window=window)
#     transform = src.window_transform(window)
#     out_meta.update({"height": CHUNK_SIZE, "width": CHUNK_SIZE, "transform": transform})
#     with rasterio.open(OUTPUT_NAME, "w", **out_meta) as dst:
#         dst.write(data)
# print(f"Single-threaded time: {time.time() - start_time:.2f}s")
