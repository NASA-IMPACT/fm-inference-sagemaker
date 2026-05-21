import morecantile
import os
import rasterio
import requests

from multiprocessing import Pool, cpu_count

BASE_URL = "https://openveda.cloud/api/titiler-cmr"
PROJECTION = "WorldCRS84Quad"  # EPSG:4326 tile matrix set
TILE_ENDPOINT = f"{BASE_URL}/rasterio/tiles/{PROJECTION}/{{z}}/{{x}}/{{y}}.tif"

# CMR collection concept ids for HLS v2.0 (LP DAAC cloud archive)
COLLECTION_CONCEPT_ID = {
    "HLSL30": "C2021957657-LPCLOUD",
    "HLSS30": "C2021957295-LPCLOUD",
}

# Band composition (kept identical to the previous titiler implementation)
ASSETS = {
    "HLSL30": ["B02", "B03", "B04", "B05", "B06", "B07"],
    "HLSS30": ["B02", "B03", "B04", "B8A", "B11", "B12"],
}

# Matches HLS asset tokens (B02..B12, B8A) in CMR asset filenames
ASSETS_REGEX = r"B[0-9][0-9A-Z]"

TMS = morecantile.tms.get(PROJECTION)
# z=11 in WorldCRS84Quad ≈ 38 m/px at the equator, closest match to HLS native 30 m.
ZOOM_LEVEL = 11
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../data")


WIDTH, HEIGHT = (224, 224)

class Downloader:
    def __init__(self, date, layer="HLSL30"):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.layer = layer
        self.date = date
        self.collection_concept_id = COLLECTION_CONCEPT_ID[layer]
        self.assets = ASSETS[layer]
        self.temporal = f"{self.date}T00:00:00Z/{self.date}T23:59:59Z"

    def tile_params(self):
        params = [
            ("collection_concept_id", self.collection_concept_id),
            ("temporal", self.temporal),
            ("assets_regex", ASSETS_REGEX),
        ]
        params.extend(("assets", asset) for asset in self.assets)
        return params

    def download_tile(self, x_index, y_index, filename):
        if os.path.exists(filename):
            return filename
        response = requests.get(
            TILE_ENDPOINT.format(z=ZOOM_LEVEL, x=x_index, y=y_index),
            params=self.tile_params(),
        )
        if response.status_code != 200:
            return ""

        # Titiler returns the 6 spectral bands followed by an alpha/mask band.
        # Strip the alpha band by re-reading and writing only the first 6.
        tmp_path = filename + ".tmp"
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(response.content)
        with rasterio.open(tmp_path) as src:
            profile = src.profile.copy()
            data = src.read(indexes=[1, 2, 3, 4, 5, 6])
        profile["count"] = 6
        with rasterio.open(filename, "w", **profile) as dst:
            dst.write(data)
        os.remove(tmp_path)
        return filename

    def mkdir(self, foldername):
        if not (os.path.exists(foldername)):
            os.makedirs(foldername)

    def download_tiles(self, bounding_box):
        x_tiles, y_tiles = self.tile_indices(bounding_box)
        downloaded_files = list()
        tile_infos = list()
        for x_index in range(x_tiles[0], x_tiles[1] + 1):
            for y_index in range(y_tiles[0], y_tiles[1] + 1):
                self.mkdir(f"{DOWNLOAD_FOLDER}/{self.layer}")
                filename = f"{DOWNLOAD_FOLDER}/{self.layer}/{self.date}-{x_index}-{y_index}.tif"
                tile_infos.append((x_index, y_index, filename))
        # parallelize download here
        pool = Pool(cpu_count() - 1)
        downloaded_files = pool.starmap(self.download_tile, tile_infos)
        downloaded_files = [
            downloaded_file for downloaded_file in downloaded_files if downloaded_file
        ]
        pool.close()
        pool.join()
        return downloaded_files

    def tile_indices(self, bounding_box):
        """
            Extract tile indices based on bounding_box

        Args:
            bounding_box (list): [left, down, right, top]

        Returns:
            list: [[start_x, end_x], [start_y, end_y]]
        """
        start_x, start_y, _ = TMS.tile(bounding_box[0], bounding_box[3], ZOOM_LEVEL)
        end_x, end_y, _ = TMS.tile(bounding_box[2], bounding_box[1], ZOOM_LEVEL)
        return [[start_x, end_x], [start_y, end_y]]
