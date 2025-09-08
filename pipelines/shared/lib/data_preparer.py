import math
import numpy as np
import os
import rasterio

from lib.consts import DOWNLOAD_FOLDER
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds, Window

SHAPE = (512, 512)

QA_INDICES = {
    'cloud': 1,
    'adjacent_cloud': 2,
    'shadow': 3,
    'snow': 4,
    'water': 5,
    'aerosol': 6
}

class DataPreparer:
    def __init__(self, filename, batch_size=120, overlap=0, scale=False, qa_flags=['cloud', 'shadow', 'snow', 'water']):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = overlap
        self.qa_flags = qa_flags
        self.scale = scale

    def generate_tiles(self):
        """
        Generate tiles of given shape from the input file.
        Yields (tile_array, window, tile_index) for each tile.
        Args:
            filename: path to the multi-band raster file
            shape: (height, width) of each tile
        """
        height, width = SHAPE
        with rasterio.open(self.filename) as src:
            step_y = height - self.overlap
            step_x = width - self.overlap
            nrows = max(1, math.ceil((src.height - self.overlap) / step_y))
            ncols = max(1, math.ceil((src.width - self.overlap) / step_x))
            batch = []
            for i in range(nrows):
                for j in range(ncols):
                    row_off = i * step_y
                    col_off = j * step_x
                    window = Window(col_off, row_off, width, height)
                    # Calculate actual window shape
                    win_height = min(height, src.height - row_off)
                    if win_height <= 0:
                        win_height = 0
                    win_width = min(width, src.width - col_off)
                    if win_width <= 0:
                        win_width = 0
                    window = Window(col_off, row_off, win_width, win_height)
                    tile = src.read(window=window)
                    if self.scale:
                        tile = tile / 10000.0
                        tile = np.clip(tile, 0, 1)

                    # Zero pad if needed
                    if win_height < height or win_width < width:
                        pad_shape = (tile.shape[0], height, width)
                        padded = np.zeros(pad_shape, dtype=tile.dtype)
                        padded[:, :win_height, :win_width] = tile
                        tile = padded

                    combined = np.zeros_like(tile).astype('uint')
                    for qa_flag in self.qa_flags:
                        qa_index = QA_INDICES.get(qa_flag)
                        flag = tile[6].astype('uint') & (1 << qa_index) != 0
                        combined |= flag
                    for index in range(6):
                        tile[index][combined] = 0.0001

                    # Prepare metadata for memory file
                    meta = src.meta.copy()
                    meta.update({
                        "height": height,
                        "width": width
                    })
                    # Calculate correct transform for padded window
                    base_transform = src.window_transform(window)
                    # Assign transform for the window (same for padded and non-padded)
                    meta["transform"] = base_transform
                    memfile = MemoryFile()
                    with memfile.open(**meta) as dst:
                        dst.write(tile)
                    batch.append(memfile)
                    if len(batch) == self.batch_size:
                        yield np.asarray(batch)
                        batch = []
            if batch:
                yield np.asarray(batch)
