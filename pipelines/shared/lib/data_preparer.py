import math
import numpy as np
import os
import rasterio
import torch

from lib.consts import DOWNLOAD_FOLDER
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds, Window
import rasterio.windows

SHAPE = (512, 512)

QA_INDICES = {
    'cloud': 1,
    'adjacent_cloud': 2,
    'shadow': 3,
    'snow': 4,
    'water': 5,
    'aerosol': 6
}

class ArrayPool:
    """Memory pool for reusing arrays to reduce allocation overhead."""

    def __init__(self, max_pool_size=50):
        self.pool = {}  # (shape, dtype) -> list of arrays
        self.max_pool_size = max_pool_size

    def get_array(self, shape, dtype):
        """Get an array from pool or create new one."""
        key = (shape, dtype)
        if key in self.pool and self.pool[key]:
            array = self.pool[key].pop()
            array.fill(0)  # Clear the array
            return array
        else:
            return np.zeros(shape, dtype=dtype)

    def return_array(self, array):
        """Return array to pool for reuse."""
        key = (array.shape, array.dtype)
        if key not in self.pool:
            self.pool[key] = []

        # Limit pool size to prevent excessive memory usage
        if len(self.pool[key]) < self.max_pool_size:
            self.pool[key].append(array)

    def clear(self):
        """Clear all arrays from pool."""
        self.pool.clear()

class DataPreparer:
    def __init__(self, filename, batch_size=120, overlap=0, scale=False, qa_flags=['cloud', 'shadow', 'snow', 'water'], timeseries=False):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.filename = filename
        # Set batch_size based on available GPU memory if possible
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)  # in MB
            # Example heuristic: use larger batch if GPU has more memory

            if gpu_mem >= 81152:
                self.batch_size = max(batch_size, 240)
            elif gpu_mem <= 81152:
                self.batch_size = max(batch_size, 120)
            else:
                self.batch_size = batch_size
        except Exception:
            self.batch_size = batch_size
        self.overlap = overlap
        self.qa_flags = qa_flags
        self.scale = scale
        self.timeseries = timeseries
        # Initialize memory pool for array reuse
        self._array_pool = ArrayPool(max_pool_size=batch_size * 2)

    def handle_qa(self, tile):
        def get_qa_mask_vectorized(qa_band):
            # Vectorized QA mask calculation using broadcasting
            if not self.qa_flags:
                return np.zeros_like(qa_band, dtype=bool)

            qa_indices = np.array([QA_INDICES.get(flag, 0) for flag in self.qa_flags], dtype=np.uint32)
            qa_band_uint = qa_band.astype(np.uint32)

            # Create shift values with proper shape for broadcasting
            shift_values = (1 << qa_indices).astype(np.uint32)  # Shape: (n_flags,)

            # Reshape for broadcasting: qa_band_uint[..., newaxis] has shape (..., 1)
            # shift_values has shape (n_flags,) - this should broadcast correctly
            flags = (qa_band_uint[..., np.newaxis] & shift_values[np.newaxis, ...]) != 0
            return np.any(flags, axis=-1)

        if self.timeseries:
            # Vectorized operations for timeseries
            pre_combined = get_qa_mask_vectorized(tile[6])
            for band_idx in range(6):
                tile[band_idx][pre_combined] = 0.0001

            combined = get_qa_mask_vectorized(tile[15])  # QA band for middle period
            for band_idx in range(9, 15):
                tile[band_idx][combined] = 0.0001

            post_combined = get_qa_mask_vectorized(tile[24])  # QA band for post period
            for band_idx in range(18, 24):
                tile[band_idx][post_combined] = 0.0001
        else:
            # Vectorized operations for single time
            combined = get_qa_mask_vectorized(tile[6])
            for band_idx in range(6):
                tile[band_idx][combined] = 0.0001
        return tile

    def _process_tile_vectorized(self, tile):
        """Vectorized tile processing combining scaling and QA operations."""
        # Apply scaling first if needed (in-place for memory efficiency)
        if self.scale:
            # Use in-place operations to minimize memory allocations
            tile = tile.astype(np.float32, copy=False)
            np.divide(tile, 10000.0, out=tile)
            np.clip(tile, 0, 1, out=tile)

        # Apply QA processing
        tile = self.handle_qa(tile)
        return tile

    def _pad_tile_optimized(self, tile, actual_height, actual_width, target_height, target_width):
        """Optimized tile padding with memory pool support."""
        if actual_height == target_height and actual_width == target_width:
            return tile

        # For small padding, use memory pool for better performance
        if target_height - actual_height <= 64 and target_width - actual_width <= 64:
            # Get padded array from pool
            target_shape = (tile.shape[0], target_height, target_width)
            padded = self._array_pool.get_array(target_shape, tile.dtype)
            # Copy original data
            padded[:, :actual_height, :actual_width] = tile
            return padded
        else:
            # For larger padding, use numpy.pad which is more memory efficient
            pad_width = [(0, 0), (0, target_height - actual_height), (0, target_width - actual_width)]
            return np.pad(tile, pad_width, mode='constant', constant_values=0)

    def _get_block_aligned_tiles(self, src, padding_info, block_height, block_width):
        """Group tiles by rasterio blocks for efficient I/O."""
        from collections import defaultdict

        # Group tiles by which block(s) they intersect with
        block_to_tiles = defaultdict(list)

        for idx, (i, j, row_off, col_off, win_height, win_width, needs_padding) in enumerate(padding_info):
            # Calculate which blocks this tile intersects
            start_block_row = row_off // block_height
            end_block_row = (row_off + win_height - 1) // block_height
            start_block_col = col_off // block_width
            end_block_col = (col_off + win_width - 1) // block_width

            # Add this tile to all blocks it intersects
            for block_row in range(start_block_row, end_block_row + 1):
                for block_col in range(start_block_col, end_block_col + 1):
                    block_key = (block_row, block_col)
                    block_to_tiles[block_key].append(idx)

        return block_to_tiles

    def _read_tiles_block_optimized(self, src, padding_info, block_to_tiles, height, width):
        """Read tiles using block-optimized strategy."""
        # Pre-allocate array to store processed tiles
        processed_tiles = [None] * len(padding_info)
        processed_blocks = set()

        # Process tiles grouped by blocks
        for block_key, tile_indices in block_to_tiles.items():
            if block_key in processed_blocks:
                continue

            # Read tiles that share this block efficiently
            for tile_idx in tile_indices:
                if processed_tiles[tile_idx] is not None:
                    continue  # Already processed

                i, j, row_off, col_off, win_height, win_width, needs_padding = padding_info[tile_idx]
                window = rasterio.windows.Window(col_off, row_off, win_width, win_height)

                # Read the tile
                tile = src.read(window=window)

                # Process the tile
                tile = self._process_tile_vectorized(tile)

                # Apply padding if needed
                if needs_padding:
                    tile = self._pad_tile_optimized(tile, win_height, win_width, height, width)
                else:
                    if tile.shape[1:] != (height, width):
                        tile = self._pad_tile_optimized(tile, win_height, win_width, height, width)

                processed_tiles[tile_idx] = tile

            processed_blocks.add(block_key)

        return processed_tiles

    def _return_tiles_to_pool(self, tiles):
        """Return tiles to memory pool after processing."""
        for tile in tiles:
            if tile is not None:
                # Only return padded tiles to pool (they're the ones we allocated)
                if tile.shape == (tile.shape[0], SHAPE[0], SHAPE[1]):
                    self._array_pool.return_array(tile)

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

            # Get block information for optimized I/O
            try:
                block_height, block_width = src.block_shapes[0]  # Assuming all bands have same block structure
            except (IndexError, AttributeError):
                # Fallback if block information is not available
                block_height, block_width = 512, 512

            # Pre-calculate metadata template to avoid copying for each tile
            meta_template = src.meta.copy()
            meta_template.update({
                "height": height,
                "width": width
            })

            # Pre-calculate padding requirements for all tiles
            padding_info = []
            for i in range(nrows):
                for j in range(ncols):
                    row_off = i * step_y
                    col_off = j * step_x
                    win_height = min(height, src.height - row_off)
                    win_width = min(width, src.width - col_off)

                    # Store padding info: (i, j, row_off, col_off, win_height, win_width, needs_padding)
                    needs_padding = win_height < height or win_width < width
                    padding_info.append((i, j, row_off, col_off, win_height, win_width, needs_padding))

            # Use block-optimized reading strategy for better I/O performance
            if len(padding_info) > 10 and block_height > 0 and block_width > 0:
                # For larger datasets, use block-aligned reading
                block_to_tiles = self._get_block_aligned_tiles(src, padding_info, block_height, block_width)
                processed_tiles = self._read_tiles_block_optimized(src, padding_info, block_to_tiles, height, width)
            else:
                # For smaller datasets, use direct reading (avoid overhead)
                processed_tiles = []
                for i, j, row_off, col_off, win_height, win_width, needs_padding in padding_info:
                    window = Window(col_off, row_off, win_width, win_height)
                    tile = src.read(window=window)
                    tile = self._process_tile_vectorized(tile)

                    if needs_padding:
                        tile = self._pad_tile_optimized(tile, win_height, win_width, height, width)
                    elif tile.shape[1:] != (height, width):
                        tile = self._pad_tile_optimized(tile, win_height, win_width, height, width)

                    processed_tiles.append(tile)

            # Collect tiles and transforms separately to reduce metadata overhead
            tiles_batch = []
            transforms_batch = []

            # Process tiles and transforms in batches
            for idx, (i, j, row_off, col_off, win_height, win_width, needs_padding) in enumerate(padding_info):
                tile = processed_tiles[idx]

                # Store tile and transform separately
                tiles_batch.append(tile)

                # Calculate and store only the transform (defer metadata creation)
                window = Window(col_off, row_off, win_width, win_height)
                base_transform = src.window_transform(window)
                transforms_batch.append(base_transform)

                # Create MemoryFiles only when batch is full
                if len(tiles_batch) == self.batch_size:
                    memfiles_batch = []
                    for tile_data, transform in zip(tiles_batch, transforms_batch):
                        # Create metadata only when needed for MemoryFile
                        meta = meta_template.copy()
                        meta["transform"] = transform
                        memfile = MemoryFile()
                        with memfile.open(**meta) as dst:
                            dst.write(tile_data)
                        memfiles_batch.append(memfile)

                    yield np.asarray(memfiles_batch)

                    # Return tiles to memory pool after yielding
                    self._return_tiles_to_pool(tiles_batch)
                    tiles_batch = []
                    transforms_batch = []

            # Handle remaining tiles
            if tiles_batch:
                memfiles_batch = []
                for tile_data, transform in zip(tiles_batch, transforms_batch):
                    # Create metadata only when needed for MemoryFile
                    meta = meta_template.copy()
                    meta["transform"] = transform
                    memfile = MemoryFile()
                    with memfile.open(**meta) as dst:
                        dst.write(tile_data)
                    memfiles_batch.append(memfile)

                yield np.asarray(memfiles_batch)

                # Return final batch tiles to memory pool
                self._return_tiles_to_pool(tiles_batch)

            # Clear memory pool after processing complete file
            self._array_pool.clear()

    def cleanup(self):
        """Clean up resources and memory pools."""
        if hasattr(self, '_array_pool'):
            self._array_pool.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
