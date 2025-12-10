"""
Tile server for solar imagery with Helioprojective coordinates.
Serves tiles in XYZ format with custom coordinate system handling.
"""
import asyncio
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import sunpy.visualization.colormaps as cm  # Registers SunPy colormaps with matplotlib

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from functools import partial
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from rasterio.windows import Window
from typing import Optional

# Thread pool for CPU-bound operations (rasterio, numpy, PIL)
executor = ThreadPoolExecutor(max_workers=4)

root_path = os.environ.get("TILER_ROOT_PATH", "/api/tiles")

app = FastAPI(title="Solar Tile Server", root_path=root_path)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for TIF files
TILE_DIR = Path("/root/.cache/surya/inferences/")


# Solar colormap mapping using SunPy's official colormaps
# SunPy registers these colormaps with matplotlib on import
SOLAR_COLORMAPS = {
    # AIA Channels - SDO/AIA specific colormaps from SunPy
    94: 'sdoaia94',
    131: 'sdoaia131',
    171: 'sdoaia171',
    193: 'sdoaia193',
    211: 'sdoaia211',
    304: 'sdoaia304',
    335: 'sdoaia335',
    1600: 'sdoaia1600',
    # HMI Channels - Diverging colormaps for magnetic/velocity data
    'magnetogram': 'RdBu_r',
    'mag': 'RdBu_r',
    'bx': 'RdBu_r',
    'by': 'RdBu_r',
    'bz': 'RdBu_r',
    'dopplergram': 'seismic',
    'v': 'seismic',
    'continuum': 'gray',
    'ic': 'gray',
}


def get_solar_colormap(instrument_type: str, wavelength: int = None, observable: str = None):
    """
    Get the appropriate colormap for a solar instrument.

    Args:
        instrument_type: 'aia' or 'hmi'
        wavelength: AIA wavelength (e.g., 94, 171, 304)
        observable: HMI observable type (e.g., 'magnetogram', 'dopplergram')

    Returns:
        matplotlib colormap or None if not found
    """
    cmap_name = None

    if instrument_type == 'aia' and wavelength:
        cmap_name = SOLAR_COLORMAPS.get(wavelength)
    elif instrument_type == 'hmi' and observable:
        cmap_name = SOLAR_COLORMAPS.get(observable)

    if cmap_name:
        return plt.get_cmap(cmap_name)
    return None


def extract_instrument_info(filename: str, instrument: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Extract instrument type and observable/wavelength from filename.

    Returns:
        (instrument_type, wavelength, observable)
        - instrument_type: 'aia' or 'hmi'
        - wavelength: int for AIA (e.g., 94, 171), None for HMI
        - observable: None for AIA, 'magnetogram'/'continuum'/'dopplergram' for HMI
    """
    import re
    filename_lower = filename.lower()
    instrument_lower = instrument.lower()

    # Check for AIA
    aia_match = re.search(r'aia(\d+)', filename_lower)
    if aia_match or 'aia' in instrument_lower:
        wavelength = int(aia_match.group(1)) if aia_match else None
        return ('aia', wavelength, None)

    # Check for HMI
    if 'hmi' in filename_lower or 'hmi' in instrument_lower:
        # Detect observable type
        if 'magnetogram' in filename_lower or 'mag' in filename_lower:
            return ('hmi', None, 'magnetogram')
        elif 'continuum' in filename_lower or 'ic' in filename_lower or 'cont' in filename_lower:
            return ('hmi', None, 'continuum')
        elif 'dopplergram' in filename_lower or 'doppler' in filename_lower or '_v' in filename_lower:
            return ('hmi', None, 'dopplergram')
        else:
            # Default to magnetogram if HMI but observable not specified
            return ('hmi', None, 'magnetogram')

    return (None, None, None)


class SolarTileGenerator:
    """Generate tiles for solar imagery with custom coordinates."""

    def __init__(self, tif_path: str, instrument_type: Optional[str] = None,
                 wavelength: Optional[int] = None, observable: Optional[str] = None):
        self.tif_path = tif_path
        self.instrument_type = instrument_type  # 'aia' or 'hmi'
        self.wavelength = wavelength  # AIA wavelength
        self.observable = observable  # HMI observable type
        with rasterio.open(tif_path) as src:
            self.width = src.width
            self.height = src.height
            self.bounds = src.bounds
            self.transform = src.transform

            # Compute file-level normalization statistics for consistent tile rendering
            data = src.read(1)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            valid_data = data[~np.isnan(data)]

            if len(valid_data) > 0:
                if self.instrument_type == 'hmi' and self.observable in ['magnetogram', 'dopplergram', 'mag', 'v']:
                    # HMI magnetogram/dopplergram: symmetric scale around zero
                    abs_max = np.percentile(np.abs(valid_data), 99.5)
                    self.vmin, self.vmax = -abs_max, abs_max
                    self.nan_fill = 0.5  # NaN -> middle of diverging scale
                else:
                    # AIA or HMI continuum: percentile-based normalization
                    self.vmin, self.vmax = np.percentile(valid_data, [1.25, 99.5])
                    self.nan_fill = 0.0
            else:
                self.vmin, self.vmax = 0, 1
                self.nan_fill = 0.0

    def get_tile(self, z: int, x: int, y: int, tile_size: int = 256, colormap: bool = True) -> Optional[bytes]:
        """
        Generate a tile for the given z/x/y coordinates.

        For solar images, we use a simple power-of-2 tile scheme:
        - z=0: entire image in 1 tile
        - z=1: image split into 2x2 tiles
        - z=2: image split into 4x4 tiles, etc.
        """
        tiles_per_side = 2 ** z

        # Check if tile coordinates are valid
        if x < 0 or x >= tiles_per_side or y < 0 or y >= tiles_per_side:
            return None

        # Calculate the window in the source image
        tile_width_px = self.width / tiles_per_side
        tile_height_px = self.height / tiles_per_side

        # Calculate pixel coordinates
        col_off = int(x * tile_width_px)
        row_off = int(y * tile_height_px)
        width = int(min(tile_width_px, self.width - col_off))
        height = int(min(tile_height_px, self.height - row_off))

        # Read the window from the TIF
        with rasterio.open(self.tif_path) as src:
            window = Window(col_off, row_off, width, height)
            data = src.read(1, window=window)

            # Handle nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

            # Check if tile has any valid data
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                # Empty tile
                return None

            # Normalize using file-level statistics for consistent rendering across tiles
            normalized = np.clip((data - self.vmin) / (self.vmax - self.vmin), 0, 1)
            normalized = np.nan_to_num(normalized, nan=self.nan_fill)

            # Apply colormap if available and requested
            cmap = None
            if colormap:
                cmap = get_solar_colormap(self.instrument_type, self.wavelength, self.observable)

            if cmap:
                # Apply colormap
                rgba = cmap(normalized)
                # Convert to RGB (drop alpha channel)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(rgb, mode='RGB')
            else:
                # Grayscale fallback
                gray = (normalized * 255).astype(np.uint8)
                img = Image.fromarray(gray, mode='L')

            # Resize to tile_size if needed
            if width != tile_size or height != tile_size:
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            # Convert to PNG
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()


# Cache for tile generators
tile_generators = {}


def get_tile_generator(instrument: str, timestamp: str, step: int) -> SolarTileGenerator:
    """Get or create a tile generator for a given file."""
    "20140107_0348_aia304_step01"
    parsed_datetime = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    filename = f"{parsed_datetime.strftime('%Y%m%d_%H%M')}_{instrument}_step{'%02d' % step}.tif"
    key = f"{instrument}/{filename}"
    if key not in tile_generators:
        tif_path = TILE_DIR / instrument / filename
        if not tif_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {tif_path}")

        # Extract instrument info from filename for colormap selection
        instrument_type, wavelength, observable = extract_instrument_info(filename, instrument)
        tile_generators[key] = SolarTileGenerator(
            str(tif_path),
            instrument_type=instrument_type,
            wavelength=wavelength,
            observable=observable
        )
    return tile_generators[key]

# Define a model for the POST request body
class TileInfoParams(BaseModel):
    instrument: str
    timestamp: str
    step: int

class TileRequestParams(BaseModel):
    instrument: str
    timestamp: str
    z: int
    x: int
    y: int
    step: int
    colormap: Optional[bool] = True

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Solar Tile Server",
        "description": "Tile server for solar imagery with Helioprojective coordinates",
        "endpoints": {
            "tiles": "/tiles/{instrument}/{timestamp}/{step}/{z}/{x}/{y}.png",
            "info": "/info/{instrument}/{timestamp}/{step}",
        }
    }

@app.get("/tiles/{instrument}/{timestamp}/{step}/{z}/{x}/{y}.png")
async def get_tile(tile_request: TileRequestParams = Depends()):
    """
    Get a tile for the specified instrument and file.

    Example: /tiles/aia94/20140107_0348_aia94_step01.tif/2/1/1.png
    Query params:
        - colormap: bool (default True) - Apply AIA wavelength-specific colormap
    """
    try:
        instrument = tile_request.instrument
        timestamp = tile_request.timestamp
        z = tile_request.z
        x = tile_request.x
        y = tile_request.y
        step = tile_request.step
        colormap = tile_request.colormap

        generator = get_tile_generator(instrument, timestamp, step)
        # Run CPU-bound tile generation in thread pool
        loop = asyncio.get_event_loop()
        tile_data = await loop.run_in_executor(
            executor,
            partial(generator.get_tile, z, x, y, colormap=colormap)
        )

        if tile_data is None:
            raise HTTPException(status_code=404, detail="Tile not found or empty")

        return Response(content=tile_data, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info/{instrument}/{timestamp}/{step}")
async def get_info(tile_info: TileInfoParams):
    """Get information about a TIF file."""
    try:
        instrument = tile_info.instrument
        timestamp = tile_info.timestamp
        step = tile_info.step
        generator = get_tile_generator(instrument, timestamp, step)

        # Determine colormap availability
        colormap_available = False
        colormap_name = None
        if generator.instrument_type == 'aia' and generator.wavelength:
            colormap_name = SOLAR_COLORMAPS.get(generator.wavelength)
            colormap_available = colormap_name is not None
        elif generator.instrument_type == 'hmi' and generator.observable:
            colormap_name = SOLAR_COLORMAPS.get(generator.observable)
            colormap_available = colormap_name is not None

        return {
            "width": generator.width,
            "height": generator.height,
            "bounds": {
                "left": generator.bounds.left,
                "bottom": generator.bounds.bottom,
                "right": generator.bounds.right,
                "top": generator.bounds.top
            },
            "instrument_type": generator.instrument_type,
            "wavelength": generator.wavelength,  # AIA only
            "observable": generator.observable,  # HMI only
            "colormap_available": colormap_available,
            "colormap_name": colormap_name,
            "coordinate_system": "Helioprojective",
            "units": "arcseconds"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/colormaps")
async def get_available_colormaps():
    """Get list of available colormaps for both AIA and HMI."""
    aia_wavelengths = {k: v for k, v in SOLAR_COLORMAPS.items() if isinstance(k, int)}
    hmi_observables = {k: v for k, v in SOLAR_COLORMAPS.items()
                       if isinstance(k, str) and k not in ['mag', 'ic', 'v']}
    return {
        "aia": {
            "available_wavelengths": sorted(aia_wavelengths.keys()),
            "colormap_names": aia_wavelengths,
            "description": "AIA wavelength-specific colormaps from SunPy"
        },
        "hmi": {
            "available_observables": sorted(hmi_observables.keys()),
            "colormap_names": hmi_observables,
            "description": "HMI observable-specific colormaps (magnetogram, continuum, dopplergram, bx, by, bz)"
        }
    }
#
