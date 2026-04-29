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
import sunpy.visualization.colormaps  # noqa: F401 - registers SDO colormaps with matplotlib

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

app = FastAPI(
    title="Solar Tile Server",
    version=os.getenv("RELEASE_VERSION", "0.0.1"),
    root_path=root_path,
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for TIF files
TILE_DIR = Path("/root/.cache/data/surya/geotiff_forecasts/")


# Solar colormap mapping using SunPy's official colormaps
# SunPy registers these colormaps with matplotlib on import
SOLAR_COLORMAPS = {
    # AIA Channels - SDO/AIA specific colormaps from SunPy
    94: "sdoaia94",
    131: "sdoaia131",
    171: "sdoaia171",
    193: "sdoaia193",
    211: "sdoaia211",
    304: "sdoaia304",
    335: "sdoaia335",
    1600: "sdoaia1600",
    # HMI Channels - Diverging colormaps for magnetic/velocity data
    "magnetogram": "RdBu_r",
    "mag": "RdBu_r",
    "bx": "RdBu_r",
    "by": "RdBu_r",
    "bz": "RdBu_r",
    "dopplergram": "seismic",
    "v": "seismic",
    "continuum": "gray",
    "ic": "gray",
}

# AIA scaling parameters matching Helioviewer's JP2 generation
# Source: https://aia.cfa.harvard.edu/content/aia_rfilter_jp2gen.pro
# All channels use log10 scaling (dataScalingType=3) except 4500 which uses linear
AIA_SCALING_PARAMS = {
    94: {"dataMin": 0.25, "dataMax": 2080.0, "exptime": 4.99803},
    131: {"dataMin": 2.0, "dataMax": 2800.0, "exptime": 6.99685},
    171: {"dataMin": 15.0, "dataMax": 25600.0, "exptime": 4.99803},
    193: {"dataMin": 11.0, "dataMax": 18000.0, "exptime": 2.99950},
    211: {"dataMin": 8.0, "dataMax": 16220.0, "exptime": 4.99801},
    304: {"dataMin": 30.0, "dataMax": 2000.0, "exptime": 4.99441},
    335: {"dataMin": 2.0, "dataMax": 1600.0, "exptime": 6.99734},
    1600: {"dataMin": 5.0, "dataMax": 8800.0, "exptime": 2.99911},
    1700: {"dataMin": 100.0, "dataMax": 32935.0, "exptime": 1.00026},
    4500: {"dataMin": 0.25, "dataMax": 26000.0, "exptime": 1.00026, "linear": True},
}


def get_solar_colormap(
    instrument_type: str, wavelength: int = None, observable: str = None
):
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

    if instrument_type == "aia" and wavelength:
        cmap_name = SOLAR_COLORMAPS.get(wavelength)
    elif instrument_type == "hmi" and observable:
        cmap_name = SOLAR_COLORMAPS.get(observable)

    if cmap_name:
        return plt.get_cmap(cmap_name)
    return None


def extract_instrument_info(
    filename: str, instrument: str
) -> tuple[Optional[str], Optional[int], Optional[str]]:
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
    aia_match = re.search(r"aia(\d+)", filename_lower)
    if aia_match or "aia" in instrument_lower:
        wavelength = int(aia_match.group(1)) if aia_match else None
        return ("aia", wavelength, None)

    # Check for HMI
    if "hmi" in filename_lower or "hmi" in instrument_lower:
        # Detect observable type
        if "magnetogram" in filename_lower or "mag" in filename_lower:
            return ("hmi", None, "magnetogram")
        elif (
            "continuum" in filename_lower
            or "ic" in filename_lower
            or "cont" in filename_lower
        ):
            return ("hmi", None, "continuum")
        elif (
            "dopplergram" in filename_lower
            or "doppler" in filename_lower
            or "_v" in filename_lower
        ):
            return ("hmi", None, "dopplergram")
        else:
            # Default to magnetogram if HMI but observable not specified
            return ("hmi", None, "magnetogram")

    return (None, None, None)


class SolarTileGenerator:
    """Generate tiles for solar imagery with custom coordinates."""

    def __init__(
        self,
        tif_path: str,
        instrument_type: Optional[str] = None,
        wavelength: Optional[int] = None,
        observable: Optional[str] = None,
    ):
        self.tif_path = tif_path
        self.instrument_type = instrument_type  # 'aia' or 'hmi'
        self.wavelength = wavelength  # AIA wavelength
        self.observable = observable  # HMI observable type

        # Determine scaling method based on instrument
        self.use_log_scaling = False
        self.aia_params = None

        if self.instrument_type == "aia" and self.wavelength in AIA_SCALING_PARAMS:
            self.aia_params = AIA_SCALING_PARAMS[self.wavelength]
            # Use log10 scaling like Helioviewer (except 4500 which is linear)
            self.use_log_scaling = not self.aia_params.get("linear", False)

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
                if self.instrument_type == "hmi" and self.observable in [
                    "magnetogram",
                    "dopplergram",
                    "mag",
                    "v",
                ]:
                    # HMI magnetogram/dopplergram: symmetric scale around zero
                    abs_max = np.percentile(np.abs(valid_data), 99.5)
                    self.vmin, self.vmax = -abs_max, abs_max
                    self.nan_fill = 0.5  # NaN -> middle of diverging scale
                elif self.use_log_scaling and self.aia_params:
                    # AIA with Helioviewer-style log10 scaling
                    # Use the channel-specific dataMin/dataMax from Helioviewer
                    self.vmin = self.aia_params["dataMin"]
                    self.vmax = self.aia_params["dataMax"]
                    self.nan_fill = 0.0
                else:
                    # Fallback: percentile-based normalization
                    self.vmin, self.vmax = np.percentile(valid_data, [1.25, 99.5])
                    self.nan_fill = 0.0
            else:
                self.vmin, self.vmax = 0, 1
                self.nan_fill = 0.0

    def _create_zero_tile(self, tile_size: int = 256, colormap: bool = True) -> bytes:
        """Create a tile filled with zero values (black for most colormaps)."""
        # Create array of zeros normalized to 0
        normalized = np.zeros((tile_size, tile_size), dtype=np.float32)

        # Apply colormap if available and requested
        cmap = None
        if colormap:
            cmap = get_solar_colormap(
                self.instrument_type, self.wavelength, self.observable
            )

        if cmap:
            # Apply colormap to zeros
            rgba = cmap(normalized)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        else:
            # Grayscale fallback - zeros become black
            gray = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(gray, mode="L")

        # Convert to PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def get_tile(
        self, z: int, x: int, y: int, tile_size: int = 256, colormap: bool = True
    ) -> Optional[bytes]:
        """
        Generate a tile for the given z/x/y coordinates.

        For solar images, we use a simple power-of-2 tile scheme:
        - z=0: entire image in 1 tile
        - z=1: image split into 2x2 tiles
        - z=2: image split into 4x4 tiles, etc.
        """
        tiles_per_side = 2**z

        # Check if tile coordinates are beyond bounds - return zero-filled tile
        if x < 0 or x >= tiles_per_side or y < 0 or y >= tiles_per_side:
            return self._create_zero_tile(tile_size, colormap)

        # Calculate the window in the source image
        tile_width_px = self.width / tiles_per_side
        tile_height_px = self.height / tiles_per_side

        # Convert from center-origin tile coordinates to image pixel coordinates
        # Tile (0,0) at zoom 0 covers the whole image
        # At higher zooms, x increases right, y increases up (Cartesian/solar convention)
        # But image pixels have row 0 at top, so we flip y

        # Flip y-axis: tile y=0 should be bottom of image, but image row 0 is top
        y_flipped = (tiles_per_side - 1) - y

        # Calculate pixel coordinates
        col_off = int(x * tile_width_px)
        row_off = int(y_flipped * tile_height_px)
        width = int(min(tile_width_px, self.width - col_off))
        height = int(min(tile_height_px, self.height - row_off))

        # Read the window from the TIF
        with rasterio.open(self.tif_path) as src:
            window = Window(col_off, row_off, width, height)
            data = src.read(1, window=window)

            # Handle nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                # Empty tile
                return None

            # Normalize using appropriate scaling method
            if self.use_log_scaling:
                # Helioviewer-style log10 scaling for AIA
                # Clip to dataMin/dataMax range, then apply log10
                clipped = np.clip(data, self.vmin, self.vmax)
                # Apply log10 scaling: log10(data) normalized to [0,1]
                log_min = np.log10(self.vmin)
                log_max = np.log10(self.vmax)
                normalized = (np.log10(np.maximum(clipped, self.vmin)) - log_min) / (
                    log_max - log_min
                )
                normalized = np.nan_to_num(normalized, nan=self.nan_fill)
            else:
                # Linear normalization for HMI and fallback
                normalized = np.clip((data - self.vmin) / (self.vmax - self.vmin), 0, 1)
                normalized = np.nan_to_num(normalized, nan=self.nan_fill)

            # Apply colormap if available and requested
            cmap = None
            if colormap:
                cmap = get_solar_colormap(
                    self.instrument_type, self.wavelength, self.observable
                )

            if cmap:
                # Apply colormap
                rgba = cmap(normalized)
                # Convert to RGB (drop alpha channel)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(rgb, mode="RGB")
            else:
                # Grayscale fallback
                gray = (normalized * 255).astype(np.uint8)
                img = Image.fromarray(gray, mode="L")

            # Resize to tile_size if needed
            if width != tile_size or height != tile_size:
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            # Convert to PNG
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()


# Cache for tile generators
tile_generators = {}


def get_tile_generator(
    instrument: str, timestamp: str, step: int
) -> SolarTileGenerator:
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
        instrument_type, wavelength, observable = extract_instrument_info(
            filename, instrument
        )
        tile_generators[key] = SolarTileGenerator(
            str(tif_path),
            instrument_type=instrument_type,
            wavelength=wavelength,
            observable=observable,
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
        },
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
            executor, partial(generator.get_tile, z, x, y, colormap=colormap)
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
        if generator.instrument_type == "aia" and generator.wavelength:
            colormap_name = SOLAR_COLORMAPS.get(generator.wavelength)
            colormap_available = colormap_name is not None
        elif generator.instrument_type == "hmi" and generator.observable:
            colormap_name = SOLAR_COLORMAPS.get(generator.observable)
            colormap_available = colormap_name is not None

        # Calculate center coordinates (origin at Sun center)
        center_x = (generator.bounds.left + generator.bounds.right) / 2
        center_y = (generator.bounds.bottom + generator.bounds.top) / 2

        return {
            "width": generator.width,
            "height": generator.height,
            "bounds": {
                "left": generator.bounds.left,
                "bottom": generator.bounds.bottom,
                "right": generator.bounds.right,
                "top": generator.bounds.top,
            },
            "center": {"x": center_x, "y": center_y},
            "instrument_type": generator.instrument_type,
            "wavelength": generator.wavelength,  # AIA only
            "observable": generator.observable,  # HMI only
            "colormap_available": colormap_available,
            "colormap_name": colormap_name,
            "coordinate_system": "Helioprojective",
            "origin": "center",
            "tile_origin": "bottom-left",  # y=0 is at bottom, x=0 is at left
            "units": "arcseconds",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/colormaps")
async def get_available_colormaps():
    """Get list of available colormaps for both AIA and HMI."""
    aia_wavelengths = {k: v for k, v in SOLAR_COLORMAPS.items() if isinstance(k, int)}
    hmi_observables = {
        k: v
        for k, v in SOLAR_COLORMAPS.items()
        if isinstance(k, str) and k not in ["mag", "ic", "v"]
    }
    return {
        "aia": {
            "available_wavelengths": sorted(aia_wavelengths.keys()),
            "colormap_names": aia_wavelengths,
            "description": "AIA wavelength-specific colormaps from SunPy",
        },
        "hmi": {
            "available_observables": sorted(hmi_observables.keys()),
            "colormap_names": hmi_observables,
            "description": "HMI observable-specific colormaps (magnetogram, continuum, dopplergram, bx, by, bz)",
        },
    }


#
