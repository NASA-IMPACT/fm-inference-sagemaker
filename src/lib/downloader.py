from __future__ import annotations

import os
import time
import hashlib
import datetime
import traceback
import numpy as np
import rasterio
import earthaccess
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from earthaccess import search_data, download
from typing import List, Tuple, Optional, Dict
from rasterio.merge import merge as rio_merge


# -------------------------
# Constants
# -------------------------
BANDS = {
    "HLSL30": [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "Fmask",
        "SAA",
        "SZA",
    ],
    "HLSS30": [
        "B02",
        "B03",
        "B04",
        "B8A",
        "B11",
        "B12",
        "Fmask",
        "SAA",
        "SZA",
    ],
}

LAYERS = {"HLS": ["HLSS30", "HLSL30"], "MERRA2": ["M2T1NXSLV", "M2T1NXLND"]}

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", "/root/.cache/")

WIDTH, HEIGHT = (256, 256)
DELTA = 90

def _init_gdal_env():
    global gdal
    from osgeo import gdal as _gdal
    gdal = _gdal
    os.environ.update({
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_TIFF_INTERNAL_MASK": "YES",
        "GDAL_TIFF_OVR_BLOCKSIZE": "256",
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_CACHEMAX": "8192",
        "GDAL_WARP_MEMORY_LIMIT": "1073741824",
        "GDAL_TIFF_DIRECT_IO": "YES",
    })


def generate_digest(date: str, bbox: Tuple[float, float, float, float]) -> str:
    """Generate SHA256 hash for date and bbox combination."""
    key = f"{date}|{','.join(map(str, bbox))}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class WorkerConfig:
    """Configuration for worker processes."""

    download_folder: str
    bbox: Tuple[float, float, float, float]
    bands_map: Dict[str, List[str]]
    layers: List[str]
    width: int = WIDTH
    height: int = HEIGHT
    delta: int = DELTA
    thread_workers: int = 10
    merge_workers: int = 10


# -------------------------
# Core Processing Functions
# -------------------------
def convert_band_to_uint16_vrt(src_file: str, out_dir: str) -> str:
    """Convert band to UInt16 VRT (no pixel copy)."""
    vrt_path = os.path.join(
        out_dir, os.path.basename(src_file).replace(".tif", "_u16.vrt")
    )
    gdal.Translate(
        vrt_path, src_file, format="VRT", outputType=gdal.GDT_UInt16
    )
    return vrt_path


def merge_bands_crop_to_tiff(
    granule_id: str, band_vrts: list[str], cfg: WorkerConfig
) -> str:
    """
    Merge uint16 band VRTs → reproject → crop → write cropped GeoTIFF (EPSG:4326).
    Single Warp call over a stacked VRT for speed.
    """

    stacked_vrt = os.path.join(cfg.download_folder, f"{granule_id}_stack.vrt")
    gdal.BuildVRT(stacked_vrt, band_vrts, separate=True)

    minx, miny, maxx, maxy = cfg.bbox
    wkt_poly = f"POLYGON(({minx} {miny},{minx} {maxy},{maxx} {maxy},{maxx} {miny},{minx} {miny}))"

    cropped_tiff = os.path.join(cfg.download_folder, f"{granule_id}_cropped_4326.tif")

    warp_options = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        format="GTiff",
        cutlineWKT=wkt_poly,
        cropToCutline=True,
        dstNodata=-9999,
        resampleAlg="near",
        multithread=True,
        warpMemoryLimit=1073741824,
        creationOptions=[
            "TILED=YES",
            f"BLOCKXSIZE={cfg.width}",
            f"BLOCKYSIZE={cfg.height}",
            "NUM_THREADS=ALL_CPUS",
            "BIGTIFF=IF_SAFER",
        ],
    )

    gdal.Warp(cropped_tiff, stacked_vrt, options=warp_options)
    return cropped_tiff


def merge_granule_tiffs(
    cfg: WorkerConfig,
    cropped_tiffs: list[str],
    out_file: str,
    timings: dict,
    date: str,
) -> str:
    """
    1. Build VRT mosaic (no data copy)
    2. Convert VRT → final GeoTIFF (EPSG:4326)
    """
    if not cropped_tiffs:
        return ""

    # Validate inputs
    valid_tiffs = [
        t
        for t in cropped_tiffs
        if os.path.exists(t) and os.path.getsize(t) > 0
    ]
    if not valid_tiffs:
        return ""

    # 1️ Build temporary mosaic VRT
    vrt_path = os.path.join(cfg.download_folder, f"{date}_mosaic.vrt")
    try:
        gdal.BuildVRT(vrt_path, valid_tiffs, options=gdal.BuildVRTOptions())
    except Exception:
        return ""

    if not os.path.exists(vrt_path):
        return ""

    # 2️ Convert VRT → GeoTIFF

    warp_opts = gdal.TranslateOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        creationOptions=[
            "TILED=YES",
            "NUM_THREADS=ALL_CPUS",
            "BIGTIFF=IF_SAFER",
            "SPARSE_OK=TRUE",
            f"BLOCKXSIZE={cfg.width}",
            f"BLOCKYSIZE={cfg.height}",
        ],
    )

    try:
        gdal.Translate(out_file, vrt_path, options=warp_opts)

    except Exception:

        return ""

    # 3️ Clean up temporary VRT
    try:
        os.remove(vrt_path)
    except OSError:
        pass
    return out_file


def worker_merge_and_crop(
    cfg: WorkerConfig,
    filenames: List[str],
    date: str,
    uuid: str,
) -> Tuple[str, Optional[str]]:
    """
    Worker process:
    1. Convert bands → Float32 VRT
    2. Merge + reproject + crop → cropped GeoTIFF (EPSG:4326)
    3. Return cropped GeoTIFF path
    """
    _init_gdal_env()
    if not filenames:

        return "", None

    try:
        # Validate input files
        valid_files = [
            f
            for f in filenames
            if os.path.exists(f) and os.path.getsize(f) > 0
        ]
        if not valid_files:

            return "", None

        # Convert bands to Float32 VRTs
        vrt_bands = []
        for f in valid_files:
            try:
                vrt = convert_band_to_uint16_vrt(f, cfg.download_folder)
                if os.path.exists(vrt):
                    vrt_bands.append(vrt)
            except Exception:

                continue

        if not vrt_bands:
            return "", None

        # Merge, crop, and save GeoTIFF
        cropped_tiff = merge_bands_crop_to_tiff(
            uuid, vrt_bands, cfg
        )

        # Verify output
        if (
            not os.path.exists(cropped_tiff)
            or os.path.getsize(cropped_tiff) == 0
        ):
            return "", None

        return cropped_tiff, "EPSG:4326"

    except Exception:
        traceback.print_exc()
        return "", None


def _download_granule_links(
    granule_id: str, links: List[str], download_folder: str
) -> Tuple[str, List[str]]:
    """Download granule links and return list of downloaded files."""
    try:
        filenames = download(links, local_path=download_folder, threads=32)
        return (granule_id, filenames if filenames else [])
    except Exception:

        return (granule_id, [])


def search_and_download_granules(cfg, layer, date_range, bbox, cloud_cover):
    """Search and download all valid granules for a single layer."""
    try:
        granules = search_data(
            short_name=layer,
            temporal=date_range,
            bounding_box=tuple(map(float, bbox)),
            cloud_hosted=True,
            cloud_cover=cloud_cover,
            count=1000,
        )
    except Exception:
        granules = []

    if not granules:
        return []

    # Build granule links mapping
    granule_link_list = []
    for granule in granules:
        links = [
            link
            for band in cfg.bands_map.get(layer, [])
            for link in granule.data_links(access="external")
            if f".{band}." in link
        ]

        if all(any(f".{band}." in l for l in links)
               for band in cfg.bands_map.get(layer, [])):
            granule_link_list.append((granule.uuid, links))

    # Parallel downloads
    downloaded_granules = []
    with ThreadPoolExecutor(max_workers=cfg.thread_workers) as dl_executor:
        futures = {
            dl_executor.submit(_download_granule_links, gid, links, cfg.download_folder): gid
            for gid, links in granule_link_list
        }
        for f in as_completed(futures):
            try:
                gid_local, filelist = f.result()
                if filelist:
                    downloaded_granules.append((gid_local, filelist))
            except Exception:
                traceback.print_exc()
    return downloaded_granules


def process_downloaded_granules(cfg, downloaded_granules, date):
    """Merge/reproject/crop each granule in parallel."""
    merge_results = []
    if not downloaded_granules:
        return merge_results

    with ProcessPoolExecutor(max_workers=cfg.merge_workers) as merge_pool:
        futures = {
            merge_pool.submit(worker_merge_and_crop, cfg, filenames, date, uuid): uuid
            for uuid, filenames in downloaded_granules
        }
        for f in as_completed(futures):
            try:
                merged_path, crs_str = f.result()
                if merged_path:
                    merge_results.append(merged_path)
            except Exception:
                traceback.print_exc()
    return merge_results


def create_empty_fallback(cfg,current_merged_file, final_filename):
    """Create an empty GeoTIFF matching reference file."""
    try:
        with rasterio.open(current_merged_file) as src:
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "count": src.count,
                "tiled": True,
                "blockxsize": cfg.width,
                "blockysize": cfg.height,
                "dtype": "float32",
                "nodata": -9999,
            })
            empty_data = np.full(
                (src.count, src.height, src.width), -9999, dtype="float32"
            )
            with rasterio.open(final_filename, "w", **meta) as dst:
                dst.write(empty_data)
        return True
    except Exception:
        traceback.print_exc()
        return False


def prepare_merged_file_per_date(
    cfg: WorkerConfig,
    date_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    layers: List[str],
    empty: bool = False,
    current_merged_file: Optional[str] = None,
    cloud_cover: Tuple[int, int] = (0, 100),
) -> Tuple[str, Dict]:
    """Main orchestrator for one date."""
    _init_gdal_env()
    date = date_range[0].split("T")[0]
    timings = {}
    final_filename = os.path.join(
        cfg.download_folder, f"{generate_digest(date, bbox)}_merged_cropped_final.tif"
    )

    if os.path.exists(final_filename):
        return final_filename, timings

    merged_files = []
    for layer in layers:
        downloaded_granules = search_and_download_granules(cfg, layer, date_range, bbox, cloud_cover)
        merge_results = process_downloaded_granules(cfg, downloaded_granules, date)
        merged_files.extend(merge_results)

    # Handle empty case
    if not merged_files:
        if empty and current_merged_file:
            if create_empty_fallback(cfg,current_merged_file, final_filename):
                return final_filename, timings
        return "", timings

    # Final mosaic merge
    try:
        final_file = merge_granule_tiffs(cfg, merged_files, final_filename, timings, date)
        return final_file, timings
    except Exception:
        traceback.print_exc()
        return "", timings


# -------------------------
# Main Downloader Class
# -------------------------
class Downloader:
    """Main class for downloading and processing HLS data."""

    def __init__(
        self,
        dates: str,
        bbox: Tuple[float, float, float, float],
        layers: List[str] = LAYERS["HLS"],
        timeseries: bool = False,
        process_workers: int = 4,  # Reduced default for better memory management
        thread_workers: int = 10,
        merge_workers: int = 6,  # Reduced default
    ):
        self.dates = self.prepare_dates(dates)
        self.layers = layers
        self.bbox = bbox
        self.timeseries = timeseries
        self.process_workers = process_workers
        self.thread_workers = thread_workers
        self.merge_workers = merge_workers
        self.timings = {}

        self.cfg = WorkerConfig(
            download_folder=DOWNLOAD_FOLDER,
            bbox=bbox,
            bands_map=BANDS,
            layers=layers,
            width=WIDTH,
            height=HEIGHT,
            delta=DELTA,
            thread_workers=thread_workers,
            merge_workers=merge_workers,
        )

    @staticmethod
    def prepare_dates(dates: str) -> List[str]:
        """Parse date string into list of dates."""
        date_list = []

        if ":" in dates:
            # Date range
            start_date, end_date = [d.strip() for d in dates.split(":")]
            date_list.append(start_date)
            current_date = start_date

            try:
                while current_date != end_date:
                    dt = datetime.datetime.strptime(current_date, "%Y-%m-%d")
                    current_date = (dt + datetime.timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    )
                    date_list.append(current_date)
            except ValueError:
                raise ValueError("Incorrect date format, should be YYYY-MM-DD")

        elif "," in dates:
            # Comma-separated dates
            date_list = dates.split(",")
            for date in date_list:
                try:
                    datetime.datetime.strptime(date.strip(), "%Y-%m-%d")
                except ValueError:
                    raise ValueError(
                        "Incorrect date format, should be YYYY-MM-DD"
                    )
            date_list = [d.strip() for d in date_list]
        else:
            # Single date
            date_list = [dates]
            try:
                datetime.datetime.strptime(dates, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Incorrect date format, should be YYYY-MM-DD")

        return date_list

    def login(self) -> None:
        """Authenticate with earthaccess using environment credentials."""
        self.auth = earthaccess.login(strategy="environment")

    def find_and_prepare_data(self) -> Dict[str, str]:
        """
        Orchestrate processing of all dates in parallel.

        Returns:
            Dictionary mapping dates to processed file paths.
        """
        results: Dict[str, str] = {}
        start_global = time.perf_counter()

        if self.timeseries:
            # Timeseries mode
            with ProcessPoolExecutor(max_workers=self.process_workers) as pool:
                futures = {
                    pool.submit(
                        self.prepare_data_for_date_timeseries, date
                    ): date
                    for date in self.dates
                }
                for f in as_completed(futures):
                    date = futures[f]
                    try:
                        data, local_timings = f.result()
                        results.update(data)
                        if isinstance(local_timings, dict):
                            self.timings.update(local_timings)
                    except Exception:
                        results[date] = ""
        else:
            # Single-date mode
            with ProcessPoolExecutor(max_workers=self.process_workers) as pool:
                futures = {
                    pool.submit(
                        prepare_merged_file_per_date,
                        self.cfg,
                        self.prepare_start_end_date(date),
                        tuple(self.bbox),
                        self.layers,
                        False,
                        None,
                        (0, 100),
                    ): date
                    for date in self.dates
                }
                for f in as_completed(futures):
                    date = futures[f]
                    try:
                        path, local_timings = f.result()
                        results[date] = path
                        if isinstance(local_timings, dict):
                            self.timings.update(local_timings)
                    except Exception:

                        results[date] = ""

        return results

    @staticmethod
    def prepare_start_end_date(date: str) -> Tuple[str, str]:
        """Convert date to start/end datetime strings."""
        return (f"{date}T00:00:00Z", f"{date}T23:59:59Z")

    def prepare_date_range(
        self, date: str, delta: int = DELTA
    ) -> Tuple[str, str]:
        """Calculate date range with delta offset."""
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        start_time = date_obj + datetime.timedelta(days=delta)
        start_date = datetime.datetime.strftime(start_time, "%Y-%m-%d")
        return self.prepare_start_end_date(start_date)

    def find_first_available_file(
        self,
        base_date: str,
        buffer: int,
        delta: int = DELTA,
        direction: int = 1,
        cloud_cover: Tuple[int, int] = (0, 200),
    ) -> str:
        """Search for first available data file before or after a base date."""
        min_delta = (delta - buffer) * direction
        max_delta = delta * direction

        if direction < 0:
            min_delta, max_delta = max_delta, min_delta
        base_dt = datetime.datetime.strptime(base_date, "%Y-%m-%d")
        direction_label = "pre" if direction == -1 else "post"


        for step in range(min_delta, max_delta + 1):
            candidate_dt = base_dt + datetime.timedelta(days=step)
            candidate_str = candidate_dt.strftime("%Y-%m-%d")
            date_range = self.prepare_start_end_date(candidate_str)
            path, _ = prepare_merged_file_per_date(
                self.cfg,
                date_range,
                tuple(self.bbox),
                self.layers,
                False,
                None,
                cloud_cover,
            )
            if path:
                return path

        return ""

    def prepare_data_for_date_timeseries(
        self, date: str
    ) -> Tuple[Dict[str, str], Dict]:
        """Prepare timeseries data (pre/current/post) for a single date."""
        t0 = time.perf_counter()
        prepared_data: Dict[str, str] = {}
        local_timings: Dict = {}

        # Current date raster
        current_range = self.prepare_start_end_date(date)
        current_file, _ = prepare_merged_file_per_date(
            self.cfg, current_range, tuple(self.bbox), self.layers
        )
        if not current_file:
            return {date: ""}, local_timings

        # Pre and post dates
        pre_file = self.find_first_available_file(
            date, buffer=15, direction=-1, cloud_cover=(0, 20)
        )
        post_file = self.find_first_available_file(
            date, buffer=15, direction=1, cloud_cover=(0, 20)
        )

        # Create empty files if missing
        if not pre_file:
            pre_file, _ = prepare_merged_file_per_date(
                self.cfg,
                current_range,
                tuple(self.bbox),
                self.layers,
                empty=True,
                current_merged_file=current_file,
                cloud_cover=(0, 20),
            )
        if not post_file:
            post_file, _ = prepare_merged_file_per_date(
                self.cfg,
                current_range,
                tuple(self.bbox),
                self.layers,
                empty=True,
                current_merged_file=current_file,
                cloud_cover=(0, 20),
            )

        timeseries_files = [pre_file, current_file, post_file]

        # Stack timeseries bands in order: pre, current, post
        datasets = [rasterio.open(f) for f in timeseries_files]
        try:
            crs = datasets[0].crs
            transform = datasets[0].transform
            band_arrays = [ds.read() for ds in datasets]
            mosaic = np.concatenate(band_arrays, axis=0)
        finally:
            for ds in datasets:
                ds.close()

        # Save as timeseries COG
        output_filename = os.path.join(
            self.cfg.download_folder,
            f"{generate_digest(date, self.bbox)}_timeseries_stack.tif",
        )

        profile = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": mosaic.shape[0],
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": self.cfg.width,
            "blockysize": self.cfg.height,
            "nodata": -9999,
        }
        with rasterio.open(output_filename, "w", **profile) as dst:
            dst.write(mosaic)
        prepared_data[date] = output_filename
        return prepared_data, local_timings
