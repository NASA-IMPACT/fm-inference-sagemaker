#!/usr/bin/env python
"""
Complete SDO Data Download - Surya's Exact Pipeline
Minimal version without early aiapy import
"""

import drms
import os
import sys
import calendar
import time
import logging
import threading
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from glob import glob
from typing import List, Tuple, Optional, Dict
import re
import pandas as pd

# Suppress verbose logging
logging.getLogger('drms').setLevel(logging.WARNING)

TAI_DATETIME_FORMAT = "%Y.%m.%d_%H:%M:%S_TAI"


# Wind parameters for 2010-05-13 07:00:00 to 2019-12-31 23:00:00
WIND_MIN = 240.0
WIND_PTP = 560.0

RECORDS = {
    "aia_euv": {
        "jsoc_query": "aia.lev1_euv_12s",
        "channels": ["94", "131", "171", "193", "211", "304", "335"],
        "num_channels": 7,
        # Pattern: aia.lev1_euv_12s.2016-02-20T165948Z.WAVELENGTH.image_lev1.fits
        "file_pattern": r"aia\.lev1_euv_12s\.\d{4}-\d{2}-\d{2}T\d{6}Z\.({channels})\.image_lev1\.fits"
    },
    "aia_uv": {
        "jsoc_query": "aia.lev1_uv_24s",
        "channels": ["1600"],
        "num_channels": 1,
        "file_pattern": r"aia\.lev1_uv_24s\.\d{4}-\d{2}-\d{2}T\d{6}Z\.({channels})\.image_lev1\.fits"
    },
    "hmi_mag": {
        "jsoc_query": "hmi.M_720s",
        "num_channels": 1,
        # Pattern: hmi.M_720s.20160220_170000_TAI.magnetogram.fits
        "file_pattern": r"hmi\.M_720s\.\d{8}_\d{6}_TAI\.magnetogram\.fits"
    },
    "hmi_b": {
        "jsoc_query": "hmi.B_720s",
        "segments": ["field", "inclination", "azimuth", "disambig"],
        "num_channels": 4,
        # Pattern: hmi.B_720s.20160220_170000_TAI.field.fits
        "file_pattern": r"hmi\.B_720s\.\d{8}_\d{6}_TAI\.({segments})\.fits"
    },
    "hmi_vel": {
        "jsoc_query": "hmi.V_720s",
        "num_channels": 1,
        # Pattern: hmi.V_720s.20160220_170000_TAI.Dopplergram.fits
        "file_pattern": r"hmi\.V_720s\.\d{8}_\d{6}_TAI\.Dopplergram\.fits"
    }
}

class Downloader():
    """Downloader class for Surya's SDO data pipeline"""

    def __init__(self, email, output_dir, wind_data_dir):
        self.email = email
        self.output_dir = output_dir
        self.wind_data_dir = wind_data_dir
        self._print_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._export_cache_file = os.path.join(output_dir, ".export_cache.json")
        self._export_cache: Dict[str, str] = self._load_export_cache()

    def _load_export_cache(self) -> Dict[str, str]:
        """Load export ID cache from disk."""
        if os.path.exists(self._export_cache_file):
            try:
                with open(self._export_cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_export_cache(self):
        """Save export ID cache to disk."""
        os.makedirs(os.path.dirname(self._export_cache_file), exist_ok=True)
        with self._cache_lock:
            try:
                with open(self._export_cache_file, 'w') as f:
                    json.dump(self._export_cache, f, indent=2)
            except IOError:
                pass

    def _get_cached_export_id(self, query: str) -> Optional[str]:
        """Get cached export ID for a query."""
        with self._cache_lock:
            return self._export_cache.get(query)

    def _cache_export_id(self, query: str, export_id: str):
        """Cache an export ID for a query."""
        if export_id:
            with self._cache_lock:
                self._export_cache[query] = export_id
            self._save_export_cache()

    def _invalidate_cached_export(self, query: str):
        """Remove a cached export ID (e.g., if the export expired)."""
        with self._cache_lock:
            if query in self._export_cache:
                del self._export_cache[query]
        self._save_export_cache()

    def clear_export_cache(self):
        """Clear all cached export IDs."""
        with self._cache_lock:
            self._export_cache.clear()
        self._save_export_cache()

    def _get_client(self):
        """Create a new drms client (thread-safe)"""
        return drms.Client(email=self.email)

    def _thread_safe_print(self, message, end='\n', flush=False):
        """Thread-safe print function"""
        with self._print_lock:
            print(message, end=end, flush=flush)

    def _download_file(self, url: str, output_path: str, retries: int = 3, timeout: int = 60) -> Tuple[bool, str]:
        """Download a single file using requests with retry logic.

        Args:
            url: URL to download from
            output_path: Local path to save file
            retries: Number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Tuple of (success: bool, filename: str)
        """
        filename = os.path.basename(output_path)

        # Skip if already exists
        if os.path.exists(output_path):
            return True, filename

        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                # Write to temp file first, then rename (atomic)
                temp_path = output_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                os.rename(temp_path, output_path)
                return True, filename

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False, filename
            except Exception as e:
                return False, filename

        return False, filename

    def _download_files_parallel(
        self,
        urls: List[str],
        output_dir: str,
        max_workers: int = 10,
        label: str = ""
    ) -> Tuple[int, int, int]:
        """Download multiple files in parallel using requests.

        Args:
            urls: List of URLs to download
            output_dir: Directory to save files
            max_workers: Maximum parallel download threads
            label: Label for progress display

        Returns:
            Tuple of (total, downloaded, skipped)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build download tasks: (url, output_path)
        download_tasks = []
        skipped = []

        for url in urls:
            filename = url.split('/')[-1]
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                skipped.append(filename)
            else:
                download_tasks.append((url, output_path))

        if not download_tasks:
            return len(urls), [], skipped

        downloaded = []
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._download_file, url, path): url
                for url, path in download_tasks
            }

            for future in as_completed(future_to_url):
                success, filename = future.result()
                if success:
                    downloaded.append(filename)
                else:
                    failed.append(filename)

        return len(urls), downloaded, skipped

    def download(self, start_datetime, end_datetime, cadence="12m", parallel=True, max_workers=5):
        """Download SDO data for given datetime range and cadence

        Args:
            start_datetime: Start datetime
            end_datetime: End datetime
            cadence: Download cadence (e.g., "12m", "1h")
            parallel: If True, download bands in parallel (default: True)
            max_workers: Maximum number of parallel download threads (default: 5)
        """
        start_tai = start_datetime.strftime(TAI_DATETIME_FORMAT)
        end_tai = end_datetime.strftime(TAI_DATETIME_FORMAT)
        t_key_base = f"{start_datetime.strftime('%Y%m%d_%H%M%S')}_{end_datetime.strftime('%Y%m%d_%H%M%S')}"

        # self.download_aia_uv(start_tai, end_tai, '12m', t_key_base)

        return self._download(start_tai, end_tai, cadence, t_key_base, max_workers)
        # self.download_wind_params()

    def _download(self, start_tai, end_tai, cadence, t_key_base, max_workers=5):
        """Download all bands in parallel using ThreadPoolExecutor"""
        download_tasks = [
            ("AIA EUV", self.download_aia_euv),
            ("AIA UV", self.download_aia_uv),
            ("HMI Magnetogram", self.download_hmi_magnetogram),
            ("HMI Vector", self.download_hmi_vector_components),
            ("HMI Velocity", self.download_hmi_velocity),
        ]

        results = {}

        for name, func in download_tasks:
            self._thread_safe_print(f"Scheduling download task: {name}")
            count, downloaded, skipped = func(start_tai, end_tai, cadence, t_key_base)
            results[name] = {
                'total': count,
                'downloaded': downloaded,
                'skipped': skipped
            }

        return results

    def build_query(self, record, start_tai, end_tai=None, cadence=None, wavelengths=None, segment=None):
        """Build JSOC query string for given record.

        Args:
            record: JSOC record series name (e.g., "aia.lev1_euv_12s")
            start_tai: Start time in TAI format (or single timestamp if end_tai is None)
            end_tai: End time in TAI format (None for single timestamp query)
            cadence: Cadence string (e.g., "12m") - required if end_tai is provided
            wavelengths: List of wavelength channels to filter
            segment: Segment name for HMI vector data

        Returns:
            JSOC query string
        """
        # Build time part of query
        if end_tai and cadence:
            time_part = f"[{start_tai}-{end_tai}@{cadence}]"
        else:
            # Single timestamp query
            time_part = f"[{start_tai}]"

        # Build full query
        if segment:
            return f"{record}{time_part}{{{segment}}}"
        elif wavelengths:
            wave_str = ",".join(wavelengths)
            return f"{record}{time_part}[{wave_str}]{{image}}"
        else:
            return f"{record}{time_part}"

    def _get_export_urls(self, query: str, label: str, timeout: int = 3600) -> Optional[List[str]]:
        """Get download URLs from JSOC export request.

        Uses cached export IDs when available to avoid redundant server-side processing.

        Args:
            query: JSOC query string
            label: Display label for logging
            timeout: Export request timeout in seconds

        Returns:
            List of URLs or None if failed
        """
        client = None
        export_request = None
        try:
            client = self._get_client()

            # Check for cached export ID
            cached_export_id = self._get_cached_export_id(query)
            if cached_export_id:
                try:
                    export_request = client.export_from_id(cached_export_id)
                    if export_request and export_request.status == 0:
                        urls = list(export_request.urls.url) if hasattr(export_request.urls, 'url') else list(export_request.urls)
                        self._thread_safe_print(f"(cached) ", end='')
                        return urls
                    else:
                        # Export no longer valid, invalidate cache
                        self._invalidate_cached_export(query)
                except Exception:
                    # Cached export may have expired, invalidate and proceed with new request
                    self._invalidate_cached_export(query)

            # Make new export request
            export_request = client.export(query, method='url', protocol='fits')
            export_request.wait(timeout=timeout)

            if export_request and export_request.status == 0:
                # Cache the export ID for future requests
                if export_request.id:
                    self._cache_export_id(query, export_request.id)

                urls = list(export_request.urls.url) if hasattr(export_request.urls, 'url') else list(export_request.urls)
                return urls

            return None

        except Exception as e:
            self._thread_safe_print(f"  {label} export failed: {str(e)}")
            return None
        finally:
            del export_request
            del client

    def _download_query(self, query: str, label: str, max_workers: int = 10) -> Tuple[int, int]:
        """Generic download method using drms for URL discovery and parallel requests for download.

        Args:
            query: JSOC query string
            label: Display label for progress output
            max_workers: Number of parallel download threads

        Returns:
            Tuple of (total_files, skipped_files)
        """
        self._thread_safe_print(f"  {label}... ", end='')

        # Step 1: Get URLs from JSOC
        urls = self._get_export_urls(query, label)

        if not urls:
            self._thread_safe_print(f"No data found")
            return 0, 0

        total = len(urls)

        # Step 2: Download files in parallel using requests
        output_dir = os.path.join(self.output_dir, "raw_fits")
        total, downloaded, skipped = self._download_files_parallel(
            urls, output_dir, max_workers=max_workers, label=label
        )

        # Report results
        if len(skipped) == total:
            self._thread_safe_print(f"(already downloaded, {total} files)")
        elif len(downloaded) > 0:
            self._thread_safe_print(f"({downloaded} downloaded, {skipped} skipped)")
        else:
            self._thread_safe_print(f"(failed to download)")

        return total, downloaded, skipped

    def download_aia_euv(self, start_tai, end_tai=None, cadence=None, t_key_base=None):
        """Download AIA EUV channels"""
        query = self.build_query(
            RECORDS["aia_euv"]["jsoc_query"], start_tai, end_tai, cadence,
            wavelengths=RECORDS["aia_euv"]["channels"]
        )
        count, downloaded, skipped = self._download_query(query, "AIA EUV (7 wavelengths)")
        return count, downloaded, skipped

    def download_aia_uv(self, start_tai, end_tai=None, cadence=None, t_key_base=None):
        """Download AIA UV 1600Å"""
        query = self.build_query(
            RECORDS["aia_uv"]["jsoc_query"], start_tai, end_tai, cadence,
            wavelengths=RECORDS["aia_uv"]["channels"]
        )
        count, downloaded, skipped = self._download_query(query, "AIA UV 1600Å")
        return count, downloaded, skipped

    def download_hmi_magnetogram(self, start_tai, end_tai=None, cadence=None, t_key_base=None):
        """Download HMI magnetogram"""
        query = self.build_query(
            RECORDS["hmi_mag"]["jsoc_query"], start_tai, end_tai, cadence
        )
        count, downloaded, skipped = self._download_query(query, "HMI Magnetogram")
        return count, downloaded, skipped

    def download_hmi_vector_components(self, start_tai, end_tai=None, cadence=None, t_key_base=None, max_workers=10):
        """Download HMI vector components using parallel requests.

        Args:
            start_tai: Start time in TAI format
            end_tai: End time in TAI format (optional)
            cadence: Cadence string (optional)
            t_key_base: Time key base (unused, kept for API compatibility)
            max_workers: Number of parallel download threads
        """
        segments = RECORDS["hmi_b"]["segments"]
        label = f"HMI Vector ({len(segments)} components)"

        self._thread_safe_print(f"  {label}... ", end='')

        # Collect all URLs from all segments
        all_urls = []
        for segment in segments:
            query = self.build_query(
                RECORDS["hmi_b"]["jsoc_query"], start_tai, end_tai, cadence,
                segment=segment
            )
            urls = self._get_export_urls(query, f"{label}/{segment}")
            if urls:
                all_urls.extend(urls)

        if not all_urls:
            self._thread_safe_print(f"No data found")
            return 0, [], []

        # Download all files in parallel
        output_dir = os.path.join(self.output_dir, "raw_fits")
        total, downloaded, skipped = self._download_files_parallel(
            all_urls, output_dir, max_workers=max_workers, label=label
        )

        if len(skipped) == total:
            self._thread_safe_print(f"(already downloaded, {total} files)")
        elif len(downloaded) > 0:
            self._thread_safe_print(f"({downloaded} downloaded, {skipped} skipped)")
        else:
            self._thread_safe_print(f"(failed to download)")

        return total, downloaded, skipped

    def download_hmi_velocity(self, start_tai, end_tai=None, cadence=None, t_key_base=None):
        """Download HMI velocity"""
        query = self.build_query(
            RECORDS["hmi_vel"]["jsoc_query"], start_tai, end_tai, cadence
        )
        count, downloaded, skipped = self._download_query(query, "HMI Velocity")
        return count, downloaded, skipped
