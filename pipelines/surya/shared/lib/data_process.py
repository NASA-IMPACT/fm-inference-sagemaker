"""
SURYA-EXACT PREPROCESSING - JSOC-FRESH VERSION
Matches Surya's pipeline but fetches correction tables and pointing tables
directly from JSOC (not pre-downloaded versions)

This ensures you always use the latest calibration data.
"""

import os
import re
import gc
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from sunpy.map import Map, contains_full_disk

# AIA/HMI processing
import aiapy.calibrate as ac
from aiapy.calibrate import (
    normalize_exposure,
    register,
    update_pointing,
    correct_degradation,
)
from skimage.transform import SimilarityTransform, warp
from sunpy.util.exceptions import SunpyUserWarning, SunpyMetadataWarning
from aiapy.util.exceptions import AiapyUserWarning
from lib.consts import SATURATION, TARGET_SOLAR_RADIUS, IMAGE_SHAPE, FITS_DIR

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=SunpyMetadataWarning)
warnings.simplefilter("ignore", category=AiapyUserWarning)
warnings.simplefilter("ignore", category=SunpyUserWarning)

OUTPUT_DIR = os.path.join(FITS_DIR, "processed_netcdf")

HIA_MAP_VAR_DICT = {
    "V": {"var_name": "hmi_v", "unit": "m/s", "description": "HMI LOS Dopplergrams"},
    "M": {"var_name": "hmi_m", "unit": "Gauss", "description": "HMI LOS Magnetograms"},
    "Bx": {
        "var_name": "hmi_bx",
        "unit": "Gauss",
        "description": "x-component of HMI vector magnetic field",
    },
    "By": {
        "var_name": "hmi_by",
        "unit": "Gauss",
        "description": "y-component of HMI vector magnetic field",
    },
    "Bz": {
        "var_name": "hmi_bz",
        "unit": "Gauss",
        "description": "z-component of HMI vector magnetic field",
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_correction_table():
    try:
        correction_table = ac.util.get_correction_table()
        return correction_table
    except Exception:
        return None


CORRECTION_TABLE = load_correction_table()


class DataProcess:
    def __init__(self, aia_files, hmi_files):
        self.map = None
        self.aia_files = aia_files
        self.hmi_files = hmi_files

    def scale_solardisk(self, m, map_type, target_solar_radius=TARGET_SOLAR_RADIUS):
        """EXACT from Surya's helio.py"""
        mdata = m.data

        if map_type == "aia":
            valid_mask = (mdata > 0).astype(float)
            mdata[mdata <= 0.0] = 0.0
        elif map_type == "hmi":
            valid_mask = (mdata < 1.0e6).astype(float)
        else:
            raise ValueError("The type of the map is not recognized.")

        rad = m.meta["RSUN_OBS"]
        scale_factor = target_solar_radius / rad

        shape_center = mdata.shape[0] / 2.0
        translation = (shape_center - scale_factor * shape_center,) * 2

        transform = SimilarityTransform(scale=scale_factor, translation=translation)

        scaled_mdata = warp(
            mdata,
            transform.inverse,
            preserve_range=True,
            mode="edge",
            output_shape=mdata.shape,
            order=0,
        )
        scaled_validmask = warp(
            valid_mask,
            transform.inverse,
            preserve_range=True,
            mode="edge",
            output_shape=mdata.shape,
            order=0,
        )

        scaled_mdata /= scaled_validmask + 1e-8

        new_meta = m.meta.copy()
        new_meta["RSUN_FIX"] = target_solar_radius
        new_meta["RFIX_COM"] = (
            "[arcsec] Target solar radius achieved by scaling the solar disk."
        )

        map_scaled = m._new_instance(scaled_mdata, new_meta)
        return map_scaled

    def process_aia_map(self, map_lev1, pointing_tbl=None):
        """
        EXACT from Surya's helio.py
        If pointing_tbl=None, aiapy will fetch it automatically from JSOC
        """
        # 1: update pointing (fetches from JSOC if pointing_tbl=None)
        m_updated_pointing = update_pointing(map_lev1, pointing_table=pointing_tbl)

        # 2: Register
        map_lev15 = register(m_updated_pointing, missing=None)

        # 3: Pad if necessary
        if (map_lev15.data).shape != IMAGE_SHAPE:
            temp_map = map_lev15
            xdata = np.pad(temp_map.data, ((1, 1), (1, 1)), mode="constant")
            temp_map.meta["naxis1"], temp_map.meta["naxis2"] = IMAGE_SHAPE
            temp_map.meta["crpix1"] += 1
            temp_map.meta["crpix2"] += 1
            padded_map = temp_map._new_instance(xdata, temp_map.meta)
            map_lev15 = padded_map

        # 4: Normalize exposure
        aia_map = normalize_exposure(map_lev15)

        # 5: Correct degradation
        if CORRECTION_TABLE is not None:
            aia_corrected = correct_degradation(
                aia_map, correction_table=CORRECTION_TABLE
            )
        else:
            print("Skipping degradation correction (table unavailable)")
            aia_corrected = aia_map

        # 6: Scale solar disk
        aia_map_scaled = self.scale_solardisk(
            aia_corrected, map_type="aia", target_solar_radius=TARGET_SOLAR_RADIUS
        )

        # 7: Clip to valid range
        aia_map_scaled.data[aia_map_scaled.data > SATURATION] = SATURATION
        aia_map_scaled.data[aia_map_scaled.data < 0] = 0

        return aia_map_scaled

    def process_hmi_map(self, map_lev1, aia_wcs):
        """EXACT from Surya's helio.py"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SunpyUserWarning)
            mp = map_lev1.reproject_to(aia_wcs)

        mp.meta["RSUN_OBS"] = map_lev1.meta["RSUN_OBS"]
        hmi_map_scaled = self.scale_solardisk(
            mp, map_type="hmi", target_solar_radius=TARGET_SOLAR_RADIUS
        )
        return hmi_map_scaled

    def hmi_disambig(self, azimuth_map, disambig_map, method=2):
        """EXACT from Surya's helio.py"""
        azimuth = azimuth_map.data.copy()
        disambig = disambig_map.data

        nx, ny = azimuth.shape
        nx1, ny1 = disambig.shape
        if nx != nx1 or ny != ny1:
            print("Dimensions of two images do not agree")
            return azimuth_map

        disambig = disambig.astype(int)

        if method < 0 or method > 2:
            method = 2
            print("Invalid disambiguation method, set to default method = 2")

        disambig = disambig // (2**method)
        odd_value_index_mask = disambig % 2 != 0
        azimuth[odd_value_index_mask] += 180

        disambiguated_azimuth_map = azimuth_map._new_instance(azimuth, azimuth_map.meta)
        return disambiguated_azimuth_map

    def vector2components(self, field_map, inclination_map, azimuth_disambiguated_map):
        """EXACT from Surya's helio.py"""
        field_data = field_map.data.copy()
        inclination_data = inclination_map.data.copy()
        azimuth_data = azimuth_disambiguated_map.data.copy()

        dtor = np.pi / 180.0
        bad_index_mask = field_data < -1.0e7
        field_data[bad_index_mask] = np.nan

        Bx = field_data * np.sin(inclination_data * dtor) * np.sin(azimuth_data * dtor)
        By = -field_data * np.sin(inclination_data * dtor) * np.cos(azimuth_data * dtor)
        Bz = field_data * np.cos(inclination_data * dtor)

        Bx[bad_index_mask] = np.nan
        By[bad_index_mask] = np.nan
        Bz[bad_index_mask] = np.nan

        Bx_map = field_map._new_instance(Bx, field_map.meta)
        By_map = field_map._new_instance(By, field_map.meta)
        Bz_map = field_map._new_instance(Bz, field_map.meta)

        return Bx_map, By_map, Bz_map

    def make_hmi_vector(
        self, azimuth_file, disambig_file, field_file, inclination_file
    ):
        """EXACT from Surya's helio.py"""
        field_map = Map(field_file)
        inclination_map = Map(inclination_file)
        azimuth_map = Map(azimuth_file)
        disambig_map = Map(disambig_file)

        azimuth_disambiguated_map = self.hmi_disambig(azimuth_map, disambig_map, 2)
        Bx_map, By_map, Bz_map = self.vector2components(
            field_map, inclination_map, azimuth_disambiguated_map
        )

        return Bx_map, By_map, Bz_map

    def compute_timestamp(self, fname):
        """EXACT from Surya's helio.py"""
        # AIA pattern
        pattern = re.compile(r"(s\.)(.*)Z\.(\d+)(\.image)")
        match = pattern.search(fname)
        if match:
            t = match.group(2)
            format_string = "%Y-%m-%dT%H%M%S"
            t_obs = datetime.strptime(t, format_string)
            nearest_12min = round(t_obs.minute / 12) * 12
            deltaT = nearest_12min * 60 - (60 * t_obs.minute + t_obs.second)
            t = t_obs + timedelta(seconds=deltaT)
            t_key = f"{t.year}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}"
            return t_key

        # HMI pattern
        pattern = re.compile(r"\.(\d{8})_(\d{6})_TAI")
        match = pattern.search(fname)
        if match:
            ymd = match.group(1)
            hms = match.group(2)
            return f"{ymd}_{hms[:4]}"

        raise ValueError(f"Cannot parse timestamp from {fname}")

    def process_aia_channels(self):
        encoding = {}
        data_arrays = {}
        aia_wcs = None

        for fname in sorted(self.aia_files):
            try:
                _map = Map(fname)
                wavelnth = _map.meta["wavelnth"]
                print(f"Processing AIA {wavelnth}...", end="")
                # Quality check (STRICT - skip entire timestamp if any channel is bad)
                if _map.meta.get("quality", 0) != 0:
                    return False

                # Store original metadata
                original_meta = dict(_map.meta)

                # Process AIA map (pointing table fetched automatically by aiapy)
                aia_map = self.process_aia_map(_map, pointing_tbl=None)

                # Store processed metadata
                updated_meta = dict(aia_map.meta)

                # Get data
                xdata = aia_map.data.astype(np.float32)
                np.nan_to_num(xdata, copy=False, nan=0.0)

                # Store in xarray
                var_name = f"aia{wavelnth}"
                data_arrays[var_name] = xr.DataArray(
                    xdata,
                    name=var_name,
                    dims=["y", "x"],
                    attrs={
                        "unit": "DN/s",
                        "t_obs": original_meta.get("t_obs", ""),
                        "qflag": original_meta.get("quality", 0),
                        "description": f"Level-1.5 AIA image for wavelength {wavelnth} Angstrom",
                        "meta_0": json.dumps(original_meta),
                        "meta_1": json.dumps(updated_meta),
                    },
                )

                # Compression settings
                encoding[var_name] = {
                    "zlib": True,
                    "complevel": 5,
                    "chunksizes": (4096, 4096),
                }

                # Get WCS from 171 for HMI alignment
                if wavelnth == 171:
                    aia_wcs = aia_map.wcs

                print(f"  AIA {wavelnth}: ✓ (mean={xdata.mean():.1f} DN/s)")

            except Exception as e:
                print(f"  AIA {wavelnth}: ✗ Error: {str(e)}")
                return False

        return data_arrays, encoding, aia_wcs

    def process_hmi_channels(self, hmi_files, aia_wcs):
        hmi_maps = {}

        def filter_map(map):
            """Filter out bad quality and non-full-disk maps"""
            if map.meta.get("quality", 0) != 0:
                return None
            if not contains_full_disk(map):
                return None
            return map

        # LOS Magnetogram
        if hmi_files["magnetogram"]:
            hmi_map = Map(hmi_files["magnetogram"])
            hmi_maps["M"] = filter_map(hmi_map)

        # Doppler
        if hmi_files["doppler"]:
            hmi_map = Map(hmi_files["doppler"])
            hmi_maps["V"] = filter_map(hmi_map)

        # Vector components
        if all(hmi_files[k] for k in ["azimuth", "disambig", "field", "inclination"]):
            try:
                Bx_map, By_map, Bz_map = self.make_hmi_vector(
                    hmi_files["azimuth"],
                    hmi_files["disambig"],
                    hmi_files["field"],
                    hmi_files["inclination"],
                )
                hmi_maps["Bx"] = Bx_map
                hmi_maps["By"] = By_map
                hmi_maps["Bz"] = Bz_map
                print("  HMI_Bx/By/Bz: ✓")
            except Exception as e:
                print(f"  HMI vector: ✗ Error: {str(e)[:60]}")

        return hmi_maps

    def process_timestamp(self, fits_path, timestamp, output_file=None):
        """
        Process one timestamp - EXACT match to Surya's pipeline
        """

        data_arrays = {}
        encoding = {}
        aia_wcs = None

        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"{timestamp}.nc")

        # Skip if already exists
        if os.path.exists(output_file):
            print(f"{timestamp} already exists, skipping")
            return output_file

        if len(self.aia_files) == 0:
            return False

        aia_result = self.process_aia_channels()
        if aia_result is False:
            print(f"AIA processing failed for {timestamp}")
            return False

        data_arrays, encoding, aia_wcs = aia_result

        hmi_maps = self.process_hmi_channels(self.hmi_files, aia_wcs)

        # Process all HMI maps
        for suffix, m in hmi_maps.items():
            try:
                # Store original metadata
                original_meta = dict(m.meta)

                # Process HMI map (align with AIA)
                hmi_map = self.process_hmi_map(m, aia_wcs)

                # Store processed metadata
                updated_meta = dict(hmi_map.meta)

                # Get data
                xdata = hmi_map.data.astype(np.float32)
                np.nan_to_num(xdata, copy=False, nan=0.0)

                # Determine variable name and metadata
                var_info = HIA_MAP_VAR_DICT.get(suffix, None)
                if var_info is None:
                    continue

                var_name = var_info["var_name"]
                unit = var_info["unit"]
                desc = var_info["description"]
                # Store in xarray
                data_arrays[var_name] = xr.DataArray(
                    xdata,
                    name=var_name,
                    dims=["y", "x"],
                    attrs={
                        "unit": unit,
                        "t_obs": original_meta.get("t_obs", ""),
                        "qflag": original_meta.get("quality", 0),
                        "description": desc,
                        "meta_0": json.dumps(original_meta),
                        "meta_1": json.dumps(updated_meta),
                    },
                )

                encoding[var_name] = {
                    "zlib": True,
                    "complevel": 5,
                    "chunksizes": (4096, 4096),
                }

            except Exception as e:
                print(f"  {var_name}: ✗ Error: {str(e)[:60]}")

        # Check we have 13 channels
        if len(data_arrays) != 13:
            print(f"\nOnly {len(data_arrays)}/13 channels available, skipping")
            return False

        # Create dataset
        attrs = {
            "title": "SDO data",
            "author": "R2O",
            "institution": "UAH/ODSI",
            "data_time": timestamp,
            "production_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "degradation_correction": "applied"
            if CORRECTION_TABLE is not None
            else "not_applied",
            "pointing_source": "JSOC_realtime",
        }

        ds = xr.Dataset(data_arrays, attrs=attrs)

        # Write to file
        gc.collect()
        ds.to_netcdf(
            path=output_file,
            format="NETCDF4",
            engine="h5netcdf",
            encoding=encoding,
            mode="w",
        )

        ds.close()
        del ds

        return output_file
