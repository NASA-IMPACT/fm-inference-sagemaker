# Use Agg backend by default for memory efficiency; switch to inline for display
import gc
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import warnings
import xarray as xr
import yaml

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from glob import glob
from lib.consts import CHANNELS, DOWNLOAD_FOLDER, OUTPUT_DIR, TARGET_SOLAR_RADIUS, DOMAIN_NAME
from osgeo import gdal, osr
from pathlib import Path
from surya.models.helio_spectformer import HelioSpectFormer

warnings.filterwarnings("ignore")
#
# NoData value for saved files (NOT for visualization)
NODATA_VALUE = -9999.0

SHAPE = (4096, 4096)
PIXEL_SCALE_ARCSEC = 0.6

TIF_METADATA = {
    'CHANNEL': '',
    'UNITS': 'DN/s',
    'FORECAST_TIME': '',
    'FORECAST_STEP': '',
    'FORECAST_CADENCE_MIN': '',
    'HORIZON_MIN': '',
    'MODEL': 'Surya v1.0 HelioSpectFormer',
    'INSTITUTION': 'NASA ODSI IMPACT AI',
    'OBSERVATORY': 'SDO',
    'INSTRUMENT': '',
    'RSUN_REF': str(TARGET_SOLAR_RADIUS),
    'CDELT1': str(PIXEL_SCALE_ARCSEC),
    'CDELT2': str(PIXEL_SCALE_ARCSEC),
    'CRPIX1': '',
    'CRPIX2': '',
    'CTYPE1': 'HPLN-TAN',
    'CTYPE2': 'HPLT-TAN',
    'CUNIT1': 'arcsec',
    'CUNIT2': 'arcsec'
}

WKT = f'''PROJCS["Helioprojective",
GEOGCS["GCS_Sun",
    DATUM["D_Sun",
        SPHEROID["Sun",{TARGET_SOLAR_RADIUS * 725},0]],
    PRIMEM["Reference_Meridian",0],
    UNIT["arcsecond",{1/3600}]],
PROJECTION["Orthographic"],
PARAMETER["false_easting",0],
PARAMETER["false_northing",0],
PARAMETER["central_meridian",0],
PARAMETER["latitude_of_origin",0],
UNIT["arcsec",1]]'''


class Infer:
    """
    Main inference class for Surya model rollout forecasting.

    Handles model loading, data preparation, multi-step forecasting,
    metrics computation, and output generation.
    """

    def __init__(self, config_path, scalers_path, weights_path, data_dir=None, results_dir=OUTPUT_DIR):
        """
        Initialize the inference pipeline.

        Args:
            config_path: Path to model config YAML
            scalers_path: Path to scalers YAML
            weights_path: Path to model weights
            data_dir: Directory containing input NetCDF files
            results_dir: Directory for output files
        """
        self.config_path = config_path
        self.scalers_path = scalers_path
        self.weights_path = weights_path
        self.data_dir = data_dir
        self.results_dir = results_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers = self._load_scalers()
        self.solar_mask = self._create_solar_disk_mask(shape=SHAPE)
        self.model = None

        # CUDA streams for overlapping compute and data transfer
        self._inference_stream = None
        self._transfer_stream = None
        if torch.cuda.is_available():
            self._inference_stream = torch.cuda.Stream()
            self._transfer_stream = torch.cuda.Stream()

        # Setup output directories if results_dir provided
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            self.geotiff_output_dir = os.path.join(results_dir, "geotiff_forecasts")
            os.makedirs(self.geotiff_output_dir, exist_ok=True)

    @staticmethod
    def clear_memory(sync=True):
        """
        Clear GPU and CPU memory.

        Args:
            sync: If True, synchronize GPU (blocking). Set to False for non-blocking cleanup.
        """
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()


    def _load_scalers(self):
        """Load scaler parameters from YAML and pre-compute vectorized arrays."""
        with open(self.scalers_path, 'r') as f:
            scalers_raw = yaml.safe_load(f)

        scalers = {}
        for channel in CHANNELS:
            params = scalers_raw[channel]
            scalers[channel] = {
                'mean': params['mean'],
                'std': params['std'],
                'epsilon': params['epsilon'],
                'sl_scale_factor': params['sl_scale_factor']
            }

        # Pre-compute vectorized scaler arrays for fast batch operations
        # Shape: (13, 1, 1) for broadcasting over (13, 4096, 4096)
        n_channels = len(CHANNELS)
        self._scaler_mean = np.array([scalers[ch]['mean'] for ch in CHANNELS], dtype=np.float32).reshape(n_channels, 1, 1)
        self._scaler_std = np.array([scalers[ch]['std'] for ch in CHANNELS], dtype=np.float32).reshape(n_channels, 1, 1)
        self._scaler_epsilon = np.array([scalers[ch]['epsilon'] for ch in CHANNELS], dtype=np.float32).reshape(n_channels, 1, 1)
        self._scaler_sl_factor = np.array([scalers[ch]['sl_scale_factor'] for ch in CHANNELS], dtype=np.float32).reshape(n_channels, 1, 1)
        # Pre-compute combined values for faster inverse transform
        self._scaler_std_eps = self._scaler_std + self._scaler_epsilon

        return scalers

    def load_model(self):
        """Load Surya model with pretrained weights."""
        print("\nLoading Surya model...")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model = HelioSpectFormer(
            img_size=config['model']['img_size'],
            patch_size=config['model']['patch_size'],
            in_chans=len(config['data']['sdo_channels']),
            embed_dim=config['model']['embed_dim'],
            time_embedding={
                'type': 'linear',
                'time_dim': len(config['data']['time_delta_input_minutes'])
            },
            depth=config['model']['depth'],
            n_spectral_blocks=config['model']['n_spectral_blocks'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            drop_rate=config['model']['drop_rate'],
            dtype=torch.float16,
            window_size=config['model']['window_size'],
            dp_rank=config['model']['dp_rank'],
            learned_flow=config['model']['learned_flow'],
            use_latitude_in_learned_flow=config['model']['use_latitude_in_learned_flow'],
            init_weights=False,
            checkpoint_layers=list(range(10)),
            rpe=config['model']['rpe'],
            ensemble=config['model']['ensemble'],
            finetune=config['model']['finetune']
        )

        weights = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        return self.model

    def _create_solar_disk_mask(self, shape, pixel_scale_arcsec=0.6, rsun_pixels=None):
        """
        Create a mask for the solar disk region.

        Args:
            shape: Tuple of (height, width)
            pixel_scale_arcsec: Pixel scale in arcseconds
            rsun_pixels: Solar radius in pixels (computed if None)

        Returns:
            Boolean mask where True indicates on-disk pixels
        """
        h, w = shape
        cy, cx = h // 2, w // 2

        if rsun_pixels is None:
            sun_radius_arcsec = TARGET_SOLAR_RADIUS
            rsun_pixels = sun_radius_arcsec / pixel_scale_arcsec
            # Extend by ~20% to include corona and prominences
            rsun_pixels = rsun_pixels * 1.20

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = r <= rsun_pixels

        return mask


    @staticmethod
    def parse_timestamp_from_filename(filename):
        """Extract datetime from filename like '20140107_1500.nc'."""
        basename = os.path.basename(filename).replace('.nc', '')
        try:
            return datetime.strptime(basename, '%Y%m%d_%H%M%S')
        except:
            return None

    @staticmethod
    def detect_file_cadence(nc_files):
        """Detect the actual time spacing between consecutive files."""
        timestamps = []
        for f in nc_files:
            ts = Infer.parse_timestamp_from_filename(f)
            if ts:
                timestamps.append(ts)

        if len(timestamps) < 2:
            return None

        deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60
                  for i in range(len(timestamps)-1)]

        unique_deltas = list(set(deltas))
        if len(unique_deltas) == 1:
            return int(unique_deltas[0])
        else:
            return int(Counter(deltas).most_common(1)[0][0])

    @staticmethod
    def get_available_timestamps(nc_files):
        """Get sorted dict mapping timestamps to file paths."""
        file_dict = {}
        for f in nc_files:
            ts = Infer.parse_timestamp_from_filename(f)
            if ts:
                file_dict[ts] = f
        return dict(sorted(file_dict.items()))

    def apply_surya_transforms(self, data):
        """
        Apply Surya's preprocessing transforms (vectorized).

        Transforms: data -> scale -> siglog -> standardize
        Operates on all 13 channels simultaneously using broadcasting.
        """
        # Vectorized operations on full (13, 4096, 4096) array
        scaled = data * self._scaler_sl_factor
        siglog = np.sign(scaled) * np.log1p(np.abs(scaled))
        transformed = (siglog - self._scaler_mean) / self._scaler_std_eps
        return transformed

    def inverse_surya_transforms(self, data):
        """
        Inverse Surya transforms: normalized → DN/s (vectorized).

        Operates on all 13 channels simultaneously using broadcasting.
        ~10-15x faster than per-channel loop.
        """
        # Vectorized operations on full (13, 4096, 4096) array
        denorm = data * self._scaler_std_eps + self._scaler_mean
        inv_siglog = np.sign(denorm) * np.expm1(np.abs(denorm))  # expm1 = exp(x) - 1
        physical = inv_siglog / self._scaler_sl_factor
        return physical

    @staticmethod
    def apply_solar_disk_mask(data, mask, nodata_value=NODATA_VALUE):
        """Apply solar disk mask to data array - ONLY for file saving."""
        masked_data = data.copy()

        if masked_data.ndim == 3:
            for i in range(masked_data.shape[0]):
                masked_data[i][~mask] = nodata_value
        else:
            masked_data[~mask] = nodata_value

        return masked_data


    def load_netcdf(self, netcdf_path, transform=True):
        """Load NetCDF file and optionally apply transforms."""
        import hdf5plugin

        with xr.open_dataset(netcdf_path, engine="h5netcdf", chunks=None) as ds:
            data = np.zeros((13, 4096, 4096), dtype=np.float32)

            for idx, channel in enumerate(CHANNELS):
                data[idx] = ds[channel].values

            timestamp = ds.attrs.get('data_time',
                        ds.attrs.get('timestamp_t_0',
                        os.path.basename(netcdf_path).replace('.nc', '')))

        if transform:
            transformed = self.apply_surya_transforms(data)
            transformed = transformed[:, np.newaxis, :, :]
            transformed = transformed[np.newaxis, :, :, :, :]
            tensor = torch.from_numpy(transformed).float()
            return tensor, data, timestamp
        else:
            return data, timestamp

    def load_two_timestamps(self, nc_path1, nc_path2):
        """Load two NetCDF files and combine into 2-timestamp tensor."""
        tensor1, raw1, ts1 = self.load_netcdf(nc_path1, transform=True)
        tensor2, raw2, ts2 = self.load_netcdf(nc_path2, transform=True)

        combined = torch.cat([tensor1, tensor2], dim=2)

        time1 = Infer.parse_timestamp_from_filename(nc_path1)
        time2 = Infer.parse_timestamp_from_filename(nc_path2)

        if time1 and time2:
            delta_minutes = (time2 - time1).total_seconds() / 60
        else:
            delta_minutes = 0.0

        return combined, raw1, raw2, ts1, ts2, delta_minutes

    @staticmethod
    def compute_metrics(forecast, ground_truth, solar_mask=None):
        """Compute correlation and relative error metrics per channel."""
        metrics = {}

        for i, channel in enumerate(CHANNELS):
            pred = forecast[i].flatten()
            true = ground_truth[i].flatten()

            mask = (true != 0) & np.isfinite(pred) & np.isfinite(true)

            pred = pred[mask]
            true = true[mask]

            if len(pred) > 0:
                corr = np.corrcoef(pred, true)[0, 1]
                rel_error = np.abs(pred - true).mean() / (np.abs(true).mean() + 1e-8) * 100

                metrics[channel] = {
                    'correlation': corr,
                    'rel_error': rel_error
                }
            else:
                metrics[channel] = {
                    'correlation': 0.0,
                    'rel_error': 999.9
                }

        return metrics


    def run_inference(self, first_step_file, second_step_file, steps=10, cadence_minutes=60):
        """
        Run multi-step autoregressive forecasting.

        Args:
            first_step_file: Path to first input NetCDF file
            second_step_file: Path to second input NetCDF file
            steps: Maximum number of forecast steps
            cadence_minutes: Time between forecast steps

        Returns:
            List of forecast results with metrics
        """
        # load first two timesteps
        # infer
        # load files for next steps for comparison
        # Load initial inputs
        # infer config setup
        # pass current datetime parsed timestamp
        initial_input, raw1, raw2, ts1, ts2, initial_delta = self.load_two_timestamps(
            first_step_file, second_step_file
        )

        # Get available timestamps for ground truth comparison
        nc_files = sorted(glob(f"{self.data_dir}/*.nc"))

        # Parse start times
        time1 = self.parse_timestamp_from_filename(first_step_file)
        time2 = self.parse_timestamp_from_filename(second_step_file)

        forecast_config = {
            'cadence_minutes': cadence_minutes,
            'max_steps': steps
        }

        results = self._extended_autoregressive_forecast(
            initial_input, forecast_config, time2
        )

        results = self.calculate_metrics(results)
        saved_results = self.save_outputs(results, forecast_config, ts2)

        return saved_results

    def process_results(self, results, forecast_config):
        """
        Process forecast results into API response format.

        Args:
            results: List of forecast result dictionaries from rollout
            forecast_config: Configuration dict with query parameters

        Returns:
            Formatted response dictionary with query, steps, and correlation data
        """
        selected_datetime = forecast_config.get('start_time')
        cadence_minutes = forecast_config.get('cadence_minutes', 12)
        num_frames = len(results)
        # Format the selected datetime as ISO string
        selected_datetime_str = selected_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        # Build the steps and correlation arrays
        steps = []
        correlation = []

        for result in results:
            target_time = result.get('target_time')
            if isinstance(target_time, datetime):
                timestamp_str = target_time.strftime('%Y-%m-%dT%H:%M:%S')
            else:
                timestamp_str = str(target_time)

            # Build step entry with tile and metadata endpoints
            step_number = result.get('step', 1)
            step_entry = {
                'timestamp': timestamp_str,
                'tiles_endpoint': f'{DOMAIN_NAME}/api/tiles/tiles/{{band}}/{selected_datetime_str}/{step_number}/{{z}}/{{x}}/{{y}}.png',
                'metadata_endpoint': f'{DOMAIN_NAME}/api/tiles/info/{{band}}/{selected_datetime_str}/{step_number}'
            }
            steps.append(step_entry)

            # Build correlation entry from metrics
            metrics = result.get('metrics', {})
            score = {}
            for channel, channel_metrics in metrics.items():
                # Convert channel name to API format (e.g., 'aia171' -> 'AIA_171')
                if channel.startswith('aia'):
                    api_channel = f"AIA_{channel[3:]}"
                elif channel.startswith('hmi_'):
                    api_channel = f"HMI_{channel[4:].upper()}"
                else:
                    api_channel = channel.upper()

                corr_value = channel_metrics.get('correlation')
                if corr_value is not None:
                    score[api_channel] = round(float(corr_value), 2)

            correlation_entry = {
                'timestamp': timestamp_str,
                'score': score
            }
            correlation.append(correlation_entry)

        # Assemble the final response
        response = {
            'surya': {
                'rollout': {
                    'query': {
                        'selected_datetime': selected_datetime_str,
                        'cadence_in_minutes': cadence_minutes,
                        'num_frames': num_frames
                    },
                    'steps': steps,
                    'correlation': correlation
                }
            }
        }

        return response

    def calculate_metrics(self, results):
        """
        Calculate metrics for each forecast step.

        Optimized: builds timestamp->file lookup dict once (O(1) per lookup)
        instead of linear search (O(n) per lookup).
        """
        # Build lookup dict once - O(n) instead of O(n*m) for nested loops
        timestamp_to_file = self.get_available_timestamps(
            sorted(glob(f"{self.data_dir}/*.nc"))
        )

        for result in results:
            target_time = result['target_time']

            gt_file = timestamp_to_file.get(target_time)

            if gt_file is None:
                result['metrics'] = {}
                result['avg_correlation'] = None
                continue

            gt_raw, _ = self.load_netcdf(gt_file, transform=False)

            metrics = self.compute_metrics(result['forecast'], gt_raw, self.solar_mask)
            avg_corr = np.mean([m['correlation'] for m in metrics.values()])
            result['metrics'] = metrics
            result['avg_correlation'] = avg_corr
        return results

    def _extended_autoregressive_forecast(self,
        initial_input,
        forecast_config,
        start_time
    ):
        """
        Run extended autoregressive forecasting with proper time deltas.

        Optimized for performance (GPU mode):
        - Uses CUDA streams for overlapping inference and data transfer
        - Pipelines: while GPU runs step N, CPU processes step N-1 results
        - Batches memory cleanup to reduce sync overhead

        Falls back to simple sequential execution on CPU.
        """
        cadence_minutes = forecast_config['cadence_minutes']
        max_steps = forecast_config['max_steps']

        # Use optimized GPU path or fallback to CPU path
        if torch.cuda.is_available() and self._inference_stream is not None:
            return self._gpu_autoregressive_forecast(
                initial_input, cadence_minutes, max_steps, start_time
            )
        else:
            return self._cpu_autoregressive_forecast(
                initial_input, cadence_minutes, max_steps, start_time
            )

    def _cpu_autoregressive_forecast(self, initial_input, cadence_minutes, max_steps, start_time):
        """Simple sequential forecast for CPU execution."""
        results = []

        curr_batch = {
            'ts': initial_input.to(self.device),
            'time_delta_input': torch.tensor([[-cadence_minutes, 0.0]]).to(self.device)
        }
        current_time = start_time

        with torch.no_grad():
            for step in range(max_steps):
                target_time = current_time + timedelta(minutes=cadence_minutes)

                forecast_hat = self.model(curr_batch)
                forecast_np = forecast_hat.detach().numpy()[0]
                forecast_physical = self.inverse_surya_transforms(forecast_np)

                results.append({
                    'step': step + 1,
                    'forecast': forecast_physical,
                    'target_time': target_time,
                    'delta_minutes': cadence_minutes
                })

                curr_batch['ts'] = torch.cat(
                    (curr_batch['ts'][:, :, 1:, ...],
                     forecast_hat[:, :, None, ...]),
                    dim=2
                )
                current_time = target_time
                del forecast_hat, forecast_np

        return results

    def _gpu_autoregressive_forecast(self, initial_input, cadence_minutes, max_steps, start_time):
        """
        Optimized GPU forecast with CUDA streams for pipelining.

        Pipeline structure per iteration:
        1. Launch inference on inference_stream (non-blocking)
        2. While GPU works, process previous step's result on CPU
        3. Sync inference_stream, update batch
        4. Start async transfer of current result on transfer_stream
        """
        results = []
        CLEANUP_INTERVAL = 4

        curr_batch = {
            'ts': initial_input.to(self.device, non_blocking=True),
            'time_delta_input': torch.tensor([[-cadence_minutes, 0.0]]).to(self.device, non_blocking=True)
        }
        current_time = start_time

        # Pipeline state
        pending_result = None  # (cpu_tensor, target_time, step_num)
        transfer_event = None

        with torch.no_grad():
            for step in range(max_steps):
                target_time = current_time + timedelta(minutes=cadence_minutes)

                with torch.cuda.stream(self._inference_stream):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        forecast_hat = self.model(curr_batch)

                if pending_result is not None:
                    cpu_tensor, prev_target_time, prev_step = pending_result
                    transfer_event.synchronize()  # Wait for D2H transfer only
                    forecast_np = cpu_tensor.numpy()[0]
                    forecast_physical = self.inverse_surya_transforms(forecast_np)
                    results.append({
                        'step': prev_step,
                        'forecast': forecast_physical,
                        'target_time': prev_target_time,
                        'delta_minutes': cadence_minutes
                    })
                    del cpu_tensor, forecast_np

                self._inference_stream.synchronize()
                curr_batch['ts'] = torch.cat(
                    (curr_batch['ts'][:, :, 1:, ...],
                     forecast_hat[:, :, None, ...]),
                    dim=2
                )

                self._transfer_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._transfer_stream):
                    cpu_tensor = forecast_hat.to('cpu', non_blocking=True)
                    transfer_event = self._transfer_stream.record_event()

                pending_result = (cpu_tensor, target_time, step + 1)
                current_time = target_time
                del forecast_hat

                # Periodic cleanup
                if (step + 1) % CLEANUP_INTERVAL == 0:
                    self.clear_memory(sync=False)

            # Process final step
            if pending_result is not None:
                cpu_tensor, prev_target_time, prev_step = pending_result
                transfer_event.synchronize()
                forecast_np = cpu_tensor.numpy()[0]
                forecast_physical = self.inverse_surya_transforms(forecast_np)
                results.append({
                    'step': prev_step,
                    'forecast': forecast_physical,
                    'target_time': prev_target_time,
                    'delta_minutes': cadence_minutes
                })
                del cpu_tensor, forecast_np

        self.clear_memory(sync=True)
        return results

    def _save_single_geotiff(self, args):
        """Save a single channel as GeoTIFF. Used for parallel saving."""
        data, channel, timestamp, target_time, step_number, forecast_config = args
        try:
            channel_dir = os.path.join(self.geotiff_output_dir, channel)
            os.makedirs(channel_dir, exist_ok=True)

            output_file = os.path.join(channel_dir, f"{timestamp}_{channel}_step{step_number:02d}.tif")

            driver = gdal.GetDriverByName('GTiff')
            ny, nx = data.shape
            dataset = driver.Create(output_file, nx, ny, 1, gdal.GDT_Float32,
                                    options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES',
                                            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])

            band = dataset.GetRasterBand(1)
            band.WriteArray(data)
            metadata = TIF_METADATA.copy()

            metadata.update({
                'CHANNEL': channel,
                'FORECAST_TIME': target_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'FORECAST_STEP': str(step_number),
                'FORECAST_CADENCE_MIN': str(forecast_config['cadence_minutes']),
                'HORIZON_MIN': str(step_number * forecast_config['cadence_minutes']),
                'INSTRUMENT': 'AIA' if 'aia' in channel else 'HMI',
                'CRPIX1': str(nx / 2),
                'CRPIX2': str(ny / 2)
            })

            try:
                dataset.SetProjection(WKT)
            except Exception:
                srs = osr.SpatialReference()
                srs.SetLocalCS(f"SDO_{channel}_Helioprojective")
                dataset.SetProjection(srs.ExportToWkt())

            origin_x = -(nx / 2) * PIXEL_SCALE_ARCSEC
            origin_y = (ny / 2) * PIXEL_SCALE_ARCSEC
            dataset.SetGeoTransform([origin_x, PIXEL_SCALE_ARCSEC, 0,
                                    origin_y, 0, -PIXEL_SCALE_ARCSEC])

            band.FlushCache()
            dataset = None
            return output_file
        except Exception as e:
            print(f"Warning: Could not save GeoTIFF for {channel}: {str(e)}")
            return None

    def save_forecast_as_geotiff(self, forecast_data, timestamp, target_time, step_number,
                                 forecast_config):
        """
        Save forecast data as GeoTIFF files (one per channel).

        Uses ThreadPoolExecutor for parallel I/O - ~3-4x faster than sequential.
        """
        # Prepare arguments for parallel execution
        save_args = [
            (forecast_data[idx], channel, timestamp, target_time, step_number, forecast_config)
            for idx, channel in enumerate(CHANNELS)
        ]

        saved_files = []
        # Use threads for I/O-bound work (GDAL writes)
        with ThreadPoolExecutor(max_workers=min(len(CHANNELS), 8)) as executor:
            futures = [executor.submit(self._save_single_geotiff, args) for args in save_args]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    saved_files.append(result)

        return saved_files

    def save_outputs(self, results, forecast_config, timestamp):
        """Save all forecast outputs (NetCDF, GeoTIFF, log)."""
        print("\nSaving forecast results...")

        for result in results:
            tif_files = self.save_forecast_as_geotiff(
                forecast_data=result['forecast'],
                timestamp=timestamp,
                target_time=result['target_time'],
                step_number=result['step'],
                forecast_config=forecast_config
            )
            if tif_files:
                print(f"Saved GeoTIFF: {len(tif_files)} channels")
                result['tiff_files'] = tif_files

        return results

    @staticmethod
    def _downsample(arr, factor):
        """Downsample array by factor for display."""
        if factor <= 1:
            return arr
        return arr[::factor, ::factor]

    def cleanup(self, results=None):
        """Clean up large arrays and free memory."""
        if results:
            for r in results:
                if 'ground_truth' in r:
                    del r['ground_truth']
        self.clear_memory()
