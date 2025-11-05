import json
import numpy as np
import rasterio
import torch
import yaml
import queue
import threading

from datetime import datetime
from rasterio.features import shapes

from lib.data_preparer import QA_INDICES
from lib.consts import NO_DATA, NO_DATA_FLOAT, MEANS, STDS
from terratorch.cli_tools import LightningInferenceModel

class Infer:
    def __init__(self, config, checkpoint, max_queue_size=4, num_streams=2):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cuda = torch.cuda.is_available()

        # Stream and queue initialization
        self.num_streams = num_streams if self.use_cuda else 0
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)] if self.use_cuda else []
        self.max_queue_size = max_queue_size

        # Pre-allocate tensors on device for reuse (will be sized dynamically)
        self.device_tensor_cache = {}

        self.load_model()

        # Use proper mean and std from consts if not in config.
        self.means = np.asarray(self.config['data']['init_args'].get('means', MEANS))
        self.stds = np.asarray(self.config['data']['init_args'].get('stds', STDS))
        if len(self.means) > 0 and len(self.stds) > 0:
            self.means = torch.from_numpy(self.means).view(-1, 1, 1)
            self.stds = torch.from_numpy(self.stds).view(-1, 1, 1)
            # Move to device for faster operations
            if self.use_cuda:
                self.means = self.means.to(self.device)
                self.stds = self.stds.to(self.device)

    def load_model(self):
        if not(self.model):
            inference_model = LightningInferenceModel.from_config(self.config_filename, self.checkpoint_filename)
            self.model = inference_model.model
            self.model.to(self.device)
            self.model = self.model.eval()

            # Enable cudnn benchmarking for optimized convolution algorithms
            if self.use_cuda:
                torch.backends.cudnn.benchmark = True

    def postprocess(self, bbox, date, predictions, images):
        return predictions

    def preprocess(self, images, date):
        """
        Optimized preprocessing with pinned memory for faster transfers
        """
        images_array = []
        profiles = []
        coords = []
        temporal = []
        date = datetime.strptime(date, '%Y-%m-%d')
        julian_year, julian_day = int(datetime.strftime(date, "%Y")), int(datetime.strftime(date, "%j"))

        for image in images:
            with rasterio.open(image) as raster_file:
                image = raster_file.read()[:6]  # Read first 6 bands
                image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)

                # Use pinned memory for faster CPU->GPU transfers
                if self.use_cuda:
                    image_tensor = torch.from_numpy(image).pin_memory()
                else:
                    image_tensor = torch.from_numpy(image)

                # Normalization moved to GPU if available
                if len(self.means) > 0 and len(self.stds) > 0 and self.use_cuda:
                    # Transfer to GPU and normalize there
                    image_tensor = image_tensor.to(self.device, non_blocking=True)
                    for band in range(image_tensor.shape[0]):
                        band_mask = image_tensor[band] == NO_DATA_FLOAT
                        image_tensor[band][~band_mask] = ((image_tensor[band][~band_mask].float() - self.means[band]) / self.stds[band]).to(image_tensor.dtype)
                    # Move back to CPU for batching
                    image_tensor = image_tensor.cpu()
                elif len(self.means) > 0 and len(self.stds) > 0:
                    # CPU normalization (original behavior)
                    for band in range(image_tensor.shape[0]):
                        band_mask = image_tensor[band] == NO_DATA_FLOAT
                        image_tensor[band][~band_mask] = ((image_tensor[band][~band_mask].float() - self.means[band]) / self.stds[band]).to(image_tensor.dtype)

                images_array.append(image_tensor)
                coords.append(raster_file.lnglat())
                temporal.append([julian_year, julian_day])
                profiles.append(raster_file.profile)

        # Stack into a tensor
        imgs_tensor = torch.stack(images_array).float()

        # Use pinned memory for the final tensor
        if self.use_cuda and not imgs_tensor.is_pinned():
            imgs_tensor = imgs_tensor.pin_memory()

        # Increase dimensions to match input size
        processed_images = imgs_tensor.unsqueeze(2)
        return processed_images, profiles, coords, temporal

    def preprocess_stream(self, images, date, stream_idx=0):
        """
        Preprocessing optimized for streaming with specific CUDA stream
        """
        stream = self.streams[stream_idx] if self.use_cuda and stream_idx < len(self.streams) else None

        with torch.cuda.stream(stream) if stream else torch.no_grad():
            return self.preprocess(images, date)

    def calculate_area_from_mask(self, prediction_file, mask_values=[1]):
        """
        Calculates the area in square kilometers of a mask in a GeoTIFF file.

        Args:
            tiff_file_path (str): The path to the GeoTIFF file.
            mask_value (int): The pixel value representing the masked area.
                            Defaults to 1 for binary masks.

        Returns:
            float: The calculated area in square kilometers.
        """
        try:
            # Open the GeoTIFF file
            areas = {}
            with rasterio.open(prediction_file) as src:
                for mask_value in mask_values:
                    # Read the raster data into a NumPy array
                    data = src.read(1)  # Assuming the mask is in the first band
                    # Get the cell size (resolution) in the file's units (e.g., meters)
                    cell_width, cell_height = src.res
                    # Calculate the area of a single cell in square meters
                    cell_area_sqm = abs(cell_width * cell_height)
                    # Count the number of pixels that match the mask_value
                    masked_pixel_count = np.sum(data == mask_value)
                    # Calculate the total area in square meters
                    total_area_sqm = masked_pixel_count * cell_area_sqm
                    # Convert the area from square meters to square kilometers
                    total_area_sqkm = total_area_sqm / 1_000_000
                    areas[mask_value] = total_area_sqkm
                return areas
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening or reading file: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def qa_flags_to_tif(self, image_file, qa_flags, timeseries=False):
        """
        Convert predicted masks to GeoJSON format.
        Args:
            image_files (list): List of input image file paths.

        Returns:
            list: List of GeoJSON features.
        """
        with rasterio.open(image_file) as src:
            profile = src.profile
            tile = src.read()
            if timeseries:
                mask = tile[15]
            else:
                mask = tile[6]
            transform = profile['transform']
            mask = mask.astype('uint8')  # Ensure mask is in uint8 format

            # Save mask as GeoTIFF

            output_tif = image_file.replace('.tif', '_qa_mask.tif')
            with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=mask.dtype,
                crs=profile['crs'],
                transform=transform,
            ) as dst:
                dst.write(mask, 1)
        return output_tif

    def infer_batch_streaming(self, batch_queue, result_queue, stream_idx=0):
        """
        Process batches from queue using a specific CUDA stream
        Enables pipeline parallelism between data loading and inference
        """
        stream = self.streams[stream_idx] if self.use_cuda and stream_idx < len(self.streams) else None

        while True:
            try:
                batch_data = batch_queue.get(timeout=1)
                if batch_data is None:  # Sentinel value to stop
                    break

                images, date, batch_idx = batch_data

                with torch.cuda.stream(stream) if stream else torch.no_grad():
                    processed_images, profiles, coords, temporal = self.preprocess(images, date)

                    # Transfer to device with non-blocking (asynchronous) transfer
                    images_device = processed_images.to(self.device, non_blocking=True)

                    # Forward pass
                    result = self.model(images_device)

                    # Move results back to CPU asynchronously
                    results = result.output.detach().cpu()

                    # Synchronize stream before putting in queue
                    if stream:
                        stream.synchronize()

                    result_queue.put((batch_idx, results, profiles))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in streaming inference: {e}")
                result_queue.put((None, None, None))
                break

    def infer(self, images, date):
        """
        Optimized inference with pinned memory and CUDA streams
        Args:
            images (list): List of images
        """
        # forward the model
        with torch.no_grad():
            images, profiles, coords, temporal = self.preprocess(images, date)

            # Use non-blocking transfer with pinned memory
            images_device = images.to(self.device, non_blocking=True)

            # Synchronize to ensure transfer is complete before inference
            if self.use_cuda:
                torch.cuda.synchronize()

            result = self.model(images_device)

            predicted_masks = list()
            # Use non-blocking transfer back to CPU
            results = result.output.detach()

            # Process results
            num_classes = self.config['model']['init_args']['model_args']['num_classes']
            threshold = self.config.get('threshold', 0.5)

            for index, mask in enumerate(results):
                # Already on GPU, process there then move to CPU
                if num_classes == 1:
                    updated_mask = torch.sigmoid(mask.clone()).squeeze(0)
                    predicted_mask = (updated_mask > threshold).int().cpu()
                else:
                    # Process on GPU for speed
                    probabilities = torch.softmax(mask, dim=0)
                    predicted_mask = torch.argmax(probabilities, dim=0).cpu().numpy()

                predicted_masks.append(predicted_mask)

            # Clear GPU cache to free memory
            if self.use_cuda:
                torch.cuda.empty_cache()

            return predicted_masks, profiles

    def infer_streaming(self, image_batches, date, num_workers=1):
        """
        Streaming inference using queues and multiple CUDA streams
        Enables pipeline parallelism: while one batch is being transferred, another is being processed

        Args:
            image_batches (list): List of image batches
            date (str): Date string
            num_workers (int): Number of worker threads (default 1, max num_streams)

        Returns:
            tuple: (predicted_masks, profiles)
        """
        if not self.use_cuda or not image_batches:
            # Fall back to standard inference
            all_masks = []
            all_profiles = []
            for images in image_batches:
                masks, profs = self.infer(images, date)
                all_masks.extend(masks)
                all_profiles.extend(profs)
            return all_masks, all_profiles

        num_workers = min(num_workers, self.num_streams, len(image_batches))
        batch_queue = queue.Queue(maxsize=self.max_queue_size)
        result_queue = queue.Queue()

        # Start worker threads
        workers = []
        for i in range(num_workers):
            worker = threading.Thread(
                target=self.infer_batch_streaming,
                args=(batch_queue, result_queue, i % self.num_streams)
            )
            worker.start()
            workers.append(worker)

        # Enqueue batches
        for idx, images in enumerate(image_batches):
            batch_queue.put((images, date, idx))

        # Send sentinel values to stop workers
        for _ in range(num_workers):
            batch_queue.put(None)

        # Collect results
        results_dict = {}
        for _ in range(len(image_batches)):
            batch_idx, results, profiles = result_queue.get()
            if batch_idx is not None:
                results_dict[batch_idx] = (results, profiles)

        # Wait for all workers to complete
        for worker in workers:
            worker.join()

        # Reconstruct ordered results
        all_masks = []
        all_profiles = []
        for idx in sorted(results_dict.keys()):
            results, profiles = results_dict[idx]

            # Post-process results
            num_classes = self.config['model']['init_args']['model_args']['num_classes']
            threshold = self.config.get('threshold', 0.5)

            for mask in results:
                if num_classes == 1:
                    updated_mask = torch.sigmoid(mask.clone()).squeeze(0)
                    predicted_mask = (updated_mask > threshold).int()
                else:
                    probabilities = torch.softmax(mask, dim=0)
                    predicted_mask = torch.argmax(probabilities, dim=0).numpy()
                all_masks.append(predicted_mask)

            all_profiles.extend(profiles)

        return all_masks, all_profiles
