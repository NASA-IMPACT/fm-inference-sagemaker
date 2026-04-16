import numpy as np
import rasterio
import torch
import yaml

from datetime import datetime
from torch.utils.data import DataLoader

from lib.data_preparer import QA_INDICES, TileIterableDataset
from lib.consts import NO_DATA, NO_DATA_FLOAT, MEANS, STDS
from terratorch.cli_tools import LightningInferenceModel


class Infer:
    def __init__(self, config, checkpoint):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda = torch.cuda.is_available()

        # Configure CUDA optimizations before model loading
        if self.use_cuda:
            # Disable cudnn.benchmark - since batch sizes are fixed (padded to batch_size),
            # benchmark mode just adds variance without benefit. Setting to False uses
            # deterministic algorithm selection for consistent inference times.
            torch.backends.cudnn.benchmark = False
            # Enable TF32 for faster matrix multiplications and convolutions on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set float32 matmul precision to allow mixed precision operations
            torch.set_float32_matmul_precision("high")

        self.load_model()

        # Use proper mean and std from consts if not in config.
        self.means = np.asarray(self.config["data"]["init_args"].get("means", MEANS))
        self.stds = np.asarray(self.config["data"]["init_args"].get("stds", STDS))
        if len(self.means) > 0 and len(self.stds) > 0:
            self.means = torch.from_numpy(self.means).view(-1, 1, 1).float()
            self.stds = torch.from_numpy(self.stds).view(-1, 1, 1).float()
            # Keep CPU copies for preprocessing (avoids GPU bounce)
            self.means_cpu = self.means.clone()
            self.stds_cpu = self.stds.clone()
            # Move to device for any GPU-side operations
            if self.use_cuda:
                self.means = self.means.to(self.device)
                self.stds = self.stds.to(self.device)

    def load_model(self):
        if not (self.model):
            inference_model = LightningInferenceModel.from_config(
                self.config_filename, self.checkpoint_filename
            )
            self.model = inference_model.model
            self.model.to(self.device)
            self.model = self.model.eval()
            if self.use_cuda:
                # Use "reduce-overhead" mode for CUDA graph optimization
                # Requires consistent input shapes (handled by batch padding in DataPreparer)
                self.model = torch.compile(self.model)

    def postprocess(self, bbox, date, predictions, images):
        return predictions

    def preprocess(self, tiles, profiles, date):
        """
        Preprocess tiles (numpy arrays) into tensors for inference.
        Profiles are passed through for downstream mosaic operations.

        Args:
            tiles: list of numpy arrays (C, H, W) from DataPreparer
            profiles: list of rasterio profile dicts with transforms
            date: date string in 'YYYY-MM-DD' format
        """
        images_array = []
        coords = []
        temporal = []
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
        julian_year, julian_day = (
            int(datetime.strftime(parsed_date, "%Y")),
            int(datetime.strftime(parsed_date, "%j")),
        )

        for tile, profile in zip(tiles, profiles):
            # Use first 6 bands, handle NO_DATA
            image = tile[:6]
            image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)

            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image).float()

            # Vectorized normalization on CPU
            if len(self.means) > 0 and len(self.stds) > 0:
                mask = image_tensor != NO_DATA_FLOAT
                image_tensor = torch.where(
                    mask, (image_tensor - self.means_cpu) / self.stds_cpu, image_tensor
                )

            images_array.append(image_tensor)

            # Compute center coords from transform (equivalent to rasterio lnglat())
            transform = profile["transform"]
            width, height = profile["width"], profile["height"]
            center_lng = transform.c + width * transform.a / 2
            center_lat = transform.f + height * transform.e / 2
            coords.append((center_lng, center_lat))
            temporal.append([julian_year, julian_day])

        # Stack into a single tensor, then convert to BF16 and pin once
        imgs_tensor = torch.stack(images_array)

        if self.use_cuda:
            # Convert to BF16 and pin memory once for the whole batch
            # Pinning must happen in main process, not DataLoader workers
            imgs_tensor = imgs_tensor.bfloat16().pin_memory()
        else:
            imgs_tensor = imgs_tensor.float()

        # Increase dimensions to match input size
        processed_images = imgs_tensor.unsqueeze(2)
        return processed_images, profiles, coords, temporal

    def calculate_area_from_mask(self, prediction_file, mask_values=[1]):
        """
        Calculates the area in square kilometers of a mask in a GeoTIFF file.

        Args:
            prediction_file (str): The path to the GeoTIFF file.
            mask_values (list): List of pixel values to calculate areas for.

        Returns:
            dict: Mapping of mask_value -> area in square kilometers.
        """
        try:
            with rasterio.open(prediction_file) as src:
                # Read data once
                data = src.read(1)
                # Calculate cell area once
                cell_width, cell_height = src.res
                cell_area_sqkm = abs(cell_width * cell_height) / 1_000_000

            max_val = max(mask_values) if mask_values else 0
            if data.max() >= 0:
                counts = np.bincount(
                    data.ravel().astype(np.int32), minlength=max_val + 1
                )
                areas = {
                    v: counts[v] * cell_area_sqkm if v < len(counts) else 0.0
                    for v in mask_values
                }
            else:
                areas = {v: 0.0 for v in mask_values}

            return areas
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening or reading file: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def qa_flags_to_tif(self, image_file, qa_flags, timeseries=False):
        """
        Generate QA mask GeoTIFFs from the HLS QA band.
        Optimized: reads file once, computes all masks vectorized, writes in batch.
        """
        if not qa_flags:
            return {}

        with rasterio.open(image_file) as src:
            # Read only the QA band, not all bands
            qa_band_idx = 16 if timeseries else 7  # 1-indexed for rasterio
            qa_band = src.read(qa_band_idx).astype(np.uint32)
            transform = src.transform
            crs = src.crs
            height, width = qa_band.shape

        # Pre-compute all bit indices
        valid_flags = [
            (flag, QA_INDICES.get(flag))
            for flag in qa_flags
            if QA_INDICES.get(flag) is not None
        ]

        if not valid_flags:
            return {}

        # Pre-compute all masks at once (vectorized)
        masks = {}
        output_paths = {}
        for flag, bit_index in valid_flags:
            masks[flag] = ((qa_band & (1 << bit_index)) != 0).astype(np.uint8)
            output_paths[flag] = image_file.replace(".tif", f"_qa_{flag}.tif")

        # Shared profile for all QA masks
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": np.uint8,
            "crs": crs,
            "transform": transform,
        }

        # Batch write all masks
        qa_tifs = {}
        for flag in masks:
            with rasterio.open(output_paths[flag], "w", **profile) as dst:
                dst.write(masks[flag], 1)
            qa_tifs[flag] = output_paths[flag]

        return qa_tifs

    def infer(self, tiles, profiles, date):
        """
        Optimized inference with pinned memory and CUDA streams
        Args:
            tiles (list): List of numpy arrays (C, H, W) from DataPreparer
            profiles (list): List of rasterio profile dicts with transforms
            date (str): Date string in 'YYYY-MM-DD' format
        """
        # forward the model
        with torch.no_grad():
            images, profiles, coords, temporal = self.preprocess(tiles, profiles, date)

            # Use non-blocking transfer with pinned memory
            images_device = images.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.use_cuda
            ):
                result = self.model(images_device)

            # Use non-blocking transfer back to CPU
            results = result.output.detach()

            # Batched post-processing on GPU (fewer kernel launches)
            # results shape: (batch, num_classes, H, W)
            num_classes = self.config["model"]["init_args"]["model_args"]["num_classes"]
            threshold = self.config.get("threshold", 0.5)

            if num_classes == 1:
                # Shape: (B, 1, H, W) -> sigmoid -> squeeze -> (B, H, W) -> threshold -> cpu
                probs = torch.sigmoid(results).squeeze(1)
                predicted_masks = (probs > threshold).int().cpu().numpy()
                del probs
            else:
                # Shape: (B, C, H, W) -> softmax over C -> argmax over C -> (B, H, W) -> cpu
                probabilities = torch.softmax(results, dim=1)
                predicted_masks = torch.argmax(probabilities, dim=1).cpu().numpy()
                del probabilities

            del result, results, images_device, images

            # Convert to list of 2D arrays to match original return format
            predicted_masks = list(predicted_masks)

            return predicted_masks, profiles

    def infer_dataloader(self, data_preparer, date, num_workers=2, prefetch_factor=2):
        """
        Inference using DataLoader for prefetching.

        Uses DataLoader workers to prefetch batches in parallel while GPU
        inference runs, improving GPU utilization.

        Args:
            data_preparer: DataPreparer instance
            date (str): Date string in 'YYYY-MM-DD' format
            num_workers (int): Number of DataLoader workers for prefetching (default 2)
            prefetch_factor (int): Batches to prefetch per worker (default 2)

        Returns:
            tuple: (all_masks, all_profiles)
        """
        dataset = TileIterableDataset(data_preparer)

        # DataLoader with prefetching - batches are already formed by DataPreparer
        # persistent_workers=True: keeps workers alive to avoid spawn overhead on each request
        # This significantly reduces variance in inference times across requests
        loader = DataLoader(
            dataset,
            batch_size=None,  # DataPreparer already batches
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=False,  # We handle pinning in infer()
            persistent_workers=True if num_workers > 0 else False,
        )

        all_masks = []
        all_profiles = []

        for tiles_batch, profiles_batch, actual_count in loader:
            # Reuse existing infer method
            batch_masks, batch_profiles = self.infer(
                list(tiles_batch), list(profiles_batch), date
            )
            # Only use actual results (not padded dummy tiles)
            actual = int(actual_count)
            all_masks.extend(batch_masks[:actual])
            all_profiles.extend(batch_profiles[:actual])

        return all_masks, all_profiles
