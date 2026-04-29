import numpy as np
import torch


from datetime import datetime
from einops import rearrange

from lib.infer import Infer
from lib.consts import NO_DATA, NO_DATA_FLOAT
from terratorch.tasks import SemanticSegmentationTask


class CropClassificationInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)

    def load_model(self):
        if self.model:
            return
        model_args = {
            # Backbone
            "backbone": "prithvi_eo_v2_600_tl",
            "backbone_pretrained": False,
            "backbone_num_frames": 3,
            "backbone_bands": [
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
            ],
            "backbone_coords_encoding": [],
            # Necks
            "necks": [
                {"name": "SelectIndices", "indices": [7, 15, 23, 31]},
                {"name": "ReshapeTokensToImage", "effective_time_dim": 3},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            # Head
            "head_dropout": 0.1,
            "num_classes": 13,
        }
        self.model = SemanticSegmentationTask.load_from_checkpoint(
            self.checkpoint_filename,
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
        )
        self.model.to(self.device)
        self.model = self.model.eval()
        if self.use_cuda:
            # Use "default" mode to avoid CUDA graph conflicts with DataLoader workers
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def preprocess(self, tiles, profiles, date):
        """
        Preprocess tiles (numpy arrays) into tensors for inference.

        Args:
            tiles: list of numpy arrays (C, H, W) from DataPreparer (25 bands for timeseries)
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
            # Extract 3 time frames (6 bands each) from timeseries data
            stacked = []
            stacked.append(tile[:6])  # Read pre bands
            stacked.append(tile[9:15])  # Read current bands
            stacked.append(tile[18:24])  # Read post bands
            image = np.concatenate(stacked, axis=0)
            image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)

            # Convert to float tensor for normalization
            image_tensor = torch.from_numpy(image).float()

            # Vectorized normalization on CPU (avoids GPU bounce)
            # Tile means/stds to match 18 bands (3 time frames × 6 bands)
            if len(self.means) > 0 and len(self.stds) > 0:
                num_repeats = image_tensor.shape[0] // len(self.means_cpu)
                means_tiled = self.means_cpu.repeat(num_repeats, 1, 1)
                stds_tiled = self.stds_cpu.repeat(num_repeats, 1, 1)
                mask = image_tensor != NO_DATA_FLOAT
                image_tensor = torch.where(
                    mask, (image_tensor - means_tiled) / stds_tiled, image_tensor
                )

            images_array.append(image_tensor)

            # Compute center coords from transform
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
            imgs_tensor = imgs_tensor.bfloat16().pin_memory()
        else:
            imgs_tensor = imgs_tensor.float()

        # Increase dimensions to match input size
        processed_images = rearrange(imgs_tensor, "b (c t) h w -> b c t h w", c=6, t=3)
        return processed_images, profiles, coords, temporal

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

            # Synchronize to ensure transfer is complete before inference
            if self.use_cuda:
                torch.cuda.synchronize()

            # Forward pass with BF16 autocast
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
                probs = torch.sigmoid(results).squeeze(1)
                predicted_masks = (probs > threshold).int().cpu().numpy()
            else:
                probabilities = torch.softmax(results, dim=1)
                predicted_masks = torch.argmax(probabilities, dim=1).cpu().numpy()

            # Convert to list of 2D arrays to match original return format
            predicted_masks = list(predicted_masks)

            return predicted_masks, profiles
