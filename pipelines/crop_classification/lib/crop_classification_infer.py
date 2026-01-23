import numpy as np
import rasterio
import torch



from datetime import datetime
from einops import rearrange

from lib.infer import Infer
from lib.consts import NO_DATA, NO_DATA_FLOAT, MEANS, STDS
from terratorch.tasks import SemanticSegmentationTask


class CropClassificationInfer(Infer):
    def __init__(self, config, checkpoint, max_queue_size=4, num_streams=2):
        super().__init__(config, checkpoint, max_queue_size, num_streams)
    def load_model(self):
        if self.model:
            return
        model_args = {
                # Backbone
                "backbone": "prithvi_eo_v2_600_tl",
                "backbone_pretrained": False,
                "backbone_num_frames": 3,
                "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
                "backbone_coords_encoding": [],
                # Necks
                "necks": [
                    {
                        "name": "SelectIndices",
                        "indices": [7, 15, 23, 31]
                    },
                    {
                        "name": "ReshapeTokensToImage",
                        "effective_time_dim": 3
                    },
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
                model_args=model_args
            )
        self.model.to(self.device)
        self.model = self.model.eval()

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
                stacked = []
                src = raster_file.read()
                stacked.append(src[:6])  # Read pre bands
                stacked.append(src[9:15])  # Read current bands
                stacked.append(src[18:24])  # Read post bands
                image = np.concatenate(stacked, axis=0)
                image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)

                # Use pinned memory for faster CPU->GPU transfers
                if self.use_cuda:
                    image_tensor = torch.from_numpy(image).pin_memory()
                else:
                    image_tensor = torch.from_numpy(image)

                # Normalization (match device of image tensor and statistics)
                if len(self.means) > 0 and len(self.stds) > 0 and self.use_cuda:
                    # Move to GPU for normalization
                    image_tensor = image_tensor.to(self.device, non_blocking=True)
                    for band in range(image_tensor.shape[0]):
                        band_mask = image_tensor[band] == NO_DATA_FLOAT
                        band_index = band % len(self.means)
                        image_tensor[band][~band_mask] = (
                            (image_tensor[band][~band_mask].float() - self.means[band_index])
                            / self.stds[band_index]
                        ).to(image_tensor.dtype)
                    # Move back to CPU for batching
                    image_tensor = image_tensor.cpu()
                elif len(self.means) > 0 and len(self.stds) > 0:
                    # CPU-only normalization
                    for band in range(image_tensor.shape[0]):
                        band_mask = image_tensor[band] == NO_DATA_FLOAT
                        band_index = band % len(self.means)
                        image_tensor[band][~band_mask] = (
                            (image_tensor[band][~band_mask].float() - self.means[band_index])
                            / self.stds[band_index]
                        ).to(image_tensor.dtype)

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
        processed_images = rearrange(imgs_tensor, 'b (c t) h w -> b c t h w', c=6, t=3)
        return processed_images, profiles, coords, temporal

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
                    print("Shape of predicted mask:", predicted_mask.shape)
                    print("max and min of predicted mask:", predicted_mask.max(), predicted_mask.min())

                predicted_masks.append(predicted_mask)

            # Clear GPU cache to free memory
            if self.use_cuda:
                torch.cuda.empty_cache()

            return predicted_masks, profiles
