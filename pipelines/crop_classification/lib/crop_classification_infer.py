import rasterio
import torch

from lib.infer import Infer

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

    def preprocess(self, images, date):
        images_array = []
        profiles = []
        coords = []
        temporal = []
        date = datetime.strptime(date, '%Y-%m-%d')
        julian_year, julian_day = int(datetime.strftime(date, "%Y")), int(datetime.strftime(date, "%j"))
        for image in images:
            with rasterio.open(image) as raster_file:
                stacked = []
                stacked.append(raster_file.read()[:6])  # Read pre bands
                stacked.append(raster_file.read()[9:15])  # Read current bands
                stacked.append(raster_file.read()[18:24])  # Read post bands
                image = np.concatenate(stacked, axis=0)
                image = np.where(image == NO_DATA, NO_DATA_FLOAT, image)
                image = torch.from_numpy(image)
                if len(self.means) > 0 and len(self.stds) > 0:
                    for band in range(image.shape[0]):
                        band_mask = image[band] == NO_DATA_FLOAT
                        image[band][~band_mask] = ((image[band][~band_mask].float() - self.means[band]) / self.stds[band]).to(image.dtype)
                images_array.append(image)
                coords.append(raster_file.lnglat())
                temporal.append([julian_year, julian_day])
                profiles.append(raster_file.profile)
                raster_file.close()
        # Example processing function to simulate the pipeline
        imgs_tensor = torch.from_numpy(np.asarray(images_array))  # Assuming input_array is of type np.float32
        imgs_tensor = imgs_tensor.float()

        # increase dimensions to match input size
        processed_images = imgs_tensor
        print("shape of processed images:", processed_images.shape)
        processed_images = imgs_tensor.unsqueeze(2)
        return processed_images, profiles, coords, temporal
