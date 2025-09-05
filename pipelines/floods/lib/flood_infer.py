import rasterio

from lib.infer import Infer
from lib.dem_downloader import DEMDownloader

from terratorch.tasks import SemanticSegmentationTask



class FloodInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)

    def load_model(self):
        num_classes = 2
        indices = [7, 15, 23, 31]  # for prithvi_eo_v2_600
        model_args = {
            # Backbone
            "backbone": 'prithvi_eo_v2_600',
            "backbone_pretrained": True,
            "backbone_num_frames": 1,
            "backbone_img_size": 512,
            "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": indices
                },
                {"name": "ReshapeTokensToImage",},
                {"name": "LearnedInterpolateToPyramidal"}
            ],
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            # Head
            "head_dropout": 0.1,
            "num_classes": 2,
        }
        self.model = SemanticSegmentationTask.load_from_checkpoint(
            self.checkpoint_filename,
            model_factory="EncoderDecoderFactory",
            model_args=model_args
        )

        self.model = self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def postprocess(self, bbox, date, prediction, image):
        dem_downloader = DEMDownloader(bbox, date)
        dem_files = dem_downloader.download_dem_tiles()
        with rasterio.open(image) as src:
            width, height = src.profile['width'], src.profile['height']
        dem_file = dem_downloader.merge_and_clip_dems(dem_files, width, height)
        final_prediction = dem_downloader.apply_all_postprocessing(prediction, image, dem_file)
        return final_prediction
