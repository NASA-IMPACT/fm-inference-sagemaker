import rasterio
import torch

from lib.infer import Infer
from lib.dem_downloader import DEMDownloader

from terratorch.tasks import SemanticSegmentationTask


class BurnScarInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)
