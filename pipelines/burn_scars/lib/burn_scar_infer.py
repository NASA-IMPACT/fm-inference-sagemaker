import rasterio
import torch

from lib.infer import Infer
from lib.dem_downloader import DEMDownloader

from terratorch.tasks import SemanticSegmentationTask


class BurnScarInfer(Infer):
    def __init__(self, config, checkpoint, max_queue_size=4, num_streams=2):
        super().__init__(config, checkpoint, max_queue_size, num_streams)
