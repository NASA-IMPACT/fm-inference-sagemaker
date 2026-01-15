import rasterio
import torch

from lib.infer import Infer
from lib.data_preparer import DataPreparer

from terratorch.tasks import SemanticSegmentationTask


class BurnScarInfer(Infer):
    def __init__(self, config, checkpoint, max_queue_size=4, num_streams=2):
        super().__init__(config, checkpoint, max_queue_size, num_streams)

    def postprocess(self, bbox, date, prediction_filename, image_filename):
        pred_data = None
        with rasterio.open(prediction_filename) as src:
            pred_data = src.read(1)
            profile = src.profile
        with rasterio.open(image_filename) as src:
            image_data = src.read()
            qa_indices = DataPreparer.get_qa_mask_vectorized(image_data[6], ['cloud', 'adjacent_cloud', 'shadow', 'snow', 'water'])
            pred_data[qa_indices] = 0  # Mask out no-data areas based on QA

        with rasterio.open(prediction_filename, 'w', **profile) as dst:
            dst.write(pred_data, 1)
        return prediction_filename
