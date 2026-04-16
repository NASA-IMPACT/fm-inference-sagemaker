import rasterio

from lib.infer import Infer
from lib.data_preparer import DataPreparer


class BurnScarInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)

    def postprocess(self, bbox, date, prediction_filename, image_filename):
        pred_data = None
        with rasterio.open(prediction_filename) as src:
            pred_data = src.read(1)
            profile = src.profile
        with rasterio.open(image_filename) as src:
            # Only band 7 (fmask) is needed for QA masking — avoid reading all bands
            fmask = src.read(7)
            qa_indices = DataPreparer.get_qa_mask_vectorized(
                fmask, ["cloud", "adjacent_cloud", "shadow", "snow", "water"]
            )
            pred_data[qa_indices] = 0  # Mask out no-data areas based on QA

        with rasterio.open(prediction_filename, "w", **profile) as dst:
            dst.write(pred_data, 1)
        return prediction_filename
