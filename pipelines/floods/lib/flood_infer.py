from lib.infer import Infer
from lib.dem_downloader import DEMDownloader


class FloodInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)

    def postprocess(self, bbox, date, predictions, images):
        dem_downloader = DEMDownloader(bbox=bbox, date=date)
        dem_files = dem_downloader.download_dem_tiles()
        dem_file = dem_downloader.merge_and_clip_dems(dem_files)
        final_predictions = dem_file.apply_all_postprocessing(predictions, images, dem_file)
        return final_predictions
