from lib.infer import Infer
from lib.dem_downloader import DEMDownloader


class FloodInfer(Infer):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint)

    def postprocess(self, bbox, date, prediction, image):
        dem_downloader = DEMDownloader(bbox, date)
        dem_files = dem_downloader.download_dem_tiles()
        with rasterio.open(image) as src:
            width, height = src.profile['width'], src.profile['height']
        dem_file = dem_downloader.merge_and_clip_dems(dem_files, width, height)
        final_prediction = dem_downloader.apply_all_postprocessing(prediction, image, dem_file)
        return final_prediction
