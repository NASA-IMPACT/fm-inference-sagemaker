import json
import numpy as np
import rasterio
import torch
import yaml

from datetime import datetime
from rasterio.features import shapes

from lib.data_preparer import QA_INDICES
from lib.consts import NO_DATA, NO_DATA_FLOAT, MEANS, STDS
from terratorch.cli_tools import LightningInferenceModel

class Infer:
    def __init__(self, config, checkpoint):
        self.config_filename = config
        with open(self.config_filename) as config:
            self.config = yaml.safe_load(config)
        self.checkpoint_filename = checkpoint
        self.model = None
        self.load_model()
        # Use proper mean and std from consts if not in config.
        self.means = np.asarray(self.config['data']['init_args'].get('means', MEANS))
        self.stds = np.asarray(self.config['data']['init_args'].get('stds', STDS))
        if len(self.means) > 0 and len(self.stds) > 0:
            self.means = torch.from_numpy(self.means).view(-1, 1, 1)
            self.stds = torch.from_numpy(self.stds).view(-1, 1, 1)

    def load_model(self):
        if not(self.model):
            inference_model = LightningInferenceModel.from_config(self.config_filename, self.checkpoint_filename)
            self.model = inference_model.model
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.eval()

    def postprocess(self, bbox, date, predictions, images):
        return predictions

    def preprocess(self, images, date):
        images_array = []
        profiles = []
        coords = []
        temporal = []
        date = datetime.strptime(date, '%Y-%m-%d')
        julian_year, julian_day = int(datetime.strftime(date, "%Y")), int(datetime.strftime(date, "%j"))
        for image in images:
            with rasterio.open(image) as raster_file:
                image = raster_file.read()[:6]  # Read first 6 bands
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

    def calculate_area_from_mask(self, prediction_file, mask_values=[1]):
        """
        Calculates the area in square kilometers of a mask in a GeoTIFF file.

        Args:
            tiff_file_path (str): The path to the GeoTIFF file.
            mask_value (int): The pixel value representing the masked area.
                            Defaults to 1 for binary masks.

        Returns:
            float: The calculated area in square kilometers.
        """
        try:
            # Open the GeoTIFF file
            areas = {}
            with rasterio.open(prediction_file) as src:
                for mask_value in mask_values:
                    # Read the raster data into a NumPy array
                    data = src.read(1)  # Assuming the mask is in the first band
                    # Get the cell size (resolution) in the file's units (e.g., meters)
                    cell_width, cell_height = src.res
                    # Calculate the area of a single cell in square meters
                    cell_area_sqm = abs(cell_width * cell_height)
                    # Count the number of pixels that match the mask_value
                    masked_pixel_count = np.sum(data == mask_value)
                    # Calculate the total area in square meters
                    total_area_sqm = masked_pixel_count * cell_area_sqm
                    # Convert the area from square meters to square kilometers
                    total_area_sqkm = total_area_sqm / 1_000_000
                    areas[mask_value] = total_area_sqkm
                return areas
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening or reading file: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def qa_flags_to_tif(self, image_file, qa_flags, timeseries=False):
        """
        Convert predicted masks to GeoJSON format.
        Args:
            image_files (list): List of input image file paths.

        Returns:
            list: List of GeoJSON features.
        """
        geojson_features = []
        def get_qa_mask(tile, qa_flags):
            combined = np.zeros_like(tile).astype('uint')
            for qa_flag in qa_flags:
                qa_index = QA_INDICES.get(qa_flag)
                flag = tile[6].astype('uint') & (1 << qa_index) != 0
                combined |= flag
            return combined
        with rasterio.open(image_file) as src:
            profile = src.profile
            tile = src.read()
            if timeseries:
                mask = get_qa_mask(tile[9:16], qa_flags)
            else:
                mask = get_qa_mask(tile, qa_flags)
            transform = profile['transform']
            mask = mask.astype('uint8')  # Ensure mask is in uint8 format

            # Save mask as GeoTIFF

            output_tif = image_file.replace('.tif', '_qa_mask.tif')
            with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=mask.dtype,
                crs=profile['crs'],
                transform=transform,
            ) as dst:
                dst.write(mask, 1)
        return output_tif


    def infer(self, images, date):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        # forward the model
        with torch.no_grad():
            images, profiles, coords, temporal = self.preprocess(images, date)
            result = self.model(
                images.to('cuda' if torch.cuda.is_available() else 'cpu'),
            )
            predicted_masks = list()
            results = result.output.detach().cpu()
            for index, mask in enumerate(results):
                output = mask.cpu()  # [n_segmentation_class, 224, 224]
                if self.config['model']['init_args']['model_args']['num_classes'] == 1:
                    updated_mask = torch.sigmoid(output.clone()).squeeze(0)
                    predicted_mask = (updated_mask > self.config.get('threshold', 0.5)).int()
                else:
                    # predicted_mask = mask.argmax(dim=0)
                    probabilities = torch.softmax(output, dim=0)
                    predicted_mask = torch.argmax(probabilities, dim=0).cpu().numpy()
                    # flood_prob = predicted_mask[0, 1].cpu().numpy()
                    # img_size = profiles[index]['height']
                    # predicted_mask = torch.nn.functional.interpolate(
                    #         predicted_mask.unsqueeze(0).float(),
                    #         size=img_size,
                    #         mode="nearest"
                    #     )
                    print("Shape of predicted mask:", predicted_mask.shape)
                    print("max and min of predicted mask:", predicted_mask.max(), predicted_mask.min())
                predicted_masks.append(predicted_mask)
            return predicted_masks, profiles
