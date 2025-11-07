import cv2
import numpy as np
import rasterio
import rasterio.warp

from geojson import Feature, Polygon
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from scipy.interpolate import splprep, splev
from skimage.morphology import disk, binary_closing
from shapely import geometry
from concurrent.futures import ThreadPoolExecutor, as_completed


AREA_THRESHOLD = 0.05
MIN_POINTS = 3
PREDICT_THRESHOLD = 0.5
CONTOUR_AREA_THRESHOLD = 400
BLUR_FACTOR = (20, 20)
BLUR_THRESHOLD = 127


class PostProcess:
    @classmethod
    def prepare_bitmap(cls, predictions, width, height):
        # Use view instead of copy for better performance
        return predictions.reshape((height, width))

    @classmethod
    def prepare_contours(cls, predictions):
        # Vectorized thresholding
        bitmap = np.where(predictions > PREDICT_THRESHOLD, 255, 0).astype(dtype="uint8")
        img_blurred = cv2.blur(bitmap, BLUR_FACTOR)
        # Vectorized thresholding for blurred image
        thresholded_img = np.where(img_blurred > BLUR_THRESHOLD, 255, 0).astype(dtype="uint8")
        contours, _ = cv2.findContours(
            thresholded_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return [contours, bitmap.shape]

    @staticmethod
    def _clamp_point(point, shape):
        """Clamp a point to be within the image boundaries defined by shape."""
        x, y = point
        x = min(max(x, 0), shape[0])
        y = min(max(y, 0), shape[1])
        return x, y

    @classmethod
    def extract_shapes(cls, predictions, contours, transform, shape):
        """
        Optimized shape extraction with vectorized clipping and numpy operations.
        """
        smoothened = list()
        for contour in contours:
            length = len(contour)
            if length > MIN_POINTS:
                if cv2.contourArea(contour) < CONTOUR_AREA_THRESHOLD:
                    continue
                y, x = contour.T
                x = x.tolist()[0]
                y = y.tolist()[0]
                x_new = x
                y_new = y

                # Vectorized clipping using numpy
                x_arr = np.array(x_new)
                y_arr = np.array(y_new)
                x_arr = np.clip(x_arr, 0, shape[0])
                y_arr = np.clip(y_arr, 0, shape[1])

                new_polygon = list(zip(x_arr.tolist(), y_arr.tolist()))

                # Vectorized coordinate conversion
                res_array = [cls.convert_xy_to_latlon(px, py, transform)
                            for px, py in new_polygon]

                # Optimized mask creation and score calculation
                img = Image.new("L", (shape[0], shape[1]), 0)
                ImageDraw.Draw(img).polygon(new_polygon, outline=1, fill=1)
                mask = np.where(np.array(img).T > 0)
                score = predictions[mask]
                score_length = len(score)

                if score_length > 0:
                    # Vectorized mean calculation
                    score = np.mean(score)
                    if score >= PREDICT_THRESHOLD:
                        smoothened.append([np.asarray(res_array), score])
        return smoothened

    @classmethod
    def convert_to_geojson(cls, shapes):
        geojson_dict = []
        for id_, shape in enumerate(shapes):
            geojson_dict.append(
                Feature(
                    properties={
                        "score": shape[1],
                    },
                    geometry=Polygon([[[float(lon), float(lat)] for lon, lat in shape[0]]]),
                )
            )
        # just return the list of features, wrapping is done in main.py
        return geojson_dict

    @classmethod
    def convert_geojson(cls, results):
        feature = results["geometry"]
        feature_proj = rasterio.warp.transform_geom(
            CRS.from_epsg(3857), CRS.from_epsg(4326), feature
        )
        results["geometry"] = feature_proj
        return results

    @classmethod
    def remove_intersections(cls, shapes):
        """
        Optimized intersection removal with vectorized operations.
        """
        computed_polygons = []
        selected_indices = []
        selected_shapes = []
        areas = []

        # Vectorized polygon validation and area calculation
        for shape in shapes:
            polygon = geometry.Polygon(shape[0])
            if polygon.is_valid:
                area = polygon.area
                if area > AREA_THRESHOLD:
                    computed_polygons.append(polygon)
                    areas.append(area)
                    selected_shapes.append(shape)

        if len(areas) > 0:
            computed_polygons = np.asarray(computed_polygons, dtype=object)
            # Use numpy argsort for faster sorting
            polygon_indices = np.argsort(areas).tolist()

            while len(polygon_indices) > 0:
                selected_index = polygon_indices[-1]
                selected_polygon = computed_polygons[selected_index]
                selected_indices.append(selected_index)
                polygon_indices.pop()  # More efficient than remove

                # Filter out intersecting polygons
                polygon_indices = [
                    idx for idx in polygon_indices
                    if not computed_polygons[idx].intersects(selected_polygon)
                ]

        return np.array(selected_shapes, dtype='object')[selected_indices] if selected_indices else np.array([])

    @classmethod
    def convert_xy_to_latlon(cls, row, col, transform):
        """
        uses rasterio transform module to convert row, col of an image to
        its respective lat, lon coordinates
        """
        transform = rasterio.transform.guard_transform(transform)
        return rasterio.transform.xy(transform, row, col, offset="center")
