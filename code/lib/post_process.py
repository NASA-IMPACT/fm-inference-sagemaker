import cv2
import numpy as np
import rasterio

from geojson import Feature, Polygon
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev
from skimage.morphology import disk, binary_closing
from shapely import geometry


AREA_THRESHOLD = 0.05
MIN_POINTS = 3
PREDICT_THRESHOLD = 0.5
CONTOUR_AREA_THRESHOLD = 400
BLUR_FACTOR = (20, 20)
BLUR_THRESHOLD = 127


class PostProcess:
    @classmethod
    def prepare_bitmap(cls, predictions, width, height):
        predictions = predictions.reshape((height, width))
        # return reshaped raw array instead of bitmap
        return predictions

    @classmethod
    def prepare_contours(cls, predictions):
        bitmap = (predictions > PREDICT_THRESHOLD).astype(dtype="uint8") * 255
        img_blurred = cv2.blur(bitmap, BLUR_FACTOR)
        # img_blurred = binary_closing(bitmap, disk(6))
        thresholded_img = (img_blurred > BLUR_THRESHOLD).astype(dtype="uint8") * 255
        contours, _ = cv2.findContours(
            np.asarray(thresholded_img, dtype="uint8"),
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
        """Extracts and scores shapes from contours, returning those above threshold."""
        smoothened = []
        for contour in contours:
            if len(contour) <= MIN_POINTS:
                continue
            if cv2.contourArea(contour) < CONTOUR_AREA_THRESHOLD:
                continue
            y, x = contour.T
            x = x.tolist()[0]
            y = y.tolist()[0]
            # Optionally smooth the contour here if needed
            x_new, y_new = x, y
            res_array = []
            new_polygon = []
            for px, py in zip(x_new, y_new):
                clamped_x, clamped_y = cls._clamp_point((px, py), shape)
                new_polygon.append((clamped_x, clamped_y))
                res_array.append(cls.convert_xy_to_latlon(clamped_x, clamped_y, transform))
            img = Image.new("L", (shape[0], shape[1]), 0)
            ImageDraw.Draw(img).polygon(new_polygon, outline=1, fill=1)
            mask = np.where(np.array(img).T > 0)
            score_pixels = predictions[mask]
            if len(score_pixels) == 0:
                continue
            score = sum(score_pixels) / len(score_pixels)
            if score < PREDICT_THRESHOLD:
                continue
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
        # Geometry is already in EPSG:4326 from the WorldCRS84Quad tile pipeline.
        return results

    @classmethod
    def remove_intersections(cls, shapes):
        computed_polygons = list()
        selected_indices = list()
        selected_shapes = list()
        areas = list()
        for shape in shapes:
            polygon = geometry.Polygon(shape[0])
            if polygon.is_valid:
                area = polygon.area
                if area > AREA_THRESHOLD:
                    computed_polygons.append(polygon)
                    areas.append(area)
                    selected_shapes.append(shape)
        if len(areas) > 0:
            computed_polygons = np.asarray(computed_polygons)
            polygon_indices = list(np.argsort(areas))
            while len(polygon_indices) > 0:
                selected_index = polygon_indices[-1]
                selected_polygon = computed_polygons[selected_index]
                selected_indices.append(selected_index)
                polygon_indices.remove(selected_index)
                indices_holder = polygon_indices.copy()
                for index in indices_holder:
                    if computed_polygons[index].intersects(selected_polygon):
                        polygon_indices.remove(index)
        return np.array(selected_shapes, dtype='object')[selected_indices]

    @classmethod
    def convert_xy_to_latlon(cls, row, col, transform):
        """
        uses rasterio transform module to convert row, col of an image to
        its respective lat, lon coordinates
        """
        transform = rasterio.transform.guard_transform(transform)
        return rasterio.transform.xy(transform, row, col, offset="center")
