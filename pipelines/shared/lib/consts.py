import os
import json

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

CROP_SIZE = (512, 512)
SPLITS = ['training', 'validation', 'test']

LAYERS = ['HLSS30', 'HLSL30']
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MODEL_PATH = "models/{model_name}"
USECASE = os.environ.get('USECASE')
MODEL_WEIGHT_PATH = os.environ.get('MODEL_WEIGHT_PATH')
CONFIG_PATH = os.environ.get('CONFIG_PATH')
DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", '/root/.cache/')
MEANS = json.loads(os.environ.get('MEANS', '[]'))
STDS = json.loads(os.environ.get('STDS', '[]'))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 2))
