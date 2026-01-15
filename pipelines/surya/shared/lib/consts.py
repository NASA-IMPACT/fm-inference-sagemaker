import os

CHANNELS = [
    'aia94',
    'aia131',
    'aia171',
    'aia193',
    'aia211',
    'aia304',
    'aia335',
    'aia1600',
    'hmi_m',
    'hmi_bx',
    'hmi_by',
    'hmi_bz',
    'hmi_v'
]

DOWNLOAD_FOLDER = os.environ.get("DOWNLOAD_FOLDER", '/root/.cache/')
OUTPUT_DIR = os.environ.get("SDO_OUTPUT_DIR", f"{DOWNLOAD_FOLDER}/surya")
FITS_DIR = os.path.join(OUTPUT_DIR, "raw_fits")

EMAIL = os.environ.get("SDO_EMAIL", "")

# Surya model paths (downloaded from HuggingFace: nasa-ibm-ai4science/Surya-1.0)
SURYA_CONFIG_PATH = os.environ.get("SURYA_CONFIG_PATH", "/app/models/config.yaml")
SURYA_SCALERS_PATH = os.environ.get("SURYA_SCALERS_PATH", "/app/models/scalers.yaml")
SURYA_WEIGHTS_PATH = os.environ.get("SURYA_WEIGHTS_PATH", "/app/models/surya.366m.v1.pt")

IMAGE_SHAPE = (4096, 4096)
SATURATION = 16383
TARGET_SOLAR_RADIUS = 976.0
DOMAIN_NAME = os.environ.get("DOMAIN_NAME", "https://fm.prism.nasa-impact.net")
