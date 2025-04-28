import os


# Base Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "checkpoint", "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm"
)
