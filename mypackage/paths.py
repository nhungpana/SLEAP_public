import os
from pathlib import Path

# Get absolute path of the project root (one folder up from code/)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).parent.parent

# Define data folder path
# DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
# IMAGES_DIR = os.path.join(PROJECT_ROOT, "Streamlit_interface/Image")
DATA_DIR = PROJECT_ROOT / "Data"
IMAGES_DIR = PROJECT_ROOT / "Streamlit_interface/Image"
MODEL_DIR = PROJECT_ROOT / "Streamlit_interface/Model_export"