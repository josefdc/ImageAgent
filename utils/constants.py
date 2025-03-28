# streamlit_image_editor/utils/constants.py
from typing import List

IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png", "bmp"]
DEFAULT_SAVE_FORMAT: str = 'PNG'
DEFAULT_MIME_TYPE: str = 'image/png'
HIGHLIGHT_GRAY_COLOR: List[int] = [150, 150, 150] # Color for non-highlighted areas

# Canales por defecto para el multiselect
DEFAULT_CHANNELS: List[str] = ['Red', 'Green', 'Blue']