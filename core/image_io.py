# streamlit_image_editor/core/image_io.py
import streamlit as st
from PIL import Image
import io
from typing import Optional, Tuple

def load_image(uploaded_file) -> Optional[Tuple[Image.Image, str]]:
    """Loads an image from an uploaded file object, ensuring RGB format."""
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image, uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image '{uploaded_file.name}': {e}")
            return None
    return None

def prepare_image_for_download(image: Image.Image, save_format: str) -> Optional[bytes]:
    """Converts a PIL Image to bytes for downloading."""
    if image is None:
        return None
    try:
        buf = io.BytesIO()
        img_mode = image.mode
        img_to_save = image # Start with the original

        # Handle different modes for saving appropriately
        if img_mode == 'RGBA' and save_format == 'PNG':
             img_to_save.save(buf, format=save_format)
        elif img_mode == 'P': # Palette mode often needs conversion
             img_to_save = img_to_save.convert('RGB')
             img_to_save.save(buf, format=save_format)
        elif img_mode == 'L': # Grayscale is usually fine
             img_to_save.save(buf, format=save_format)
        else: # Assume RGB or convert if needed
             # Ensure it's RGB if the format requires it (like JPG)
             if save_format == 'JPEG' and img_mode != 'RGB':
                  img_to_save = img_to_save.convert('RGB')
             elif img_mode != 'RGB' and save_format not in ['PNG', 'BMP', 'GIF', 'TIFF']: # Add other formats supporting L/RGBA if needed
                 img_to_save = img_to_save.convert('RGB') # Default fallback
             img_to_save.save(buf, format=save_format)

        return buf.getvalue()
    except Exception as e:
        st.error(f"Error preparing image for download as {save_format}: {e}")
        return None