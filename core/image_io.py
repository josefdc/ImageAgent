"""
Image input/output operations module.

This module provides functionality for loading images from uploaded files
and preparing images for download in various formats, with proper format
conversion and error handling for Streamlit applications.
"""

import streamlit as st
from PIL import Image
import io
from typing import Optional, Tuple


def load_image(uploaded_file) -> Optional[Tuple[Image.Image, str]]:
    """
    Load an image from an uploaded file object and convert to RGB format.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple containing the PIL Image in RGB format and filename,
        or None if loading fails
    """
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
    """
    Convert a PIL Image to bytes for downloading in specified format.

    Args:
        image: PIL Image object to convert
        save_format: Target format for saving (e.g., 'JPEG', 'PNG', 'BMP')

    Returns:
        Image data as bytes, or None if conversion fails

    Note:
        Handles format-specific conversions (e.g., RGBA to RGB for JPEG,
        palette mode conversion) to ensure compatibility.
    """
    if image is None:
        return None

    try:
        buf = io.BytesIO()
        img_mode = image.mode
        img_to_save = image

        if img_mode == 'RGBA' and save_format == 'PNG':
            img_to_save.save(buf, format=save_format)
        elif img_mode == 'P':
            img_to_save = img_to_save.convert('RGB')
            img_to_save.save(buf, format=save_format)
        elif img_mode == 'L':
            img_to_save.save(buf, format=save_format)
        else:
            if save_format == 'JPEG' and img_mode != 'RGB':
                img_to_save = img_to_save.convert('RGB')
            elif img_mode != 'RGB' and save_format not in ['PNG', 'BMP', 'GIF', 'TIFF']:
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(buf, format=save_format)

        return buf.getvalue()

    except Exception as e:
        st.error(f"Error preparing image for download as {save_format}: {e}")
        return None