"""
Image processing operations module.

This module provides a comprehensive set of image processing functions including
basic adjustments (brightness, contrast, rotation), advanced operations (zoom,
binarization, negative), channel manipulation, highlighting, and image merging
with proper error handling for Streamlit applications.
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Optional, List
from utils.constants import HIGHLIGHT_GRAY_COLOR, DEFAULT_CHANNELS


def apply_brightness(image: Optional[Image.Image], factor: int) -> Optional[Image.Image]:
    """
    Apply brightness adjustment to an image.
    
    Args:
        image: PIL Image object to adjust
        factor: Brightness factor as integer percentage (-100 to 100)
        
    Returns:
        Brightness-adjusted image, or original image if operation fails
    """
    if image is None:
        return None
    try:
        enhancer = ImageEnhance.Brightness(image)
        mapped_factor = 1.0 + (factor / 100.0)
        return enhancer.enhance(mapped_factor)
    except Exception as e:
        st.warning(f"Could not apply brightness: {e}")
        return image


def apply_contrast(image: Optional[Image.Image], factor: float) -> Optional[Image.Image]:
    """
    Apply contrast adjustment to an image.
    
    Args:
        image: PIL Image object to adjust
        factor: Contrast enhancement factor
        
    Returns:
        Contrast-adjusted image, or original image if operation fails
    """
    if image is None:
        return None
    try:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    except Exception as e:
        st.warning(f"Could not apply contrast: {e}")
        return image


def apply_rotation(image: Optional[Image.Image], angle: int) -> Optional[Image.Image]:
    """
    Rotate an image by specified angle with canvas expansion.
    
    Args:
        image: PIL Image object to rotate
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image with expanded canvas, or original image if operation fails
    """
    if image is None or angle % 360 == 0:
        return image
    try:
        return image.rotate(angle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)
    except Exception as e:
        st.warning(f"Could not apply rotation: {e}")
        return image


def apply_zoom(image: Optional[Image.Image], x_perc: int, y_perc: int, 
               w_perc: int, h_perc: int) -> Optional[Image.Image]:
    """
    Crop image based on percentage coordinates (zoom operation).
    
    Args:
        image: PIL Image object to crop
        x_perc: X position as percentage (0-100)
        y_perc: Y position as percentage (0-100)
        w_perc: Width as percentage (0-100)
        h_perc: Height as percentage (0-100)
        
    Returns:
        Cropped image, or original image if operation fails or dimensions are invalid
    """
    if image is None:
        return None
    try:
        width, height = image.size
        left = int(width * (x_perc / 100.0))
        top = int(height * (y_perc / 100.0))
        right = left + int(width * (w_perc / 100.0))
        bottom = top + int(height * (h_perc / 100.0))

        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)

        if right <= left or bottom <= top:
            st.warning("Invalid zoom dimensions. Width and Height must result in a positive area.")
            return image
        return image.crop((left, top, right, bottom))
    except Exception as e:
        st.warning(f"Could not apply zoom: {e}")
        return image


def apply_binarization(image: Optional[Image.Image], threshold: int) -> Optional[Image.Image]:
    """
    Apply binary threshold to convert image to black and white.
    
    Args:
        image: PIL Image object to binarize
        threshold: Threshold value (0-255) for binarization
        
    Returns:
        Binarized image in RGB format, or original image if operation fails
    """
    if image is None:
        return None
    try:
        img_gray = image.convert('L')
        img_bin = img_gray.point(lambda p: 255 if p > threshold else 0, mode='1')
        return img_bin.convert('RGB')
    except Exception as e:
        st.warning(f"Could not apply binarization: {e}")
        return image


def apply_negative(image: Optional[Image.Image]) -> Optional[Image.Image]:
    """
    Invert the colors of an image to create a negative effect.
    
    Args:
        image: PIL Image object to invert
        
    Returns:
        Color-inverted image, or original image if operation fails
    """
    if image is None:
        return None
    try:
        return ImageOps.invert(image.convert('RGB'))
    except Exception as e:
        st.warning(f"Could not apply negative: {e}")
        return image


def apply_channel_manipulation(image: Optional[Image.Image], 
                             selected_channels: List[str]) -> Optional[Image.Image]:
    """
    Manipulate RGB channels by keeping selected ones and zeroing others.
    
    Args:
        image: PIL Image object to process
        selected_channels: List of channel names to keep ('Red', 'Green', 'Blue')
        
    Returns:
        Channel-manipulated image, or original image if operation fails
    """
    if image is None or set(selected_channels) == set(DEFAULT_CHANNELS):
        return image
    try:
        r, g, b = image.split()
        size = image.size
        zero_channel = Image.new('L', size, 0)

        r_channel = r if 'Red' in selected_channels else zero_channel
        g_channel = g if 'Green' in selected_channels else zero_channel
        b_channel = b if 'Blue' in selected_channels else zero_channel

        return Image.merge('RGB', (r_channel, g_channel, b_channel))
    except Exception as e:
        st.warning(f"Could not apply channel manipulation: {e}")
        return image


def apply_highlight(image: Optional[Image.Image], mode: str, threshold: int) -> Optional[Image.Image]:
    """
    Highlight light or dark areas by applying gray overlay to non-selected regions.
    
    Args:
        image: PIL Image object to process
        mode: Highlight mode ('Highlight Light Areas', 'Highlight Dark Areas', or 'None')
        threshold: Threshold value (0-255) for area selection
        
    Returns:
        Highlighted image, or original image if operation fails
    """
    if image is None or mode == "None":
        return image
    try:
        img_gray = image.convert('L')
        img_np = np.array(image).copy()
        mask = None
        
        if mode == "Highlight Light Areas":
            mask = np.array(img_gray) <= threshold
        elif mode == "Highlight Dark Areas":
            mask = np.array(img_gray) >= threshold
        else:
            return image

        if mask is not None:
            img_np[mask] = HIGHLIGHT_GRAY_COLOR

        return Image.fromarray(img_np)
    except Exception as e:
        st.warning(f"Could not apply highlight: {e}")
        return image


def merge_images(image1: Optional[Image.Image], image2: Optional[Image.Image], 
                alpha: float) -> Optional[Image.Image]:
    """
    Merge two images using alpha blending with automatic resizing.
    
    Args:
        image1: Primary PIL Image object
        image2: Secondary PIL Image object to blend
        alpha: Blending factor (0.0 to 1.0)
        
    Returns:
        Merged image in RGB format, or primary image if operation fails
        
    Note:
        The second image is automatically resized to match the first image's dimensions.
    """
    if image1 is None or image2 is None:
        st.warning("Both images must be loaded to merge.")
        return image1 or image2

    try:
        if image1.size != image2.size:
            st.info(f"Resizing second image from {image2.size} to {image1.size} for merging.")
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)

        image1_rgba = image1.convert("RGBA")
        image2_rgba = image2.convert("RGBA")

        blended = Image.blend(image1_rgba, image2_rgba, alpha=alpha)
        return blended.convert("RGB")
    except Exception as e:
        st.error(f"Error merging images: {e}")
        return image1