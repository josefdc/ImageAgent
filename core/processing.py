# streamlit_image_editor/core/processing.py
import streamlit as st # Keep for feedback messages
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Optional, List
from utils.constants import HIGHLIGHT_GRAY_COLOR, DEFAULT_CHANNELS

# --- Basic Adjustments ---

def apply_brightness(image: Optional[Image.Image], factor: int) -> Optional[Image.Image]:
    """Applies brightness adjustment."""
    if image is None: return None
    try:
        enhancer = ImageEnhance.Brightness(image)
        mapped_factor = 1.0 + (factor / 100.0)
        return enhancer.enhance(mapped_factor)
    except Exception as e:
        st.warning(f"Could not apply brightness: {e}")
        return image

def apply_contrast(image: Optional[Image.Image], factor: float) -> Optional[Image.Image]:
    """Applies contrast adjustment."""
    if image is None: return None
    try:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    except Exception as e:
        st.warning(f"Could not apply contrast: {e}")
        return image

def apply_rotation(image: Optional[Image.Image], angle: int) -> Optional[Image.Image]:
    """Rotates the image, expanding the canvas."""
    if image is None or angle % 360 == 0: return image # No-op if no rotation
    try:
        return image.rotate(angle, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC) # Use BICUBIC for better quality
    except Exception as e:
        st.warning(f"Could not apply rotation: {e}")
        return image

# --- Advanced Operations ---

def apply_zoom(image: Optional[Image.Image], x_perc: int, y_perc: int, w_perc: int, h_perc: int) -> Optional[Image.Image]:
    """Crops the image based on percentage coordinates."""
    if image is None: return None
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
            return image # Return original if dimensions are invalid
        return image.crop((left, top, right, bottom))
    except Exception as e:
        st.warning(f"Could not apply zoom: {e}")
        return image

def apply_binarization(image: Optional[Image.Image], threshold: int) -> Optional[Image.Image]:
    """Applies binarization using a threshold."""
    if image is None: return None
    try:
        img_gray = image.convert('L') # Convert to grayscale first
        # Apply threshold: pixels > threshold become white (255), others black (0)
        img_bin = img_gray.point(lambda p: 255 if p > threshold else 0, mode='1') # Use mode '1' for efficiency
        return img_bin.convert('RGB') # Convert back to RGB for display consistency
    except Exception as e:
        st.warning(f"Could not apply binarization: {e}")
        return image

def apply_negative(image: Optional[Image.Image]) -> Optional[Image.Image]:
    """Inverts the colors of the image."""
    if image is None: return None
    try:
        # Ensure image is RGB before inverting for predictable results
        return ImageOps.invert(image.convert('RGB'))
    except Exception as e:
        st.warning(f"Could not apply negative: {e}")
        return image

def apply_channel_manipulation(image: Optional[Image.Image], selected_channels: List[str]) -> Optional[Image.Image]:
    """Keeps selected RGB channels, setting others to 0."""
    if image is None or set(selected_channels) == set(DEFAULT_CHANNELS):
        return image # No-op if all channels selected
    try:
        r, g, b = image.split()
        size = image.size
        zero_channel = Image.new('L', size, 0) # Efficiently create a black channel

        r_channel = r if 'Red' in selected_channels else zero_channel
        g_channel = g if 'Green' in selected_channels else zero_channel
        b_channel = b if 'Blue' in selected_channels else zero_channel

        return Image.merge('RGB', (r_channel, g_channel, b_channel))
    except Exception as e:
        st.warning(f"Could not apply channel manipulation: {e}")
        return image

def apply_highlight(image: Optional[Image.Image], mode: str, threshold: int) -> Optional[Image.Image]:
    """Highlights light or dark areas by graying out the rest."""
    if image is None or mode == "None": return image
    try:
        img_gray = image.convert('L')
        img_np = np.array(image).copy() # Work on numpy array for masking
        mask = None
        if mode == "Highlight Light Areas":
            mask = np.array(img_gray) <= threshold # Mask areas *not* highlighted (darker/equal)
        elif mode == "Highlight Dark Areas":
            mask = np.array(img_gray) >= threshold # Mask areas *not* highlighted (lighter/equal)
        else:
             return image # Should not happen with 'None' check above but good failsafe

        if mask is not None:
            # Apply a neutral gray overlay to masked areas
            img_np[mask] = HIGHLIGHT_GRAY_COLOR

        return Image.fromarray(img_np)
    except Exception as e:
        st.warning(f"Could not apply highlight: {e}")
        return image

# --- Merging ---

def merge_images(image1: Optional[Image.Image], image2: Optional[Image.Image], alpha: float) -> Optional[Image.Image]:
    """Merges two images using alpha blending, resizing the second if needed."""
    if image1 is None or image2 is None:
        st.warning("Both images must be loaded to merge.")
        return image1 or image2 # Return whichever is available

    try:
        # Resize second image to match the first if necessary
        if image1.size != image2.size:
            st.info(f"Resizing second image from {image2.size} to {image1.size} for merging.")
            # Use LANCZOS for high-quality downscaling/upscaling
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)

        # Ensure RGBA for alpha blending
        image1_rgba = image1.convert("RGBA")
        image2_rgba = image2.convert("RGBA")

        # Blend images
        blended = Image.blend(image1_rgba, image2_rgba, alpha=alpha)
        return blended.convert("RGB") # Convert back to RGB for consistency
    except Exception as e:
        st.error(f"Error merging images: {e}")
        return image1 # Return primary image on error