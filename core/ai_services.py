#core/ai_services.py
import logging
from PIL import Image
from io import BytesIO
import requests # Example for a hypothetical upscaling API
import os

# Assuming 'rembg' is installed: pip install rembg
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    logging.warning("rembg library not found. Background removal AI tool will not work.")
    REMBG_AVAILABLE = False

# Placeholder for an upscaling service API key/URL
# You might get this from environment variables or Streamlit secrets
UPSCALE_API_URL = os.getenv("UPSCALE_API_URL", "YOUR_UPSCALE_API_ENDPOINT_HERE")
UPSCALE_API_KEY = os.getenv("UPSCALE_API_KEY", "YOUR_UPSCALE_API_KEY_HERE")

logger = logging.getLogger(__name__)

def remove_background_ai(image: Image.Image) -> Image.Image | None:
    """
    Removes the background from an image using an AI model (rembg).

    Args:
        image: The input PIL Image object.

    Returns:
        A PIL Image object with the background removed, or None if the operation fails.
    """
    if not REMBG_AVAILABLE:
        logger.error("Attempted to use remove_background_ai, but rembg library is not available.")
        return None
    if image is None:
        logger.error("Input image is None for remove_background_ai.")
        return None

    try:
        logger.info("Attempting AI background removal...")
        # Ensure image has an alpha channel for transparency, convert if necessary
        if image.mode != 'RGBA':
             image = image.convert('RGBA')

        result_image = remove(image)
        logger.info("AI background removal successful.")
        return result_image
    except Exception as e:
        logger.error(f"Error during AI background removal: {e}", exc_info=True)
        return None

def upscale_image_ai(image: Image.Image, scale_factor: int = 2) -> Image.Image | None:
    """
    Upscales an image using a hypothetical AI service API.
    Replace this with your actual upscaling implementation (e.g., using a local model or a different API).

    Args:
        image: The input PIL Image object.
        scale_factor: The factor by which to upscale the image (e.g., 2 for 2x).

    Returns:
        The upscaled PIL Image object, or None if the operation fails.
    """
    if image is None:
        logger.error("Input image is None for upscale_image_ai.")
        return None
    if not UPSCALE_API_URL or "YOUR_UPSCALE_API_ENDPOINT_HERE" in UPSCALE_API_URL:
         logger.error("UPSCALE_API_URL is not configured. Cannot upscale image.")
         # Optionally, fall back to a basic Pillow resize?
         # return image.resize((image.width * scale_factor, image.height * scale_factor), Image.Resampling.LANCZOS)
         return None

    logger.info(f"Attempting AI upscaling (factor: {scale_factor}) via API: {UPSCALE_API_URL}...")

    try:
        # Convert image to bytes
        buffer = BytesIO()
        image_format = image.format or 'PNG' # Keep original format or default to PNG
        image.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()

        headers = {
            "Authorization": f"Bearer {UPSCALE_API_KEY}" if UPSCALE_API_KEY and "YOUR_UPSCALE_API_KEY_HERE" not in UPSCALE_API_KEY else {}
        }
        files = {'image': (f'input.{image_format.lower()}', image_bytes, f'image/{image_format.lower()}')}
        data = {'scale_factor': scale_factor}

        response = requests.post(UPSCALE_API_URL, headers=headers, files=files, data=data, timeout=60) # 60 seconds timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        upscaled_image = Image.open(BytesIO(response.content))
        logger.info("AI upscaling successful.")
        return upscaled_image

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for AI upscaling: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error during AI upscaling: {e}", exc_info=True)
        return None

# You might add more AI-related image functions here later