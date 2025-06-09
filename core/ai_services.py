"""
AI Image Processing Services Module

This module provides AI-powered image processing capabilities using external APIs,
primarily Stability AI, with local fallbacks where available. It handles:

- Background removal (Stability AI + rembg fallback)
- Image upscaling (Stability AI Fast Upscaler)
- Object search and replace (Stability AI)
- Object recoloring (Stability AI)

The module automatically detects Streamlit context for error display and secret
management, while maintaining compatibility with standalone execution.

Dependencies:
- requests: HTTP API calls
- PIL (Pillow): Image processing
- rembg (optional): Local background removal fallback
- streamlit (optional): UI error display and secrets
"""

# --- Standard Library Imports ---
import os
import io
import logging
from typing import Optional, Dict, Any

# --- Third-Party Imports ---
import requests
from PIL import Image, UnidentifiedImageError

# --- Optional rembg import for local background removal ---
_REMBG_AVAILABLE = False
try:
    from rembg import remove as remove_bg_local
    _REMBG_AVAILABLE = True
    logging.info("rembg library loaded successfully for local background removal.")
except ImportError:
    logging.warning("rembg library not found. Local background removal fallback is unavailable.")
    # Define a dummy function if rembg isn't available to avoid NameError later
    def remove_bg_local(img_data, *args, **kwargs) -> None:
        logger.error("rembg function called but library is not available.")
        return None

# --- Optional Streamlit import for UI integration ---
_AI_SERVICES_IN_STREAMLIT_CONTEXT = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'): # Check if running in a context where secrets are available
        _AI_SERVICES_IN_STREAMLIT_CONTEXT = True
        _st_module = st # Store for conditional use
except (ImportError, RuntimeError, ModuleNotFoundError):
    pass # Fail silently if streamlit is not available or not running

# --- Logging setup ---
logger = logging.getLogger(__name__)
# Ensure basic logging is configured if running standalone or before main app config
if not logger.hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

# --- Configuration constants ---
STABILITY_API_HOST: str = os.environ.get("STABILITY_API_HOST", "https://api.stability.ai")
STABILITY_API_KEY_NAME: str = "STABILITY_API_KEY" # Name for secrets/env var
APP_USER_AGENT: str = "StreamlitImageEditor/1.0" # Basic User-Agent


def _get_stability_api_key() -> Optional[str]:
    """
    Retrieve Stability API key from Streamlit secrets or environment variables.
    
    Returns:
        API key if found, None otherwise
    """
    key = None
    source = "Not Found"
    
    if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
        try:
            key = _st_module.secrets.get(STABILITY_API_KEY_NAME)
            if key:
                source = "Streamlit Secrets"
        except Exception as e:
            logger.warning(f"Could not access Streamlit secrets for Stability Key: {e}")
    
    if not key:
        key = os.environ.get(STABILITY_API_KEY_NAME)
        if key:
            source = "Environment Variable"

    if not key:
        logger.warning(f"{STABILITY_API_KEY_NAME} not found in Streamlit Secrets or Environment Variables.")
    else:
        logger.debug(f"Stability API Key loaded from: {source}")
    
    return key


def _handle_stability_api_error(response: requests.Response, url: str) -> str:
    """
    Parse and handle Stability API error responses.
    
    Args:
        response: HTTP response object
        url: Request URL for context
        
    Returns:
        User-friendly error message
    """
    status_code = response.status_code
    error_message_prefix = f"Stability API Error ({status_code} for {url})"
    
    try:
        error_json = response.json()
        error_details_list = error_json.get('errors', [])
        if not error_details_list and 'message' in error_json:
            error_details_list = [error_json['message']]
        elif not error_details_list:
            error_details_list = [str(error_json)]

        error_details = '; '.join(map(str, error_details_list))
        log_message = f"{error_message_prefix}: {error_json}"
        user_message = f"{error_message_prefix}: {error_details}"
    except requests.exceptions.JSONDecodeError:
        error_details = response.text[:500]
        log_message = f"{error_message_prefix} (non-JSON): {error_details}"
        user_message = f"{error_message_prefix}. See console logs for details."

    logger.error(log_message)

    # Specific handling for common status codes
    status_messages = {
        401: f"API Authentication Error ({status_code}): Invalid Stability API Key.",
        403: f"API Permission Denied ({status_code}): Check key permissions or content moderation flags.",
        413: f"API Request Too Large ({status_code}): Input image might be too big.",
        429: f"API Rate Limit Exceeded ({status_code}): Too many requests. Please wait and try again.",
        500: f"API Internal Server Error ({status_code}): Problem on Stability AI's side. Try again later."
    }
    
    if status_code in status_messages:
        user_message = status_messages[status_code]

    if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
        _st_module.error(user_message)

    return user_message


def _call_stability_api(
    endpoint: str,
    api_key: str,
    image: Image.Image,
    data: Dict[str, Any],
    files_extra: Dict[str, Any] = None,
    timeout: int = 90
) -> Optional[Image.Image]:
    """
    Make a POST request to Stability AI API.
    
    Args:
        endpoint: API endpoint path
        api_key: Stability API key
        image: Input PIL Image
        data: Form data for the request
        files_extra: Additional files to include
        timeout: Request timeout in seconds
        
    Returns:
        Processed image or None if failed
    """
    if files_extra is None:
        files_extra = {}
        
    if not api_key:
        logger.error(f"Internal Error: _call_stability_api called without API key for {endpoint}.")
        return None
    if not isinstance(image, Image.Image):
        logger.error(f"Invalid input: 'image' must be a PIL Image object for {endpoint}.")
        return None

    url = f"{STABILITY_API_HOST}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
        "User-Agent": APP_USER_AGENT
    }

    try:
        # Prepare image data
        buffer = io.BytesIO()
        img_format = 'PNG'
        if image.mode == 'RGB':
            img_format = 'JPEG'
        
        save_kwargs = {'quality': 95} if img_format == 'JPEG' else {}

        save_image = image
        if img_format == 'JPEG' and image.mode == 'RGBA':
            save_image = image.convert('RGB')
        elif img_format == 'PNG' and image.mode not in ['RGB', 'RGBA', 'L', 'LA']:
            save_image = image.convert('RGBA')

        save_image.save(buffer, format=img_format, **save_kwargs)
        buffer.seek(0)
        files = {'image': (f'input.{img_format.lower()}', buffer, f'image/{img_format.lower()}')}
        files.update(files_extra)

        logger.info(f"Calling Stability API: POST {url}")
        logger.debug(f"Data Keys: {list(data.keys())}")
        logger.debug(f"Files: {list(files.keys())}")

        response = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

        logger.debug(f"Stability API Response Status: {response.status_code} for {endpoint}")
        if response.status_code == 200:
            logger.info(f"Stability API call successful for endpoint {endpoint}.")
            try:
                result_image = Image.open(io.BytesIO(response.content))
                output_mode = "RGBA" if result_image.mode in ['RGBA', 'LA', 'P'] else "RGB"
                return result_image.convert(output_mode)
            except (UnidentifiedImageError, ValueError, Exception) as img_err:
                logger.error(f"Failed to decode/convert image response from Stability API ({endpoint}): {img_err}", exc_info=True)
                if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
                    _st_module.error("Received invalid image data from API.")
                return None
        else:
            _handle_stability_api_error(response, url)
            return None

    except requests.exceptions.Timeout:
        logger.error(f"Timeout ({timeout}s) calling Stability API: {url}")
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Stability API ({url}): {e}", exc_info=True)
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error(f"Network error connecting to API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Stability API call ({url}): {e}", exc_info=True)
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error(f"An unexpected error occurred: {e}")
        return None


def _remove_background_stability(image: Image.Image, api_key: str) -> Optional[Image.Image]:
    """Remove background using Stability AI API."""
    logger.debug("Using Stability AI for background removal.")
    endpoint = "/v2beta/stable-image/edit/remove-background"
    data = {"output_format": "png"}
    return _call_stability_api(endpoint, api_key, image, data)


def _upscale_image_stability_fast(image: Image.Image, api_key: str) -> Optional[Image.Image]:
    """Upscale image 4x using Stability AI Fast Upscaler."""
    logger.debug("Using Stability AI Fast Upscaler (4x).")
    endpoint = "/v2beta/stable-image/upscale/fast"
    data = {"output_format": "png"}
    return _call_stability_api(endpoint, api_key, image, data)


def _search_and_replace_stability(
    image: Image.Image, 
    api_key: str, 
    search_prompt: str, 
    prompt: str, 
    negative_prompt: Optional[str] = None
) -> Optional[Image.Image]:
    """Replace objects using Stability AI Search and Replace."""
    logger.debug("Using Stability AI Search and Replace.")
    endpoint = "/v2beta/stable-image/edit/search-and-replace"
    data = {"search_prompt": search_prompt, "prompt": prompt, "output_format": "png"}
    if negative_prompt:
        data["negative_prompt"] = negative_prompt
    return _call_stability_api(endpoint, api_key, image, data)


def _recolor_object_stability(
    image: Image.Image, 
    api_key: str, 
    select_prompt: str, 
    prompt: str, 
    negative_prompt: Optional[str] = None
) -> Optional[Image.Image]:
    """Recolor objects using Stability AI Search and Recolor."""
    logger.debug("Using Stability AI Search and Recolor.")
    endpoint = "/v2beta/stable-image/edit/search-and-recolor"
    data = {"select_prompt": select_prompt, "prompt": prompt, "output_format": "png"}
    if negative_prompt:
        data["negative_prompt"] = negative_prompt
    return _call_stability_api(endpoint, api_key, image, data)


def remove_background_ai(image: Image.Image) -> Optional[Image.Image]:
    """
    Remove background from image using AI.
    
    Tries Stability AI first, then falls back to local rembg if available.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Image with background removed or None if failed
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input to remove_background_ai")
        return None

    api_key = _get_stability_api_key()
    stability_result = None
    if api_key:
        stability_result = _remove_background_stability(image, api_key)

    if stability_result is not None:
        logger.info("Background removal successful via Stability AI.")
        return stability_result
    else:
        logger.warning("Stability AI background removal failed or key unavailable. Checking rembg fallback...")
        if _REMBG_AVAILABLE:
            logger.info("Attempting fallback using local rembg...")
            try:
                rembg_result = remove_bg_local(image.convert('RGBA'))
                if rembg_result:
                    logger.info("Background removal successful via local rembg.")
                    return rembg_result.convert("RGBA")
                else:
                    logger.error("Local rembg processing returned None.")
                    return None
            except Exception as e:
                logger.error(f"Error during rembg fallback: {e}", exc_info=True)
                if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
                    _st_module.warning("Local background removal failed.")
                return None
        else:
            logger.error("Background removal failed: Stability API unusable and rembg not installed.")
            if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
                _st_module.error("Background removal requires a Stability API key or the 'rembg' library.")
            return None


def upscale_image_ai(image: Image.Image, scale_factor: int = 4) -> Optional[Image.Image]:
    """
    Upscale image using AI.
    
    Uses Stability AI Fast Upscaler (always performs 4x upscaling).
    
    Args:
        image: Input PIL Image
        scale_factor: Requested scale factor (ignored, always 4x)
        
    Returns:
        Upscaled image or None if failed
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input to upscale_image_ai")
        return None

    if scale_factor != 4:
        logger.warning(f"Requested upscale factor {scale_factor} ignored; Stability Fast Upscaler performs 4x.")

    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error("Upscaling requires a Stability API Key.")
        return None

    result = _upscale_image_stability_fast(image, api_key)
    if result is None:
        logger.error("AI Upscaling failed via Stability Fast Upscaler.")
    return result


def search_and_replace_ai(
    image: Image.Image, 
    search_prompt: str, 
    prompt: str, 
    negative_prompt: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Search and replace objects in image using AI.
    
    Args:
        image: Input PIL Image
        search_prompt: Description of object to find
        prompt: Description of replacement object
        negative_prompt: What to avoid in the result
        
    Returns:
        Modified image or None if failed
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input to search_and_replace_ai")
        return None
    
    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error("Search and Replace requires a Stability API Key.")
        return None
    
    return _search_and_replace_stability(image, api_key, search_prompt, prompt, negative_prompt)


def recolor_object_ai(
    image: Image.Image, 
    select_prompt: str, 
    prompt: str, 
    negative_prompt: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Recolor objects in image using AI.
    
    Args:
        image: Input PIL Image
        select_prompt: Description of object to recolor
        prompt: Description of new color/appearance
        negative_prompt: What to avoid in the result
        
    Returns:
        Recolored image or None if failed
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input to recolor_object_ai")
        return None
    
    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
            _st_module.error("Recolor Object requires a Stability API Key.")
        return None
    
    return _recolor_object_stability(image, api_key, select_prompt, prompt, negative_prompt)


if __name__ == "__main__":
    """Test module functionality when run directly."""
    from PIL import ImageDraw
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logger.info("--- Running ai_services.py directly for Integration Testing ---")
    logger.info(f"Streamlit Context Detected: {_AI_SERVICES_IN_STREAMLIT_CONTEXT}")
    logger.info(f"rembg Available: {_REMBG_AVAILABLE}")

    API_KEY_FOR_TEST = _get_stability_api_key()
    if not API_KEY_FOR_TEST:
        logger.warning("STABILITY_API_KEY not found. API calls will fail. Testing only mock/fallback paths.")

    test_img_path = "test_image_ai_service.png"
    test_img = None
    
    try:
        if not os.path.exists(test_img_path):
            img_create = Image.new('RGB', (200, 150), 'lightgrey')
            draw = ImageDraw.Draw(img_create)
            draw.rectangle((20, 30, 80, 90), fill='red', outline='black')
            draw.ellipse((110, 40, 180, 110), fill='blue', outline='black')
            img_create.save(test_img_path)
            logger.info(f"Created dummy test image: {test_img_path}")
        test_img = Image.open(test_img_path)
        logger.info(f"Loaded test image: {test_img.size}, {test_img.mode}")
    except Exception as e:
        logger.error(f"Error loading/creating test image: {e}", exc_info=True)

    if test_img:
        logger.info("\n--- Testing Background Removal ---")
        bg_removed = remove_background_ai(test_img.copy())
        if bg_removed:
            logger.info(f"BG Remove Result: Mode={bg_removed.mode}, Size={bg_removed.size}")
        else:
            logger.error("Background Removal FAILED.")

        logger.info("\n--- Testing Upscale (Stability Fast 4x) ---")
        upscaled = upscale_image_ai(test_img.copy(), scale_factor=4)
        if upscaled:
            logger.info(f"Upscale Result: Mode={upscaled.mode}, Size={upscaled.size}")
        else:
            logger.error("Upscaling FAILED.")

        logger.info("\n--- Testing Search and Replace ---")
        replaced = search_and_replace_ai(test_img.copy(), search_prompt="the red shape", prompt="a green star")
        if replaced:
            logger.info(f"Search/Replace Result: Mode={replaced.mode}, Size={replaced.size}")
        else:
            logger.error("Search and Replace FAILED.")

        logger.info("\n--- Testing Recolor ---")
        recolored = recolor_object_ai(test_img.copy(), select_prompt="the blue object", prompt="make it yellow")
        if recolored:
            logger.info(f"Recolor Result: Mode={recolored.mode}, Size={recolored.size}")
        else:
            logger.error("Recolor FAILED.")
    else:
        logger.warning("Skipping direct API call tests as test image is unavailable.")

    logger.info("--- Finished ai_services.py direct execution checks ---")