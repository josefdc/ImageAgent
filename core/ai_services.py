# streamlit_image_editor/core/ai_services.py
# Implements AI image processing functions using external APIs (Stability AI, rembg).

# --- Standard Library Imports ---
import os
import io
import logging
from typing import Optional, Dict, Any

# --- Third-Party Imports ---
import requests
from PIL import Image, UnidentifiedImageError

# --- Optional Local AI Model Import (rembg) ---
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

# --- Streamlit Import (Conditional for Secrets/Error Display) ---
_AI_SERVICES_IN_STREAMLIT_CONTEXT = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'): # Check if running in a context where secrets are available
        _AI_SERVICES_IN_STREAMLIT_CONTEXT = True
        _st_module = st # Store for conditional use
except (ImportError, RuntimeError, ModuleNotFoundError):
    pass # Fail silently if streamlit is not available or not running

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Ensure basic logging is configured if running standalone or before main app config
if not logger.hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=_log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Stability AI Configuration ---
STABILITY_API_HOST: str = os.environ.get("STABILITY_API_HOST", "https://api.stability.ai")
STABILITY_API_KEY_NAME: str = "STABILITY_API_KEY" # Name for secrets/env var
APP_USER_AGENT: str = "StreamlitImageEditor/1.0" # Basic User-Agent

# --- Helper Functions ---

def _get_stability_api_key() -> Optional[str]:
    """Safely retrieves the Stability API key from Streamlit secrets or environment variables."""
    key = None
    source = "Not Found"
    if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
        try:
            key = _st_module.secrets.get(STABILITY_API_KEY_NAME)
            if key: source = "Streamlit Secrets"
        except Exception as e: # Catch potential errors accessing secrets
            logger.warning(f"Could not access Streamlit secrets for Stability Key: {e}")
    if not key:
        key = os.environ.get(STABILITY_API_KEY_NAME)
        if key: source = "Environment Variable"

    if not key:
        logger.warning(f"{STABILITY_API_KEY_NAME} not found in Streamlit Secrets or Environment Variables.")
    else:
        logger.debug(f"Stability API Key loaded from: {source}")
    return key

def _handle_stability_api_error(response: requests.Response, url: str) -> str:
    """Parses Stability API error responses for logging and user feedback."""
    status_code = response.status_code
    error_message_prefix = f"Stability API Error ({status_code} for {url})"
    try:
        error_json = response.json()
        # Stability API often uses an 'errors' list or a 'message' field
        error_details_list = error_json.get('errors', [])
        if not error_details_list and 'message' in error_json:
             error_details_list = [error_json['message']]
        elif not error_details_list:
             error_details_list = [str(error_json)] # Fallback

        error_details = '; '.join(map(str, error_details_list))
        log_message = f"{error_message_prefix}: {error_json}" # Log full JSON
        user_message = f"{error_message_prefix}: {error_details}" # More concise for UI
    except requests.exceptions.JSONDecodeError:
        error_details = response.text[:500] # Show raw text if not JSON
        log_message = f"{error_message_prefix} (non-JSON): {error_details}"
        user_message = f"{error_message_prefix}. See console logs for details."

    logger.error(log_message) # Log detailed error

    # Specific handling for common status codes
    if status_code == 401: user_message = f"API Authentication Error ({status_code}): Invalid Stability API Key."
    elif status_code == 403: user_message = f"API Permission Denied ({status_code}): Check key permissions or content moderation flags."
    elif status_code == 413: user_message = f"API Request Too Large ({status_code}): Input image might be too big."
    elif status_code == 429: user_message = f"API Rate Limit Exceeded ({status_code}): Too many requests. Please wait and try again."
    elif status_code == 500: user_message = f"API Internal Server Error ({status_code}): Problem on Stability AI's side. Try again later."
    # Add more specific messages for 400/422 based on common 'errors' content if needed

    if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module:
         _st_module.error(user_message) # Show structured error in Streamlit UI

    return user_message # Return the user-facing message

def _call_stability_api(
    endpoint: str,
    api_key: str, # API key is now passed in
    image: Image.Image,
    data: Dict[str, Any],
    files_extra: Dict[str, Any] = {},
    timeout: int = 90
) -> Optional[Image.Image]:
    """Helper: Makes POST request to Stability AI, handles image data, auth, errors."""
    if not api_key: # Should be checked before calling, but double-check
        logger.error(f"Internal Error: _call_stability_api called without API key for {endpoint}.")
        return None
    if not isinstance(image, Image.Image):
        logger.error(f"Invalid input: 'image' must be a PIL Image object for {endpoint}.")
        return None

    url = f"{STABILITY_API_HOST}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*", # Request raw image bytes
        "User-Agent": APP_USER_AGENT # Identify our app
    }

    try:
        # --- Prepare Image Data ---
        buffer = io.BytesIO()
        # Use PNG for quality and potential transparency, fallback to JPEG for specific modes if needed
        img_format = 'PNG' # Default to PNG
        if image.mode == 'RGB': img_format = 'JPEG' # JPEG is fine for RGB
        save_kwargs = {'quality': 95} if img_format == 'JPEG' else {}

        # Ensure image is in a suitable mode before saving
        save_image = image
        if img_format == 'JPEG' and image.mode == 'RGBA':
             save_image = image.convert('RGB') # Remove alpha for JPEG
        elif img_format == 'PNG' and image.mode not in ['RGB', 'RGBA', 'L', 'LA']:
             save_image = image.convert('RGBA') # Convert complex modes to RGBA for PNG

        save_image.save(buffer, format=img_format, **save_kwargs)
        buffer.seek(0)
        files = {'image': (f'input.{img_format.lower()}', buffer, f'image/{img_format.lower()}')}
        files.update(files_extra) # Add mask etc.

        # --- Log Request Details ---
        logger.info(f"Calling Stability API: POST {url}")
        # Avoid logging sensitive data like prompts directly in production INFO logs
        logged_data_keys = list(data.keys())
        logger.debug(f"  Headers: Authorization=Bearer ***, Accept=image/*, User-Agent={APP_USER_AGENT}")
        logger.debug(f"  Data Keys: {logged_data_keys}")
        logger.debug(f"  Files: {list(files.keys())}")

        # --- Make API Request ---
        response = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

        # --- Handle Response ---
        logger.debug(f"Stability API Response Status: {response.status_code} for {endpoint}")
        if response.status_code == 200:
            logger.info(f"Stability API call successful for endpoint {endpoint}.")
            try:
                result_image = Image.open(io.BytesIO(response.content))
                # Ensure a consistent output mode (prefer RGBA if alpha exists, else RGB)
                output_mode = "RGBA" if result_image.mode in ['RGBA', 'LA', 'P'] else "RGB" # 'P' might have transparency
                return result_image.convert(output_mode)
            except (UnidentifiedImageError, ValueError, Exception) as img_err:
                logger.error(f"Failed to decode/convert image response from Stability API ({endpoint}): {img_err}", exc_info=True)
                if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("Received invalid image data from API.")
                return None
        else:
            # Use helper to handle specific error codes and display messages
            _handle_stability_api_error(response, url)
            return None # Indicate failure

    except requests.exceptions.Timeout:
        logger.error(f"Timeout ({timeout}s) calling Stability API: {url}")
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Stability API ({url}): {e}", exc_info=True)
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error(f"Network error connecting to API: {e}")
        return None
    except Exception as e: # Catch-all for unexpected errors during the process
        logger.error(f"Unexpected error during Stability API call ({url}): {e}", exc_info=True)
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error(f"An unexpected error occurred: {e}")
        return None

# --- Internal Stability Service Functions (Now require API Key) ---

def _remove_background_stability(image: Image.Image, api_key: str) -> Optional[Image.Image]:
    """Internal: Removes background using Stability AI."""
    logger.debug("Using Stability AI for background removal.")
    endpoint = "/v2beta/stable-image/edit/remove-background"
    # API requires png or webp for output to preserve transparency
    data = {"output_format": "png"}
    return _call_stability_api(endpoint, api_key, image, data)

def _upscale_image_stability_fast(image: Image.Image, api_key: str) -> Optional[Image.Image]:
    """Internal: Upscales image 4x using Stability AI Fast Upscaler."""
    logger.debug("Using Stability AI Fast Upscaler (4x).")
    endpoint = "/v2beta/stable-image/upscale/fast"
    data = {"output_format": "png"} # Choose preferred output
    return _call_stability_api(endpoint, api_key, image, data)

def _search_and_replace_stability(image: Image.Image, api_key: str, search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Optional[Image.Image]:
    """Internal: Replaces object using Stability AI Search and Replace."""
    logger.debug("Using Stability AI Search and Replace.")
    endpoint = "/v2beta/stable-image/edit/search-and-replace"
    data = {"search_prompt": search_prompt, "prompt": prompt, "output_format": "png"}
    if negative_prompt: data["negative_prompt"] = negative_prompt
    return _call_stability_api(endpoint, api_key, image, data)

def _recolor_object_stability(image: Image.Image, api_key: str, select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Optional[Image.Image]:
    """Internal: Recolors object using Stability AI Search and Recolor."""
    logger.debug("Using Stability AI Search and Recolor.")
    endpoint = "/v2beta/stable-image/edit/search-and-recolor"
    data = {"select_prompt": select_prompt, "prompt": prompt, "output_format": "png"}
    if negative_prompt: data["negative_prompt"] = negative_prompt
    return _call_stability_api(endpoint, api_key, image, data)


# --- Combined Public Functions (Called by agent/tools.py) ---
# These functions fetch the API key once and then call the internal implementations.

def remove_background_ai(image: Image.Image) -> Optional[Image.Image]:
    """
    Tries Stability API first for background removal, then falls back to local rembg if available.
    Requires STABILITY_API_KEY to be configured for Stability AI.
    """
    if not isinstance(image, Image.Image): logger.error("Invalid image input to remove_background_ai"); return None

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
                # rembg works best with RGBA input
                rembg_result = remove_bg_local(image.convert('RGBA'))
                if rembg_result:
                    logger.info("Background removal successful via local rembg.")
                    return rembg_result.convert("RGBA") # Ensure alpha channel
                else:
                    logger.error("Local rembg processing returned None.")
                    return None
            except Exception as e:
                logger.error(f"Error during rembg fallback: {e}", exc_info=True)
                if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.warning("Local background removal failed.")
                return None
        else:
            logger.error("Background removal failed: Stability API unusable and rembg not installed.")
            if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("Background removal requires a Stability API key or the 'rembg' library.")
            return None

def upscale_image_ai(image: Image.Image, scale_factor: int = 4) -> Optional[Image.Image]:
    """
    Uses Stability AI Fast Upscaler (always performs 4x).
    Logs a warning if scale_factor != 4 is requested but still proceeds.
    Requires STABILITY_API_KEY to be configured.
    """
    if not isinstance(image, Image.Image): logger.error("Invalid image input to upscale_image_ai"); return None

    if scale_factor != 4:
        logger.warning(f"Requested upscale factor {scale_factor} ignored; Stability Fast Upscaler performs 4x.")

    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("Upscaling requires a Stability API Key.")
        return None # Cannot proceed without API key

    result = _upscale_image_stability_fast(image, api_key)
    if result is None:
        logger.error("AI Upscaling failed via Stability Fast Upscaler.")
        # Error likely already shown by helper function if in Streamlit context
    return result

def search_and_replace_ai(image: Image.Image, search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Optional[Image.Image]:
    """
    Wrapper for Stability AI Search and Replace.
    Requires STABILITY_API_KEY to be configured.
    """
    if not isinstance(image, Image.Image): logger.error("Invalid image input to search_and_replace_ai"); return None
    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("Search and Replace requires a Stability API Key.")
        return None
    return _search_and_replace_stability(image, api_key, search_prompt, prompt, negative_prompt)

def recolor_object_ai(image: Image.Image, select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Optional[Image.Image]:
    """
    Wrapper for Stability AI Search and Recolor.
    Requires STABILITY_API_KEY to be configured.
    """
    if not isinstance(image, Image.Image): logger.error("Invalid image input to recolor_object_ai"); return None
    api_key = _get_stability_api_key()
    if not api_key:
        if _AI_SERVICES_IN_STREAMLIT_CONTEXT and _st_module: _st_module.error("Recolor Object requires a Stability API Key.")
        return None
    return _recolor_object_stability(image, api_key, select_prompt, prompt, negative_prompt)


# --- Direct Execution Test Block (Improved) ---
if __name__ == "__main__":
    # Configure logging for direct script execution
    logging.basicConfig(level=logging.DEBUG, # Set to DEBUG for detailed output during test
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("--- Running ai_services.py directly for Integration Testing ---")
    logger.info(f"Streamlit Context Detected: {_AI_SERVICES_IN_STREAMLIT_CONTEXT}")
    logger.info(f"rembg Available: {_REMBG_AVAILABLE}")

    # Check for Stability API Key for meaningful tests
    API_KEY_FOR_TEST = _get_stability_api_key()
    if not API_KEY_FOR_TEST:
        logger.warning("STABILITY_API_KEY not found. API calls will fail. Testing only mock/fallback paths.")

    # Define test image path
    test_img_path = "test_image_ai_service.png" # Use a unique name

    # Attempt to create or load a test image
    test_img = None
    try:
        if not os.path.exists(test_img_path):
            # Create a simple image with distinct features for testing replace/recolor
            img_create = Image.new('RGB', (200, 150), 'lightgrey')
            draw = ImageDraw.Draw(img_create)
            draw.rectangle((20, 30, 80, 90), fill='red', outline='black') # Red rectangle
            draw.ellipse((110, 40, 180, 110), fill='blue', outline='black') # Blue ellipse
            img_create.save(test_img_path)
            logger.info(f"Created dummy test image: {test_img_path}")
        test_img = Image.open(test_img_path)
        logger.info(f"Loaded test image: {test_img.size}, {test_img.mode}")
    except NameError: # PIL/ImageDraw might not be available if test runs very early
        logger.error("PIL library (for Image/ImageDraw) not available for creating test image.")
    except FileNotFoundError:
        logger.error(f"Test image '{test_img_path}' not found and couldn't be created.")
    except Exception as e:
        logger.error(f"Error loading/creating test image: {e}", exc_info=True)

    # Run tests only if image loaded
    if test_img:
        # Test BG Remove (will try Stability then rembg)
        logger.info("\n--- Testing Background Removal ---")
        bg_removed = remove_background_ai(test_img.copy()) # Use copy
        if bg_removed: logger.info(f"BG Remove Result: Mode={bg_removed.mode}, Size={bg_removed.size}")
        else: logger.error("Background Removal FAILED.")

        # Test Upscale (Stability Fast 4x)
        logger.info("\n--- Testing Upscale (Stability Fast 4x) ---")
        upscaled = upscale_image_ai(test_img.copy(), scale_factor=4)
        if upscaled: logger.info(f"Upscale Result: Mode={upscaled.mode}, Size={upscaled.size}")
        else: logger.error("Upscaling FAILED.")

        # Test Search and Replace
        logger.info("\n--- Testing Search and Replace ---")
        replaced = search_and_replace_ai(test_img.copy(), search_prompt="the red shape", prompt="a green star")
        if replaced: logger.info(f"Search/Replace Result: Mode={replaced.mode}, Size={replaced.size}")
        else: logger.error("Search and Replace FAILED.")

        # Test Recolor
        logger.info("\n--- Testing Recolor ---")
        recolored = recolor_object_ai(test_img.copy(), select_prompt="the blue object", prompt="make it yellow")
        if recolored: logger.info(f"Recolor Result: Mode={recolored.mode}, Size={recolored.size}")
        else: logger.error("Recolor FAILED.")
    else:
        logger.warning("Skipping direct API call tests as test image is unavailable.")

    logger.info("--- Finished ai_services.py direct execution checks ---")