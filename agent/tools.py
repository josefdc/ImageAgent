#gent/tools.py
# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Literal, Callable
import logging

# --- Path Setup (Add Project Root) ---
try:
    # Project root is the parent directory of the 'agent' directory
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        # Use print for initial setup diagnostics as logger might not be configured yet
        print(f"DEBUG (tools.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (tools.py): Failed during sys.path setup: {e}")

# --- Third-Party Imports ---
from langchain_core.tools import tool
from PIL import Image, ImageFilter, UnidentifiedImageError
from pydantic import BaseModel, Field

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Set default logging level if not configured elsewhere
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Streamlit/State Imports ---
_IN_STREAMLIT_CONTEXT_TOOLS = False
try:
    import streamlit as st_state_access # Use alias
    # Check for a streamlit-specific attribute to confirm context more reliably
    if hasattr(st_state_access, 'secrets'):
        _IN_STREAMLIT_CONTEXT_TOOLS = True
        logger.debug("Tools: Streamlit context detected.")
    else:
        logger.warning("Tools: Streamlit imported but might not be in a running app context.")
except (ImportError, RuntimeError, ModuleNotFoundError):
    logger.warning("Tools: Not running within Streamlit context or Streamlit module not found.")

# --- Core Modules Import (Use Paths Relative to Project Root) ---
_CORE_MODULES_LOADED = False
try:
    # Use import relative to the project root added to sys.path
    from core import processing, ai_services
    logger.info("Tools: Successfully imported CORE processing and AI services.")
    _CORE_MODULES_LOADED = True
except ImportError as e:
     logger.error(f"Tools: FAILED to import core modules using absolute path: {e}. Using MOCKS.", exc_info=True)
     logger.error(f"Current sys.path: {sys.path}") # Log path on failure
     # Define Mocks only if import failed
     class MockProcessing:
          def __getattr__(self, name):
               def mock_func(*args, **kwargs):
                    logger.info(f"[MOCK] processing.{name} called")
                    img = kwargs.get('input_image') or (args[0] if args and isinstance(args[0], Image.Image) else None)
                    if img and name != 'get_image_info':
                        logger.debug(f"[MOCK] Returning copy for {name}")
                        return img.copy()
                    elif name == 'get_image_info':
                         logger.debug("[MOCK] Returning mock info string")
                         return "Mock Info: 100x100 RGB"
                    else:
                         logger.debug(f"[MOCK] Returning None for {name}")
                         return None
               return mock_func
     processing = MockProcessing()
     class MockAIServices:
          def remove_background_removebg(self, img, api_key):
               logger.info(f"[MOCK] ai_services.remove_background_removebg called")
               return Image.new('RGBA', (100, 100), (0,0,0,0)) # Return transparent mock
          def upscale_image_clipdrop(self, img, api_key, factor=2):
               logger.info(f"[MOCK] ai_services.upscale_image_clipdrop called with factor {factor}")
               if isinstance(img, Image.Image): return img.resize((img.width*factor, img.height*factor))
               return Image.new('RGB', (200, 200), color='cyan')
     ai_services = MockAIServices()


# --- Helper Function API Key ---
def _get_api_key(service_key_name: str) -> Optional[str]:
     """Gets API key from Streamlit secrets or environment variables."""
     key = None; source = "Not Found"
     if _IN_STREAMLIT_CONTEXT_TOOLS:
          try:
              # Use .get() for safer access to secrets dictionary
              key = st_state_access.secrets.get(service_key_name)
              if key: source = "Streamlit Secrets"
          except Exception as e: logger.warning(f"Error accessing St secrets for {service_key_name}: {e}")
     # Fallback to environment variable
     if not key:
          key = os.environ.get(service_key_name)
          if key: source = "Environment Var"

     if key: logger.info(f"API Key {service_key_name} loaded from: {source}")
     else: logger.warning(f"API Key '{service_key_name}' not found in Secrets or Env Vars.")
     return key


# --- Pydantic Models for Tool Arguments ---
class BrightnessArgs(BaseModel): factor: int = Field(..., description="Integer brightness adjustment factor, range -100 (darkest) to 100 (brightest). 0 means no change.")
class ContrastArgs(BaseModel): factor: float = Field(..., description="Contrast adjustment factor. Float >= 0. 1.0 is original contrast. <1 decreases, >1 increases.")
class FilterArgs(BaseModel): filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour'] = Field(..., description="The name of the filter to apply.")
class RotateArgs(BaseModel): degrees: float = Field(..., description="Angle in degrees to rotate the image clockwise.")
class CropArgs(BaseModel):
    x: int = Field(..., ge=0, description="Left coordinate (X) of the crop area, starting from 0.")
    y: int = Field(..., ge=0, description="Top coordinate (Y) of the crop area, starting from 0.")
    width: int = Field(..., gt=0, description="Width of the crop area (must be positive).")
    height: int = Field(..., gt=0, description="Height of the crop area (must be positive).")
class BinarizeArgs(BaseModel): threshold: int = Field(default=128, ge=0, le=255, description="Threshold value (0-255). Pixels above become white, others black.")
class UpscaleArgs(BaseModel): factor: Literal[2, 4] = Field(default=2, description="Upscaling factor (supports 2 or 4).")


# --- Implementation Functions (Internal Logic) ---
# These functions perform the actual image processing and accept the PIL Image object.
# They return a tuple: (result_string, optional_processed_image)
# They now use the imported 'processing' and 'ai_services' (which are real if import succeeded)

def _adjust_brightness_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image]]:
    logger.info(f"Impl: adjust_brightness running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image provided.", None
    try:
        # Pass a copy to avoid modifying the original potentially shared object
        new_image = processing.apply_brightness(input_image.copy(), factor)
        if new_image: return f"Success: Brightness adjusted.", new_image
        else: return "Error: Failed applying brightness (processing returned None).", None
    except Exception as e: logger.error(f"Brightness impl error: {e}", exc_info=True); return f"Error: {str(e)}", None

def _adjust_contrast_impl(input_image: Image.Image, factor: float) -> Tuple[str, Optional[Image.Image]]:
    logger.info(f"Impl: adjust_contrast running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
    try:
        new_image = processing.apply_contrast(input_image.copy(), factor)
        if new_image: return f"Success: Contrast adjusted.", new_image
        else: return "Error: Failed applying contrast (processing returned None).", None
    except Exception as e: logger.error(f"Contrast impl error: {e}", exc_info=True); return f"Error: {str(e)}", None

def _apply_filter_impl(input_image: Image.Image, filter_name: str) -> Tuple[str, Optional[Image.Image]]:
    logger.info(f"Impl: apply_filter running with filter: {filter_name}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
    filter_map = {
        'blur': ImageFilter.BLUR, 'sharpen': ImageFilter.SHARPEN, 'smooth': ImageFilter.SMOOTH,
        'edge_enhance': ImageFilter.EDGE_ENHANCE, 'emboss': ImageFilter.EMBOSS, 'contour': ImageFilter.CONTOUR,
    }
    selected_filter = filter_map.get(filter_name.lower())
    if not selected_filter:
         valid_filters = ", ".join(filter_map.keys())
         return f"Error: Invalid filter name '{filter_name}'. Valid options: {valid_filters}", None
    try:
        # Apply filter to a copy
        processed = input_image.copy().filter(selected_filter)
        return f"Success: Applied '{filter_name}' filter.", processed
    except Exception as e: logger.error(f"Filter impl error: {e}", exc_info=True); return f"Error applying filter '{filter_name}': {str(e)}", None

def _rotate_image_impl(input_image: Image.Image, degrees: float) -> Tuple[str, Optional[Image.Image]]:
    logger.info(f"Impl: rotate_image running with degrees: {degrees}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
    try:
        angle = int(degrees % 360)
        new_image = processing.apply_rotation(input_image.copy(), angle)
        if new_image: return f"Success: Image rotated by {angle} degrees.", new_image
        else: return "Error: Failed applying rotation (processing returned None).", None
    except Exception as e: logger.error(f"Rotate impl error: {e}", exc_info=True); return f"Error: {str(e)}", None

def _invert_colors_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image]]:
     logger.info("Impl: invert_colors running")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
     try:
        new_image = processing.apply_negative(input_image.copy())
        if new_image: return "Success: Colors inverted.", new_image
        else: return "Error: Failed inverting colors (processing returned None).", None
     except Exception as e: logger.error(f"Invert impl error: {e}", exc_info=True); return f"Error: {str(e)}", None

def _crop_image_impl(input_image: Image.Image, x: int, y: int, width: int, height: int) -> Tuple[str, Optional[Image.Image]]:
     logger.info(f"Impl: crop_image running with x={x}, y={y}, w={width}, h={height}")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
     try:
        img_width, img_height = input_image.size
        if x < 0 or y < 0 or (x + width) > img_width or (y + height) > img_height:
            return (f"Error: Crop rectangle (x={x}, y={y}, w={width}, h={height}) "
                    f"is outside image bounds (W={img_width}, H={img_height})."), None
        box = (x, y, x + width, y + height)
        # Crop doesn't have a dedicated processing function, use PIL directly
        new_image = input_image.crop(box)
        return f"Success: Image cropped.", new_image
     except Exception as e: logger.error(f"Crop impl error: {e}", exc_info=True); return f"Error during cropping: {str(e)}", None

def _apply_binarization_impl(input_image: Image.Image, threshold: int) -> Tuple[str, Optional[Image.Image]]:
     logger.info(f"Impl: apply_binarization running with threshold: {threshold}")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None
     try:
        new_image = processing.apply_binarization(input_image.copy(), threshold)
        if new_image: return f"Success: Image binarized.", new_image
        else: return "Error: Failed binarization (processing returned None).", None
     except Exception as e: logger.error(f"Binarize impl error: {e}", exc_info=True); return f"Error: {str(e)}", None

def _remove_background_ai_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image]]:
    logger.info("Impl: remove_background_ai running")
    if not isinstance(input_image, Image.Image): return f"Error: Invalid input image type.", None
    api_key = _get_api_key("REMOVEBG_API_KEY")
    if not api_key: return "Error: Remove.bg API Key is not configured.", None
    try:
        logger.info("Info: Calling Remove.bg API...")
        result_img = ai_services.remove_background_removebg(input_image.copy(), api_key)
        if result_img: return "Success: Background removed via AI.", result_img
        else: return "Error: Background removal via AI failed (check API response in core/ai_services).", None
    except Exception as e: logger.error(f"Remove BG impl error: {e}", exc_info=True); return f"Error calling Remove.bg API: {str(e)}", None

def _upscale_image_ai_impl(input_image: Image.Image, factor: Literal[2, 4]) -> Tuple[str, Optional[Image.Image]]:
    logger.info(f"Impl: upscale_image_ai running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return f"Error: Invalid input image type.", None
    api_key = _get_api_key("CLIPDROP_API_KEY")
    if not api_key: return "Error: ClipDrop API Key is not configured.", None
    try:
        logger.info(f"Info: Calling ClipDrop Upscale API (x{factor})...")
        result_img = ai_services.upscale_image_clipdrop(input_image.copy(), api_key, factor)
        if result_img: return f"Success: Image upscaled x{factor} via AI.", result_img
        else: return f"Error: Upscaling x{factor} via AI failed (check API response in core/ai_services).", None
    except Exception as e: logger.error(f"Upscale impl error: {e}", exc_info=True); return f"Error calling ClipDrop Upscale API: {str(e)}", None


# --- Tool Definitions exposed to LLM (Schema Definitions) ---
# These functions are decorated with @tool and define the interface for the LLM.
# They DO NOT contain the image processing logic directly.
# They DO NOT accept Image.Image as an argument visible to the LLM.

@tool(args_schema=BrightnessArgs)
def adjust_brightness(factor: int) -> str:
    """Adjusts the brightness of the current image. Provide a factor between -100 (darker) and 100 (brighter). 0 means no change."""
    logger.debug("adjust_brightness tool schema called by LLM binding.")
    return "Brightness adjustment tool schema invoked."

@tool(args_schema=ContrastArgs)
def adjust_contrast(factor: float) -> str:
    """Adjusts the contrast of the current image. Factor >= 0. 1.0 is original contrast."""
    logger.debug("adjust_contrast tool schema called by LLM binding.")
    return "Contrast adjustment tool schema invoked."

@tool(args_schema=FilterArgs)
def apply_filter(filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour']) -> str:
    """Applies a standard image filter (blur, sharpen, smooth, edge_enhance, emboss, contour) to the current image."""
    logger.debug("apply_filter tool schema called by LLM binding.")
    return "Filter application tool schema invoked."

@tool(args_schema=RotateArgs)
def rotate_image(degrees: float) -> str:
    """Rotates the current image clockwise by the specified degrees."""
    logger.debug("rotate_image tool schema called by LLM binding.")
    return "Rotation tool schema invoked."

@tool # No args needed for schema
def invert_colors() -> str:
    """Inverts the colors of the current image (creates a negative)."""
    logger.debug("invert_colors tool schema called by LLM binding.")
    return "Color inversion tool schema invoked."

@tool(args_schema=CropArgs)
def crop_image(x: int, y: int, width: int, height: int) -> str:
    """Crops the current image to a specified rectangular area using top-left coordinates (x, y) and dimensions (width, height)."""
    logger.debug("crop_image tool schema called by LLM binding.")
    return "Cropping tool schema invoked."

@tool(args_schema=BinarizeArgs)
def apply_binarization(threshold: int) -> str:
    """Converts the current image to black and white based on a threshold (0-255)."""
    logger.debug("apply_binarization tool schema called by LLM binding.")
    return "Binarization tool schema invoked."

@tool # No args needed for schema
def remove_background_ai() -> str:
    """Removes the background from the current image using an AI service (Remove.bg). Requires API Key configuration."""
    logger.debug("remove_background_ai tool schema called by LLM binding.")
    return "AI background removal tool schema invoked."

@tool(args_schema=UpscaleArgs)
def upscale_image_ai(factor: Literal[2, 4]) -> str:
    """Upscales the current image by a factor of 2 or 4 using an AI service (ClipDrop). Requires API Key configuration."""
    logger.debug("upscale_image_ai tool schema called by LLM binding.")
    return "AI upscaling tool schema invoked."

@tool
def get_image_info() -> str:
    """Gets information about the current image (dimensions, color mode). Does not modify the image."""
    logger.info("Tool: get_image_info executing logic directly")
    img = None
    if _IN_STREAMLIT_CONTEXT_TOOLS:
        try:
            # Access session state safely using .get()
            img = st_state_access.session_state.get('processed_image')
        except Exception as e:
            logger.warning(f"Couldn't access session_state for get_image_info: {e}")
    if img is None:
        return "Error: Cannot access current image information (not in state or not in Streamlit context)."
    if not isinstance(img, Image.Image):
        return f"Error: Object in state 'processed_image' is not a PIL Image (Type: {type(img)})."
    try:
        width, height = img.size
        mode = img.mode
        return f"Current image info: Size={width}x{height}, Mode={mode}."
    except Exception as e:
        logger.error(f"Error formatting image info: {e}", exc_info=True)
        return f"Error getting image info: {str(e)}"


# --- Tool Dictionary (For LLM Binding) ---
# Contains the @tool decorated functions (schemas)
available_tools: Dict[str, Any] = {
    t.name: t for t in [
        adjust_brightness, adjust_contrast, apply_filter, rotate_image,
        invert_colors, get_image_info, crop_image, apply_binarization,
        remove_background_ai, upscale_image_ai,
    ]
}

# --- Implementation Dictionary (For Graph Execution) ---
# Maps tool names to their actual implementation functions (_impl)
tool_implementations: Dict[str, Callable[..., Tuple[str, Optional[Image.Image]]]] = {
    "adjust_brightness": _adjust_brightness_impl,
    "adjust_contrast": _adjust_contrast_impl,
    "apply_filter": _apply_filter_impl,
    "rotate_image": _rotate_image_impl,
    "invert_colors": _invert_colors_impl,
    "crop_image": _crop_image_impl,
    "apply_binarization": _apply_binarization_impl,
    "remove_background_ai": _remove_background_ai_impl,
    "upscale_image_ai": _upscale_image_ai_impl,
    # get_image_info is special: its logic is in the @tool function itself.
    # The execute_tool_and_update node in agent_graph.py handles this case.
}


# --- Direct Execution Test Block ---
if __name__ == "__main__":
    # Ensure logging is setup for direct execution testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running tools.py directly for Testing ---")
    # Check if project root was added correctly
    project_root_in_path = False
    if '_PROJECT_ROOT_DIR' in globals() and isinstance(_PROJECT_ROOT_DIR, Path):
         project_root_in_path = str(_PROJECT_ROOT_DIR) in sys.path
    logger.info(f"Project Root added to sys.path: {project_root_in_path}")
    logger.info(f"Core Modules Loaded: {_CORE_MODULES_LOADED}")
    logger.info(f"Streamlit Context Detected: {_IN_STREAMLIT_CONTEXT_TOOLS}")

    logger.info("\nAvailable tools (for LLM):")
    for t_name, t_func in available_tools.items():
        try: logger.info(f"- {t_name}: {t_func.description} | Args: {t_func.args}")
        except Exception as e: logger.error(f"Could not get args for {t_name}: {e}")

    logger.info("\nAvailable implementations:")
    for t_name, t_impl in tool_implementations.items():
        logger.info(f"- {t_name} -> {t_impl.__name__}")

    # Test calling an implementation function (will use mock if core failed)
    logger.info("\nTesting _adjust_brightness_impl...")
    mock_img_test = Image.new("RGB", (50, 50), "red")
    result_txt, result_img = _adjust_brightness_impl(mock_img_test, 50)
    logger.info(f"Result Text: {result_txt}")
    logger.info(f"Result Image Type: {type(result_img)}")
    if result_img: logger.info(f"Result Image Size: {result_img.size}")

    logger.info("\nTesting get_image_info tool (via available_tools)...")
    # This will return an error when run directly because _IN_STREAMLIT_CONTEXT_TOOLS is False.
    info_result = get_image_info.invoke({})
    logger.info(f"get_image_info Result: {info_result}")
    logger.info("--- End tools.py direct test ---")