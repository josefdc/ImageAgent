# streamlit_image_editor/agent/tools.py
# Define the tools available to the LLM agent with improved robustness and typing.

# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Literal, Callable
import logging

# --- Path Setup (Add Project Root) ---
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        print(f"DEBUG (tools.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (tools.py): Failed during sys.path setup: {e}")

# --- Third-Party Imports ---
from langchain_core.tools import tool
from PIL import Image, ImageFilter, UnidentifiedImageError
from pydantic import BaseModel, Field

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Streamlit/State Imports ---
_IN_STREAMLIT_CONTEXT_TOOLS = False
try:
    import streamlit as st_state_access # Use alias
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
    from core import processing, ai_services
    logger.info("Tools: Successfully imported CORE processing and AI services.")
    _CORE_MODULES_LOADED = True
except ImportError as e:
     logger.error(f"Tools: FAILED to import core modules: {e}. Using MOCKS.", exc_info=True)
     # Define Mocks only if import failed
     class MockProcessing:
          def __getattr__(self, name):
               def mock_func(*args, **kwargs):
                    logger.info(f"[MOCK] processing.{name} called")
                    img = kwargs.get('input_image') or (args[0] if args and isinstance(args[0], Image.Image) else None)
                    if img and name != 'get_image_info': return img.copy()
                    elif name == 'get_image_info': return "Mock Info: 100x100 RGB"
                    else: return None
               return mock_func
     processing = MockProcessing()
     class MockAIServices:
          def remove_background_ai(self, img): logger.info(f"[MOCK] ai_services.remove_background_ai called"); return Image.new('RGBA', (100, 100), (0,0,0,0))
          def upscale_image_ai(self, img, scale_factor=2): logger.info(f"[MOCK] ai_services.upscale_image_ai called with factor {scale_factor}"); return Image.new('RGB', (img.width*scale_factor, img.height*scale_factor), color='cyan') if isinstance(img, Image.Image) else Image.new('RGB', (200,200), 'cyan')
     ai_services = MockAIServices()

# --- Helper Function API Key ---
def _get_api_key(service_key_name: str) -> Optional[str]:
     key = None; source = "Not Found"
     if _IN_STREAMLIT_CONTEXT_TOOLS:
          try: key = st_state_access.secrets.get(service_key_name); source = "Streamlit Secrets" if key else source
          except Exception: pass
     if not key: key = os.environ.get(service_key_name); source = "Environment Var" if key else source
     if key: logger.debug(f"API Key {service_key_name} loaded from: {source}")
     else: logger.warning(f"API Key '{service_key_name}' not found.")
     return key

# --- Pydantic Models for Tool Arguments (Sin cambios) ---
class BrightnessArgs(BaseModel): factor: int = Field(..., description="Integer brightness adjustment factor, range -100 (darker) to 100 (brighter). 0 means no change.")
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
# Return Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]

def _adjust_brightness_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info(f"Impl: adjust_brightness running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image provided.", None, None
    try:
        new_image = processing.apply_brightness(input_image.copy(), factor)
        if new_image:
            ui_updates = {"brightness_slider": factor}
            return f"Success: Brightness adjusted.", new_image, ui_updates
        else: return "Error: Failed applying brightness.", None, None
    except Exception as e: logger.error(f"Brightness impl error: {e}", exc_info=True); return f"Error: {str(e)}", None, None

def _adjust_contrast_impl(input_image: Image.Image, factor: float) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info(f"Impl: adjust_contrast running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
    try:
        new_image = processing.apply_contrast(input_image.copy(), factor)
        if new_image:
            ui_updates = {"contrast_slider": factor}
            return f"Success: Contrast adjusted.", new_image, ui_updates
        else: return "Error: Failed applying contrast.", None, None
    except Exception as e: logger.error(f"Contrast impl error: {e}", exc_info=True); return f"Error: {str(e)}", None, None

def _apply_filter_impl(input_image: Image.Image, filter_name: str) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info(f"Impl: apply_filter running with filter: {filter_name}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
    filter_map = {
        'blur': ImageFilter.BLUR, 'sharpen': ImageFilter.SHARPEN, 'smooth': ImageFilter.SMOOTH,
        'edge_enhance': ImageFilter.EDGE_ENHANCE, 'emboss': ImageFilter.EMBOSS, 'contour': ImageFilter.CONTOUR,
    }
    selected_filter = filter_map.get(filter_name.lower())
    if not selected_filter:
         valid_filters = ", ".join(filter_map.keys()); return f"Error: Invalid filter name '{filter_name}'. Valid: {valid_filters}", None, None
    try:
        processed = input_image.copy().filter(selected_filter)
        return f"Success: Applied '{filter_name}' filter.", processed, None # No UI update
    except Exception as e: logger.error(f"Filter impl error: {e}", exc_info=True); return f"Error applying filter '{filter_name}': {str(e)}", None, None

def _rotate_image_impl(input_image: Image.Image, degrees: float) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info(f"Impl: rotate_image running with degrees: {degrees}")
    if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
    try:
        angle = int(degrees % 360)
        new_image = processing.apply_rotation(input_image.copy(), angle)
        if new_image:
            ui_updates = {"rotation_slider": angle}
            return f"Success: Image rotated by {angle} degrees.", new_image, ui_updates
        else: return "Error: Failed applying rotation.", None, None
    except Exception as e: logger.error(f"Rotate impl error: {e}", exc_info=True); return f"Error: {str(e)}", None, None

def _invert_colors_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
     logger.info("Impl: invert_colors running")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
     try:
        new_image = processing.apply_negative(input_image.copy())
        if new_image: return "Success: Colors inverted.", new_image, None # No UI update
        else: return "Error: Failed inverting colors.", None, None
     except Exception as e: logger.error(f"Invert impl error: {e}", exc_info=True); return f"Error: {str(e)}", None, None

def _crop_image_impl(input_image: Image.Image, x: int, y: int, width: int, height: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
     logger.info(f"Impl: crop_image running with x={x}, y={y}, w={width}, h={height}")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
     try:
        img_width, img_height = input_image.size
        if x < 0 or y < 0 or (x + width) > img_width or (y + height) > img_height:
            return (f"Error: Crop rectangle invalid."), None, None
        box = (x, y, x + width, y + height)
        new_image = input_image.crop(box)
        ui_updates = {'zoom_x': 0, 'zoom_y': 0, 'zoom_w': 100, 'zoom_h': 100} # Reset zoom UI
        return f"Success: Image cropped.", new_image, ui_updates
     except Exception as e: logger.error(f"Crop impl error: {e}", exc_info=True); return f"Error during cropping: {str(e)}", None, None

def _apply_binarization_impl(input_image: Image.Image, threshold: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
     logger.info(f"Impl: apply_binarization running with threshold: {threshold}")
     if not isinstance(input_image, Image.Image): return "Error: Invalid input image type.", None, None
     try:
        new_image = processing.apply_binarization(input_image.copy(), threshold)
        if new_image:
            ui_updates = {"binarize_thresh_slider": threshold, "apply_binarization_cb": True}
            return f"Success: Image binarized.", new_image, ui_updates
        else: return "Error: Failed binarization.", None, None
     except Exception as e: logger.error(f"Binarize impl error: {e}", exc_info=True); return f"Error: {str(e)}", None, None

def _remove_background_ai_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info("Impl: remove_background_ai running")
    if not isinstance(input_image, Image.Image): return f"Error: Invalid input image type.", None, None
    # API key retrieval moved inside to use helper
    # api_key = _get_api_key("REMOVEBG_API_KEY") # Assuming this name for remove.bg key
    # if not api_key: return "Error: Remove.bg API Key is not configured.", None, None
    try:
        logger.info("Info: Calling AI background removal...")
        # Use the processing function which might contain the API call or local model
        result_img = ai_services.remove_background_ai(input_image.copy()) # Pass only image
        if result_img: return "Success: Background removed via AI.", result_img, None # No UI update
        else: return "Error: Background removal via AI failed.", None, None
    except Exception as e: logger.error(f"Remove BG impl error: {e}", exc_info=True); return f"Error calling background removal service: {str(e)}", None, None

def _upscale_image_ai_impl(input_image: Image.Image, factor: Literal[2, 4]) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    logger.info(f"Impl: upscale_image_ai running with factor: {factor}")
    if not isinstance(input_image, Image.Image): return f"Error: Invalid input image type.", None, None
    # API key retrieval moved inside if needed by the service function
    # api_key = _get_api_key("UPSCALE_API_KEY") # Use appropriate key name
    # if not api_key: return "Error: Upscale API Key is not configured.", None, None
    try:
        logger.info(f"Info: Calling AI Upscale (x{factor})...")
        result_img = ai_services.upscale_image_ai(input_image.copy(), scale_factor=factor) # Pass factor
        if result_img: return f"Success: Image upscaled x{factor} via AI.", result_img, None # No UI update
        else: return f"Error: Upscaling x{factor} via AI failed.", None, None
    except Exception as e: logger.error(f"Upscale impl error: {e}", exc_info=True); return f"Error calling Upscale service: {str(e)}", None, None

# --- Tool Definitions exposed to LLM (Schema Definitions) ---
@tool(args_schema=BrightnessArgs)
def adjust_brightness(factor: int) -> str:
    """Adjusts the brightness of the current image. Provide a factor between -100 (darker) and 100 (brighter). 0 means no change."""
    # This is just the schema definition, logic is in _impl
    pass

@tool(args_schema=ContrastArgs)
def adjust_contrast(factor: float) -> str:
    """Adjusts the contrast of the current image. Factor >= 0. 1.0 is original contrast."""
    pass

@tool(args_schema=FilterArgs)
def apply_filter(filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour']) -> str:
    """Applies a standard image filter (blur, sharpen, smooth, edge_enhance, emboss, contour) to the current image."""
    pass

@tool(args_schema=RotateArgs)
def rotate_image(degrees: float) -> str:
    """Rotates the current image clockwise by the specified degrees."""
    pass

@tool # No args needed for schema
def invert_colors() -> str:
    """Inverts the colors of the current image (creates a negative)."""
    pass

@tool(args_schema=CropArgs)
def crop_image(x: int, y: int, width: int, height: int) -> str:
    """Crops the current image to a specified rectangular area using top-left coordinates (x, y) and dimensions (width, height)."""
    pass

@tool(args_schema=BinarizeArgs)
def apply_binarization(threshold: int) -> str:
    """Converts the current image to black and white based on a threshold (0-255)."""
    pass

@tool # No args needed for schema
def remove_background_ai() -> str:
    """Removes the background from the current image using an AI service. Requires API Key configuration."""
    pass

@tool(args_schema=UpscaleArgs)
def upscale_image_ai(factor: Literal[2, 4]) -> str:
    """Upscales the current image by a factor of 2 or 4 using an AI service. Requires API Key configuration."""
    pass

@tool
def get_image_info() -> str:
    """Gets information about the current image (dimensions, color mode). Does not modify the image."""
    # Logic remains here as it only reads state
    logger.info("Tool: get_image_info executing logic directly")
    img = None
    # Use alias for safe access within function scope
    local_st_state_access = None
    if _IN_STREAMLIT_CONTEXT_TOOLS:
        try:
            import streamlit as local_st_state_access
            img = local_st_state_access.session_state.get('processed_image')
        except Exception as e:
            logger.warning(f"Couldn't access session_state for get_image_info: {e}")

    if img is None: return "Error: Cannot access current image information."
    if not isinstance(img, Image.Image): return f"Error: Invalid image object in state."
    try:
        width, height = img.size
        mode = img.mode
        return f"Current image info: Size={width}x{height}, Mode={mode}."
    except Exception as e: logger.error(f"Error formatting image info: {e}", exc_info=True); return f"Error getting image info: {str(e)}"


# --- Tool Dictionary (For LLM Binding) ---
# Contains the @tool decorated functions (schemas)
available_tools: Dict[str, Callable] = { # Changed type hint to Callable
    t.name: t for t in [
        adjust_brightness, adjust_contrast, apply_filter, rotate_image,
        invert_colors, get_image_info, crop_image, apply_binarization,
        remove_background_ai, upscale_image_ai,
    ]
}

# --- Implementation Dictionary (For Graph Execution) ---
# Maps tool names to their actual implementation functions (_impl)
tool_implementations: Dict[str, Callable[..., Tuple[str, Optional[Image.Image], Optional[Dict]]]] = {
    "adjust_brightness": _adjust_brightness_impl,
    "adjust_contrast": _adjust_contrast_impl,
    "apply_filter": _apply_filter_impl,
    "rotate_image": _rotate_image_impl,
    "invert_colors": _invert_colors_impl,
    "crop_image": _crop_image_impl,
    "apply_binarization": _apply_binarization_impl,
    "remove_background_ai": _remove_background_ai_impl,
    "upscale_image_ai": _upscale_image_ai_impl,
    # get_image_info's logic is within its @tool decorated function, no separate _impl needed here
}


# --- Direct Execution Test Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running tools.py directly for Testing ---")
    logger.info(f"Project Root added to sys.path: {str(_PROJECT_ROOT_DIR) in sys.path if '_PROJECT_ROOT_DIR' in globals() else 'N/A'}")
    logger.info(f"Core Modules Loaded: {_CORE_MODULES_LOADED}")
    logger.info(f"Streamlit Context Detected: {_IN_STREAMLIT_CONTEXT_TOOLS}")

    logger.info("\nAvailable tools (for LLM):")
    for t_name, t_func in available_tools.items():
        try: logger.info(f"- {t_name}: {t_func.description} | Args: {t_func.args}")
        except Exception as e: logger.error(f"Could not get args for {t_name}: {e}")

    logger.info("\nAvailable implementations:")
    for t_name, t_impl in tool_implementations.items():
        logger.info(f"- {t_name} -> {t_impl.__name__}")

    logger.info("\nTesting _adjust_brightness_impl...")
    mock_img_test = Image.new("RGB", (50, 50), "red")
    result_txt, result_img, ui_updates = _adjust_brightness_impl(mock_img_test, 50)
    logger.info(f"Result Text: {result_txt}")
    logger.info(f"Result Image Type: {type(result_img)}")
    logger.info(f"UI Updates: {ui_updates}")
    if result_img: logger.info(f"Result Image Size: {result_img.size}")

    logger.info("\nTesting get_image_info tool (via available_tools)...")
    info_result = get_image_info.invoke({})
    logger.info(f"get_image_info Result: {info_result}") # Will show error if not in Streamlit context

    logger.info("\nTesting _remove_background_ai_impl (will use mock)...")
    result_txt_bg, result_img_bg, _ = _remove_background_ai_impl(mock_img_test)
    logger.info(f"Remove BG Result Text: {result_txt_bg}")
    logger.info(f"Remove BG Image Type: {type(result_img_bg)}")
    if result_img_bg: logger.info(f"Remove BG Image Mode: {result_img_bg.mode}")

    logger.info("--- End tools.py direct test ---")