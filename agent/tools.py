# streamlit_image_editor/agent/tools.py
# Defines tool schemas for the LLM and maps them to implementation functions.
# Implementation functions fetch necessary context (like current image) and
# return results including the processed image and any UI state updates needed.

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
from PIL import Image, ImageFilter, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=_log_level,
                        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')

# --- Streamlit Import (Conditional for State/Secrets Access) ---
_IN_STREAMLIT_CONTEXT_TOOLS = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        _IN_STREAMLIT_CONTEXT_TOOLS = True
        _st_module = st
        logger.debug("Tools: Streamlit context detected.")
    else:
        logger.warning("Tools: Streamlit imported but might not be in a running app context.")
except (ImportError, RuntimeError, ModuleNotFoundError):
    logger.warning("Tools: Not running within Streamlit context or Streamlit module not found.")

# --- Core Modules & State Manager Import ---
_CORE_MODULES_LOADED = False
_STATE_MANAGER_LOADED = False # Flag for state manager import
try:
    from core import processing, ai_services
    logger.info("Tools: Successfully imported CORE processing and AI services.")
    _CORE_MODULES_LOADED = True
    # Import state manager function Needed for direct updates
    from state.session_state_manager import update_processed_image
    _STATE_MANAGER_LOADED = True
    logger.info("Tools: Successfully imported State Manager.")
except ImportError as e:
     logger.error(f"Tools: FAILED to import core/state modules: {e}. Using MOCKS.", exc_info=True)
     # Define Mocks if core imports failed
     if not _CORE_MODULES_LOADED:
         class MockProcessing:
              def __getattr__(self, name):
                   def mock_func(*args, **kwargs):
                        logger.info(f"[MOCK] processing.{name} called")
                        img = kwargs.get('input_image') or (args[0] if args and isinstance(args[0], Image.Image) else None)
                        return img.copy() if img else Image.new('RGB', (100, 100), color='gray')
                   return mock_func
         processing = MockProcessing()
         class MockAIServices:
              def remove_background_ai(self, img): logger.info(f"[MOCK] ai_services.remove_background_ai called"); return Image.new('RGBA', (100, 100), (0,0,0,0))
              def upscale_image_ai(self, img, scale_factor=4): logger.info(f"[MOCK] ai_services.upscale_image_ai called (factor {scale_factor})"); return Image.new('RGB', (img.width*scale_factor, img.height*scale_factor), color='cyan') if isinstance(img, Image.Image) else Image.new('RGB', (400,400), 'cyan')
              def search_and_replace_ai(self, img, search_prompt, prompt, negative_prompt=None): logger.info(f"[MOCK] ai_services.search_and_replace_ai called"); return img.copy() if isinstance(img, Image.Image) else Image.new('RGB', (100, 100), color='lime')
              def recolor_object_ai(self, img, select_prompt, prompt, negative_prompt=None): logger.info(f"[MOCK] ai_services.recolor_object_ai called"); return img.copy() if isinstance(img, Image.Image) else Image.new('RGB', (100, 100), color='magenta')
         ai_services = MockAIServices()
     # Mock update_processed_image if state manager failed
     if not _STATE_MANAGER_LOADED:
          def update_processed_image(img): logger.info("[MOCK] update_processed_image called"); return True # Mock success

# --- Helper Functions ---

def _get_current_image() -> Optional[Image.Image]:
    """Safely gets the current processed image from session state (if available)."""
    if not _IN_STREAMLIT_CONTEXT_TOOLS or not _st_module:
        logger.warning("Cannot get image: Not in Streamlit context.")
        return None
    try:
        img = _st_module.session_state.get('processed_image')
        if isinstance(img, Image.Image):
            logger.debug(f"Retrieved image from state: Mode={img.mode}, Size={img.size}")
            return img
        else:
             logger.warning(f"State['processed_image'] invalid (Type: {type(img)}).")
             return None
    except Exception as e:
        logger.error(f"Error accessing session state for image: {e}", exc_info=True)
        return None

# --- Pydantic Models for Tool Arguments ---
class BrightnessArgs(BaseModel): factor: int = Field(..., description="Integer brightness adjustment factor, range -100 (darker) to 100 (brighter). 0 means no change.")
class ContrastArgs(BaseModel): factor: float = Field(..., ge=0.0, description="Contrast adjustment factor. Float >= 0. 1.0 is original contrast. <1 decreases, >1 increases.")
class FilterArgs(BaseModel): filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour'] = Field(..., description="The name of the filter to apply.")
class RotateArgs(BaseModel): degrees: float = Field(..., description="Angle in degrees to rotate the image clockwise.")
class CropArgs(BaseModel):
    x: int = Field(..., ge=0, description="Left coordinate (X pixel) of the crop area, starting from 0.")
    y: int = Field(..., ge=0, description="Top coordinate (Y pixel) of the crop area, starting from 0.")
    width: int = Field(..., gt=0, description="Width of the crop area in pixels (must be positive).")
    height: int = Field(..., gt=0, description="Height of the crop area in pixels (must be positive).")
class BinarizeArgs(BaseModel): threshold: int = Field(default=128, ge=0, le=255, description="Threshold value (0-255). Pixels brighter than this become white, others black.")
class RemoveBackgroundArgs(BaseModel): pass
class UpscaleArgs(BaseModel): factor: int = Field(default=4, description="The desired upscaling factor (e.g., 2 for 2x, 4 for 4x). Note: Current AI service only supports 4x.")
class SearchReplaceArgs(BaseModel):
    search_prompt: str = Field(..., description="Short description of the object/area to find and replace.")
    prompt: str = Field(..., description="Description of what to replace the found object with.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional: Describe elements to avoid.")
class RecolorArgs(BaseModel):
    select_prompt: str = Field(..., description="Short description of the object/area to find and recolor.")
    prompt: str = Field(..., description="Description of the new color(s) or style.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional: Describe colors/elements to avoid.")


# --- Tool Implementation Wrapper (_execute_impl) - MODIFIED ---
# Now expects impl_func to return Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]
# It will call update_processed_image directly if impl_func modifies the image.
def _execute_impl(
    impl_func: Callable[..., Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]], # Keep original signature for impl
    tool_name: str,
    needs_image: bool,
    args: Dict[str, Any]
) -> Tuple[str, Optional[Dict[str, Any]]]: # MODIFIED RETURN TYPE (str, ui_updates)
    """
    Helper to execute implementation functions.
    Handles image fetching, error trapping, and DIRECTLY updates Streamlit image state.
    Returns: (result_message: str, ui_updates: Optional[Dict])
    """
    logger.info(f"Executing implementation: {impl_func.__name__} for tool '{tool_name}'")
    input_image_copy: Optional[Image.Image] = None
    ui_updates: Optional[Dict[str, Any]] = None
    impl_result_msg: str = f"Error: Default error in _execute_impl for '{tool_name}'." # Default error

    # 1. Fetch Image if needed
    if needs_image:
        original_image = _get_current_image()
        if original_image is None:
            return f"Error executing '{tool_name}': No image available.", None
        input_image_copy = original_image.copy()
        impl_args = {"input_image": input_image_copy, **args}
    else:
        impl_args = args

    # 2. Execute Implementation
    try:
        # --- Call the actual logic function ---
        result_tuple = impl_func(**impl_args)
        # --------------------------------------

        # 3. Validate and Process Return Tuple (str, Optional[Image], Optional[Dict])
        if not (isinstance(result_tuple, tuple) and len(result_tuple) == 3):
             logger.error(f"Tool '{tool_name}' impl returned invalid format: {type(result_tuple)}")
             impl_result_msg = f"Error: Tool '{tool_name}' internal error (bad return format)."
             # Proceed without image update or UI updates
        else:
            msg_str, new_image, ui_updates_dict = result_tuple
            impl_result_msg = msg_str # Store the message

            # 4. *** DIRECTLY Update Streamlit Image State if new image returned ***
            if new_image is not None:
                if isinstance(new_image, Image.Image):
                    if _IN_STREAMLIT_CONTEXT_TOOLS and _STATE_MANAGER_LOADED:
                        logger.info(f"Attempting to update Streamlit image state from '{tool_name}'...")
                        try:
                            # Call the imported state manager function
                            update_success = update_processed_image(new_image)
                            if update_success: logger.info("Streamlit image state updated successfully.")
                            else: logger.warning("update_processed_image returned False (state update might have failed).")
                        except NameError:
                             logger.error("update_processed_image function not loaded. Cannot update image.")
                             impl_result_msg += " (Warning: Image processed but UI state update failed - function missing)"
                        except Exception as e:
                             logger.error(f"Error calling update_processed_image: {e}", exc_info=True)
                             impl_result_msg += f" (Warning: Image processed but UI state update failed - {e})"
                    elif not _IN_STREAMLIT_CONTEXT_TOOLS:
                        logger.warning(f"Tool '{tool_name}' produced an image but not in Streamlit context to update state.")
                        impl_result_msg += " (Info: Image processed but cannot update UI state)"
                    elif not _STATE_MANAGER_LOADED:
                        logger.warning(f"Tool '{tool_name}' produced an image but state manager not loaded.")
                        impl_result_msg += " (Info: Image processed but cannot update UI state - manager missing)"

                else: # new_image was not None, but not an Image
                     logger.error(f"Tool '{tool_name}' impl returned invalid image type: {type(new_image)}")
                     impl_result_msg = f"Error: Tool '{tool_name}' internal error (bad image return)."
                     # Discard potential UI updates if image failed
                     ui_updates_dict = None

            # 5. Validate and Store UI Updates
            if ui_updates_dict is not None:
                if isinstance(ui_updates_dict, dict):
                    ui_updates = ui_updates_dict # Store validated UI updates
                else:
                    logger.error(f"Tool '{tool_name}' impl returned invalid ui_updates type: {type(ui_updates_dict)}")
                    # Keep message, discard invalid UI updates
                    ui_updates = None

        logger.info(f"Tool '{tool_name}' implementation finished.")
        # Return only the message and UI updates
        return impl_result_msg, ui_updates

    # --- Handle Exceptions ---
    except ValidationError as e: # Catch Pydantic validation errors if args were somehow invalid
        logger.error(f"Tool '{tool_name}' failed Pydantic validation within impl: {e}", exc_info=True)
        return f"Error: Invalid arguments passed internally to '{tool_name}': {e}", None
    except TypeError as e: # Catch errors if args don't match function signature
        logger.error(f"Tool '{tool_name}' implementation called with incorrect signature arguments: {e}", exc_info=True)
        return f"Error: Internal tool argument mismatch for '{tool_name}'.", None
    except Exception as e: # Catch-all for other errors during implementation logic
        logger.error(f"Error during '{impl_func.__name__}' execution: {e}", exc_info=True)
        return f"Error executing '{tool_name}': {str(e)}", None


# --- Concrete Implementation Functions (_impl) ---
# These return Tuple[str, Optional[Image.Image], Optional[Dict]]

def _adjust_brightness_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    new_image = processing.apply_brightness(input_image, factor)
    ui_updates = {"brightness_slider": factor} if new_image else None
    return ("Success: Brightness adjusted.", new_image, ui_updates) if new_image else ("Error: Failed applying brightness.", None, None)

def _adjust_contrast_impl(input_image: Image.Image, factor: float) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    new_image = processing.apply_contrast(input_image, factor)
    ui_updates = {"contrast_slider": factor} if new_image else None
    return ("Success: Contrast adjusted.", new_image, ui_updates) if new_image else ("Error: Failed applying contrast.", None, None)

def _apply_filter_impl(input_image: Image.Image, filter_name: str) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    filter_map = {'blur': ImageFilter.BLUR, 'sharpen': ImageFilter.SHARPEN, 'smooth': ImageFilter.SMOOTH,
                  'edge_enhance': ImageFilter.EDGE_ENHANCE, 'emboss': ImageFilter.EMBOSS, 'contour': ImageFilter.CONTOUR}
    selected_filter = filter_map.get(filter_name.lower())
    if selected_filter is None:
        return f"Error: Unknown filter name '{filter_name}'.", None, None
    processed = input_image.filter(selected_filter)
    return f"Success: Applied '{filter_name}' filter.", processed, None

def _rotate_image_impl(input_image: Image.Image, degrees: float) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    angle = int(degrees % 360)
    new_image = processing.apply_rotation(input_image, angle)
    ui_updates = {"rotation_slider": angle} if new_image else None
    return (f"Success: Image rotated by {angle} degrees.", new_image, ui_updates) if new_image else ("Error: Failed applying rotation.", None, None)

def _invert_colors_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    new_image = processing.apply_negative(input_image)
    return ("Success: Colors inverted.", new_image, None) if new_image else ("Error: Failed inverting colors.", None, None)

def _crop_image_impl(input_image: Image.Image, x: int, y: int, width: int, height: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    img_width, img_height = input_image.size
    if not ( (x + width) <= img_width and (y + height) <= img_height ):
         return f"Error: Crop rectangle (x={x}, y={y}, w={width}, h={height}) exceeds image bounds ({img_width}x{img_height}).", None, None
    box = (x, y, x + width, y + height)
    new_image = input_image.crop(box)
    # Reset zoom state in UI after crop
    ui_updates = {'zoom_x': 0, 'zoom_y': 0, 'zoom_w': 100, 'zoom_h': 100}
    return f"Success: Image cropped to box {box}.", new_image, ui_updates

def _apply_binarization_impl(input_image: Image.Image, threshold: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    new_image = processing.apply_binarization(input_image, threshold)
    ui_updates = {"binarize_thresh_slider": threshold, "apply_binarization_cb": True} if new_image else None
    return ("Success: Image binarized.", new_image, ui_updates) if new_image else ("Error: Failed binarization.", None, None)

# --- AI Tool Implementations ---
# These call the public functions in ai_services, which handle API key retrieval internally.

def _remove_background_ai_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    try:
        result_img = ai_services.remove_background_ai(input_image)
        return ("Success: Background removal processing attempted via AI.", result_img, None) if result_img else ("Error: Background removal via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.remove_background_ai: {e}", exc_info=True)
        return f"Error during background removal: {e}", None, None

def _upscale_image_ai_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    try:
        # Pass the factor, ai_services will handle logic/warnings if factor != 4
        result_img = ai_services.upscale_image_ai(input_image, scale_factor=factor)
        # Determine actual factor if possible, otherwise assume requested or default
        actual_factor = factor # Or potentially get from result_img metadata if available
        return (f"Success: Image upscale (x{actual_factor}) attempted via AI.", result_img, None) if result_img else (f"Error: Upscaling via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.upscale_image_ai: {e}", exc_info=True)
        return f"Error during upscaling: {e}", None, None

def _search_and_replace_ai_impl(input_image: Image.Image, search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    try:
        result_img = ai_services.search_and_replace_ai(input_image, search_prompt, prompt, negative_prompt)
        return ("Success: Search and Replace attempted via AI.", result_img, None) if result_img else ("Error: Search and Replace via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.search_and_replace_ai: {e}", exc_info=True)
        return f"Error during search and replace: {e}", None, None

def _recolor_object_ai_impl(input_image: Image.Image, select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Tuple[str, Optional[Image.Image], Optional[Dict]]:
    try:
        result_img = ai_services.recolor_object_ai(input_image, select_prompt, prompt, negative_prompt)
        return ("Success: Recolor Object attempted via AI.", result_img, None) if result_img else ("Error: Recolor Object via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.recolor_object_ai: {e}", exc_info=True)
        return f"Error during recolor object: {e}", None, None


# --- Tool Schemas Exposed to LLM (@tool decorated functions) - MODIFIED ---
# These now call _execute_impl and return ONLY the message string.
# UI updates are handled by the graph node via the return from _execute_impl.

@tool(args_schema=BrightnessArgs)
def adjust_brightness(factor: int) -> str:
    """Adjusts the brightness of the current image. Provide an integer factor between -100 (darker) and 100 (brighter). 0 means no change."""
    msg, _ = _execute_impl(_adjust_brightness_impl, "adjust_brightness", True, {"factor": factor})
    return msg

@tool(args_schema=ContrastArgs)
def adjust_contrast(factor: float) -> str:
    """Adjusts the contrast of the current image. Factor >= 0. 1.0 is original contrast. Less than 1 decreases, greater than 1 increases."""
    msg, _ = _execute_impl(_adjust_contrast_impl, "adjust_contrast", True, {"factor": factor})
    return msg

@tool(args_schema=FilterArgs)
def apply_filter(filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour']) -> str:
    """Applies a standard image filter selected from the allowed list: blur, sharpen, smooth, edge_enhance, emboss, contour."""
    msg, _ = _execute_impl(_apply_filter_impl, "apply_filter", True, {"filter_name": filter_name})
    return msg

@tool(args_schema=RotateArgs)
def rotate_image(degrees: float) -> str:
    """Rotates the current image clockwise by the specified angle in degrees."""
    msg, _ = _execute_impl(_rotate_image_impl, "rotate_image", True, {"degrees": degrees})
    return msg

@tool # No args needed for schema
def invert_colors() -> str:
    """Inverts the colors of the current image, creating a negative effect."""
    msg, _ = _execute_impl(_invert_colors_impl, "invert_colors", True, {})
    return msg

@tool(args_schema=CropArgs)
def crop_image(x: int, y: int, width: int, height: int) -> str:
    """Crops the current image to a specified rectangular area using top-left pixel coordinates (x, y) and dimensions (width, height). Coordinates must be within image bounds."""
    args = locals() # Capture args passed to this schema function
    msg, _ = _execute_impl(_crop_image_impl, "crop_image", True, args)
    return msg

@tool(args_schema=BinarizeArgs)
def apply_binarization(threshold: int) -> str:
    """Converts the current image to black and white (binarizes) based on a pixel brightness threshold (0-255)."""
    msg, _ = _execute_impl(_apply_binarization_impl, "apply_binarization", True, {"threshold": threshold})
    return msg

@tool(args_schema=RemoveBackgroundArgs)
def remove_background_ai() -> str:
    """Removes the background from the current image using an AI service (e.g., Stability AI or rembg). Requires API Key configuration for cloud services."""
    msg, _ = _execute_impl(_remove_background_ai_impl, "remove_background_ai", True, {})
    return msg

@tool(args_schema=UpscaleArgs)
def upscale_image_ai(factor: int = 4) -> str:
    """Upscales the current image using an AI service. Provide desired integer factor (e.g., 2, 4). Note: Current AI service only supports 4x."""
    # Pass the factor from the schema to the implementation
    msg, _ = _execute_impl(_upscale_image_ai_impl, "upscale_image_ai", True, {"factor": factor})
    return msg

@tool(args_schema=SearchReplaceArgs)
def search_and_replace_ai(search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> str:
    """Replaces an object described by 'search_prompt' with something new described by 'prompt', using Stability AI. Requires API Key configuration."""
    args = locals()
    msg, _ = _execute_impl(_search_and_replace_ai_impl, "search_and_replace_ai", True, args)
    return msg

@tool(args_schema=RecolorArgs)
def recolor_object_ai(select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> str:
    """Changes the color of an object described by 'select_prompt' according to 'prompt', using Stability AI. Requires API Key configuration."""
    args = locals()
    msg, _ = _execute_impl(_recolor_object_ai_impl, "recolor_object_ai", True, args)
    return msg

@tool # No args needed for schema
def get_image_info() -> str:
    """Gets information about the current image (dimensions, color mode). Does not modify the image."""
    # Logic is self-contained, doesn't need _impl or _execute_impl
    logger.info("Tool: get_image_info executing logic directly")
    img = _get_current_image()
    if img is None: return "Error: Cannot access current image information."
    if not isinstance(img, Image.Image): return f"Error: Invalid image object in state."
    try:
        width, height = img.size; mode = img.mode
        return f"Current image info: Size={width}x{height}, Mode={mode}."
    except Exception as e: logger.error(f"Error getting image info: {e}", exc_info=True); return f"Error: {str(e)}"


# --- Tool Dictionary (For LLM Binding - Schemas) ---
available_tools_list: List[Callable] = [
    adjust_brightness, adjust_contrast, apply_filter, rotate_image,
    invert_colors, get_image_info, crop_image, apply_binarization,
    remove_background_ai, upscale_image_ai, search_and_replace_ai, recolor_object_ai,
]
available_tools: Dict[str, Callable] = {t.name: t for t in available_tools_list}


# --- Implementation Dictionary (For Graph Execution - Logic) ---
# Maps tool names (str) to the *actual* implementation functions (_impl)
# These _impl functions return (str, Optional[Image], Optional[Dict])
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
    "search_and_replace_ai": _search_and_replace_ai_impl,
    "recolor_object_ai": _recolor_object_ai_impl,
    # get_image_info is handled directly by its @tool function via invoke,
    # so it doesn't need an entry in this implementation map.
}


# --- Direct Execution Test Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')
    logger.info(f"--- Running {Path(__file__).name} directly for Testing ---")
    logger.info(f"Core Modules Loaded: {_CORE_MODULES_LOADED}")
    logger.info(f"State Manager Loaded: {_STATE_MANAGER_LOADED}")
    logger.info(f"Streamlit Context Detected: {_IN_STREAMLIT_CONTEXT_TOOLS}")

    logger.info("\nAvailable tools (Schemas for LLM):")
    for t_name, t_func in available_tools.items():
        try: logger.info(f"- {t_name}: {t_func.description} | Args Schema: {getattr(t_func, 'args_schema', 'N/A')}")
        except Exception as e: logger.error(f"Could not get details for {t_name}: {e}")

    logger.info("\nAvailable implementation functions (Logic):")
    for t_name, t_impl in tool_implementations.items():
        logger.info(f"- {t_name} -> {t_impl.__name__}")

    logger.info("\n--- Testing Implementations (_impl) ---")
    try: test_img_impl = Image.new("RGB", (60, 40), "lightblue"); logger.info("Created mock image.")
    except Exception as e: logger.error("Could not create mock image."); test_img_impl = None

    if test_img_impl:
        logger.info("\nTesting _adjust_brightness_impl...")
        res_msg, res_img, res_ui = _adjust_brightness_impl(test_img_impl.copy(), -30) # Use copy
        logger.info(f"  Result: '{res_msg}', Img Type: {type(res_img)}, UI Updates: {res_ui}")

        logger.info("\nTesting _remove_background_ai_impl...")
        # This will now call the public ai_services function which handles keys/mocks
        res_msg_bg, res_img_bg, _ = _remove_background_ai_impl(test_img_impl.copy())
        logger.info(f"  Result: '{res_msg_bg}', Img Type: {type(res_img_bg)}")
        if res_img_bg: logger.info(f"  Result Img Mode: {res_img_bg.mode}")

        logger.info("\nTesting _upscale_image_ai_impl...")
        res_msg_up, res_img_up, _ = _upscale_image_ai_impl(test_img_impl.copy(), factor=4)
        logger.info(f"  Result: '{res_msg_up}', Img Type: {type(res_img_up)}")
        if res_img_up: logger.info(f"  Result Img Size: {res_img_up.size}")


    logger.info("\n--- Testing Schema Tools (via invoke) ---")
    # These test if the @tool decorated functions correctly call _execute_impl
    # Note: Outside Streamlit, _get_current_image returns None, and update_processed_image is mocked.

    logger.info("Testing adjust_brightness schema tool...")
    # Expected: Error message about no image, as _get_current_image returns None
    result_invoke = adjust_brightness.invoke({"factor": 30})
    logger.info(f"  Result: {result_invoke}")

    logger.info("\nTesting get_image_info schema tool...")
    # Expected: Error message about no image
    result_invoke_info = get_image_info.invoke({})
    logger.info(f"  Result: {result_invoke_info}")

    logger.info(f"--- Finished {Path(__file__).name} direct test ---")