"""
Agent Tool Definitions and Implementations

This module defines tool schemas for the LLM agent and maps them to implementation functions.
It provides:
- Pydantic models for tool argument validation
- LangChain @tool decorated functions that the LLM can call
- Implementation functions that handle actual image processing
- Helper functions for image state management and execution

The tools cover basic image operations (brightness, contrast, filters) and AI-powered
operations (background removal, upscaling, search/replace, recoloring).
"""

# --- Standard Library Imports ---
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Literal, Callable

# --- Path Setup (Add Project Root) ---
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
except Exception as e:
    print(f"ERROR: Failed during sys.path setup: {e}")

# --- Third-Party Imports ---
from langchain_core.tools import tool
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field, ValidationError

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
    )

# --- Streamlit Import (Conditional for State/Secrets Access) ---
_IN_STREAMLIT_CONTEXT_TOOLS = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        _IN_STREAMLIT_CONTEXT_TOOLS = True
        _st_module = st
        logger.debug("Streamlit context detected")
except (ImportError, RuntimeError, ModuleNotFoundError):
    logger.warning("Not running within Streamlit context")

# --- Core Modules & State Manager Import ---
_CORE_MODULES_LOADED = False
_STATE_MANAGER_LOADED = False # Flag for state manager import
try:
    from core import processing, ai_services
    from state.session_state_manager import update_processed_image
    _CORE_MODULES_LOADED = _STATE_MANAGER_LOADED = True
    logger.info("Successfully imported core processing and state management modules")
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}. Using mocks.")
    
    # Mock implementations
    class MockProcessing:
        def __getattr__(self, name: str) -> Callable:
            def mock_func(*args, **kwargs):
                logger.info(f"[MOCK] processing.{name} called")
                img = kwargs.get('input_image') or (args[0] if args and isinstance(args[0], Image.Image) else None)
                return img.copy() if img else Image.new('RGB', (100, 100), color='gray')
            return mock_func
    
    class MockAIServices:
        def remove_background_ai(self, img: Image.Image) -> Image.Image:
            logger.info("[MOCK] ai_services.remove_background_ai called")
            return Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        
        def upscale_image_ai(self, img: Image.Image, scale_factor: int = 4) -> Image.Image:
            logger.info(f"[MOCK] ai_services.upscale_image_ai called (factor {scale_factor})")
            return Image.new('RGB', (img.width * scale_factor, img.height * scale_factor), color='cyan')
        
        def search_and_replace_ai(self, img: Image.Image, search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Image.Image:
            logger.info("[MOCK] ai_services.search_and_replace_ai called")
            return img.copy()
        
        def recolor_object_ai(self, img: Image.Image, select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Image.Image:
            logger.info("[MOCK] ai_services.recolor_object_ai called")
            return img.copy()
    
    processing = MockProcessing()
    ai_services = MockAIServices()
    
    def update_processed_image(img: Image.Image) -> bool:
        logger.info("[MOCK] update_processed_image called")
        return True


def _get_current_image() -> Optional[Image.Image]:
    """
    Get the current processed image from Streamlit session state.
    
    Returns:
        Current image or None if not available or not in Streamlit context
    """
    if not _IN_STREAMLIT_CONTEXT_TOOLS or not _st_module:
        logger.warning("Cannot get image: Not in Streamlit context")
        return None
    
    try:
        img = _st_module.session_state.get('processed_image')
        if isinstance(img, Image.Image):
            logger.debug(f"Retrieved image: Mode={img.mode}, Size={img.size}")
            return img
        else:
            logger.warning(f"Invalid image in session state: {type(img)}")
            return None
    except Exception as e:
        logger.error(f"Error accessing session state for image: {e}")
        return None


# Pydantic models for tool arguments
class BrightnessArgs(BaseModel):
    """Arguments for brightness adjustment tool."""
    factor: int = Field(..., description="Integer brightness adjustment factor, range -100 (darker) to 100 (brighter). 0 means no change.")


class ContrastArgs(BaseModel):
    """Arguments for contrast adjustment tool."""
    factor: float = Field(..., ge=0.0, description="Contrast adjustment factor. Float >= 0. 1.0 is original contrast. <1 decreases, >1 increases.")


class FilterArgs(BaseModel):
    """Arguments for filter application tool."""
    filter_name: Literal['blur', 'sharpen', 'smooth', 'edge_enhance', 'emboss', 'contour'] = Field(
        ..., description="The name of the filter to apply."
    )


class RotateArgs(BaseModel):
    """Arguments for image rotation tool."""
    degrees: float = Field(..., description="Angle in degrees to rotate the image clockwise.")


class CropArgs(BaseModel):
    """Arguments for image cropping tool."""
    x: int = Field(..., ge=0, description="Left coordinate (X pixel) of the crop area, starting from 0.")
    y: int = Field(..., ge=0, description="Top coordinate (Y pixel) of the crop area, starting from 0.")
    width: int = Field(..., gt=0, description="Width of the crop area in pixels (must be positive).")
    height: int = Field(..., gt=0, description="Height of the crop area in pixels (must be positive).")


class BinarizeArgs(BaseModel):
    """Arguments for binarization tool."""
    threshold: int = Field(default=128, ge=0, le=255, description="Threshold value (0-255). Pixels brighter than this become white, others black.")


class RemoveBackgroundArgs(BaseModel):
    """Arguments for background removal tool (no parameters needed)."""
    pass


class UpscaleArgs(BaseModel):
    """Arguments for AI upscaling tool."""
    factor: int = Field(default=4, description="The desired upscaling factor (e.g., 2 for 2x, 4 for 4x). Note: Current AI service only supports 4x.")


class SearchReplaceArgs(BaseModel):
    """Arguments for AI search and replace tool."""
    search_prompt: str = Field(..., description="Short description of the object/area to find and replace.")
    prompt: str = Field(..., description="Description of what to replace the found object with.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional: Describe elements to avoid.")


class RecolorArgs(BaseModel):
    """Arguments for AI recoloring tool."""
    select_prompt: str = Field(..., description="Short description of the object/area to find and recolor.")
    prompt: str = Field(..., description="Description of the new color(s) or style.")
    negative_prompt: Optional[str] = Field(default=None, description="Optional: Describe colors/elements to avoid.")


def _execute_impl(
    impl_func: Callable[..., Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]],
    tool_name: str,
    needs_image: bool,
    args: Dict[str, Any]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Execute implementation functions with error handling and state management.
    
    Args:
        impl_func: The implementation function to call
        tool_name: Name of the tool being executed
        needs_image: Whether the tool requires an input image
        args: Arguments to pass to the implementation function
        
    Returns:
        Tuple of (result_message, ui_updates)
    """
    logger.info(f"Executing implementation: {impl_func.__name__} for tool '{tool_name}'")
    
    # Fetch image if needed
    if needs_image:
        original_image = _get_current_image()
        if original_image is None:
            return f"Error executing '{tool_name}': No image available.", None
        impl_args = {"input_image": original_image.copy(), **args}
    else:
        impl_args = args

    # Execute implementation
    try:
        result_tuple = impl_func(**impl_args)
        
        # Validate return format
        if not (isinstance(result_tuple, tuple) and len(result_tuple) == 3):
            logger.error(f"Tool '{tool_name}' returned invalid format: {type(result_tuple)}")
            return f"Error: Tool '{tool_name}' internal error (bad return format).", None
        
        msg_str, new_image, ui_updates_dict = result_tuple
        
        # Update Streamlit image state if new image returned
        if new_image is not None:
            if isinstance(new_image, Image.Image):
                if _IN_STREAMLIT_CONTEXT_TOOLS and _STATE_MANAGER_LOADED:
                    try:
                        update_success = update_processed_image(new_image)
                        if update_success:
                            logger.info("Streamlit image state updated successfully")
                        else:
                            logger.warning("Image state update returned False")
                    except Exception as e:
                        logger.error(f"Error updating image state: {e}")
                        msg_str += f" (Warning: Image processed but UI state update failed - {e})"
                else:
                    logger.warning(f"Tool '{tool_name}' produced image but cannot update UI state")
                    msg_str += " (Info: Image processed but cannot update UI state)"
            else:
                logger.error(f"Tool '{tool_name}' returned invalid image type: {type(new_image)}")
                return f"Error: Tool '{tool_name}' internal error (bad image return).", None
        
        # Validate UI updates
        ui_updates = None
        if ui_updates_dict is not None:
            if isinstance(ui_updates_dict, dict):
                ui_updates = ui_updates_dict
            else:
                logger.error(f"Tool '{tool_name}' returned invalid ui_updates type: {type(ui_updates_dict)}")
        
        return msg_str, ui_updates
        
    except ValidationError as e:
        logger.error(f"Tool '{tool_name}' validation error: {e}")
        return f"Error: Invalid arguments for '{tool_name}': {e}", None
    except TypeError as e:
        logger.error(f"Tool '{tool_name}' signature error: {e}")
        return f"Error: Internal tool argument mismatch for '{tool_name}'.", None
    except Exception as e:
        logger.error(f"Error during '{impl_func.__name__}' execution: {e}")
        return f"Error executing '{tool_name}': {str(e)}", None


# Implementation functions
def _adjust_brightness_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for brightness adjustment."""
    new_image = processing.apply_brightness(input_image, factor)
    ui_updates = {"brightness_slider": factor} if new_image else None
    return ("Success: Brightness adjusted.", new_image, ui_updates) if new_image else ("Error: Failed applying brightness.", None, None)


def _adjust_contrast_impl(input_image: Image.Image, factor: float) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for contrast adjustment."""
    new_image = processing.apply_contrast(input_image, factor)
    ui_updates = {"contrast_slider": factor} if new_image else None
    return ("Success: Contrast adjusted.", new_image, ui_updates) if new_image else ("Error: Failed applying contrast.", None, None)


def _apply_filter_impl(input_image: Image.Image, filter_name: str) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for filter application."""
    filter_map = {
        'blur': ImageFilter.BLUR,
        'sharpen': ImageFilter.SHARPEN,
        'smooth': ImageFilter.SMOOTH,
        'edge_enhance': ImageFilter.EDGE_ENHANCE,
        'emboss': ImageFilter.EMBOSS,
        'contour': ImageFilter.CONTOUR
    }
    
    selected_filter = filter_map.get(filter_name.lower())
    if selected_filter is None:
        return f"Error: Unknown filter name '{filter_name}'.", None, None
    
    processed = input_image.filter(selected_filter)
    return f"Success: Applied '{filter_name}' filter.", processed, None


def _rotate_image_impl(input_image: Image.Image, degrees: float) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for image rotation."""
    angle = int(degrees % 360)
    new_image = processing.apply_rotation(input_image, angle)
    ui_updates = {"rotation_slider": angle} if new_image else None
    return (f"Success: Image rotated by {angle} degrees.", new_image, ui_updates) if new_image else ("Error: Failed applying rotation.", None, None)


def _invert_colors_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for color inversion."""
    new_image = processing.apply_negative(input_image)
    return ("Success: Colors inverted.", new_image, None) if new_image else ("Error: Failed inverting colors.", None, None)


def _crop_image_impl(input_image: Image.Image, x: int, y: int, width: int, height: int) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for image cropping."""
    img_width, img_height = input_image.size
    if not ((x + width) <= img_width and (y + height) <= img_height):
        return f"Error: Crop rectangle (x={x}, y={y}, w={width}, h={height}) exceeds image bounds ({img_width}x{img_height}).", None, None
    
    box = (x, y, x + width, y + height)
    new_image = input_image.crop(box)
    ui_updates = {'zoom_x': 0, 'zoom_y': 0, 'zoom_w': 100, 'zoom_h': 100}
    return f"Success: Image cropped to box {box}.", new_image, ui_updates


def _apply_binarization_impl(input_image: Image.Image, threshold: int) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for image binarization."""
    new_image = processing.apply_binarization(input_image, threshold)
    ui_updates = {"binarize_thresh_slider": threshold, "apply_binarization_cb": True} if new_image else None
    return ("Success: Image binarized.", new_image, ui_updates) if new_image else ("Error: Failed binarization.", None, None)


def _remove_background_ai_impl(input_image: Image.Image) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for AI background removal."""
    try:
        result_img = ai_services.remove_background_ai(input_image)
        return ("Success: Background removal processing attempted via AI.", result_img, None) if result_img else ("Error: Background removal via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.remove_background_ai: {e}")
        return f"Error during background removal: {e}", None, None


def _upscale_image_ai_impl(input_image: Image.Image, factor: int) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for AI image upscaling."""
    try:
        result_img = ai_services.upscale_image_ai(input_image, scale_factor=factor)
        return (f"Success: Image upscale (x{factor}) attempted via AI.", result_img, None) if result_img else ("Error: Upscaling via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.upscale_image_ai: {e}")
        return f"Error during upscaling: {e}", None, None


def _search_and_replace_ai_impl(input_image: Image.Image, search_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for AI search and replace."""
    try:
        result_img = ai_services.search_and_replace_ai(input_image, search_prompt, prompt, negative_prompt)
        return ("Success: Search and Replace attempted via AI.", result_img, None) if result_img else ("Error: Search and Replace via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.search_and_replace_ai: {e}")
        return f"Error during search and replace: {e}", None, None


def _recolor_object_ai_impl(input_image: Image.Image, select_prompt: str, prompt: str, negative_prompt: Optional[str] = None) -> Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]:
    """Implementation for AI object recoloring."""
    try:
        result_img = ai_services.recolor_object_ai(input_image, select_prompt, prompt, negative_prompt)
        return ("Success: Recolor Object attempted via AI.", result_img, None) if result_img else ("Error: Recolor Object via AI failed.", None, None)
    except Exception as e:
        logger.error(f"Error calling ai_services.recolor_object_ai: {e}")
        return f"Error during recolor object: {e}", None, None


# Tool schemas exposed to LLM
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


@tool
def invert_colors() -> str:
    """Inverts the colors of the current image, creating a negative effect."""
    msg, _ = _execute_impl(_invert_colors_impl, "invert_colors", True, {})
    return msg


@tool(args_schema=CropArgs)
def crop_image(x: int, y: int, width: int, height: int) -> str:
    """Crops the current image to a specified rectangular area using top-left pixel coordinates (x, y) and dimensions (width, height). Coordinates must be within image bounds."""
    args = locals()
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


@tool
def get_image_info() -> str:
    """Gets information about the current image (dimensions, color mode). Does not modify the image."""
    logger.info("Executing get_image_info")
    img = _get_current_image()
    if img is None:
        return "Error: Cannot access current image information."
    if not isinstance(img, Image.Image):
        return "Error: Invalid image object in state."
    
    try:
        width, height = img.size
        mode = img.mode
        return f"Current image info: Size={width}x{height}, Mode={mode}."
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return f"Error: {str(e)}"


# Tool dictionaries
available_tools_list: List[Callable] = [
    adjust_brightness, adjust_contrast, apply_filter, rotate_image,
    invert_colors, get_image_info, crop_image, apply_binarization,
    remove_background_ai, upscale_image_ai, search_and_replace_ai, recolor_object_ai,
]

available_tools: Dict[str, Callable] = {t.name: t for t in available_tools_list}

tool_implementations: Dict[str, Callable[..., Tuple[str, Optional[Image.Image], Optional[Dict[str, Any]]]]] = {
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
}


if __name__ == "__main__":
    """Direct execution for testing purposes."""
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
    )
    
    logger.info(f"Running {Path(__file__).name} directly for testing")
    logger.info(f"Core Modules Loaded: {_CORE_MODULES_LOADED}")
    logger.info(f"State Manager Loaded: {_STATE_MANAGER_LOADED}")
    logger.info(f"Streamlit Context: {_IN_STREAMLIT_CONTEXT_TOOLS}")

    logger.info("Available tools (Schemas for LLM):")
    for t_name, t_func in available_tools.items():
        try:
            logger.info(f"- {t_name}: {t_func.description}")
        except Exception as e:
            logger.error(f"Could not get details for {t_name}: {e}")

    logger.info("Available implementation functions:")
    for t_name, t_impl in tool_implementations.items():
        logger.info(f"- {t_name} -> {t_impl.__name__}")

    # Test implementation
    try:
        test_img = Image.new("RGB", (60, 40), "lightblue")
        logger.info("Created test image")
        
        logger.info("Testing brightness adjustment...")
        res_msg, res_img, res_ui = _adjust_brightness_impl(test_img.copy(), -30)
        logger.info(f"Result: '{res_msg}', Image: {type(res_img)}, UI: {res_ui}")
        
        logger.info("Testing get_image_info tool...")
        result = get_image_info.invoke({})
        logger.info(f"Info result: {result}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

    logger.info("Finished testing")