"""
Session State Manager for Streamlit Image Editor

This module provides centralized management of Streamlit session state for the image editor application.
It handles initialization, updates, and resets of all application state including:

- Image data (original, processed, secondary images)
- UI widget states (sliders, checkboxes, selections)
- Processing flags and triggers
- Chat history for AI assistant
- Image processing parameters

The module ensures consistent state management across all pages and components of the application,
providing safe operations for updating image data and maintaining state integrity.
"""

import streamlit as st
import logging
import os
from typing import Dict, Any, List, Optional
from PIL import Image

try:
    from utils.constants import DEFAULT_CHANNELS
except ImportError:
    DEFAULT_CHANNELS = ['Red', 'Green', 'Blue']

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=_log_level, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')


def get_default_session_values() -> Dict[str, Any]:
    """
    Get default values for all session state variables.
    
    Returns:
        Dictionary containing default values for session state initialization
    """
    return {
        'original_image': None,
        'processed_image': None,
        'image_filename': None,
        'second_image': None,
        'second_image_filename': None,
        'show_histogram': False,
        'brightness_slider': 0,
        'contrast_slider': 1.0,
        'rotation_slider': 0,
        'zoom_x': 0, 'zoom_y': 0, 'zoom_w': 100, 'zoom_h': 100,
        'binarize_thresh_slider': 128,
        'apply_binarization_cb': False,
        'channel_multiselect': DEFAULT_CHANNELS.copy(),
        'highlight_radio': 'None',
        'highlight_thresh_slider': 128,
        'merge_alpha_slider': 0.5,
        'apply_zoom_triggered': False,
        'apply_negative_triggered': False,
        'apply_merge_triggered': False,
        'last_processed_image_state': None,
        'ai_remove_bg_triggered': False,
        'ai_upscale_triggered': False,
        'chat_history': [],
        'current_graph_thread_id': "default_thread_1"
    }


def initialize_session_state() -> None:
    """
    Initialize session state with default values for any missing keys.
    
    Safely initializes all required session state variables without overwriting
    existing values. Lists and dictionaries are properly copied to avoid reference issues.
    """
    defaults = get_default_session_values()
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, list):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value


def reset_triggered_flags() -> None:
    """
    Reset all trigger flags to False.
    
    Used to clear processing flags after operations complete
    to prevent repeated execution.
    """
    st.session_state.apply_zoom_triggered = False
    st.session_state.apply_negative_triggered = False
    st.session_state.apply_merge_triggered = False
    st.session_state.ai_remove_bg_triggered = False
    st.session_state.ai_upscale_triggered = False


def reset_all_state(keep_chat_history: bool = False) -> None:
    """
    Reset all session state to default values.
    
    Args:
        keep_chat_history: If True, preserves chat history during reset
    """
    defaults = get_default_session_values()
    
    st.session_state.original_image = st.session_state.get('original_image', None)
    st.session_state.processed_image = st.session_state.original_image.copy() if st.session_state.original_image else None
    
    for key, value in defaults.items():
        if key not in ['original_image', 'processed_image']:
            if key == 'chat_history' and keep_chat_history:
                continue
            if isinstance(value, list):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value
    
    reset_triggered_flags()
    st.success("Editor State Reset!")


def update_processed_image(new_image: Optional[Image.Image]) -> bool:
    """
    Safely update the processed image in session state.
    
    Stores the current processed image as backup before updating to the new image.
    Provides error handling and logging for safe state management.
    
    Args:
        new_image: New PIL Image object to set as processed image
        
    Returns:
        True if update was successful, False otherwise
    """
    if new_image is not None and isinstance(new_image, Image.Image):
        try:
            if 'processed_image' in st.session_state and isinstance(st.session_state.processed_image, Image.Image):
                st.session_state.last_processed_image_state = st.session_state.processed_image.copy()
            else:
                st.session_state.last_processed_image_state = None

            st.session_state.processed_image = new_image
            logger.info("Session state 'processed_image' updated successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to update session state for processed image: {e}", exc_info=True)
            try:
                st.error("Internal error: Failed to save the processed image state.")
            except:
                pass
            return False
    else:
        logger.error(f"update_processed_image called with invalid input (Type: {type(new_image)}). State not updated.")
        try:
            st.error("Internal error: Attempted to update image with invalid data.")
        except:
            pass
        return False


def revert_processed_image() -> None:
    """
    Revert processed image to previous state.
    
    Restores the processed image from the last saved state, or falls back
    to the original image if no previous state exists.
    """
    if st.session_state.get('last_processed_image_state') is not None:
        st.session_state.processed_image = st.session_state.last_processed_image_state
    elif st.session_state.get('original_image') is not None:
        st.session_state.processed_image = st.session_state.original_image.copy()
    logger.info("Reverted processed image to previous state.")