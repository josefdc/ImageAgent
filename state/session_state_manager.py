# streamlit_image_editor/state/session_state_manager.py
import streamlit as st
from typing import Dict, Any, List, Optional # Added Optional
from PIL import Image # Import Image for type checking
import logging # Import logging
import os # Import os for environment variable access

# --- Constants Import ---
# Ensure constants are accessible if needed, assuming utils/constants.py exists
try:
    from utils.constants import DEFAULT_CHANNELS
except ImportError:
    DEFAULT_CHANNELS = ['Red', 'Green', 'Blue'] # Fallback

# --- Logger Setup ---
# Set up logger for this module specifically
logger = logging.getLogger(__name__)
# Ensure basicConfig is called, potentially redundant but safe
if not logging.getLogger().hasHandlers():
     _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
     logging.basicConfig(level=_log_level, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')


# --- Existing Functions (get_default_session_values, initialize_session_state, etc.) ---
# (Keep the rest of the file as it was, including imports at the top)

def get_default_session_values() -> Dict[str, Any]:
    # ... (implementation as before) ...
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

def initialize_session_state():
    # ... (implementation as before) ...
    defaults = get_default_session_values()
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, list): st.session_state[key] = value.copy()
            elif isinstance(value, dict): st.session_state[key] = value.copy()
            else: st.session_state[key] = value

def reset_triggered_flags():
    # ... (implementation as before) ...
    st.session_state.apply_zoom_triggered = False
    # ... etc ...
    st.session_state.ai_upscale_triggered = False

def reset_all_state(keep_chat_history: bool = False):
    # ... (implementation as before) ...
    defaults = get_default_session_values()
    # ... reset images ...
    st.session_state.original_image = st.session_state.get('original_image', None)
    st.session_state.processed_image = st.session_state.original_image.copy() if st.session_state.original_image else None
    # ... etc ...
    if not keep_chat_history: st.session_state.chat_history = []
    # ... reset widgets ...
    reset_triggered_flags()
    st.success("Editor State Reset!")


# --- CORRECTED update_processed_image ---
def update_processed_image(new_image: Optional[Image.Image]) -> bool:
    """
    Safely updates the 'processed_image' and 'last_processed_image_state'
    in Streamlit's session state.

    Args:
        new_image: The new PIL Image object to set as processed_image.

    Returns:
        True if the state was successfully updated, False otherwise.
    """
    if new_image is not None and isinstance(new_image, Image.Image):
        try:
            # Store the *current* processed image as the last state *before* overwriting
            if 'processed_image' in st.session_state and isinstance(st.session_state.processed_image, Image.Image):
                st.session_state.last_processed_image_state = st.session_state.processed_image.copy()
            else:
                # If there was no previous valid processed image, clear the last state
                st.session_state.last_processed_image_state = None

            # Update the processed image state
            st.session_state.processed_image = new_image
            logger.info("Session state 'processed_image' updated successfully.")
            return True # Indicate success
        except Exception as e:
             # Catch potential errors during copy or assignment (though unlikely)
             logger.error(f"Failed to update session state for processed image: {e}", exc_info=True)
             # Show error in UI if possible
             try: st.error("Internal error: Failed to save the processed image state.")
             except: pass # Ignore errors if streamlit isn't fully available
             return False # Indicate failure
    else:
        # Log error if input is invalid
        logger.error(f"update_processed_image called with invalid input (Type: {type(new_image)}). State not updated.")
        # Optionally show an error in the UI
        try: st.error("Internal error: Attempted to update image with invalid data.")
        except: pass
        return False # Indicate failure


def revert_processed_image():
    # ... (implementation as before) ...
    if st.session_state.get('last_processed_image_state') is not None:
        st.session_state.processed_image = st.session_state.last_processed_image_state
    elif st.session_state.get('original_image') is not None:
        st.session_state.processed_image = st.session_state.original_image.copy()
    logger.info("Reverted processed image to previous state.") # Add log