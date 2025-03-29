# image_editor/state/session_state_manager.py
import streamlit as st
from typing import Dict, Any, List
from utils.constants import DEFAULT_CHANNELS

def get_default_session_values() -> Dict[str, Any]:
    return {
        'original_image': None,
        'processed_image': None,
        'image_filename': None,
        'second_image': None,
        'second_image_filename': None,
        'show_histogram': False,
        # Widget state keys
        'brightness_slider': 0,
        'contrast_slider': 1.0,
        'rotation_slider': 0,
        'zoom_x': 0,
        'zoom_y': 0,
        'zoom_w': 100,
        'zoom_h': 100,
        'binarize_thresh_slider': 128,
        'apply_binarization_cb': False,
        'channel_multiselect': DEFAULT_CHANNELS.copy(),
        'highlight_radio': 'None',
        'highlight_thresh_slider': 128,
        'merge_alpha_slider': 0.5,
        # Button click flags
        'apply_zoom_triggered': False,
        'apply_negative_triggered': False,
        'apply_merge_triggered': False,
        'last_processed_image_state': None,
        # AI operation flags
        'ai_remove_bg_triggered': False,
        'ai_upscale_triggered': False,
        # Chat history
        'chat_history': [],  # List to store messages {'role': 'user'/'assistant', 'content': ...}
        'current_graph_thread_id': "default_thread_1"  # ID for current LangGraph conversation
    }

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = get_default_session_values()
    for key, value in defaults.items():
        if key not in st.session_state:
            # Use deep copies for mutable defaults like lists/dicts if necessary
            if isinstance(value, list):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value

def reset_triggered_flags():
    """Resets all action trigger flags in session state."""
    st.session_state.apply_zoom_triggered = False
    st.session_state.apply_negative_triggered = False
    st.session_state.apply_merge_triggered = False
    st.session_state.ai_remove_bg_triggered = False
    st.session_state.ai_upscale_triggered = False

def reset_all_state(keep_chat_history: bool = False):
    """Resets processed image and control values to defaults, optionally keeping chat history."""
    defaults = get_default_session_values()
    # Reset images
    st.session_state.original_image = st.session_state.get('original_image', None)  # Keep original if exists
    st.session_state.processed_image = st.session_state.original_image.copy() if st.session_state.original_image else None
    st.session_state.last_processed_image_state = None
    st.session_state.second_image = None  # Clear second image on full reset
    st.session_state.second_image_filename = None

    # Reset other non-widget states
    st.session_state.show_histogram = False

    # Save current chat history if needed
    saved_chat_history = st.session_state.chat_history.copy() if keep_chat_history else []
    saved_thread_id = st.session_state.current_graph_thread_id if keep_chat_history else "default_thread_1"

    # Reset widget-related states to their defaults
    for key, value in defaults.items():
        # Avoid resetting image data, flags handled above, and possibly chat history
        if ('image' not in key and 
            'triggered' not in key and 
            'last_processed' not in key and
            (not keep_chat_history or 'chat' not in key)):
            # Use list.copy() for mutable defaults
            if isinstance(value, list):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value

    # Restore chat history if keeping it
    if keep_chat_history:
        st.session_state.chat_history = saved_chat_history
        st.session_state.current_graph_thread_id = saved_thread_id

    # Ensure flags are reset
    reset_triggered_flags()

    st.success("Editor State Reset!")

def update_processed_image(new_image):
    """Safely updates the processed image and the last state."""
    if new_image is not None:
        if st.session_state.get('processed_image') is not None:
            st.session_state.last_processed_image_state = st.session_state.processed_image.copy()
        else:
            st.session_state.last_processed_image_state = None
        st.session_state.processed_image = new_image

def revert_processed_image():
    """Reverts processed image to the last known good state."""
    if st.session_state.get('last_processed_image_state') is not None:
        st.session_state.processed_image = st.session_state.last_processed_image_state
    elif st.session_state.get('original_image') is not None:
        st.session_state.processed_image = st.session_state.original_image.copy()