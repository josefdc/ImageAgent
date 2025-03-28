# streamlit_image_editor/app.py
import streamlit as st
from PIL import Image
from typing import Optional

# Import modules using relative paths (adjust if structure differs)
from state.session_state_manager import (
    initialize_session_state,
    reset_triggered_flags,
    update_processed_image,
    revert_processed_image
)
from core import processing, image_io, histogram # Import processing functions
from ui.interface import build_sidebar, display_main_area # Import UI functions

# --- Page Config ---
# Moved to top level, must be the first Streamlit command
st.set_page_config(
    page_title="Enhanced Image Editor++ using Streamlit & Pillow.",
    page_icon="🖼️",
    layout="wide"
)

# --- Central Image Processing Logic ---
def process_image_updates() -> bool:
    """
    Applies all sequential processing based on current session state.
    Returns True if the image was updated, False otherwise.
    Manages trigger flags internally.
    """
    # Check if processing is possible/needed
    if not st.session_state.get('original_image') or not st.session_state.get('processed_image'):
        reset_triggered_flags() # Ensure flags are reset even if no image
        return False

    img = st.session_state.processed_image.copy() # Work on a copy
    initial_img_bytes = img.tobytes() # For change detection

    processing_error_occurred = False
    try:
        # Apply continuous adjustments (order might matter)
        img = processing.apply_brightness(img, st.session_state.brightness_slider)
        img = processing.apply_contrast(img, st.session_state.contrast_slider)
        img = processing.apply_rotation(img, st.session_state.rotation_slider)
        img = processing.apply_channel_manipulation(img, st.session_state.channel_multiselect)
        img = processing.apply_highlight(img, st.session_state.highlight_radio, st.session_state.highlight_thresh_slider)
        if st.session_state.apply_binarization_cb:
            img = processing.apply_binarization(img, st.session_state.binarize_thresh_slider)

        # Apply triggered adjustments (buttons)
        if st.session_state.apply_zoom_triggered:
            img = processing.apply_zoom(img, st.session_state.zoom_x, st.session_state.zoom_y, st.session_state.zoom_w, st.session_state.zoom_h)
            if img.tobytes() != initial_img_bytes: # Basic check if zoom actually changed image
                st.success("Zoom applied!")
            st.session_state.apply_zoom_triggered = False # Reset flag

        if st.session_state.apply_negative_triggered:
            img = processing.apply_negative(img)
            st.success("Colors inverted!")
            st.session_state.apply_negative_triggered = False # Reset flag

        if st.session_state.apply_merge_triggered:
            if st.session_state.get('second_image'):
                with st.spinner("Merging images..."):
                    merged_img = processing.merge_images(img, st.session_state.second_image, st.session_state.merge_alpha_slider)
                if merged_img is not None and merged_img.tobytes() != img.tobytes(): # Check merge success/change
                     img = merged_img
                     st.success("Images merged!")
                elif merged_img is None: # merge_images signals error by returning original or None
                     st.error("Merging failed.") # Error message likely already shown in merge_images
            else:
                st.warning("Cannot merge: Second image not loaded.")
            st.session_state.apply_merge_triggered = False # Reset flag

        # --- Final Update ---
        # Update state only if the image content actually changed
        final_img_bytes = img.tobytes()
        if final_img_bytes != initial_img_bytes:
            update_processed_image(img) # Use state manager function
            return True # Image was updated
        else:
            # If no change, ensure any potentially set flags are reset if they didn't cause a change
            # (This might happen if zoom was invalid, merge failed softly, etc.)
            reset_triggered_flags()
            return False # No effective change

    except Exception as e:
        st.error(f"An error occurred during image processing:")
        st.exception(e) # Show detailed traceback
        revert_processed_image() # Attempt to revert to last known good state
        reset_triggered_flags() # Ensure flags are reset on error
        processing_error_occurred = True
        return False # Indicate no successful update

# --- Main Application Flow ---
def main():
    st.title("🖼️ Modular Image Editor")

    # 1. Initialize State (must run before accessing state)
    initialize_session_state()

    # 2. Build Sidebar (handles loading and sets widget states/flags)
    build_sidebar()

    # 3. Process Image Updates based on state from sidebar/previous run
    # Wrap in spinner for user feedback during processing
    with st.spinner("Applying adjustments..."):
         process_image_updates()

    # 4. Display Main Area (shows images, download, histogram based on current state)
    display_main_area()

    # 5. Footer
    st.markdown("---")
    st.caption("Enhanced Image Editor++ using Streamlit & Pillow.")

if __name__ == "__main__":
    main()