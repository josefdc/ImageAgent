# streamlit_image_editor/app.py
# Main application entry point and orchestrator.

# --- Standard Library Imports ---
import time
from typing import Optional # Used for type hinting, even if PIL isn't directly used here

# --- Third-Party Imports ---
import streamlit as st
# from PIL import Image # PIL Image objects are handled in other modules now

# --- Local Application Imports ---
# Use relative imports based on the defined project structure
from state.session_state_manager import (
    initialize_session_state,
    reset_triggered_flags,
    update_processed_image,
    revert_processed_image
)
# Import the processing module itself to call its functions
from core import processing
from ui.interface import build_sidebar, display_main_area

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title=" Image Editor ",  
    page_icon="🎨", 
    layout="wide",
    initial_sidebar_state="expanded", # Start with sidebar open
    menu_items={
        'Get Help': "https://github.com/josefdc/image-editor", 
        'Report a bug': "https://github.com/josefdc/image-editor/issues",
        'About': """
        ## Streamlit Image Editor Pro

        An interactive web application for image viewing and editing,
        built with Streamlit and Pillow.

        Modular design for easier maintenance and extension.
        """
    }
)

# --- Central Image Processing Logic ---
def run_image_processing_pipeline() -> None:
    """
    Orchestrates the image processing based on the current session state.

    - Applies continuous adjustments (sliders, etc.) starting from the ORIGINAL image.
    - Applies triggered adjustments (button actions) sequentially after continuous ones.
    - Updates the 'processed_image' in session state only if changes occurred.
    - Handles errors and state reversion gracefully.
    - Resets action trigger flags after processing.
    """

    # 1. Pre-checks: Ensure processing is possible
    if not st.session_state.get('original_image'):
        reset_triggered_flags()
        return # Nothing to process

    # Ensure processed_image exists, initializing from original if needed
    if not st.session_state.get('processed_image'):
        st.session_state.processed_image = st.session_state.original_image.copy()

    # 2. Get initial state for comparison (detect if changes actually happen)
    current_processed_bytes = st.session_state.processed_image.tobytes()
    img = None # Initialize img variable

    # 3. Processing Pipeline within a try block for error handling
    processing_error_occurred = False
    try:
        # --- Start processing from the ORIGINAL image for continuous adjustments ---
        img = st.session_state.original_image.copy()

        # === Apply Continuous Adjustments (Always run based on current widget state) ===
        with st.spinner("Applying continuous adjustments..."): # More specific spinner
            img = processing.apply_brightness(img, st.session_state.brightness_slider)
            if img is None: raise ValueError("Brightness processing failed.") # Defensive check

            img = processing.apply_contrast(img, st.session_state.contrast_slider)
            if img is None: raise ValueError("Contrast processing failed.")

            img = processing.apply_rotation(img, st.session_state.rotation_slider)
            if img is None: raise ValueError("Rotation processing failed.")

            img = processing.apply_channel_manipulation(img, st.session_state.channel_multiselect)
            if img is None: raise ValueError("Channel manipulation failed.")

            img = processing.apply_highlight(img, st.session_state.highlight_radio, st.session_state.highlight_thresh_slider)
            if img is None: raise ValueError("Highlight processing failed.")

            if st.session_state.apply_binarization_cb:
                img = processing.apply_binarization(img, st.session_state.binarize_thresh_slider)
                if img is None: raise ValueError("Binarization processing failed.")

        # === Apply Triggered Adjustments (Only if flags are set) ===
        # These operate sequentially on the result of the continuous adjustments
        trigger_spinner_needed = any([
            st.session_state.apply_zoom_triggered,
            st.session_state.apply_negative_triggered,
            st.session_state.apply_merge_triggered
        ])

        if trigger_spinner_needed:
            with st.spinner("Applying triggered actions..."):
                if st.session_state.apply_zoom_triggered:
                    img_before = img.tobytes()
                    img = processing.apply_zoom(img, st.session_state.zoom_x, st.session_state.zoom_y, st.session_state.zoom_w, st.session_state.zoom_h)
                    if img is None: raise ValueError("Zoom processing failed.")
                    if img.tobytes() != img_before: st.toast("Zoom applied!", icon="🔎")
                    st.session_state.apply_zoom_triggered = False # Reset flag

                if st.session_state.apply_negative_triggered:
                    img_before = img.tobytes()
                    img = processing.apply_negative(img)
                    if img is None: raise ValueError("Negative processing failed.")
                    if img.tobytes() != img_before: st.toast("Colors inverted!", icon="🌗")
                    st.session_state.apply_negative_triggered = False # Reset flag

                if st.session_state.apply_merge_triggered:
                    if st.session_state.get('second_image'):
                        img_before = img.tobytes()
                        # Merging can be slow, spinner is already active
                        merged_img = processing.merge_images(img, st.session_state.second_image, st.session_state.merge_alpha_slider)

                        if merged_img is not None: # merge_images returns original on error, None on critical failure
                           if merged_img.tobytes() != img_before:
                               img = merged_img # Update img only if merge succeeded and changed it
                               st.toast("Images merged!", icon="🧬")
                           # else: soft failure or no change, no toast
                        else:
                           st.warning("Merging failed critically.") # Should be rare
                           # Keep img as it was before merge attempt
                    else:
                        st.warning("Cannot merge: Second image not available.")
                    st.session_state.apply_merge_triggered = False # Reset flag


        # === 4. Final Comparison and State Update ===
        if img is not None: # Ensure img is valid after all processing
            final_img_bytes = img.tobytes()
            if final_img_bytes != current_processed_bytes:
                update_processed_image(img) # Update state via manager
            else:
                # No change detected, but reset flags just in case they were triggered
                # without causing a visible change (e.g., invalid zoom)
                reset_triggered_flags()
        else:
            # This should not happen if defensive checks above work, but is a safeguard
             st.error("Image processing resulted in an invalid state.")
             revert_processed_image()
             reset_triggered_flags()


    except Exception as e:
        # --- Graceful Error Handling ---
        processing_error_occurred = True
        st.error("An error occurred during image processing pipeline.")
        st.exception(e) # Show detailed traceback in the app
        revert_processed_image() # Attempt to revert to the last known good state
        reset_triggered_flags() # Reset flags to prevent loop on rerun

    finally:
        # --- Cleanup / Debug ---
        # Ensure flags are always reset *after* processing attempt, regardless of outcome
        # This prevents flags remaining True if an error happened before their reset point
        if not processing_error_occurred:
             reset_triggered_flags() # Reset any flags that might have been missed

        # For debugging performance: uncomment the print statement below
        # end_time = time.time()
        # duration = end_time - start_time
        # print(f"DEBUG: Image processing pipeline took: {duration:.4f} seconds")
        pass # Keep finally block clean unless needed


# --- Main Application Flow ---
def main() -> None:
    """Defines the main execution flow of the Streamlit application."""

    st.title("🖼️ Image Editor ")

    # 1. Initialize Session State: Ensures all keys exist on each run. Crucial.
    initialize_session_state()

    # 2. Build Sidebar UI: Renders widgets, handles uploads, sets trigger flags in state.
    build_sidebar()

    # 3. Run Image Processing Pipeline: Reads state, applies effects, updates processed_image state.
    run_image_processing_pipeline() # Spinner logic is now inside this function

    # 4. Display Main Area UI: Shows images, download, histogram based on the latest state.
    display_main_area()

    # 5. Footer
    st.markdown("---")
    st.caption("Streamlit Image Editor Pro | Modular Design")


# --- Script Execution Guard ---
if __name__ == "__main__":
    main()