# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
import time
import logging
# PIL Image is used for type hinting and potentially direct operations if needed
from PIL import Image
from typing import Optional # Used for type hinting

# --- Path Setup (Add Project Root) ---
try:
    # The project root is the directory containing app.py
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        print(f"DEBUG (app.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")

except Exception as e:
    print(f"ERROR (app.py): Failed during sys.path setup: {e}")


# --- Streamlit Page Config (MUST be the FIRST Streamlit command) ---
import streamlit as st
st.set_page_config(
    page_title="Image Editor Pro", # Main page title
    page_icon="🖼️", # Main page icon
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "https://github.com/josefdc/image-editor", # Replace with your repo
        'Report a bug': "https://github.com/josefdc/image-editor/issues", # Replace with your repo
        'About': """
        ## Streamlit Image Editor Pro

        An interactive web application for image viewing and editing.
        Includes manual controls and an AI Assistant page.
        Built with Streamlit, Pillow, and LangChain/LangGraph.
        """
    }
)

# --- Local Application Imports (Use Paths Relative to Project Root) ---
try:
    # Imports are now relative to the project root added to sys.path
    from state.session_state_manager import (
        initialize_session_state,
        reset_triggered_flags,
        update_processed_image,
        revert_processed_image
    )
    # Import the core processing module itself
    from core import processing
    # Import UI components
    from ui.interface import build_sidebar, display_main_area
    _APP_DEPENDENCIES_LOADED = True
    print("DEBUG (app.py): Successfully imported app dependencies.")
except ImportError as e:
    st.error(f"Critical Error: Could not import application modules needed for the main editor: {e}. Check console logs and project structure/sys.path.")
    print(f"ERROR (app.py): Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    _APP_DEPENDENCIES_LOADED = False
    # Stop the app if core components fail to load
    st.stop()
except Exception as e:
     st.error(f"Critical Error during application import: {e}")
     st.exception(e)
     _APP_DEPENDENCIES_LOADED = False
     st.stop()

# --- Logging Setup ---
# Configure a basic logger for the main app if needed
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Central Image Processing Logic ---
def run_image_processing_pipeline() -> None:
    """
    Orchestrates manual image processing based on the current session state widgets.
    This function handles the non-AI edits triggered by sliders, buttons etc.
    """
    start_time = time.time() # For performance monitoring

    # 1. Pre-checks
    if not st.session_state.get('original_image'):
        reset_triggered_flags() # Ensure flags are reset even if no image
        return # Nothing to process

    # Ensure processed_image exists, initializing from original if needed
    if 'processed_image' not in st.session_state or not isinstance(st.session_state.processed_image, Image.Image):
        if isinstance(st.session_state.original_image, Image.Image):
            logger.info("Initializing processed_image from original_image.")
            st.session_state.processed_image = st.session_state.original_image.copy()
        else:
            logger.error("Original image is missing or invalid. Cannot initialize processed image.")
            st.error("Cannot process: Original image is missing.")
            reset_triggered_flags()
            return

    # 2. Get initial state for comparison (detect if changes actually happen)
    # Use a copy for comparison base, as 'img' will be modified
    initial_processed_image = st.session_state.processed_image.copy()
    img = None # Initialize img variable

    # 3. Processing Pipeline
    processing_error_occurred = False
    try:
        # --- Start processing from the ORIGINAL image for continuous adjustments ---
        # This ensures sliders always apply relative to the base image, preventing drift.
        if not isinstance(st.session_state.original_image, Image.Image):
             raise ValueError("Original image in session state is not a valid PIL Image.")
        img = st.session_state.original_image.copy()
        logger.debug("Starting processing pipeline from original image copy.")

        # === Apply Continuous Adjustments (Always run based on current widget state) ===
        # Use a single spinner for all continuous adjustments for better UX
        with st.spinner("Applying adjustments..."):
            # Brightness
            if st.session_state.brightness_slider != 0: # Optimization: skip if no change
                img = processing.apply_brightness(img, st.session_state.brightness_slider)
                if img is None: raise ValueError("Brightness processing failed.")

            # Contrast
            if st.session_state.contrast_slider != 1.0: # Optimization: skip if no change
                img = processing.apply_contrast(img, st.session_state.contrast_slider)
                if img is None: raise ValueError("Contrast processing failed.")

            # Rotation
            if st.session_state.rotation_slider != 0: # Optimization: skip if no change
                img = processing.apply_rotation(img, st.session_state.rotation_slider)
                if img is None: raise ValueError("Rotation processing failed.")

            # Channel Manipulation (Example - adapt if needed)
            # Assuming default is all channels selected, so only apply if different
            # if set(st.session_state.channel_multiselect) != {'R', 'G', 'B'}: # Adjust default based on actual implementation
            #     img = processing.apply_channel_manipulation(img, st.session_state.channel_multiselect)
            #     if img is None: raise ValueError("Channel manipulation failed.")

            # Highlight (Example - adapt if needed)
            # if st.session_state.highlight_radio != 'Off': # Adjust default
            #     img = processing.apply_highlight(img, st.session_state.highlight_radio, st.session_state.highlight_thresh_slider)
            #     if img is None: raise ValueError("Highlight processing failed.")

            # Binarization (Checkbox controls application)
            if st.session_state.get('apply_binarization_cb', False): # Safely get checkbox state
                img = processing.apply_binarization(img, st.session_state.binarize_thresh_slider)
                if img is None: raise ValueError("Binarization processing failed.")

        logger.debug("Continuous adjustments applied.")

        # === Apply Triggered Adjustments (Only if flags are set) ===
        # These operate sequentially on the result of the continuous adjustments
        trigger_spinner_needed = any([
            st.session_state.get('apply_zoom_triggered', False),
            st.session_state.get('apply_negative_triggered', False),
            st.session_state.get('apply_merge_triggered', False)
            # Add other trigger flags here
        ])

        if trigger_spinner_needed:
            with st.spinner("Applying triggered actions..."):
                # Zoom
                if st.session_state.get('apply_zoom_triggered', False):
                    logger.info("Applying triggered zoom.")
                    img_before_zoom = img.copy() # Copy before potential modification
                    zoom_params = (st.session_state.zoom_x, st.session_state.zoom_y, st.session_state.zoom_w, st.session_state.zoom_h)
                    img = processing.apply_zoom(img, *zoom_params) # Use the processing function
                    if img is None: raise ValueError("Zoom processing failed.")
                    if img.tobytes() != img_before_zoom.tobytes(): st.toast("Zoom applied!", icon="🔎")
                    st.session_state.apply_zoom_triggered = False # Reset flag

                # Negative (Invert)
                if st.session_state.get('apply_negative_triggered', False):
                    logger.info("Applying triggered negative.")
                    img_before_neg = img.copy()
                    img = processing.apply_negative(img)
                    if img is None: raise ValueError("Negative processing failed.")
                    if img.tobytes() != img_before_neg.tobytes(): st.toast("Colors inverted!", icon="🌗")
                    st.session_state.apply_negative_triggered = False # Reset flag

                # Merge
                if st.session_state.get('apply_merge_triggered', False):
                    logger.info("Applying triggered merge.")
                    second_image = st.session_state.get('second_image')
                    if second_image and isinstance(second_image, Image.Image):
                        img_before_merge = img.copy()
                        alpha = st.session_state.merge_alpha_slider
                        merged_img = processing.merge_images(img, second_image, alpha)

                        if merged_img is not None:
                           # Check if merge actually changed the image
                           if merged_img.tobytes() != img_before_merge.tobytes():
                               img = merged_img # Update img only if merge succeeded and changed it
                               st.toast("Images merged!", icon="🧬")
                           else:
                               logger.info("Merge completed but resulted in no change to the primary image.")
                        else:
                           # merge_images might return None on critical failure
                           st.warning("Merging failed critically. Image remains unchanged.")
                    else:
                        st.warning("Cannot merge: Second image not available or invalid.")
                    st.session_state.apply_merge_triggered = False # Reset flag

                # Add other triggered actions here...

        logger.debug("Triggered adjustments applied (if any).")

        # === 4. Final Comparison and State Update ===
        if img is not None and isinstance(img, Image.Image):
            # Compare the final result 'img' with the 'initial_processed_image' before this run
            if img.tobytes() != initial_processed_image.tobytes():
                logger.info("Image changed, updating session state.")
                update_success = update_processed_image(img) # Update state via manager
                if not update_success:
                     logger.warning("update_processed_image returned False. State update might have failed.")
                     st.warning("Failed to update the processed image state.")
            else:
                logger.debug("No change detected in image after processing pipeline.")
        else:
            # This indicates a severe error where 'img' became None or invalid
             logger.error("Image processing resulted in an invalid state (None or wrong type). Reverting.")
             st.error("Image processing failed critically. Reverting changes.")
             revert_processed_image() # Attempt to revert
             processing_error_occurred = True # Mark error

    except Exception as e:
        # --- Graceful Error Handling ---
        processing_error_occurred = True
        logger.error(f"Error during image processing pipeline: {e}", exc_info=True)
        st.error(f"An error occurred during image processing: {e}")
        # st.exception(e) # Optionally show full traceback in the app for debugging
        revert_processed_image() # Attempt to revert to the last known good state

    finally:
        # --- Cleanup: Reset Trigger Flags ---
        # Ensure flags are always reset *after* processing attempt, regardless of outcome
        reset_triggered_flags()
        logger.debug("Triggered flags reset.")

        # --- Performance Logging ---
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"Image processing pipeline took: {duration:.4f} seconds. Error occurred: {processing_error_occurred}")


# --- Main Application Flow ---
def main() -> None:
    """Defines the main execution flow of the Streamlit application."""

    st.title("🖼️ Image Editor Pro")
    st.caption("Manual controls and AI assistance for your images.")

    # 1. Initialize Session State: Ensures all keys exist. Crucial.
    initialize_session_state()

    # 2. Build Sidebar UI: Renders widgets, handles uploads, sets trigger flags.
    build_sidebar()

    # 3. Run Image Processing Pipeline: Reads state, applies manual effects, updates processed_image.
    run_image_processing_pipeline() # Spinner logic is now inside this function

    # 4. Display Main Area UI: Shows images, download button, histogram etc. based on the latest state.
    display_main_area()

    # 5. Footer or other info
    st.markdown("---")
    st.caption("Navigate to the 'AI Assistant' page in the sidebar for AI-powered edits.")


# --- Script Execution Guard ---
if __name__ == "__main__":
    if _APP_DEPENDENCIES_LOADED:
        logger.info("Starting Streamlit Image Editor Pro application.")
        main()
    else:
        logger.critical("Application cannot start because core dependencies failed to load.")
        # Error message already shown via st.error