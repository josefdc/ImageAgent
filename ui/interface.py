# streamlit_image_editor/ui/interface.py
import streamlit as st
from state.session_state_manager import reset_all_state, reset_triggered_flags
from core.image_io import load_image, prepare_image_for_download
from utils.constants import IMAGE_TYPES, DEFAULT_SAVE_FORMAT, DEFAULT_MIME_TYPE, DEFAULT_CHANNELS
from core.histogram import generate_histogram_figure
import numpy as np
import matplotlib.pyplot as plt # Import plt for close

def build_sidebar():
    """Creates and manages the sidebar widgets and logic."""
    with st.sidebar:
        st.header("Image Loading")
        st.info("Processing very large images may consume significant memory/time.")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=IMAGE_TYPES,
            key="main_uploader" # Keep key for potential external access if needed
        )

        # --- Image Loading Logic ---
        if uploaded_file:
            is_new_upload = (st.session_state.original_image is None or
                             uploaded_file.name != st.session_state.image_filename)
            if is_new_upload:
                with st.spinner('Loading image...'):
                    load_result = load_image(uploaded_file) # Use function from image_io
                    if load_result:
                        img, fname = load_result
                        st.session_state.original_image = img
                        st.session_state.image_filename = fname
                        reset_all_state() # Reset controls and derived state
                        # Specifically set processed image after reset
                        st.session_state.processed_image = img.copy()
                        st.success(f"Image '{fname}' loaded.")
                        # Rerun recommended after state changes affecting UI structure/defaults
                        st.rerun()
                    else:
                        # Clear state if loading failed
                        st.session_state.original_image = None
                        st.session_state.image_filename = None
                        st.session_state.processed_image = None

        # --- Show controls only if an image is loaded ---
        if st.session_state.get('original_image') is not None:
            st.caption(f"Loaded: {st.session_state.image_filename}")

            # --- Reset Button ---
            if st.button("🔄 Reset All Adjustments", key="reset_button"):
                with st.spinner('Resetting...'):
                    reset_all_state() # Use the state manager function
                st.rerun() # Rerun to reflect the reset state in widgets

            st.divider()

            # --- Basic Adjustments ---
            st.header("Basic Adjustments")
            st.slider(
                "Brightness", -100, 100, key='brightness_slider',
                help="Adjust brightness (-100=black, 0=original, 100=max)."
            )
            st.slider(
                "Contrast", 0.1, 3.0, step=0.1, key='contrast_slider',
                help="Adjust contrast (1.0=original)."
            )
            st.slider(
                "Rotation (Degrees)", 0, 360, key='rotation_slider',
                help="Rotate clockwise. Note: May change dimensions affecting zoom/merge."
            )

            st.divider()

            # --- Advanced Operations Expander ---
            with st.expander("🔬 Advanced Operations", expanded=False):
                # Zoom
                st.subheader("Zoom (Crop)")
                col_z1, col_z2 = st.columns(2)
                with col_z1:
                    st.number_input("X Start (%)", 0, 100, key='zoom_x')
                    st.number_input("Width (%)", 1, 100, key='zoom_w')
                with col_z2:
                    st.number_input("Y Start (%)", 0, 100, key='zoom_y')
                    st.number_input("Height (%)", 1, 100, key='zoom_h')
                if st.button("Apply Zoom"):
                    st.session_state.apply_zoom_triggered = True # Set flag

                # Binarization
                st.subheader("Binarization")
                st.slider("Threshold", 0, 255, key='binarize_thresh_slider')
                st.checkbox("Apply Binarization", key='apply_binarization_cb')

                # Negative
                st.subheader("Negative")
                if st.button("Invert Colors (Negative)"):
                     st.session_state.apply_negative_triggered = True # Set flag

                # Channel Manipulation
                st.subheader("Color Channels")
                st.multiselect("Select RGB Channels", DEFAULT_CHANNELS, key='channel_multiselect')

                # Highlight Zones
                st.subheader("Highlight Zones")
                st.radio("Mode", ["None", "Highlight Light Areas", "Highlight Dark Areas"], key='highlight_radio')
                st.slider("Highlight Threshold", 0, 255, key='highlight_thresh_slider')

            st.divider()

             # --- Image Merging Expander ---
            with st.expander("🧬 Image Merging", expanded=False):
                st.subheader("Merge with Second Image")
                uploaded_file_2 = st.file_uploader("Upload second image...", type=IMAGE_TYPES, key="uploader2")

                # Second Image Loading Logic
                if uploaded_file_2:
                    if st.session_state.get('second_image') is None or uploaded_file_2.name != st.session_state.get('second_image_filename'):
                        with st.spinner("Loading second image..."):
                             load_result_2 = load_image(uploaded_file_2)
                             if load_result_2:
                                 st.session_state.second_image, st.session_state.second_image_filename = load_result_2
                                 st.caption(f"Second image ready: {st.session_state.second_image_filename}")
                             else: # Clear if load failed
                                  st.session_state.second_image = None
                                  st.session_state.second_image_filename = None
                elif st.session_state.get('second_image') is not None: # Show name if already loaded
                     st.caption(f"Second image ready: {st.session_state.second_image_filename}")

                # Merge Controls (only if second image is loaded)
                if st.session_state.get('second_image') is not None:
                    st.slider("Fusion Alpha", 0.0, 1.0, step=0.05, key='merge_alpha_slider',
                              help="0.0: Main image, 0.5: Equal blend, 1.0: Second image.")
                    if st.button("Merge Images"):
                         st.session_state.apply_merge_triggered = True # Set flag
                else:
                    st.info("Upload a second image to enable merging.")

            st.divider()

            # --- Histogram Toggle ---
            if st.button("📊 Show/Update Histogram"):
                st.session_state.show_histogram = not st.session_state.get('show_histogram', False)

def display_main_area():
    """Displays the original/processed images, download button, and histogram."""
    if st.session_state.get('original_image') is None:
        st.info("☝️ Upload an image using the sidebar to get started!")
        return # Exit early if no image

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        processed_img = st.session_state.get('processed_image')
        if processed_img:
            st.image(processed_img, use_container_width=True)

            # --- Download Button ---
            img_bytes = prepare_image_for_download(processed_img, DEFAULT_SAVE_FORMAT)
            if img_bytes:
                base_name = st.session_state.get('image_filename', "image.png").split('.')[0]
                download_filename = f"processed_{base_name}.{DEFAULT_SAVE_FORMAT.lower()}"
                st.download_button(
                    label=f"💾 Save Processed ({DEFAULT_SAVE_FORMAT})",
                    data=img_bytes,
                    file_name=download_filename,
                    mime=DEFAULT_MIME_TYPE
                 )
            # Error message handled in prepare_image_for_download
        else:
            st.warning("No processed image available to display or download.")

    # --- Histogram Display ---
    if st.session_state.get('show_histogram') and processed_img:
        with st.expander("📊 Histogram", expanded=True):
            hist_fig = None # Ensure fig is defined
            try:
                with st.spinner("Generating histogram..."):
                    # Prepare data for caching function (bytes + metadata)
                    img_bytes_hist = processed_img.tobytes()
                    img_shape_hist = np.array(processed_img).shape
                    img_mode_hist = processed_img.mode

                    # Call the cached function
                    hist_fig = generate_histogram_figure(img_bytes_hist, img_shape_hist, img_mode_hist)

                if hist_fig:
                    st.pyplot(hist_fig)
                # Error handled within generate_histogram_figure
            except Exception as e:
                 st.error(f"Could not prepare/display histogram: {e}")
            finally:
                 # Ensure plot is closed regardless of success/failure in this block
                 if hist_fig:
                     plt.close(hist_fig)