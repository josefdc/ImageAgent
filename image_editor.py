import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import io
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

# --- Constants ---
IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png", "bmp"]
DEFAULT_SAVE_FORMAT: str = 'PNG'
DEFAULT_MIME_TYPE: str = 'image/png'
HIGHLIGHT_GRAY_COLOR: List[int] = [150, 150, 150] # Color for non-highlighted areas

# --- Page Configuration ---
st.set_page_config(
    page_title="Streamlit Image Editor++", # Incremented version :)
    page_icon="🖼️",
    layout="wide"
)

# --- Session State Initialization ---
def initialize_session_state(defaults: Dict[str, Any]):
    """Initializes session state variables if they don't exist."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Define default values for session state keys
default_session_values: Dict[str, Any] = {
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
    'channel_multiselect': ['Red', 'Green', 'Blue'],
    'highlight_radio': 'None',
    'highlight_thresh_slider': 128,
    'merge_alpha_slider': 0.5,
    # Button click flags - useful for triggering actions once
    # These flags should be reset after the action is processed.
    'apply_zoom_triggered': False,
    'apply_negative_triggered': False,
    'apply_merge_triggered': False,
    'last_processed_image_state': None # Store previous state for potential revert on error
}

initialize_session_state(default_session_values)


# --- Image Processing Functions (Mostly unchanged, added HIGHLIGHT_GRAY_COLOR) ---

def load_image(uploaded_file) -> Optional[Tuple[Image.Image, str]]:
    """Loads an image from an uploaded file object."""
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image, uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image '{uploaded_file.name}': {e}")
            return None
    return None

def apply_brightness(image: Optional[Image.Image], factor: int) -> Optional[Image.Image]:
    if image is None: return None
    try:
        enhancer = ImageEnhance.Brightness(image)
        mapped_factor = 1.0 + (factor / 100.0)
        return enhancer.enhance(mapped_factor)
    except Exception as e:
        st.warning(f"Could not apply brightness: {e}")
        return image # Return original on error

def apply_contrast(image: Optional[Image.Image], factor: float) -> Optional[Image.Image]:
    if image is None: return None
    try:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    except Exception as e:
        st.warning(f"Could not apply contrast: {e}")
        return image

def apply_rotation(image: Optional[Image.Image], angle: int) -> Optional[Image.Image]:
    if image is None or angle == 0: return image # No-op if no angle
    try:
        return image.rotate(angle, expand=True, fillcolor='white')
    except Exception as e:
        st.warning(f"Could not apply rotation: {e}")
        return image

def apply_zoom(image: Optional[Image.Image], x_perc: int, y_perc: int, w_perc: int, h_perc: int) -> Optional[Image.Image]:
    if image is None: return None
    try:
        width, height = image.size
        left = int(width * (x_perc / 100.0))
        top = int(height * (y_perc / 100.0))
        right = left + int(width * (w_perc / 100.0))
        bottom = top + int(height * (h_perc / 100.0))

        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)

        if right <= left or bottom <= top:
            st.warning("Invalid zoom dimensions. Width and Height must result in a positive area.")
            return image
        return image.crop((left, top, right, bottom))
    except Exception as e:
        st.warning(f"Could not apply zoom: {e}")
        return image

def apply_binarization(image: Optional[Image.Image], threshold: int) -> Optional[Image.Image]:
    if image is None: return None
    try:
        img_gray = image.convert('L')
        img_bin = img_gray.point(lambda p: 255 if p > threshold else 0)
        return img_bin.convert('RGB')
    except Exception as e:
        st.warning(f"Could not apply binarization: {e}")
        return image

def apply_negative(image: Optional[Image.Image]) -> Optional[Image.Image]:
    if image is None: return None
    try:
        return ImageOps.invert(image.convert('RGB'))
    except Exception as e:
        st.warning(f"Could not apply negative: {e}")
        return image

def apply_channel_manipulation(image: Optional[Image.Image], selected_channels: List[str]) -> Optional[Image.Image]:
    if image is None or set(selected_channels) == {'Red', 'Green', 'Blue'}:
        return image # No-op if all channels selected
    try:
        r, g, b = image.split()
        size = image.size
        zero_channel = Image.new('L', size, 0)
        r_channel = r if 'Red' in selected_channels else zero_channel
        g_channel = g if 'Green' in selected_channels else zero_channel
        b_channel = b if 'Blue' in selected_channels else zero_channel
        return Image.merge('RGB', (r_channel, g_channel, b_channel))
    except Exception as e:
        st.warning(f"Could not apply channel manipulation: {e}")
        return image

def apply_highlight(image: Optional[Image.Image], mode: str, threshold: int) -> Optional[Image.Image]:
    if image is None or mode == "None": return image
    try:
        img_gray = image.convert('L')
        img_np = np.array(image).copy()
        mask = None
        if mode == "Highlight Light Areas":
            mask = np.array(img_gray) <= threshold
        elif mode == "Highlight Dark Areas":
            mask = np.array(img_gray) >= threshold

        if mask is not None:
            img_np[mask] = HIGHLIGHT_GRAY_COLOR # Use constant

        return Image.fromarray(img_np)
    except Exception as e:
        st.warning(f"Could not apply highlight: {e}")
        return image

def merge_images(image1: Optional[Image.Image], image2: Optional[Image.Image], alpha: float) -> Optional[Image.Image]:
    if image1 is None or image2 is None:
        st.warning("Both images must be loaded to merge.")
        return image1 or image2
    try:
        if image1.size != image2.size:
            st.info(f"Resizing second image from {image2.size} to {image1.size} for merging.")
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)

        image1 = image1.convert("RGBA")
        image2 = image2.convert("RGBA")
        blended = Image.blend(image1, image2, alpha=alpha)
        return blended.convert("RGB")
    except Exception as e:
        st.error(f"Error merging images: {e}")
        return image1 # Return primary image on error

@st.cache_data(show_spinner=False)
def generate_histogram_figure(image_np_bytes: bytes, shape: tuple, mode: str) -> Optional[plt.Figure]:
    """Generates histogram from image bytes, shape, and mode for better caching."""
    if not image_np_bytes: return None
    fig = None # Initialize fig to None
    try:
        image_np = np.frombuffer(image_np_bytes, dtype=np.uint8).reshape(shape)
        # Ensure it's RGB for standard processing, handle grayscale separately
        if mode == 'L':
             image_np = np.stack((image_np,)*3, axis=-1) # Convert L to RGB-like for processing
        elif mode != 'RGB':
             # Handle other modes or raise error if unsupported by hist logic
             st.warning(f"Histogram generation for mode '{mode}' might be inaccurate. Trying RGB conversion.")
             temp_img = Image.frombytes(mode, (shape[1], shape[0]), image_np_bytes)
             image_np = np.array(temp_img.convert('RGB'))


        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ('r', 'g', 'b')
        labels = ('Red', 'Green', 'Blue')

        for i, color in enumerate(colors):
            # Check if image_np has 3 dimensions before accessing the third one
            if len(image_np.shape) == 3 and image_np.shape[2] > i:
                 channel_data = image_np[:, :, i].ravel()
                 hist, bin_edges = np.histogram(channel_data, bins=256, range=[0, 256])
                 ax.plot(bin_edges[:-1], hist, color=color, label=labels[i], alpha=0.7)
            elif i == 0 and (len(image_np.shape) == 2 or image_np.shape[2] == 1): # Handle grayscale if passed directly
                 channel_data = image_np.ravel()
                 hist, bin_edges = np.histogram(channel_data, bins=256, range=[0, 256])
                 ax.plot(bin_edges[:-1], hist, color='gray', label='Intensity', alpha=0.9)
                 # No need for luminosity calculation if already gray

        # Calculate Luminosity only if it was originally RGB
        if mode == 'RGB' and len(image_np.shape) == 3 and image_np.shape[2] == 3:
             img_gray_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
             hist_gray, bin_edges_gray = np.histogram(img_gray_np.ravel(), bins=256, range=[0, 256])
             ax.plot(bin_edges_gray[:-1], hist_gray, color='gray', label='Luminosity', alpha=0.5, linestyle='--')

        ax.set_title("Color Histogram")
        ax.set_xlabel("Pixel Value (0-255)")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 255)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating histogram: {e}")
        if fig: # Ensure figure is closed even if error occurs after creation
            plt.close(fig)
        return None
    # No finally block needed as plt.close should happen after st.pyplot in main code


# --- Helper Functions ---
def reset_all():
    """Resets processed image and control values to defaults."""
    st.session_state.processed_image = st.session_state.original_image.copy() if st.session_state.original_image else None
    st.session_state.show_histogram = False
    for key, value in default_session_values.items():
        # Avoid resetting image data or flags handled elsewhere
        if 'image' not in key and 'triggered' not in key:
            st.session_state[key] = value
    # Explicitly reset flags here if managed globally
    reset_triggered_flags()
    st.success("Adjustments Reset!")

def reset_triggered_flags():
    """Resets all action trigger flags in session state."""
    st.session_state.apply_zoom_triggered = False
    st.session_state.apply_negative_triggered = False
    st.session_state.apply_merge_triggered = False

def apply_image_processing() -> Optional[Image.Image]:
    """
    Applies all sequential processing based on current session state.
    Returns the newly processed image or None if no processing was done/needed.
    Manages trigger flags.
    """
    if not st.session_state.original_image or not st.session_state.processed_image:
        return None

    # Store the state *before* this processing run
    st.session_state.last_processed_image_state = st.session_state.processed_image.copy()
    img = st.session_state.processed_image.copy() # Work on a copy

    try:
        # Apply continuous adjustments (sliders, radio)
        img = apply_brightness(img, st.session_state.brightness_slider)
        img = apply_contrast(img, st.session_state.contrast_slider)
        img = apply_rotation(img, st.session_state.rotation_slider)
        img = apply_channel_manipulation(img, st.session_state.channel_multiselect)
        img = apply_highlight(img, st.session_state.highlight_radio, st.session_state.highlight_thresh_slider)
        if st.session_state.apply_binarization_cb:
            img = apply_binarization(img, st.session_state.binarize_thresh_slider)

        # Apply triggered adjustments (buttons)
        if st.session_state.apply_zoom_triggered:
            img = apply_zoom(img, st.session_state.zoom_x, st.session_state.zoom_y, st.session_state.zoom_w, st.session_state.zoom_h)
            st.success("Zoom applied!")
            st.session_state.apply_zoom_triggered = False # Reset flag

        if st.session_state.apply_negative_triggered:
            img = apply_negative(img)
            st.success("Colors inverted!")
            st.session_state.apply_negative_triggered = False # Reset flag

        if st.session_state.apply_merge_triggered:
            if st.session_state.second_image:
                with st.spinner("Merging images..."):
                    img = merge_images(img, st.session_state.second_image, st.session_state.merge_alpha_slider)
                if img != st.session_state.last_processed_image_state: # Check if merge actually happened/succeeded
                     st.success("Images merged!")
            else:
                st.warning("Cannot merge: Second image not loaded.")
            st.session_state.apply_merge_triggered = False # Reset flag

        return img

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        # Revert to the state before this processing attempt
        if st.session_state.last_processed_image_state:
             st.session_state.processed_image = st.session_state.last_processed_image_state
        reset_triggered_flags() # Ensure flags are reset on error too
        st.exception(e) # Show detailed traceback
        return None # Indicate failure

# --- Main App UI ---
st.title("🖼️ Interactive Image Viewer & Editor++")

# --- Sidebar UI ---
with st.sidebar:
    st.header("Image Loading")
    st.info("Processing very large images may be slow or consume significant memory.")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=IMAGE_TYPES,
        key="main_uploader"
    )

    # Image Loading Logic
    if uploaded_file:
        is_new_upload = (st.session_state.original_image is None or
                         uploaded_file.name != st.session_state.image_filename)
        if is_new_upload:
            with st.spinner('Loading image...'):
                load_result = load_image(uploaded_file)
                if load_result:
                    img, fname = load_result
                    st.session_state.original_image = img
                    st.session_state.image_filename = fname
                    reset_all() # Reset controls and processed image
                    st.session_state.processed_image = img.copy() # Set initial processed state
                    st.success(f"Image '{fname}' loaded.")
                    st.rerun()
                else:
                    st.session_state.original_image = None # Ensure cleanup on load fail
                    st.session_state.image_filename = None
                    st.session_state.processed_image = None

    # Controls shown only if an image is loaded
    if st.session_state.original_image:
        st.caption(f"Loaded: {st.session_state.image_filename}")

        if st.button("🔄 Reset All Adjustments", key="reset_button"):
            with st.spinner('Resetting...'):
                reset_all()
            st.rerun()

        st.divider()
        st.header("Basic Adjustments")
        st.slider("Brightness", -100, 100, key='brightness_slider',
                  help="Adjust image brightness (-100=black, 0=original, 100=max).")
        st.slider("Contrast", 0.1, 3.0, step=0.1, key='contrast_slider',
                  help="Adjust image contrast (1.0=original).")
        st.slider("Rotation (Degrees)", 0, 360, key='rotation_slider',
                  help="Rotate the image clockwise. Note: May change dimensions.") # Added note

        st.divider()
        with st.expander("🔬 Advanced Operations", expanded=False):
            st.subheader("Zoom (Crop)")
            col_zoom1, col_zoom2 = st.columns(2)
            with col_zoom1:
                st.number_input("X Start (%)", 0, 100, key='zoom_x')
                st.number_input("Width (%)", 1, 100, key='zoom_w')
            with col_zoom2:
                st.number_input("Y Start (%)", 0, 100, key='zoom_y')
                st.number_input("Height (%)", 1, 100, key='zoom_h')
            if st.button("Apply Zoom"):
                st.session_state.apply_zoom_triggered = True # Set flag

            st.subheader("Binarization")
            st.slider("Threshold", 0, 255, key='binarize_thresh_slider')
            st.checkbox("Apply Binarization", key='apply_binarization_cb')

            st.subheader("Negative")
            if st.button("Invert Colors (Negative)"):
                 st.session_state.apply_negative_triggered = True # Set flag

            st.subheader("Color Channels")
            st.multiselect("Select RGB Channels", ["Red", "Green", "Blue"], key='channel_multiselect')

            st.subheader("Highlight Zones")
            st.radio("Highlight Mode", ["None", "Highlight Light Areas", "Highlight Dark Areas"], key='highlight_radio')
            st.slider("Highlight Threshold", 0, 255, key='highlight_thresh_slider')

        st.divider()
        with st.expander("🧬 Image Merging", expanded=False):
            st.subheader("Merge with Second Image")
            uploaded_file_2 = st.file_uploader("Upload second image...", type=IMAGE_TYPES, key="uploader2")

            # Second Image Loading
            if uploaded_file_2:
                if st.session_state.second_image is None or uploaded_file_2.name != st.session_state.second_image_filename:
                    with st.spinner("Loading second image..."):
                         load_result_2 = load_image(uploaded_file_2)
                         if load_result_2:
                             st.session_state.second_image, st.session_state.second_image_filename = load_result_2
                             st.caption(f"Second image ready: {st.session_state.second_image_filename}")
                         else:
                              st.session_state.second_image = None
                              st.session_state.second_image_filename = None
            elif st.session_state.second_image:
                 st.caption(f"Second image ready: {st.session_state.second_image_filename}")

            # Merge Controls
            if st.session_state.second_image:
                st.slider("Fusion Transparency (Alpha)", 0.0, 1.0, step=0.05, key='merge_alpha_slider',
                          help="0.0: Main image, 0.5: Equal blend, 1.0: Second image.")
                if st.button("Merge Images"):
                     st.session_state.apply_merge_triggered = True # Set flag
            else:
                st.info("Upload a second image to enable merging.")

        st.divider()
        if st.button("📊 Show/Update Histogram"):
            st.session_state.show_histogram = not st.session_state.show_histogram

# --- Central Processing Trigger ---
# This runs *after* all widgets have been rendered and state potentially updated by them
processed_image_result = apply_image_processing()
if processed_image_result is not None:
    # Update the main state only if processing function returned a valid image
    st.session_state.processed_image = processed_image_result
# Flags are reset within apply_image_processing()


# --- Main Area Display ---
if st.session_state.original_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, use_container_width=True)

            # --- Download Button ---
            try:
                buf = io.BytesIO()
                # Use the *currently displayed* processed image
                processed_img_to_save = st.session_state.processed_image
                save_format = DEFAULT_SAVE_FORMAT
                mime_type = DEFAULT_MIME_TYPE

                img_mode = processed_img_to_save.mode
                if img_mode == 'RGBA' and save_format == 'PNG':
                     processed_img_to_save.save(buf, format=save_format)
                elif img_mode == 'P':
                     processed_img_to_save.convert('RGB').save(buf, format=save_format)
                elif img_mode == 'L':
                     processed_img_to_save.save(buf, format=save_format)
                else: # Assume RGB or convert
                     processed_img_to_save.convert('RGB').save(buf, format=save_format)

                byte_im = buf.getvalue()
                base_name = st.session_state.image_filename.split('.')[0] if st.session_state.image_filename else "image"
                download_filename = f"processed_{base_name}.{save_format.lower()}"

                st.download_button(
                    label=f"💾 Save Processed Image ({save_format})",
                    data=byte_im,
                    file_name=download_filename,
                    mime=mime_type
                 )
            except Exception as e:
                 st.error(f"Error preparing image for download: {e}")
        else:
            st.warning("No processed image available to display or download.")


    # --- Histogram Display ---
    if st.session_state.show_histogram and st.session_state.processed_image:
        with st.expander("📊 Histogram", expanded=True):
            hist_fig = None
            try:
                with st.spinner("Generating histogram..."):
                    # Prepare data for caching function (bytes + metadata)
                    img_for_hist = st.session_state.processed_image
                    img_bytes = img_for_hist.tobytes()
                    img_shape = np.array(img_for_hist).shape
                    img_mode = img_for_hist.mode

                    hist_fig = generate_histogram_figure(img_bytes, img_shape, img_mode)

                if hist_fig:
                    st.pyplot(hist_fig)
                # Error handled within generate_histogram_figure
            except Exception as e:
                 st.error(f"Could not prepare image data for histogram: {e}")
            finally:
                 # Ensure plot is closed regardless of success/failure
                 if hist_fig:
                     plt.close(hist_fig)

# Initial Welcome Message (Simplified)
elif st.session_state.original_image is None:
     st.info("☝️ Upload an image using the sidebar to get started!")


# --- Footer ---
st.markdown("---")
st.caption("Enhanced Image Editor++ using Streamlit & Pillow.")