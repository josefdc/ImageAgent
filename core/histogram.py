#image_editor/core/histogram.py
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Optional, Tuple

# Cache histogram generation based on image data bytes and metadata
@st.cache_data(show_spinner=False)
def generate_histogram_figure(image_np_bytes: bytes, shape: Tuple[int, ...], mode: str) -> Optional[plt.Figure]:
    """
    Generates a histogram plot for RGB + Luminosity from image bytes and metadata.
    More cache-friendly than passing NumPy arrays directly.
    """
    if not image_np_bytes or not shape: return None
    fig = None # Initialize fig
    try:
        # Reconstruct numpy array from bytes
        dtype = np.uint8 # Assuming standard image data type
        image_np = np.frombuffer(image_np_bytes, dtype=dtype).reshape(shape)

        # Handle different modes - try to get to 3 channels (RGB) for consistent plotting
        num_channels = 1
        if len(shape) == 3:
             num_channels = shape[2]

        plot_gray_only = False
        if mode == 'L' or (len(shape) == 2) or (num_channels == 1):
             # Convert grayscale to 3-channel gray for consistent processing below
             if len(shape) == 2:
                 image_np = np.stack((image_np,)*3, axis=-1)
             # Update shape if needed
             shape = image_np.shape
             num_channels = 3
             plot_gray_only = True # Flag to only plot the gray intensity
        elif mode != 'RGB':
             # Attempt conversion for other modes like RGBA, P, CMYK etc.
             try:
                 temp_img = Image.frombytes(mode, (shape[1], shape[0]), image_np_bytes) # Assuming width, height order
                 image_np = np.array(temp_img.convert('RGB'))
                 shape = image_np.shape
                 num_channels = 3
                 st.info(f"Converted image mode '{mode}' to RGB for histogram.")
             except Exception as conv_e:
                 st.warning(f"Could not convert mode '{mode}' for histogram: {conv_e}. Plot may be inaccurate.")
                 # Proceed with original data if conversion fails, might lead to errors below

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_gray_only or num_channels == 1:
             # Plot single intensity histogram for grayscale images
             hist_gray, bin_edges_gray = np.histogram(image_np.ravel(), bins=256, range=[0, 256])
             ax.plot(bin_edges_gray[:-1], hist_gray, color='gray', label='Intensity', alpha=0.8)
        elif num_channels >= 3: # Plot RGB + Luminosity
            colors = ('r', 'g', 'b')
            labels = ('Red', 'Green', 'Blue')
            # RGB Histograms
            for i, color in enumerate(colors):
                 if i < shape[2]: # Ensure channel exists
                     channel_data = image_np[:, :, i].ravel()
                     hist, bin_edges = np.histogram(channel_data, bins=256, range=[0, 256])
                     ax.plot(bin_edges[:-1], hist, color=color, label=labels[i], alpha=0.7)

            # Luminosity Histogram (calculated from RGB)
            img_gray_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140]) # Use first 3 channels for luminosity
            hist_gray, bin_edges_gray = np.histogram(img_gray_np.ravel(), bins=256, range=[0, 256])
            ax.plot(bin_edges_gray[:-1], hist_gray, color='gray', label='Luminosity', alpha=0.5, linestyle='--')
        else:
             st.warning(f"Cannot generate standard histogram for image shape {shape} and mode {mode}.")
             plt.close(fig) # Close the empty figure
             return None


        # --- Final plot setup ---
        ax.set_title("Color Histogram")
        ax.set_xlabel("Pixel Value (0-255)")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 255) # Ensure x-axis is 0-255
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error generating histogram: {e}")
        if fig: # Close figure if partially created before error
            plt.close(fig)
        return None