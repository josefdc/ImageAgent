"""
Histogram generation module for image processing.

This module provides functionality to generate RGB and luminosity histograms
from image data, with caching support for improved performance in Streamlit applications.
"""

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


@st.cache_data(show_spinner=False)
def generate_histogram_figure(
    image_np_bytes: bytes, 
    shape: Tuple[int, ...], 
    mode: str
) -> Optional[plt.Figure]:
    """
    Generate a histogram plot for RGB channels and luminosity from image bytes.
    
    Args:
        image_np_bytes: Raw image data as bytes
        shape: Shape tuple of the original numpy array
        mode: PIL image mode (e.g., 'RGB', 'L', 'RGBA')
        
    Returns:
        matplotlib Figure object with histogram plot, or None if generation fails
        
    Note:
        This function is cache-friendly by using bytes instead of numpy arrays directly.
        Supports various image modes with automatic conversion to RGB when possible.
    """
    if not image_np_bytes or not shape:
        return None
        
    fig = None
    
    try:
        dtype = np.uint8
        image_np = np.frombuffer(image_np_bytes, dtype=dtype).reshape(shape)

        num_channels = 1
        if len(shape) == 3:
            num_channels = shape[2]

        plot_gray_only = False
        
        if mode == 'L' or len(shape) == 2 or num_channels == 1:
            if len(shape) == 2:
                image_np = np.stack((image_np,)*3, axis=-1)
            shape = image_np.shape
            num_channels = 3
            plot_gray_only = True
            
        elif mode != 'RGB':
            try:
                temp_img = Image.frombytes(mode, (shape[1], shape[0]), image_np_bytes)
                image_np = np.array(temp_img.convert('RGB'))
                shape = image_np.shape
                num_channels = 3
                st.info(f"Converted image mode '{mode}' to RGB for histogram.")
            except Exception as conv_e:
                st.warning(f"Could not convert mode '{mode}' for histogram: {conv_e}")

        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_gray_only or num_channels == 1:
            hist_gray, bin_edges_gray = np.histogram(
                image_np.ravel(), bins=256, range=[0, 256]
            )
            ax.plot(bin_edges_gray[:-1], hist_gray, color='gray', 
                   label='Intensity', alpha=0.8)
                   
        elif num_channels >= 3:
            colors = ('r', 'g', 'b')
            labels = ('Red', 'Green', 'Blue')
            
            for i, color in enumerate(colors):
                if i < shape[2]:
                    channel_data = image_np[:, :, i].ravel()
                    hist, bin_edges = np.histogram(
                        channel_data, bins=256, range=[0, 256]
                    )
                    ax.plot(bin_edges[:-1], hist, color=color, 
                           label=labels[i], alpha=0.7)

            img_gray_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
            hist_gray, bin_edges_gray = np.histogram(
                img_gray_np.ravel(), bins=256, range=[0, 256]
            )
            ax.plot(bin_edges_gray[:-1], hist_gray, color='gray', 
                   label='Luminosity', alpha=0.5, linestyle='--')
        else:
            st.warning(f"Cannot generate histogram for shape {shape} and mode {mode}")
            plt.close(fig)
            return None

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
        if fig:
            plt.close(fig)
        return None