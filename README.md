# 🖼️ Streamlit Image Editor++

A powerful, modular image editor built with Streamlit and Pillow (PIL). This application allows you to perform a variety of image processing operations through an intuitive web interface.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11%2B-FF4B4B)](https://streamlit.io/)
[![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-yellow)](https://pillow.readthedocs.io/)

![Screenshot of application](https://via.placeholder.com/800x450.png?text=Image+Editor+Screenshot)

## ✨ Features

### Basic Adjustments
- **Brightness**: Adjust the brightness of your image (-100 to +100)
- **Contrast**: Enhance or reduce image contrast (0.1 to 3.0)
- **Rotation**: Rotate the image by any angle (0° to 360°)

### Advanced Operations
- **Zoom/Crop**: Select a specific region of the image to crop
- **Binarization**: Convert image to black and white with an adjustable threshold
- **Negative**: Invert all colors in the image
- **Channel Manipulation**: Enable/disable specific RGB color channels
- **Highlight Zones**: Emphasize light or dark areas of the image

### Image Merging
- **Alpha Blend**: Combine two images with a controllable transparency level
- **Automatic Resizing**: Second image is automatically resized to match the dimensions of the first

### Analysis
- **RGB Histogram**: Visualize the distribution of color values with an interactive histogram 
- **Luminosity Analysis**: Analyze the brightness distribution across your image

### Export
- **Download Processed Images**: Save your edited images in PNG format

## 📋 Requirements

- Python 3.8 or newer
- Streamlit
- Pillow (PIL)
- NumPy
- Matplotlib

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/image-editor.git
cd image-editor
```

2. Create and activate a virtual environment (recommended):
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file yet, create one with:
```
streamlit>=1.11.0
Pillow>=9.0.0
numpy>=1.20.0
matplotlib>=3.5.0
```

## 🏃‍♀️ Running the Application

Launch the application with:
```bash
streamlit run app.py
```

This will start the application and open it in your default web browser (typically at http://localhost:8501).

## 📷 Usage Guide

1. **Upload an image** using the sidebar file uploader
2. Apply **basic adjustments** using the sliders in the sidebar
3. Explore **advanced operations** in the expandable sections
4. Upload a **second image** for merging if desired
5. Toggle the **histogram display** to analyze color distribution
6. **Download** your edited image using the button below the preview

### Example Workflow

1. Load an image
2. Increase brightness by +20
3. Enhance contrast to 1.5
4. Apply a 90° rotation
5. Use "Highlight Light Areas" to emphasize bright regions
6. View the histogram to analyze color distribution
7. Download the processed result

## 🔍 Project Structure

```
image-editor/
├── app.py                 # Main Streamlit application entry point
├── core/                  # Core processing modules
│   ├── __init__.py
│   ├── histogram.py       # Histogram generation functions
│   ├── image_io.py        # Image loading and saving utilities
│   └── processing.py      # Image processing operations
├── state/                 # State management
│   ├── __init__.py
│   └── session_state_manager.py  # Streamlit session state handling
├── ui/                    # User interface components
│   ├── __init__.py
│   └── interface.py       # UI building functions
├── utils/                 # Utility functions and constants
│   ├── __init__.py
│   └── constants.py       # Application constants
└── image_editor.py        # Legacy version (now modularized)
```

## 🧩 Architecture

The application follows a modular architecture:

1. **app.py**: Orchestrates the application flow
2. **state**: Manages session state for consistent user experience
3. **core**: Contains the core image processing functionality
4. **ui**: Handles user interface components
5. **utils**: Provides constants and utility functions

This separation of concerns makes the code more maintainable and easier to extend.

## 🛠️ Development

### Adding New Image Processing Functions

1. Add your new function to `core/processing.py`
2. Update the UI in `ui/interface.py` to expose the functionality
3. Modify `app.py` to call your function if needed

### Testing Changes

Run the application with:
```bash
streamlit run app.py
```

## 📄 License

[MIT License](LICENSE) 

## 👏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the interactive web framework
- [Pillow](https://pillow.readthedocs.io/) for the Python imaging library
- [NumPy](https://numpy.org/) for numerical processing
- [Matplotlib](https://matplotlib.org/) for histogram visualization

---

Made with ❤️ by @josefdc @Esteban8482
