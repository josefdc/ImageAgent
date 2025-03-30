# 🖼️ AI-Powered Image Editor with LangGraph Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11%2B-FF4B4B)](https://streamlit.io/) [![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-yellow)](https://pillow.readthedocs.io/) [![LangChain](https://img.shields.io/badge/LangChain-LangGraph-blueviolet)](https://python.langchain.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green)](https://opencv.org/) 

**Edit images intelligently using natural language!** This application combines a standard Streamlit image editor with a conversational AI assistant powered by **LangGraph**. The agent understands your requests and uses the editor's functions (like brightness, contrast, filters) as tools to modify the image.

-----


![image](https://github.com/user-attachments/assets/ecc51798-e926-43f8-bc5c-7afead1a80b3)


-----

## Table of Contents

- [🖼️ AI-Powered Image Editor with LangGraph Agent](#️-ai-powered-image-editor-with-langgraph-agent)
  - [Table of Contents](#table-of-contents)
  - [✨ Key Features](#-key-features)
    - [Core Image Editing (Manual Interface)](#core-image-editing-manual-interface)
    - [🤖 AI Assistant Page (Powered by LangGraph)](#-ai-assistant-page-powered-by-langgraph)
  - [📋 Requirements](#-requirements)
  - [🚀 Installation](#-installation)
  - [🏃‍♀️ Running the Application](#️-running-the-application)
  - [🗣️ Usage Guide (AI Assistant)](#️-usage-guide-ai-assistant)
  - [🔍 Project Structure](#-project-structure)
  - [🧩 Architecture \& Agent Flow](#-architecture--agent-flow)
  - [🛠️ Development \& Contributions](#️-development--contributions)
    - [Modifying the Agent](#modifying-the-agent)
    - [Adding New Agent Tools](#adding-new-agent-tools)
    - [Contributing](#contributing)
  - [📄 License](#-license)
  - [👏 Acknowledgements](#-acknowledgements)

## ✨ Key Features

### Core Image Editing (Manual Interface)

  * **Adjustments**: Brightness, Contrast, Rotation.
  * **Operations**: Zoom/Crop, Binarization, Negative (Invert), RGB Channel Selection, Highlight Light/Dark Areas.
  * **Merging**: Alpha blend the primary image with a second uploaded image (automatic resizing).
  * **Analysis**: View RGB & Luminosity Histogram.
  * **Export**: Download the edited image (default: PNG).

### 🤖 AI Assistant Page (Powered by LangGraph)

  * **Natural Language Control**: Instruct the editor via chat (e.g., "Make it brighter", "Apply a blur filter then rotate 90 degrees").
  * **LangGraph Agent**: Manages the conversation flow and decides which image editing tools to use based on your request.
  * **Tool Integration**: Uses functions from `core/processing.py` and potentially `core/ai_services.py` as tools (e.g., `adjust_brightness`, `apply_filter`, `remove_background_ai`).
  * **Multi-Step Execution**: Handles sequential commands within a single conversational turn.
  * **Context Aware**: Remembers the conversation history within the current session using LangGraph's state management.
  * **(Optional) Voice Interaction**: Includes foundations for Text-to-Speech (TTS) output. Voice input (STT) can be integrated.

## 📋 Requirements

  * Python 3.8+
  * Streamlit (`pip install streamlit`) - Check `requirements.txt` for specific version.
  * Core Libraries: `pillow>=9.5.0`, `numpy`, `matplotlib`, `opencv-python>=4.7.0`.
  * AI & Agent Libraries: `langchain`, `langchain-core`, `langgraph`, `openai`.
  * An **OpenAI API Key** is required for the AI Assistant.
  * Potentially other API keys (e.g., Stability AI) if using features from `core/ai_services.py`.
  * `ffmpeg` might be needed if audio features are extended (check specific library requirements).

*See `requirements.txt` for a detailed list of Python packages and versions.*

## 🚀 Installation

1.  **Clone the Repository:**
    ```bash
    # Replace with your actual repository URL
    git clone https://github.com/yourusername/image-editor.git
    cd image-editor
    ```
2.  **Create and Activate Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API Keys:**
      * The AI Assistant requires an OpenAI API key.
      * **Method 1: Environment Variable:** Set the `OPENAI_API_KEY` environment variable in your system or terminal session.
      * **Method 2: Streamlit Secrets:** Create a file named `.streamlit/secrets.toml` in your project root and add the key:
        ```toml
        # .streamlit/secrets.toml
        OPENAI_API_KEY="sk-..."
        ```
        *(This method is recommended, especially for deployment).*
      * Add any other required keys (e.g., `STABILITY_API_KEY`) similarly if using those AI services.

## 🏃‍♀️ Running the Application

1.  Ensure your virtual environment is active and API keys are configured.
2.  Run the Streamlit app from the project's root directory:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your browser. Use the sidebar to navigate between the "Image Editor Pro" (manual editor) and "AI Assistant" pages.

## 🗣️ Usage Guide (AI Assistant)

1.  First, load an image using the **"Image Editor Pro"** page via the sidebar uploader.
2.  Navigate to the **"🤖 AI Assistant"** page using the sidebar.
3.  In the chat input, type your image editing commands using natural language (e.g., "Increase contrast by 0.5", "Rotate 90 degrees clockwise", "Apply a sharpen filter", "Remove the background").
4.  Press Enter.
5.  Observe the chat history for the conversation flow: your request, the assistant's status/tool usage messages, and the final response.
6.  The **image preview** on the AI Assistant page updates automatically after the agent successfully applies edits.
7.  (If TTS implemented) Use controls (likely in the sidebar) to manage voice output.

## 🔍 Project Structure

```text
image-editor/
├── app.py                     # Main Streamlit app entry point
├── pages/
│   └── 1_🤖_AI_Assistant.py    # UI & Logic for the AI Assistant page
├── agent/                     # LangGraph Agent implementation
│   ├── __init__.py
│   ├── agent_graph.py         # Defines & compiles the LangGraph graph, nodes, edges
│   ├── graph_state.py         # Defines the AgentState schema for the graph
│   └── tools.py               # Defines tools (schemas & mapping to implementations) for the LLM
├── core/                      # Core image processing & AI service logic
│   ├── __init__.py
│   ├── ai_services.py         # Functions calling external AI APIs (Stability, rembg)
│   ├── histogram.py           # Histogram generation logic
│   ├── image_io.py            # Image loading/saving utilities
│   └── processing.py          # Image manipulation functions (used as agent tools)
├── state/                     # Streamlit session state management
│   ├── __init__.py
│   └── session_state_manager.py # Helper functions for managing st.session_state
├── ui/                        # UI components (potentially reusable)
│   ├── __init__.py
│   └── interface.py           # UI elements primarily for the manual editor page
├── utils/                     # Utility functions and constants
│   ├── __init__.py
│   └── constants.py           # Shared constants (e.g., image types)
├── tests/                     # Unit/Integration tests (pytest setup included)
│   ├── ...                    # Test files mirroring project structure
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🧩 Architecture & Agent Flow

1. The user interacts with the Streamlit UI (`app.py` and `pages/1_🤖_AI_Assistant.py`).
2. Session state (`st.session_state`) holds the current images, widget values, and chat history, managed via helpers in `state/session_state_manager.py`.
3. On the AI Assistant page, user chat input is formatted and passed to the compiled LangGraph graph (`agent/agent_graph.py`).
4. The graph, using the AgentState schema (`agent/graph_state.py`), manages the execution flow:
   * An LLM node (using an OpenAI model) interprets the user request and decides the next action (respond directly or use a tool).
   * Conditional edges route the flow based on the LLM's decision.
   * Tool nodes execute specific actions defined in `agent/tools.py`. These tools call underlying image processing functions (`core/processing.py`, `core/ai_services.py`).
   * Image modifications happen by updating the image data within `st.session_state`.
   * Tool results are passed back into the graph.
5. The graph continues until the task is complete, culminating in a final response from the LLM node.
6. The Streamlit UI updates reactively based on changes in `st.session_state` (displaying new chat messages and the modified image).

## 🛠️ Development & Contributions

### Modifying the Agent

1. Edit `agent/agent_graph.py` to change:
   * The agent's prompt or LLM models
   * Node functions or routing logic
   * Graph structure and flow
2. Tools: Modify tool definitions (schemas) or how they map to implementation functions in `agent/tools.py`.
3. State: Adjust `agent/graph_state.py` if the information flowing through the graph needs to change.

### Adding New Agent Tools

1. Implement the core image processing logic in `core/processing.py` or `core/ai_services.py`.
2. Define the tool schema and its link to the implementation function within `agent/tools.py`.
3. Ensure the new tool is registered and available to the agent graph in `agent/agent_graph.py`.
4. (Optional but recommended) Update the agent's system prompt to make it aware of the new tool's capabilities.
5. Test thoroughly by asking the assistant to use the new tool.

### Contributing

* Found a bug? Open an issue on GitHub.
* Have an idea? Suggest it via GitHub Issues.
* Want to contribute code? Fork the repository and submit a pull request.

## 📄 License

[MIT License](LICENSE)

## 👏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pillow](https://pillow.readthedocs.io/)
- [LangChain & LangGraph](https://python.langchain.com/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenAI](https://openai.com/) (or other LLM/TTS providers)

---
Made with ❤️ by @josefdc @Esteban8482
