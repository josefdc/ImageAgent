# 🖼️ AI-Powered Image Editor with LangGraph Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11%2B-FF4B4B)](https://streamlit.io/)
[![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-yellow)](https://pillow.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-blueviolet)](https://python.langchain.com/)

An intelligent image editor built with Streamlit and Pillow, featuring a conversational AI agent powered by **LangGraph**. Interact with the editor using natural language through text or voice commands, leveraging the agent's ability to use image processing functions as tools.

![Screenshot of application](https://via.placeholder.com/800x450.png?text=AI+Image+Editor+Screenshot)

## ✨ Features

### Core Image Editing (Manual)
- **Basic Adjustments**: Brightness, Contrast, Rotation.
- **Advanced Operations**: Zoom/Crop, Binarization, Negative, Channel Manipulation, Highlight Zones.
- **Image Merging**: Alpha Blend two images with automatic resizing.
- **Analysis**: RGB Histogram, Luminosity Analysis.
- **Export**: Download processed images in PNG format.

### 🤖 AI Assistant (Powered by LangGraph)
- **Conversational Control**: Use natural language (text/voice) to command image edits.
- **LangGraph Agent**: A stateful agent built with LangGraph manages the conversation and tool execution flow.
- **Intelligent Tool Use**: The agent analyzes requests and selects the appropriate image processing functions (defined in `core/processing.py`) as tools to achieve the desired outcome.
    - *Example*: Saying "Make the image brighter" triggers the agent to call the `adjust_brightness` tool with the appropriate parameters.
- **Multi-Step Operations**: The agent can handle sequences of operations within a conversation (e.g., "Rotate 90 degrees then increase contrast").
- **Contextual Awareness**: Maintains conversation history within a session using the LangGraph state (`agent/graph_state.py`).
- **Voice Interaction**: Optional voice output (TTS) for assistant responses and potential for voice input (STT) in the future.

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pillow (PIL)
- NumPy
- Matplotlib
- LangChain (`langchain`, `langchain-core`, `langgraph`)
- OpenAI (`openai` Python package - or other LLM/TTS provider)
- (Any other specific dependencies for your chosen LLM or tools)

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-editor.git
    cd image-editor
    ```
2.  Set up a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Keys**: Configure necessary API keys (e.g., `OPENAI_API_KEY`) as environment variables or in a `.env` file.

## 🏃‍♀️ Running the Application

```bash
streamlit run app.py
```
Navigate to the "🗣️ Voice Assistant" page in the sidebar.

## 🗣️ Usage Guide

1.  Upload an image via the main editor page.
2.  Go to the **"🗣️ Voice Assistant"** page.
3.  Type your request (e.g., "Increase contrast by 0.5", "Convert to grayscale and show the histogram", "What tools can you use?").
4.  Observe the chat history:
    *   Your request (User).
    *   Agent's thinking/status messages (Assistant).
    *   Tool execution details (Assistant, often formatted as code/result).
    *   Agent's final response (Assistant).
5.  The image preview updates based on successful tool executions.
6.  Enable/disable voice output and customize the voice via the sidebar.

## 🔍 Project Structure

```
image-editor/
├── app.py                 # Main Streamlit app
├── pages/
│   └── 2_🗣️_Voice_Assistant.py # UI for the AI Assistant
├── agent/                 # LangGraph Agent implementation
│   ├── __init__.py
│   ├── agent_graph.py     # Defines and compiles the LangGraph graph
│   └── graph_state.py     # Defines the AgentState for the graph
├── core/                  # Core image processing logic
│   ├── __init__.py
│   ├── histogram.py
│   ├── image_io.py
│   └── processing.py      # Image functions exposed as TOOLS to the agent
├── state/                 # Session state management
│   └── session_state_manager.py
├── ui/                    # UI components
│   ├── interface.py       # Manual editor UI elements
│   └── voice/
│       └── voice_manager.py # TTS handling
├── utils/                 # Utilities
│   └── constants.py
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🧩 Architecture & Agent Interaction

1.  The user interacts via the chat interface in `2_🗣️_Voice_Assistant.py`.
2.  User input is formatted as a `HumanMessage` and sent to the compiled **LangGraph graph** (`agent.agent_graph.compiled_graph`).
3.  The graph processes the input based on its defined nodes and edges:
    *   An LLM node likely determines user intent and required tools.
    *   A conditional edge routes to specific **tool-calling nodes** or directly to a response generation node.
4.  Tool-calling nodes execute functions from `core/processing.py` (e.g., `adjust_brightness`, `rotate_image`). These functions modify the image stored in session state.
5.  The results of tool execution are formatted as `ToolMessage` objects and fed back into the graph.
6.  The graph continues execution, potentially calling more tools or generating a final `AIMessage` response for the user.
7.  The state (`AgentState`) is updated throughout the graph's execution, maintaining conversation history and intermediate results.
8.  The UI (`2_🗣️_Voice_Assistant.py`) displays messages (User, AI, Tool) and updates the image preview based on changes in session state.

## 🛠️ Development

### Modifying the Agent (LangGraph)

1.  Edit `agent/agent_graph.py` to change:
    *   The agent's prompt or LLM.
    *   The graph structure (nodes, edges, conditional logic).
    *   Tool definitions and how they map to functions.
2.  Adjust `agent/graph_state.py` if the required state changes.
3.  Ensure tools defined in the graph correctly reference functions in `core/processing.py`.

### Adding New Image Processing Tools for the Agent

1.  Implement the image processing logic in `core/processing.py`. The function should ideally take the current image (from session state) and parameters, returning the modified image.
2.  Define a new tool (e.g., using LangChain's `@tool` decorator or manual definition) that wraps your function from step 1. Place this definition where your other tools are defined (likely near or within `agent/agent_graph.py` or a dedicated tools file).
3.  Add the new tool to the agent's available tools within the LangGraph definition (`agent/agent_graph.py`).
4.  Update the agent's system prompt (if applicable) to make it aware of the new tool and its purpose.
5.  Test by asking the assistant to use the new functionality.

## 📄 License

[MIT License](LICENSE)

## 👏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pillow](https://pillow.readthedocs.io/)
- [LangChain & LangGraph](https://python.langchain.com/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenAI](https://openai.com/) (or other LLM/TTS providers)

---
Made with ❤️ by @josefdc @Esteban8482
```# filepath: /home/josefdc/projects/image-editor/README.md
# 🖼️ AI-Powered Image Editor with LangGraph Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11%2B-FF4B4B)](https://streamlit.io/)
[![Pillow](https://img.shields.io/badge/Pillow-9.0%2B-yellow)](https://pillow.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-blueviolet)](https://python.langchain.com/)

An intelligent image editor built with Streamlit and Pillow, featuring a conversational AI agent powered by **LangGraph**. Interact with the editor using natural language through text or voice commands, leveraging the agent's ability to use image processing functions as tools.

![Screenshot of application](https://via.placeholder.com/800x450.png?text=AI+Image+Editor+Screenshot)

## ✨ Features

### Core Image Editing (Manual)
- **Basic Adjustments**: Brightness, Contrast, Rotation.
- **Advanced Operations**: Zoom/Crop, Binarization, Negative, Channel Manipulation, Highlight Zones.
- **Image Merging**: Alpha Blend two images with automatic resizing.
- **Analysis**: RGB Histogram, Luminosity Analysis.
- **Export**: Download processed images in PNG format.

### 🤖 AI Assistant (Powered by LangGraph)
- **Conversational Control**: Use natural language (text/voice) to command image edits.
- **LangGraph Agent**: A stateful agent built with LangGraph manages the conversation and tool execution flow.
- **Intelligent Tool Use**: The agent analyzes requests and selects the appropriate image processing functions (defined in `core/processing.py`) as tools to achieve the desired outcome.
    - *Example*: Saying "Make the image brighter" triggers the agent to call the `adjust_brightness` tool with the appropriate parameters.
- **Multi-Step Operations**: The agent can handle sequences of operations within a conversation (e.g., "Rotate 90 degrees then increase contrast").
- **Contextual Awareness**: Maintains conversation history within a session using the LangGraph state (`agent/graph_state.py`).
- **Voice Interaction**: Optional voice output (TTS) for assistant responses and potential for voice input (STT) in the future.

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pillow (PIL)
- NumPy
- Matplotlib
- LangChain (`langchain`, `langchain-core`, `langgraph`)
- OpenAI (`openai` Python package - or other LLM/TTS provider)
- (Any other specific dependencies for your chosen LLM or tools)

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-editor.git
    cd image-editor
    ```
2.  Set up a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Keys**: Configure necessary API keys (e.g., `OPENAI_API_KEY`) as environment variables or in a `.env` file.

## 🏃‍♀️ Running the Application

```bash
streamlit run app.py
```
Navigate to the "🗣️ Voice Assistant" page in the sidebar.

## 🗣️ Usage Guide

1.  Upload an image via the main editor page.
2.  Go to the **"🗣️ Voice Assistant"** page.
3.  Type your request (e.g., "Increase contrast by 0.5", "Convert to grayscale and show the histogram", "What tools can you use?").
4.  Observe the chat history:
    *   Your request (User).
    *   Agent's thinking/status messages (Assistant).
    *   Tool execution details (Assistant, often formatted as code/result).
    *   Agent's final response (Assistant).
5.  The image preview updates based on successful tool executions.
6.  Enable/disable voice output and customize the voice via the sidebar.

## 🔍 Project Structure

```
image-editor/
├── app.py                 # Main Streamlit app
├── pages/
│   └── 2_🗣️_Voice_Assistant.py # UI for the AI Assistant
├── agent/                 # LangGraph Agent implementation
│   ├── __init__.py
│   ├── agent_graph.py     # Defines and compiles the LangGraph graph
│   └── graph_state.py     # Defines the AgentState for the graph
├── core/                  # Core image processing logic
│   ├── __init__.py
│   ├── histogram.py
│   ├── image_io.py
│   └── processing.py      # Image functions exposed as TOOLS to the agent
├── state/                 # Session state management
│   └── session_state_manager.py
├── ui/                    # UI components
│   ├── interface.py       # Manual editor UI elements
│   └── voice/
│       └── voice_manager.py # TTS handling
├── utils/                 # Utilities
│   └── constants.py
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🧩 Architecture & Agent Interaction

1.  The user interacts via the chat interface in `2_🗣️_Voice_Assistant.py`.
2.  User input is formatted as a `HumanMessage` and sent to the compiled **LangGraph graph** (`agent.agent_graph.compiled_graph`).
3.  The graph processes the input based on its defined nodes and edges:
    *   An LLM node likely determines user intent and required tools.
    *   A conditional edge routes to specific **tool-calling nodes** or directly to a response generation node.
4.  Tool-calling nodes execute functions from `core/processing.py` (e.g., `adjust_brightness`, `rotate_image`). These functions modify the image stored in session state.
5.  The results of tool execution are formatted as `ToolMessage` objects and fed back into the graph.
6.  The graph continues execution, potentially calling more tools or generating a final `AIMessage` response for the user.
7.  The state (`AgentState`) is updated throughout the graph's execution, maintaining conversation history and intermediate results.
8.  The UI (`2_🗣️_Voice_Assistant.py`) displays messages (User, AI, Tool) and updates the image preview based on changes in session state.

## 🛠️ Development

### Modifying the Agent (LangGraph)

1.  Edit `agent/agent_graph.py` to change:
    *   The agent's prompt or LLM.
    *   The graph structure (nodes, edges, conditional logic).
    *   Tool definitions and how they map to functions.
2.  Adjust `agent/graph_state.py` if the required state changes.
3.  Ensure tools defined in the graph correctly reference functions in `core/processing.py`.

### Adding New Image Processing Tools for the Agent

1.  Implement the image processing logic in `core/processing.py`. The function should ideally take the current image (from session state) and parameters, returning the modified image.
2.  Define a new tool (e.g., using LangChain's `@tool` decorator or manual definition) that wraps your function from step 1. Place this definition where your other tools are defined (likely near or within `agent/agent_graph.py` or a dedicated tools file).
3.  Add the new tool to the agent's available tools within the LangGraph definition (`agent/agent_graph.py`).
4.  Update the agent's system prompt (if applicable) to make it aware of the new tool and its purpose.
5.  Test by asking the assistant to use the new functionality.

## 📄 License

[MIT License](LICENSE)

## 👏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pillow](https://pillow.readthedocs.io/)
- [LangChain & LangGraph](https://python.langchain.com/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenAI](https://openai.com/) (or other LLM/TTS providers)

Made with ❤️ by @josefdc @Esteban84
