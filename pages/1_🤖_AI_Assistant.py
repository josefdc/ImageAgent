# streamlit_image_editor/pages/1_🤖_AI_Assistant.py
# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Generator # Keep for potential type hints

# --- Path Setup (Add Project Root) ---
# Ensure the main project directory is in the path
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        print(f"DEBUG (1_🤖_AI_Assistant.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (1_🤖_AI_Assistant.py): Failed during sys.path setup: {e}")

# --- Streamlit Page Config (MUST be FIRST Streamlit command) ---
import streamlit as st
st.set_page_config(page_title="AI Image Assistant", page_icon="🛠️", layout="wide")

# --- Third-Party Imports ---
from PIL import Image # For displaying image
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, SystemMessage, HumanMessage # Import message types

# --- Local Application Imports (Use Paths Relative to Project Root) ---
_AGENT_AVAILABLE = False
_COMPILED_GRAPH = None
try:
    # Imports relative to the project root added to sys.path
    from state.session_state_manager import initialize_session_state # Removed update_processed_image - not directly needed here
    # AgentState might not be strictly needed here unless used for type hints
    # from agent.graph_state import AgentState
    from agent.agent_graph import compiled_graph # Import the compiled graph instance
    from ui.interface import get_api_key_input # Assuming this exists and works
    _COMPILED_GRAPH = compiled_graph # Assign to local variable
    _AGENT_AVAILABLE = _COMPILED_GRAPH is not None
    print(f"DEBUG (1_🤖_AI_Assistant.py): Agent available: {_AGENT_AVAILABLE}")
except ImportError as e:
    st.error(f"Critical Error: Could not import application modules needed for the AI Assistant: {e}. Check console logs and project structure/sys.path.")
    print(f"ERROR (1_🤖_AI_Assistant.py): Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define initialize_session_state as a dummy if import fails
    def initialize_session_state(): pass
    # Define get_api_key_input as a dummy if needed
    def get_api_key_input(*args, **kwargs): st.warning("API Input unavailable due to import errors.")
except Exception as e:
     st.error(f"Critical Error during import or agent graph compilation: {e}")
     st.exception(e) # Show traceback in Streamlit app
     def initialize_session_state(): pass # Dummy function on other exceptions too
     def get_api_key_input(*args, **kwargs): st.warning("API Input unavailable due to import errors.")


# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- State Initialization ---
# Ensure all required keys exist in session state
initialize_session_state()
# Ensure chat history and thread ID exist specifically
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_graph_thread_id' not in st.session_state:
    st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
    logger.info(f"Initialized new graph thread ID: {st.session_state.current_graph_thread_id}")

# --- UI Elements ---
st.title("🛠️ AI Image Editing Assistant")
st.info("Chat with the AI to apply edits to the image currently loaded in the main editor page.")

# Display Current Image Preview
st.subheader("Current Image Preview")
current_processed_image = st.session_state.get('processed_image')

if current_processed_image and isinstance(current_processed_image, Image.Image):
    st.image(current_processed_image, width=300, caption="Image being edited by AI")
elif current_processed_image:
     st.warning(f"The object in 'processed_image' state is not a valid Image (type: {type(current_processed_image)}). Cannot display preview.")
else:
    st.warning("No image loaded. Please load an image on the 'Image Editor Pro' page first.")
    # Optionally disable chat if no image is loaded
    # Consider disabling chat_input instead of st.stop() for better UX
    # st.stop()

# Chat Interface Container
assistant_container = st.container(height=450, border=True)

# Sidebar for Controls
with st.sidebar:
    st.header("AI Assistant Controls")
    st.markdown("---")
    st.subheader("API Keys")
    st.caption("(Needed if not set as environment variables or Streamlit secrets)")
    # Use the imported UI function with error handling
    try:
        get_api_key_input("OpenAI", "OPENAI_API_KEY", "https://platform.openai.com/api-keys")
        get_api_key_input("RemoveBG", "REMOVEBG_API_KEY", "https://www.remove.bg/dashboard#api-key")
        # Example for hypothetical upscale key, adjust name if needed
        get_api_key_input("UpscaleService", "UPSCALE_API_KEY", "https://example.com/upscale_api") # Replace with actual if using one
    except NameError:
        st.warning("API Key input fields could not be loaded due to import errors.")
    st.markdown("---")

    # Clear Chat Button
    if st.button("Clear Chat & Reset Thread", type="primary"):
        st.session_state.chat_history = []
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Chat cleared. New graph thread ID: {st.session_state.current_graph_thread_id}")
        # Clear potentially stuck graph state keys for this thread
        if 'tool_invocation_request' in st.session_state: del st.session_state['tool_invocation_request']
        if 'pending_ui_updates' in st.session_state: del st.session_state['pending_ui_updates']
        st.success("Chat history cleared and thread reset.")
        time.sleep(0.5) # Brief pause
        st.rerun()

# --- Chat Logic ---

# 1. Display existing chat messages from history
with assistant_container:
    if not st.session_state.chat_history:
        st.caption("Chat history is empty. Ask the AI to edit the image!")
    else:
        if not isinstance(st.session_state.chat_history, list):
             logger.error("Chat history is not a list! Resetting.")
             st.session_state.chat_history = []
             st.warning("Chat history was corrupted and has been reset.")

        for i, msg_data in enumerate(st.session_state.chat_history):
            if isinstance(msg_data, dict) and "role" in msg_data and "content" in msg_data:
                role = msg_data["role"]
                content = msg_data.get("content", "")
                if isinstance(content, BaseMessage): content = content.content # Extract if BaseMessage stored

                if role in ["user", "assistant", "system", "tool"]:
                    avatar = {"user": "👤", "assistant": "✨", "system": "⚙️", "tool": "🛠️"}.get(role)
                    try:
                        with st.chat_message(name=role, avatar=avatar):
                            st.markdown(content)
                    except Exception as display_e:
                        logger.error(f"Failed displaying message {i} (role={role}): {display_e}")
                        st.error(f"Error displaying message: {str(content)[:100]}...")
            else:
                 logger.warning(f"Skipping invalid chat history item at index {i}: {msg_data}")

    # Display agent unavailable message inside container if needed
    if not _AGENT_AVAILABLE:
        with st.chat_message("assistant", avatar="⚠️"):
            st.warning("AI Assistant is currently unavailable. Please check configuration and console logs.")

# 2. Get user input via chat_input
chat_input_placeholder = "e.g., 'Make it 50% brighter', 'Apply blur filter', 'Remove the background'"
# Disable input if agent is unavailable OR if no image is loaded
chat_disabled = not _AGENT_AVAILABLE or not current_processed_image
user_prompt = st.chat_input(
    chat_input_placeholder,
    key="ai_chat_input", # Consistent key
    disabled=chat_disabled
)

# 3. Process user input if provided and agent is ready
if user_prompt and _AGENT_AVAILABLE and _COMPILED_GRAPH:
    logger.info(f"User prompt received: {user_prompt}")

    # Add user message to session state history immediately
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Display user message in the chat container
    with assistant_container:
         with st.chat_message("user", avatar="👤"):
              st.markdown(user_prompt)

    # Prepare input for the LangGraph agent stream
    graph_input = {"messages": [HumanMessage(content=user_prompt)]}
    config = {"configurable": {"thread_id": st.session_state.current_graph_thread_id}}
    logger.info(f"Streaming graph for thread: {config['configurable']['thread_id']}")

    # Variables to store results from the stream
    final_assistant_message_content = None
    ui_updates_from_run = None
    final_state_from_run = None

    try:
        # Stream the graph execution and display intermediate/final responses
        with assistant_container:
            with st.chat_message("assistant", avatar="✨"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Thinking... ⏳")
                full_response_stream_display = ""

                # Use stream_mode="values" to get the full AgentState at each step
                for step_output in _COMPILED_GRAPH.stream(graph_input, config, stream_mode="values"):
                    # step_output is the full AgentState dictionary
                    messages = step_output.get("messages", [])
                    if not messages: continue

                    last_message = messages[-1]
                    logger.debug(f"Stream step output - last message type: {type(last_message).__name__}")

                    # Extract content for display during streaming
                    current_chunk = ""
                    is_final_ai_text = False # Flag if this step contains the final text
                    if isinstance(last_message, AIMessage):
                        if last_message.tool_calls:
                            tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
                            current_chunk = f"🛠️ Calling tool(s): `{', '.join(tool_names)}`..."
                        else:
                            current_chunk = last_message.content
                            is_final_ai_text = True # This is likely the final text response
                    elif isinstance(last_message, ToolMessage):
                        tool_name_guess = getattr(last_message, 'name', 'tool')
                        current_chunk = f"⚙️ `{tool_name_guess}` result: {last_message.content}"
                    elif isinstance(last_message, SystemMessage):
                         current_chunk = f"⚠️ System: {last_message.content}"
                    # Ignore HumanMessage during streaming output

                    # Update the placeholder incrementally
                    if current_chunk:
                        full_response_stream_display = current_chunk # Show latest status/response
                        response_placeholder.markdown(full_response_stream_display + " ▌")

                    # Store the final text response content if identified
                    if is_final_ai_text:
                         final_assistant_message_content = current_chunk

                    # Capture UI updates from the state of this step
                    step_ui_updates = step_output.get("pending_ui_updates")
                    if step_ui_updates:
                        ui_updates_from_run = step_ui_updates
                        logger.debug(f"Captured pending UI updates from graph step: {ui_updates_from_run}")

                    # Store the very last state dictionary seen
                    final_state_from_run = step_output

                # Clean up the placeholder after the stream ends
                if full_response_stream_display:
                     response_placeholder.markdown(full_response_stream_display)
                else:
                     response_placeholder.markdown("Processing complete.")

        # --- Post-Stream Processing ---

        # 1. Add final assistant text response to chat history
        if final_assistant_message_content:
             # Avoid adding duplicates if the last message is already identical
             if not st.session_state.chat_history or \
                st.session_state.chat_history[-1].get("role") != "assistant" or \
                st.session_state.chat_history[-1].get("content") != final_assistant_message_content:
                 st.session_state.chat_history.append({"role": "assistant", "content": final_assistant_message_content})
                 logger.debug("Added final assistant text message to chat history.")

        # 2. Apply pending UI updates captured during the stream
        if ui_updates_from_run and isinstance(ui_updates_from_run, dict):
            logger.info(f"Applying UI updates captured from graph run: {ui_updates_from_run}")
            for key, value in ui_updates_from_run.items():
                if key in st.session_state:
                    try:
                         st.session_state[key] = value
                         logger.debug(f"Updated st.session_state['{key}'] = {value}")
                    except Exception as ui_update_e:
                         logger.error(f"Failed to apply UI update st.session_state['{key}'] = {value}: {ui_update_e}")
                         st.warning(f"Could not apply UI update for '{key}'.")
                else:
                    logger.warning(f"Attempted to update non-existent session state key from UI updates: '{key}'")
            # It's generally safe NOT to explicitly clear pending_ui_updates from the
            # LangGraph checkpointer state here, as the execute_tool_and_update node
            # should clear it on its next run. Explicitly clearing might require:
            # graph.update_state(config, {"pending_ui_updates": None})

        # 3. Rerun the page to reflect all changes
        logger.info("Rerunning Streamlit page after AI interaction and potential UI updates.")
        # Use a shorter delay or remove if updates seem reliable
        # time.sleep(0.1)
        st.rerun()

    except Exception as e:
        logger.error(f"An error occurred during AI assistant interaction: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")
        error_msg_for_chat = f"⚠️ Error processing your request. Please check logs or try again."
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg_for_chat})
        st.rerun() # Rerun to display the error

elif user_prompt and not _AGENT_AVAILABLE:
     # Handle case where user typed but agent isn't ready
     st.warning("The AI Assistant is not available. Cannot process the request.")
     st.session_state.chat_history.append({"role": "user", "content": user_prompt})
     st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I am currently unavailable. Please check the configuration."})
     st.rerun()