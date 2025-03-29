# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Generator # Keep for potential type hints

# --- Path Setup (Add Project Root) ---
try:
    # Project root is the parent directory of the 'pages' directory
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
    # Imports are now relative to the project root added to sys.path
    from state.session_state_manager import initialize_session_state
    # AgentState might not be strictly needed here unless used for type hints
    # from agent.graph_state import AgentState
    from agent.agent_graph import compiled_graph # Import the compiled graph instance
    from ui.interface import get_api_key_input # Assuming this exists
    _COMPILED_GRAPH = compiled_graph # Assign to local variable
    _AGENT_AVAILABLE = _COMPILED_GRAPH is not None
    print(f"DEBUG (1_🤖_AI_Assistant.py): Agent available: {_AGENT_AVAILABLE}")
except ImportError as e:
    st.error(f"Critical Error: Could not import application modules needed for the AI Assistant: {e}. Check console logs and project structure/sys.path.")
    print(f"ERROR (1_🤖_AI_Assistant.py): Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define initialize_session_state as a dummy if import fails, to prevent NameError later
    # This allows the rest of the page UI to potentially load, showing the error message.
    def initialize_session_state(): pass
except Exception as e:
     st.error(f"Critical Error during import or agent graph compilation: {e}")
     st.exception(e) # Show traceback in Streamlit app
     def initialize_session_state(): pass # Dummy function on other exceptions too

# --- Logging Setup ---
# Configure logger for this page specifically if needed, or rely on root logger
logger = logging.getLogger(__name__)
# Basic config if no handlers are present (e.g., if run standalone)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- State Initialization ---
# Ensure all required keys exist in session state
# This will now call the real or dummy function depending on import success
initialize_session_state()
# Ensure chat history and thread ID exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_graph_thread_id' not in st.session_state:
    # Initialize with a unique ID based on time
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
    # st.stop() # Or just disable the input later

# Chat Interface Container
assistant_container = st.container(height=450, border=True) # Added border for clarity

# Sidebar for Controls
with st.sidebar:
    st.header("AI Assistant Controls")
    st.markdown("---")
    st.subheader("API Keys")
    st.caption("(Needed if not set as environment variables or Streamlit secrets)")
    # Example using the imported UI function
    # Add safety check in case get_api_key_input also failed to import
    try:
        get_api_key_input("OpenAI", "OPENAI_API_KEY", "https://platform.openai.com/api-keys")
        get_api_key_input("RemoveBG", "REMOVEBG_API_KEY", "https://www.remove.bg/dashboard#api-key")
        get_api_key_input("ClipDrop", "CLIPDROP_API_KEY", "https://clipdrop.co/apis")
    except NameError:
        st.warning("API Key input fields could not be loaded due to import errors.")
    st.markdown("---")

    if st.button("Clear Chat & Reset Thread", type="primary"):
        st.session_state.chat_history = []
        # Generate a new thread ID to start fresh conversation context
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Chat cleared. New graph thread ID: {st.session_state.current_graph_thread_id}")
        # Clear any potentially stuck tool request from previous runs (though graph should handle this)
        if 'tool_invocation_request' in st.session_state:
             del st.session_state['tool_invocation_request']
        st.success("Chat history cleared and thread reset.")
        time.sleep(1) # Brief pause to show message
        st.rerun()

# --- Chat Logic ---

# 1. Display existing chat messages from history
with assistant_container:
    if not st.session_state.chat_history:
        st.caption("Chat history is empty. Ask the AI to edit the image!")
    else:
        # Ensure chat_history is a list (it should be due to initialization)
        if not isinstance(st.session_state.chat_history, list):
             logger.error("Chat history is not a list! Resetting.")
             st.session_state.chat_history = []
             st.warning("Chat history was corrupted and has been reset.")

        for i, msg_data in enumerate(st.session_state.chat_history):
            # Validate message structure
            if isinstance(msg_data, dict) and "role" in msg_data and "content" in msg_data:
                role = msg_data["role"]
                content = msg_data["content"]
                # Handle potential BaseMessage objects stored directly (though shouldn't happen with current logic)
                if isinstance(content, BaseMessage): content = content.content

                if role in ["user", "assistant", "system", "tool"]: # Allow tool role display
                    avatar = {"user": "👤", "assistant": "✨", "system": "⚙️", "tool": "🛠️"}.get(role)
                    try:
                        with st.chat_message(name=role, avatar=avatar):
                            st.markdown(content) # Display content as Markdown
                    except Exception as display_e:
                        logger.error(f"Failed to display message {i} (role={role}): {display_e}")
                        st.error(f"Error displaying message: {content[:100]}...")
            else:
                 logger.warning(f"Skipping invalid chat history item at index {i}: {msg_data}")
                 # Optionally display a warning in the chat UI
                 # st.warning(f"Ignoring corrupted message at index {i}")

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
    key="ai_chat_input",
    disabled=chat_disabled
)

# 3. Process user input if provided and agent is ready
if user_prompt and _AGENT_AVAILABLE and _COMPILED_GRAPH:
    logger.info(f"User prompt received: {user_prompt}")

    # Add user message to session state history immediately for UI update
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Display user message in the chat container
    with assistant_container:
         with st.chat_message("user", avatar="👤"):
              st.markdown(user_prompt)

    # Prepare input for the LangGraph agent stream
    # The graph's checkpointer (MemorySaver) handles loading the full history for the thread_id
    graph_input = {"messages": [HumanMessage(content=user_prompt)]} # Pass only the new message
    config = {"configurable": {"thread_id": st.session_state.current_graph_thread_id}}
    logger.info(f"Streaming graph for thread: {config['configurable']['thread_id']}")

    # Stream the graph execution and display intermediate/final responses
    try:
        with assistant_container:
            with st.chat_message("assistant", avatar="✨"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Thinking... ⏳")
                full_response_content = ""
                final_assistant_message_content = None # Store the actual final AI message text

                # Use stream_mode="values" to get the full state at each step
                for step_output in _COMPILED_GRAPH.stream(graph_input, config, stream_mode="values"):
                    # step_output is the full AgentState dictionary for the current step
                    messages = step_output.get("messages", [])
                    if not messages: continue # Skip if no messages in this step's state

                    last_message = messages[-1] # Get the newest message from the state
                    logger.debug(f"Stream step output - last message type: {type(last_message).__name__}")

                    # Determine what to display based on the last message type
                    current_chunk = ""
                    if isinstance(last_message, AIMessage):
                        if last_message.tool_calls:
                            # Display that a tool is being called
                            tool_names = [tc['name'] for tc in last_message.tool_calls]
                            current_chunk = f"🛠️ Calling tool(s): `{', '.join(tool_names)}`..."
                            final_assistant_message_content = None # Reset final content as this isn't the final text response
                        else:
                            # Display the AI's text response content
                            current_chunk = last_message.content
                            final_assistant_message_content = current_chunk # Store this as potential final response
                    elif isinstance(last_message, ToolMessage):
                        # Display the result of a tool execution
                        # Extract tool name if possible (might require changes in how ToolMessage is constructed if name isn't standard)
                        tool_name_guess = getattr(last_message, 'name', 'unknown_tool') # LangChain ToolMessage might not have 'name' directly
                        current_chunk = f"⚙️ Tool `{tool_name_guess}` result: {last_message.content}"
                        final_assistant_message_content = None # Tool result isn't the final text response
                    elif isinstance(last_message, SystemMessage):
                         # Display system messages (e.g., errors from graph)
                         current_chunk = f"⚠️ System: {last_message.content}"
                         final_assistant_message_content = None # Errors aren't the final text response

                    # Update the placeholder with the latest chunk + thinking indicator
                    if current_chunk:
                        full_response_content = current_chunk # Replace previous content for now
                        response_placeholder.markdown(full_response_content + " ▌")

                # Final update after stream ends
                if full_response_content:
                     response_placeholder.markdown(full_response_content) # Remove the thinking indicator
                else:
                     response_placeholder.markdown("Processing complete. (No text response)") # Fallback

        # Add the *final assistant text message* (if any) to the chat history for persistence
        if final_assistant_message_content:
             # Check if it's different from the last message already in history (to avoid duplicates on rerun)
             if not st.session_state.chat_history or \
                st.session_state.chat_history[-1].get("role") != "assistant" or \
                st.session_state.chat_history[-1].get("content") != final_assistant_message_content:
                 st.session_state.chat_history.append({"role": "assistant", "content": final_assistant_message_content})
                 logger.debug("Added final assistant text message to chat history.")

        # Rerun the page to reflect potential image changes and update history display cleanly
        logger.info("Rerunning Streamlit page after AI interaction.")
        time.sleep(0.1) # Small delay might help ensure state updates settle
        st.rerun()

    except Exception as e:
        logger.error(f"An error occurred during AI assistant interaction: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")
        # Add error message to chat history for visibility
        error_msg_for_chat = f"⚠️ Error processing your request: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg_for_chat})
        st.rerun() # Rerun to display the error in the chat history

elif user_prompt and not _AGENT_AVAILABLE:
     # Handle case where user typed something but agent isn't ready
     st.warning("The AI Assistant is not available. Cannot process the request.")
     # Optionally add user message and warning to history
     st.session_state.chat_history.append({"role": "user", "content": user_prompt})
     st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I am currently unavailable."})
     st.rerun()