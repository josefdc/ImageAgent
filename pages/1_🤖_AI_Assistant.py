# streamlit_image_editor/pages/1_🤖_AI_Assistant.py
# Implements the AI Assistant chat interface page using Streamlit and LangGraph.

# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Generator, Optional

# --- Path Setup (Add Project Root) ---
# Ensures local modules can be imported when Streamlit runs this page script
try:
    # Assumes this file is in project_root/pages/
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        # Use print for early debugging as logger might not be configured yet
        print(f"DEBUG (1_🤖_AI_Assistant.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (1_🤖_AI_Assistant.py): Failed during sys.path setup: {e}")

# --- Streamlit Page Config (MUST be FIRST Streamlit command) ---
import streamlit as st
st.set_page_config(
    page_title="AI Image Assistant",
    page_icon="✨", # Using sparkle icon
    layout="wide"
)

# --- Third-Party Imports ---
from PIL import Image # For displaying image preview
# Import specific message types needed
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- Local Application Imports (Use Paths Relative to Project Root) ---
# Wrap imports in try-except for robustness, especially during development
_AGENT_AVAILABLE = False
_COMPILED_GRAPH = None
_ui_module_loaded = False
_state_module_loaded = False

try:
    # Imports are relative to the project root added to sys.path
    from state.session_state_manager import initialize_session_state
    _state_module_loaded = True
    from agent.agent_graph import compiled_graph # Import the compiled graph instance
    # Import UI helper only if needed within this file directly
    from ui.interface import get_api_key_input as display_api_key_input # Use alias
    _ui_module_loaded = True

    _COMPILED_GRAPH = compiled_graph # Assign to local variable for use
    _AGENT_AVAILABLE = _COMPILED_GRAPH is not None
    print(f"DEBUG (1_🤖_AI_Assistant.py): Agent available check: {_AGENT_AVAILABLE}")
except ImportError as e:
    # Use st.error for visibility in the app if Streamlit has loaded
    st.error(f"Critical Error: Could not import application modules: {e}. Check console logs and project structure.")
    print(f"ERROR (1_🤖_AI_Assistant.py): Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define dummy functions if imports fail to prevent NameErrors later
    if not _state_module_loaded:
        def initialize_session_state(): print("CRITICAL: State module failed to load.")
    if not _ui_module_loaded:
        def display_api_key_input(*args, **kwargs): st.warning("API Input unavailable due to import errors.")
except Exception as e:
     st.error(f"Critical Error during import or agent graph compilation: {e}")
     st.exception(e) # Show traceback in Streamlit app
     # Define dummies again
     if not _state_module_loaded:
         def initialize_session_state(): print("CRITICAL: State module failed to load after exception.")
     if not _ui_module_loaded:
         def display_api_key_input(*args, **kwargs): st.warning("API Input unavailable due to other errors.")

# --- Logging Setup ---
# Configure logger for this specific page module
logger = logging.getLogger(__name__)
# Set level and format if not already configured by the main app or another module
if not logger.hasHandlers():
     _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
     # Use a more detailed format for debugging agent interactions
     _log_format = '%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)'
     logging.basicConfig(level=_log_level, format=_log_format)
     logger.info(f"Logger initialized for {__name__} with level {_log_level}")

# --- State Initialization ---
# Ensure all required session state keys exist, especially chat-related ones
try:
    initialize_session_state()
    # Explicitly ensure chat history and thread ID exist after initialization
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'current_graph_thread_id' not in st.session_state:
        # Use a more robust default thread ID if needed, or generate one
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Initialized new graph thread ID: {st.session_state.current_graph_thread_id}")
    if 'assistant_thinking' not in st.session_state: st.session_state.assistant_thinking = False
    if 'example_prompt_clicked' not in st.session_state: st.session_state.example_prompt_clicked = None
except NameError:
    # This happens if initialize_session_state failed to import
    st.error("Failed to initialize session state due to previous import errors. App cannot function.")
    # Stop execution if state cannot be initialized
    st.stop()
except Exception as state_init_e:
     st.error(f"An unexpected error occurred during state initialization: {state_init_e}")
     st.exception(state_init_e)
     st.stop()


# --- Helper Function for Displaying Chat ---
def display_chat_history(container):
    """Renders the chat history within the provided Streamlit container."""
    with container:
        chat_history = st.session_state.get('chat_history', [])
        # Validate chat_history type
        if not isinstance(chat_history, list):
            logger.error("Chat history state is not a list! Resetting.")
            st.session_state.chat_history = []
            st.warning("Chat history was corrupted and has been reset.")
            chat_history = []

        if not chat_history:
            st.caption("Chat history is empty. Ask the AI to edit the image!")
            return

        # Iterate through stored messages
        for i, msg_data in enumerate(chat_history):
            # Basic validation of message structure
            # Allow content to be None if tool_calls exist (for AIMessage)
            if isinstance(msg_data, dict) and "role" in msg_data:
                role = msg_data["role"]
                content = msg_data.get("content") # Can be None
                tool_calls = msg_data.get("tool_calls") # Check for tool calls

                # Ensure content is string type for display if it exists
                display_content = ""
                if content is not None:
                    if isinstance(content, BaseMessage): display_content = str(content.content)
                    elif not isinstance(content, str): display_content = str(content)
                    else: display_content = content

                # Assign avatar based on role
                avatar = {"user": "👤", "assistant": "✨", "system": "⚙️", "tool": "🛠️"}.get(role, "❓")

                try:
                    # Display message using st.chat_message
                    with st.chat_message(name=role, avatar=avatar):
                        # Apply specific formatting
                        if role == "tool":
                            st.markdown(f"```\nTool Result:\n{display_content}\n```")
                        elif role == "system": # General system messages
                             st.warning(display_content, icon="⚙️")
                        elif role == "assistant" and tool_calls:
                             # Display AI message that contains tool calls (might have text too)
                             if display_content: st.markdown(display_content)
                             tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
                             st.markdown(f"*Assistant decided to use tool(s): `{', '.join(tool_names)}`*")
                        else: # User and normal Assistant messages
                            st.markdown(display_content)
                except Exception as display_e:
                    logger.error(f"Failed displaying message index {i} (role={role}): {display_e}")
                    try: st.error(f"Error displaying message #{i+1}...") # Minimal error in UI
                    except: pass # Prevent error loops
            else:
                 logger.warning(f"Skipping invalid chat history item at index {i}: Type={type(msg_data)}, Value={str(msg_data)[:100]}")

        # Display agent unavailable message at the end if needed
        if not _AGENT_AVAILABLE:
            with st.chat_message("assistant", avatar="⚠️"):
                st.warning("AI Assistant is currently unavailable. Check configuration/logs.")

# --- Helper Function to Add Message (Updated) ---
def add_message_to_history(role: str, content: Any, tool_call_id: Optional[str] = None, tool_calls: Optional[List[Dict]] = None):
    """Adds a message dictionary to the chat history state if content or tool_calls are valid."""
    # Allow messages with only tool_calls (AIMessage) or only content
    if content is None and not tool_calls:
        logger.warning(f"Attempted to add message with empty content/tool_calls for role {role}. Skipping.")
        return

    # Convert content to string if it exists and isn't already
    msg_content_str = None
    if content is not None:
        if isinstance(content, BaseMessage): msg_content_str = str(content.content)
        elif isinstance(content, str): msg_content_str = content
        else:
            try: msg_content_str = str(content)
            except: msg_content_str = "[Error converting content to string]"
            logger.warning(f"Converted non-string content (Type: {type(content)}) to string for chat history.")

    message_dict = {"role": role, "content": msg_content_str} # Content can be None here
    if tool_call_id:
        message_dict["tool_call_id"] = tool_call_id
    if tool_calls:
        # Ensure tool_calls are serializable if needed, though they often are dicts
        message_dict["tool_calls"] = tool_calls

    # Ensure chat_history exists and is a list
    if 'chat_history' not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    st.session_state.chat_history.append(message_dict)
    logger.debug(f"Added message to history: Role={role}, Content='{str(msg_content_str)[:50] if msg_content_str else 'None'}...', ToolCallId={tool_call_id}, HasToolCalls={bool(tool_calls)}")


# --- Helper Function to Get Image Status ---
def get_image_status_message() -> str:
    """Checks session state and returns a status message for the system prompt."""
    img = st.session_state.get('processed_image')
    if isinstance(img, Image.Image):
        try:
            return f"CONTEXT: An image ({img.width}x{img.height}, mode {img.mode}) is currently loaded and ready for editing. You can use tools like adjust_brightness, apply_filter, etc. directly."
        except Exception: # Handle potential issues accessing image properties
             return "CONTEXT: An image is currently loaded and ready for editing. You can use tools like adjust_brightness, apply_filter, etc. directly."
    else:
        return "CONTEXT: No image is currently loaded. You must ask the user to upload one before attempting any image operations."

# --- Page Title and Description ---
st.title("✨ AI Image Editing Assistant ✨")
st.caption("Use natural language to apply edits to the image currently loaded on the main editor page.")
st.divider()

# --- Main Layout: Image Preview + Chat Interface ---
col_preview, col_chat = st.columns([1, 2])

with col_preview:
    st.subheader("Current Image")
    current_processed_image = st.session_state.get('processed_image')
    image_available = current_processed_image and isinstance(current_processed_image, Image.Image)

    if image_available:
        img_display_col, img_info_col = st.columns([3, 2])
        with img_display_col:
            st.image(current_processed_image, use_container_width=True, caption="Image affected by AI")
        with img_info_col:
             st.caption("**Image Info:**")
             st.caption(f"Size: {current_processed_image.width}x{current_processed_image.height}")
             st.caption(f"Mode: {current_processed_image.mode}")
             filename = st.session_state.get('image_filename', 'N/A')
             display_filename = filename if len(filename) < 25 else f"{filename[:10]}...{filename[-10:]}"
             st.caption(f"File: `{display_filename}`")
             if current_processed_image.mode == 'RGBA': st.info("Has transparency", icon="ℹ️")
    elif current_processed_image:
        st.error(f"Invalid image data in state (Type: {type(current_processed_image)}).", icon="🚫")
    else:
        st.warning("No image loaded. Load one on the **Image Editor Pro** page.", icon="👈")


with col_chat:
    st.subheader("Chat with AI Assistant")
    # Chat message display container
    assistant_container = st.container(height=550, border=True) # Consistent height
    display_chat_history(assistant_container) # Render existing messages

    # --- Example Prompts ---
    if _AGENT_AVAILABLE and image_available:
        st.caption("💡 Quick Actions / Examples:")
        examples = [
            "Make it much brighter", "Apply a sharpen filter", "Remove the background",
            "Invert the colors", "Upscale the image", "Recolor the red shape to purple"
        ]
        example_cols = st.columns(3)
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(example, key=f"example_{i}", use_container_width=True,
                             disabled=st.session_state.assistant_thinking,
                             on_click=lambda ex=example: st.session_state.update(
                                 example_prompt_clicked=ex,
                                 assistant_thinking=True)
                             ):
                     pass # State update triggers rerun, logic handled below

# --- Chat Input ---
chat_input_placeholder = "e.g., 'Increase contrast by 0.2' or 'Replace the sky with a sunset'"
chat_disabled = st.session_state.assistant_thinking or not _AGENT_AVAILABLE or not image_available
user_prompt_from_input = st.chat_input(
    chat_input_placeholder,
    key="ai_chat_input_field",
    disabled=chat_disabled,
    on_submit=lambda: st.session_state.update(assistant_thinking=True)
)

# --- Sidebar ---
with st.sidebar:
    st.header("Assistant Controls")
    st.divider()
    with st.expander("🔑 API Keys", expanded=False):
        st.caption("Optional: Provide keys here for local use if not set as Secrets/Env Vars.")
        try:
            display_api_key_input("OpenAI", "OPENAI_API_KEY", "https://platform.openai.com/api-keys")
            display_api_key_input("StabilityAI", "STABILITY_API_KEY", "https://platform.stability.ai/account/keys")
        except NameError: st.warning("UI module error loading API inputs.")
        except Exception as e: st.error(f"Error displaying API inputs: {e}")
    st.divider()

    if st.button("🗑️ Clear Chat & Reset Thread", type="secondary", use_container_width=True, help="Clears conversation history and starts a new AI session."):
        st.session_state.chat_history = []
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Chat cleared. New thread ID: {st.session_state.current_graph_thread_id}")
        st.session_state.pop('tool_invocation_request', None) # Clear pending graph state keys
        st.session_state.pop('pending_ui_updates', None)
        # st.session_state.pop('updated_image', None) # No longer in graph state
        st.session_state.assistant_thinking = False
        st.toast("Chat history cleared!", icon="🧹")
        time.sleep(0.2)
        st.rerun()

    st.divider()
    st.page_link("app.py", label="Back to Manual Editor", icon="✏️")
    st.divider()
    status_color = "green" if _AGENT_AVAILABLE else "red"
    status_text = "Available" if _AGENT_AVAILABLE else "Unavailable"
    st.sidebar.markdown(f"**Agent Status:** <span style='color:{status_color};'>●</span> {status_text}", unsafe_allow_html=True)
    st.sidebar.caption(f"Thread ID: `{st.session_state.current_graph_thread_id}`")

# --- Processing Logic ---

# Determine the prompt to process
prompt_to_process: Optional[str] = None
if user_prompt_from_input:
    prompt_to_process = user_prompt_from_input
elif st.session_state.get('example_prompt_clicked'):
    prompt_to_process = st.session_state.example_prompt_clicked
    st.session_state.example_prompt_clicked = None # Consume the click event

# Run agent if prompt exists, agent ready, graph compiled, and not already thinking
if prompt_to_process and _AGENT_AVAILABLE and _COMPILED_GRAPH and st.session_state.assistant_thinking:

    logger.info(f"Processing prompt: {prompt_to_process}")
    # Add user message to history if it's not already the last message
    # Use the updated add_message_to_history function
    if not st.session_state.chat_history or \
       st.session_state.chat_history[-1].get("role") != "user" or \
       st.session_state.chat_history[-1].get("content") != prompt_to_process:
        add_message_to_history("user", prompt_to_process)
        # Trigger a quick rerun to display the user message immediately *before* processing
        # This can make the UI feel more responsive.
        # st.rerun() # Removed this immediate rerun to avoid potential double processing issues

    # --- Construct input for the graph (Updated with Pydantic Fix) ---
    lc_messages: List[BaseMessage] = []

    # *** ADD DYNAMIC SYSTEM MESSAGE ***
    image_status_msg = get_image_status_message()
    lc_messages.append(SystemMessage(content=f"You are a helpful image editing assistant. {image_status_msg}"))
    # *********************************

    # Convert rest of Streamlit message history to LangChain BaseMessages
    chat_history_list = st.session_state.get('chat_history', [])
    if not isinstance(chat_history_list, list): chat_history_list = []

    for message_data in chat_history_list:
        role = message_data.get("role")
        content = message_data.get("content") # Content can be None
        tool_call_id = message_data.get("tool_call_id")
        tool_calls = message_data.get("tool_calls") # Get stored tool calls

        # Avoid adding the system message again if it was stored previously
        if role == "system": continue
        elif role == "user":
            lc_messages.append(HumanMessage(content=content or "")) # Ensure content is string
        elif role == "assistant":
            # Reconstruct AIMessage (Pydantic Fix Applied)
            ai_kwargs = {"content": content or ""}
            # Only add tool_calls if it's a non-empty list
            if tool_calls and isinstance(tool_calls, list):
                ai_kwargs["tool_calls"] = tool_calls
            try:
                lc_messages.append(AIMessage(**ai_kwargs))
            except Exception as e:
                 logger.error(f"Failed to reconstruct AIMessage from history: {message_data}. Error: {e}", exc_info=True)
                 continue # Skip adding corrupted message
        elif role == "tool":
             # Reconstruct ToolMessage using content and tool_call_id
             if tool_call_id:
                 lc_messages.append(ToolMessage(content=content or "", tool_call_id=tool_call_id)) # Ensure content is string
             else:
                 logger.warning(f"Skipping tool message from history - missing tool_call_id: {message_data}")

    graph_input = {"messages": lc_messages}
    config = {"configurable": {"thread_id": st.session_state.current_graph_thread_id}}
    logger.info(f"Streaming graph for thread: {config['configurable']['thread_id']}")
    logger.debug(f"Graph Input Messages: {[m.type + ': ' + (str(m.content)[:50] if m.content else '[No Content]') + (' (TC)' if getattr(m, 'tool_calls', None) else '') + (' (TID:' + getattr(m, 'tool_call_id', '') + ')' if getattr(m, 'tool_call_id', None) else '') for m in lc_messages]}")


    # Initialize variables for this run
    final_assistant_message_content: Optional[str] = None
    stream_error: Optional[Exception] = None
    # ui_updates_from_run is handled by the graph now

    # Use status context for better feedback
    with st.status(f"AI processing '{prompt_to_process[:30]}...'") as status_box:
        try:
            # Display placeholder within the chat container
            with assistant_container:
                 with st.chat_message("assistant", avatar="✨"):
                      response_placeholder = st.empty()
                      response_placeholder.markdown("Thinking... ⏳")

            # --- Add Log Here ---
            logger.info(f"Attempting to stream graph. Input keys: {list(graph_input.keys())}, Config: {config}, Graph object: {_COMPILED_GRAPH is not None}")
            # Stream graph execution using 'values' mode to get full state
            for step_output in _COMPILED_GRAPH.stream(graph_input, config, stream_mode="values"):
                logger.debug(f"Raw step_output keys: {step_output.keys()}")
                messages = step_output.get("messages", [])
                if not messages: continue
                last_message = messages[-1]
                logger.debug(f"Stream step: Last msg type={type(last_message).__name__}")

                # --- ADD THIS BLOCK (Updated) ---
                # Add the message received from the graph step to the persistent chat history
                current_chat_history = st.session_state.get('chat_history', [])
                new_message_added = False
                if isinstance(last_message, AIMessage):
                    # Check if this exact AIMessage (content + tool_calls) is already the last one
                    # Need to handle None content correctly
                    msg_content = getattr(last_message, 'content', None)
                    msg_tool_calls = getattr(last_message, 'tool_calls', None)
                    temp_dict = {"role": "assistant", "content": msg_content, "tool_calls": msg_tool_calls}
                    last_hist_entry = current_chat_history[-1] if current_chat_history else {}
                    if not current_chat_history or \
                       last_hist_entry.get("role") != "assistant" or \
                       last_hist_entry.get("content") != temp_dict["content"] or \
                       last_hist_entry.get("tool_calls") != temp_dict["tool_calls"]:
                         add_message_to_history("assistant", msg_content, tool_calls=msg_tool_calls)
                         new_message_added = True
                elif isinstance(last_message, ToolMessage):
                     # Check if this exact ToolMessage is already the last one
                    temp_dict = {"role": "tool", "content": str(last_message.content), "tool_call_id": last_message.tool_call_id}
                    if not current_chat_history or current_chat_history[-1] != temp_dict:
                        add_message_to_history("tool", str(last_message.content), tool_call_id=last_message.tool_call_id)
                        new_message_added = True
                # Add other message types if necessary (e.g., SystemMessage from errors)
                elif isinstance(last_message, SystemMessage):
                     # Check if this exact SystemMessage is already the last one
                    temp_dict = {"role": "system", "content": str(last_message.content)}
                    if not current_chat_history or current_chat_history[-1] != temp_dict:
                        add_message_to_history("system", str(last_message.content))
                        new_message_added = True

                if new_message_added:
                    logger.debug(f"Added {last_message.type} message from stream to chat history.")
                # --- END ADDED BLOCK ---

                status_label_update = "Processing..." # Default status label
                is_final_ai_text_step = False

                # Update status box and potentially capture final AI response text
                if isinstance(last_message, AIMessage):
                    if last_message.tool_calls:
                        tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
                        status_label_update = f"Assistant using tool(s): `{', '.join(tool_names)}`..."
                        # Don't capture final content if it's just a tool call message
                        final_assistant_message_content = None
                    else:
                        # Only capture content if it's a final text response (no tool calls)
                        final_assistant_message_content = str(last_message.content)
                        is_final_ai_text_step = True
                        status_label_update = "Assistant generating response..."
                elif isinstance(last_message, ToolMessage):
                     tool_name = getattr(last_message, 'name', 'tool') # ToolMessage doesn't have 'name' directly, use ID or content?
                     status_label_update = f"Processing tool result (ID: {last_message.tool_call_id})..."
                     # Don't capture final content from tool message
                     final_assistant_message_content = None
                elif isinstance(last_message, SystemMessage):
                     status_label_update = f"System message received..."
                     # Don't capture final content from system message
                     final_assistant_message_content = None

                # Update the placeholder with the latest AI text *only if it's the final one*
                # Otherwise, just show "Thinking..." or "Using tool..."
                display_text = "Thinking... ⏳"
                if final_assistant_message_content and is_final_ai_text_step:
                     display_text = final_assistant_message_content + " ▌" # Add cursor only if streaming text
                elif isinstance(last_message, AIMessage) and last_message.tool_calls:
                     display_text = status_label_update # Show tool call status
                elif isinstance(last_message, ToolMessage):
                     display_text = status_label_update # Show tool processing status
                response_placeholder.markdown(display_text)

                # Update the status box label
                status_box.update(label=status_label_update)
                # Note: UI updates are now handled INSIDE the graph by 'update_app_state' node

            # --- After Stream Loop ---
            if final_assistant_message_content:
                response_placeholder.markdown(final_assistant_message_content) # Final text
                status_box.update(label="Assistant response complete.", state="complete", expanded=False)
            elif not stream_error: # Finished without specific final text (e.g., after tool)
                 # Check last message type to provide better status
                 last_msg_type = type(messages[-1]).__name__ if messages else "Unknown"
                 status_msg = f"Processing finished ({last_msg_type})."
                 response_placeholder.markdown("Task completed.") # Clear placeholder
                 status_box.update(label=status_msg, state="complete", expanded=False)

        except Exception as e:
            stream_error = e
            logger.error(f"Error during AI assistant stream: {e}", exc_info=True)
            error_msg_content = f"⚠️ Error processing request: {str(e)[:200]}"
            status_box.update(label="Processing failed!", state="error", expanded=True)
            # Add error message to history using the updated function
            add_message_to_history("system", error_msg_content)
            # Ensure placeholder shows error
            if 'response_placeholder' in locals(): response_placeholder.error(error_msg_content)
            else:
                 with assistant_container: st.error(error_msg_content)

    # --- Post-Stream Processing ---
    st.session_state.assistant_thinking = False # Reset thinking flag

    # --- Rerun is now ALWAYS needed after processing ---
    # Because the graph's 'update_app_state' node might have modified
    # session_state (image or widgets) even if this page doesn't see it directly.
    # Rerunning ensures the image preview and any potentially affected widget
    # (if user navigates back) are up-to-date.
    # Also ensures the chat history display is updated with messages added during the stream.
    logger.info(f"Post-stream: Final AI Msg='{final_assistant_message_content[:50] if final_assistant_message_content else 'None'}...', Error='{stream_error}'")
    logger.info(f"Rerunning page after AI stream/processing. Stream error={bool(stream_error)}")
    # Optional short delay for user to see final status/error
    if stream_error: time.sleep(0.5)
    st.rerun()


elif prompt_to_process and not _AGENT_AVAILABLE:
     # Handle prompt but agent unavailable
     st.warning("The AI Assistant is unavailable.")
     add_message_to_history("user", prompt_to_process)
     add_message_to_history("assistant", "Sorry, I am currently unavailable.")
     st.session_state.assistant_thinking = False
     st.rerun()

# Reset thinking flag if no prompt was processed (e.g., initial load, cleared input)
if not prompt_to_process and st.session_state.get('assistant_thinking'):
     st.session_state.assistant_thinking = False
     logger.debug("Resetting assistant_thinking flag as no prompt was processed.")