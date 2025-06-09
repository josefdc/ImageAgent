"""
AI Assistant Chat Interface for Image Editing

This module implements a Streamlit page that provides a conversational AI interface
for image editing operations. Users can interact with an AI assistant using natural
language to apply various image processing operations through a LangGraph-based agent.

The page features:
- Natural language image editing commands
- Real-time chat interface with message history  
- Image preview and status display
- Tool-based AI agent integration
- Session state management for conversations

The AI assistant can perform operations like brightness adjustment, filtering,
background removal, color manipulation, and image enhancement based on user prompts.
"""

# --- Standard Library Imports ---
import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Generator, Optional, Union

# --- Path Setup (Add Project Root) ---
# Ensures local modules can be imported when Streamlit runs this page script
try:
    # Assumes this file is in project_root/pages/
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
except Exception as e:
    print(f"ERROR: Failed during sys.path setup: {e}")

# --- Streamlit Page Config (MUST be FIRST Streamlit command) ---
import streamlit as st
st.set_page_config(
    page_title="AI Image Assistant",
    page_icon="✨",
    layout="wide"
)

# --- Third-Party Imports ---
from PIL import Image
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
    from agent.agent_graph import compiled_graph
    from ui.interface import get_api_key_input as display_api_key_input
    _ui_module_loaded = True

    _COMPILED_GRAPH = compiled_graph
    _AGENT_AVAILABLE = _COMPILED_GRAPH is not None
except ImportError as e:
    st.error(f"Critical Error: Could not import application modules: {e}")
    if not _state_module_loaded:
        def initialize_session_state() -> None: 
            print("CRITICAL: State module failed to load.")
    if not _ui_module_loaded:
        def display_api_key_input(*args, **kwargs) -> None: 
            st.warning("API Input unavailable due to import errors.")
except Exception as e:
    st.error(f"Critical Error during import or agent graph compilation: {e}")
    st.exception(e)
    if not _state_module_loaded:
        def initialize_session_state() -> None: 
            print("CRITICAL: State module failed to load after exception.")
    if not _ui_module_loaded:
        def display_api_key_input(*args, **kwargs) -> None: 
            st.warning("API Input unavailable due to other errors.")

# --- Logging Setup ---
# Configure logger for this specific page module
logger = logging.getLogger(__name__)
# Set level and format if not already configured by the main app or another module
if not logger.hasHandlers():
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    _log_format = '%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)'
    logging.basicConfig(level=_log_level, format=_log_format)
    logger.info(f"Logger initialized for {__name__} with level {_log_level}")

# --- State Initialization ---
# Ensure all required session state keys exist, especially chat-related ones
try:
    initialize_session_state()
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = []
    if 'current_graph_thread_id' not in st.session_state:
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Initialized new graph thread ID: {st.session_state.current_graph_thread_id}")
    if 'assistant_thinking' not in st.session_state: 
        st.session_state.assistant_thinking = False
    if 'example_prompt_clicked' not in st.session_state: 
        st.session_state.example_prompt_clicked = None
except NameError:
    st.error("Failed to initialize session state due to previous import errors. App cannot function.")
    st.stop()
except Exception as state_init_e:
    st.error(f"An unexpected error occurred during state initialization: {state_init_e}")
    st.exception(state_init_e)
    st.stop()


def display_chat_history(container: st.container) -> None:
    """
    Render the chat history within the provided Streamlit container.
    
    Args:
        container: Streamlit container to render messages in
    """
    with container:
        chat_history = st.session_state.get('chat_history', [])
        
        if not isinstance(chat_history, list):
            logger.error("Chat history state is not a list! Resetting.")
            st.session_state.chat_history = []
            st.warning("Chat history was corrupted and has been reset.")
            chat_history = []

        if not chat_history:
            st.caption("Chat history is empty. Ask the AI to edit the image!")
            return

        for i, msg_data in enumerate(chat_history):
            if isinstance(msg_data, dict) and "role" in msg_data:
                role = msg_data["role"]
                content = msg_data.get("content")
                tool_calls = msg_data.get("tool_calls")

                display_content = ""
                if content is not None:
                    if isinstance(content, BaseMessage): 
                        display_content = str(content.content)
                    elif not isinstance(content, str): 
                        display_content = str(content)
                    else: 
                        display_content = content

                avatar = {"user": "👤", "assistant": "✨", "system": "⚙️", "tool": "🛠️"}.get(role, "❓")

                try:
                    with st.chat_message(name=role, avatar=avatar):
                        if role == "tool":
                            st.markdown(f"```\nTool Result:\n{display_content}\n```")
                        elif role == "system":
                            st.warning(display_content, icon="⚙️")
                        elif role == "assistant" and tool_calls:
                            if display_content: 
                                st.markdown(display_content)
                            tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
                            st.markdown(f"*Assistant decided to use tool(s): `{', '.join(tool_names)}`*")
                        else:
                            st.markdown(display_content)
                except Exception as display_e:
                    logger.error(f"Failed displaying message index {i} (role={role}): {display_e}")
                    try: 
                        st.error(f"Error displaying message #{i+1}...")
                    except: 
                        pass
            else:
                logger.warning(f"Skipping invalid chat history item at index {i}: Type={type(msg_data)}, Value={str(msg_data)[:100]}")

        if not _AGENT_AVAILABLE:
            with st.chat_message("assistant", avatar="⚠️"):
                st.warning("AI Assistant is currently unavailable. Check configuration/logs.")


def add_message_to_history(
    role: str, 
    content: Union[str, BaseMessage, None], 
    tool_call_id: Optional[str] = None, 
    tool_calls: Optional[List[Dict]] = None
) -> None:
    """
    Add a message dictionary to the chat history state.
    
    Args:
        role: Message role ('user', 'assistant', 'system', 'tool')
        content: Message content (can be None for tool-only messages)
        tool_call_id: ID for tool messages
        tool_calls: List of tool calls for assistant messages
    """
    if content is None and not tool_calls:
        logger.warning(f"Attempted to add message with empty content/tool_calls for role {role}. Skipping.")
        return

    msg_content_str = None
    if content is not None:
        if isinstance(content, BaseMessage): 
            msg_content_str = str(content.content)
        elif isinstance(content, str): 
            msg_content_str = content
        else:
            try: 
                msg_content_str = str(content)
            except: 
                msg_content_str = "[Error converting content to string]"
            logger.warning(f"Converted non-string content (Type: {type(content)}) to string for chat history.")

    message_dict = {"role": role, "content": msg_content_str}
    if tool_call_id:
        message_dict["tool_call_id"] = tool_call_id
    if tool_calls:
        message_dict["tool_calls"] = tool_calls

    if 'chat_history' not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    st.session_state.chat_history.append(message_dict)
    logger.debug(f"Added message to history: Role={role}, Content='{str(msg_content_str)[:50] if msg_content_str else 'None'}...', ToolCallId={tool_call_id}, HasToolCalls={bool(tool_calls)}")


def get_image_status_message() -> str:
    """
    Check session state and return a status message for the system prompt.
    
    Returns:
        Status message describing current image state
    """
    img = st.session_state.get('processed_image')
    if isinstance(img, Image.Image):
        try:
            return f"CONTEXT: An image ({img.width}x{img.height}, mode {img.mode}) is currently loaded and ready for editing. You can use tools like adjust_brightness, apply_filter, etc. directly."
        except Exception:
            return "CONTEXT: An image is currently loaded and ready for editing. You can use tools like adjust_brightness, apply_filter, etc. directly."
    else:
        return "CONTEXT: No image is currently loaded. You must ask the user to upload one before attempting any image operations."


# Page layout
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
            if current_processed_image.mode == 'RGBA': 
                st.info("Has transparency", icon="ℹ️")
    elif current_processed_image:
        st.error(f"Invalid image data in state (Type: {type(current_processed_image)}).", icon="🚫")
    else:
        st.warning("No image loaded. Load one on the **Image Editor Pro** page.", icon="👈")


with col_chat:
    st.subheader("Chat with AI Assistant")
    # Chat message display container
    assistant_container = st.container(height=550, border=True)
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
                    pass

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
        except NameError: 
            st.warning("UI module error loading API inputs.")
        except Exception as e: 
            st.error(f"Error displaying API inputs: {e}")
    st.divider()

    if st.button("🗑️ Clear Chat & Reset Thread", type="secondary", use_container_width=True, 
                 help="Clears conversation history and starts a new AI session."):
        st.session_state.chat_history = []
        st.session_state.current_graph_thread_id = f"thread_{int(time.time())}"
        logger.info(f"Chat cleared. New thread ID: {st.session_state.current_graph_thread_id}")
        st.session_state.pop('tool_invocation_request', None)
        st.session_state.pop('pending_ui_updates', None)
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
prompt_to_process: Optional[str] = None
if user_prompt_from_input:
    prompt_to_process = user_prompt_from_input
elif st.session_state.get('example_prompt_clicked'):
    prompt_to_process = st.session_state.example_prompt_clicked
    st.session_state.example_prompt_clicked = None

# Run agent if prompt exists, agent ready, graph compiled, and not already thinking
if prompt_to_process and _AGENT_AVAILABLE and _COMPILED_GRAPH and st.session_state.assistant_thinking:
    logger.info(f"Processing prompt: {prompt_to_process}")
    
    if not st.session_state.chat_history or \
       st.session_state.chat_history[-1].get("role") != "user" or \
       st.session_state.chat_history[-1].get("content") != prompt_to_process:
        add_message_to_history("user", prompt_to_process)

    # Construct input for the graph
    lc_messages: List[BaseMessage] = []

    image_status_msg = get_image_status_message()
    lc_messages.append(SystemMessage(content=f"You are a helpful image editing assistant. {image_status_msg}"))

    chat_history_list = st.session_state.get('chat_history', [])
    if not isinstance(chat_history_list, list): 
        chat_history_list = []

    for message_data in chat_history_list:
        role = message_data.get("role")
        content = message_data.get("content")
        tool_call_id = message_data.get("tool_call_id")
        tool_calls = message_data.get("tool_calls")

        if role == "system": 
            continue
        elif role == "user":
            lc_messages.append(HumanMessage(content=content or ""))
        elif role == "assistant":
            ai_kwargs = {"content": content or ""}
            if tool_calls and isinstance(tool_calls, list):
                ai_kwargs["tool_calls"] = tool_calls
            try:
                lc_messages.append(AIMessage(**ai_kwargs))
            except Exception as e:
                logger.error(f"Failed to reconstruct AIMessage from history: {message_data}. Error: {e}", exc_info=True)
                continue
        elif role == "tool":
            if tool_call_id:
                lc_messages.append(ToolMessage(content=content or "", tool_call_id=tool_call_id))
            else:
                logger.warning(f"Skipping tool message from history - missing tool_call_id: {message_data}")

    graph_input = {"messages": lc_messages}
    config = {"configurable": {"thread_id": st.session_state.current_graph_thread_id}}
    logger.info(f"Streaming graph for thread: {config['configurable']['thread_id']}")

    final_assistant_message_content: Optional[str] = None
    stream_error: Optional[Exception] = None

    with st.status(f"AI processing '{prompt_to_process[:30]}...'") as status_box:
        try:
            with assistant_container:
                with st.chat_message("assistant", avatar="✨"):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("Thinking... ⏳")

            logger.info(f"Attempting to stream graph. Input keys: {list(graph_input.keys())}, Config: {config}, Graph object: {_COMPILED_GRAPH is not None}")
            
            for step_output in _COMPILED_GRAPH.stream(graph_input, config, stream_mode="values"):
                logger.debug(f"Raw step_output keys: {step_output.keys()}")
                messages = step_output.get("messages", [])
                if not messages: 
                    continue
                last_message = messages[-1]
                logger.debug(f"Stream step: Last msg type={type(last_message).__name__}")

                # Add message to chat history
                current_chat_history = st.session_state.get('chat_history', [])
                new_message_added = False
                
                if isinstance(last_message, AIMessage):
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
                    temp_dict = {"role": "tool", "content": str(last_message.content), "tool_call_id": last_message.tool_call_id}
                    if not current_chat_history or current_chat_history[-1] != temp_dict:
                        add_message_to_history("tool", str(last_message.content), tool_call_id=last_message.tool_call_id)
                        new_message_added = True
                elif isinstance(last_message, SystemMessage):
                    temp_dict = {"role": "system", "content": str(last_message.content)}
                    if not current_chat_history or current_chat_history[-1] != temp_dict:
                        add_message_to_history("system", str(last_message.content))
                        new_message_added = True

                if new_message_added:
                    logger.debug(f"Added {last_message.type} message from stream to chat history.")

                status_label_update = "Processing..."
                is_final_ai_text_step = False

                if isinstance(last_message, AIMessage):
                    if last_message.tool_calls:
                        tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
                        status_label_update = f"Assistant using tool(s): `{', '.join(tool_names)}`..."
                        final_assistant_message_content = None
                    else:
                        final_assistant_message_content = str(last_message.content)
                        is_final_ai_text_step = True
                        status_label_update = "Assistant generating response..."
                elif isinstance(last_message, ToolMessage):
                    status_label_update = f"Processing tool result (ID: {last_message.tool_call_id})..."
                    final_assistant_message_content = None
                elif isinstance(last_message, SystemMessage):
                    status_label_update = f"System message received..."
                    final_assistant_message_content = None

                display_text = "Thinking... ⏳"
                if final_assistant_message_content and is_final_ai_text_step:
                    display_text = final_assistant_message_content + " ▌"
                elif isinstance(last_message, AIMessage) and last_message.tool_calls:
                    display_text = status_label_update
                elif isinstance(last_message, ToolMessage):
                    display_text = status_label_update
                response_placeholder.markdown(display_text)

                status_box.update(label=status_label_update)

            if final_assistant_message_content:
                response_placeholder.markdown(final_assistant_message_content)
                status_box.update(label="Assistant response complete.", state="complete", expanded=False)
            elif not stream_error:
                last_msg_type = type(messages[-1]).__name__ if messages else "Unknown"
                status_msg = f"Processing finished ({last_msg_type})."
                response_placeholder.markdown("Task completed.")
                status_box.update(label=status_msg, state="complete", expanded=False)

        except Exception as e:
            stream_error = e
            logger.error(f"Error during AI assistant stream: {e}", exc_info=True)
            error_msg_content = f"⚠️ Error processing request: {str(e)[:200]}"
            status_box.update(label="Processing failed!", state="error", expanded=True)
            add_message_to_history("system", error_msg_content)
            if 'response_placeholder' in locals(): 
                response_placeholder.error(error_msg_content)
            else:
                with assistant_container: 
                    st.error(error_msg_content)

    st.session_state.assistant_thinking = False

    logger.info(f"Post-stream: Final AI Msg='{final_assistant_message_content[:50] if final_assistant_message_content else 'None'}...', Error='{stream_error}'")
    logger.info(f"Rerunning page after AI stream/processing. Stream error={bool(stream_error)}")
    if stream_error: 
        time.sleep(0.5)
    st.rerun()

elif prompt_to_process and not _AGENT_AVAILABLE:
    st.warning("The AI Assistant is unavailable.")
    add_message_to_history("user", prompt_to_process)
    add_message_to_history("assistant", "Sorry, I am currently unavailable.")
    st.session_state.assistant_thinking = False
    st.rerun()

# Reset thinking flag if no prompt was processed (e.g., initial load, cleared input)
if not prompt_to_process and st.session_state.get('assistant_thinking'):
    st.session_state.assistant_thinking = False
    logger.debug("Resetting assistant_thinking flag as no prompt was processed.")