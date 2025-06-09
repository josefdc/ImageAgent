"""
LangGraph Agent Workflow Builder

This module builds and compiles the LangGraph agent workflow that orchestrates
the execution of image processing tools through an AI agent. It handles:
- Tool execution and state management
- Communication with OpenAI's GPT model
- Integration with Streamlit's session state
- Graph-based workflow orchestration for image editing operations

The workflow consists of nodes that handle agent communication, tool preparation,
tool execution, and UI state updates in a coordinated manner.
"""

# --- Standard Library Imports ---
import os
import sys
import inspect
import logging
from pathlib import Path
from typing import Literal, Optional, Dict, Any

# --- Path Setup (Add Project Root) ---
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
except Exception as e:
    print(f"ERROR: Failed during sys.path setup: {e}")

# --- Third-Party Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage, SystemMessage
from PIL import Image

# --- Local Application Imports ---
_DEPENDENCIES_LOADED = False
_STATE_MANAGER_LOADED = False
_TOOLS_LOADED = False

try:
    # Import the state definition WITHOUT 'updated_image'
    from agent.graph_state import AgentState, ToolInvocationRequest
    # Import schemas (available_tools), implementation map (tool_implementations),
    # and the helpers (_execute_impl, _get_current_image)
    from agent.tools import available_tools, tool_implementations, _execute_impl, _get_current_image
    # Import the function to update Streamlit's global state (used by _execute_impl)
    from state.session_state_manager import update_processed_image
    _DEPENDENCIES_LOADED = _STATE_MANAGER_LOADED = _TOOLS_LOADED = True
except ImportError as e:
    print(f"ERROR: Failed to import agent dependencies: {e}")
    # Define minimal fallbacks
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def _get_current_image(): return None
    def _execute_impl(*args, **kwargs): return "Mock execution result", None
    def update_processed_image(img): pass
except Exception as e:
    print(f"ERROR (agent_graph.py): Unexpected error during dependency import: {e}")
    # Define minimal fallbacks
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def _get_current_image(): return None
    def _execute_impl(*args, **kwargs): return "Mock execution result", None
    def update_processed_image(img): pass

# --- Streamlit Import (Conditional) ---
_IN_STREAMLIT_CONTEXT = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        _IN_STREAMLIT_CONTEXT = True
        _st_module = st
except (ImportError, RuntimeError):
    pass

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
    )
logger = logging.getLogger(__name__)

# --- Global Caches ---
_cached_llm = None
_cached_agent_executor = None


def get_llm() -> Optional[ChatOpenAI]:
    """
    Initialize and return the ChatOpenAI model with caching.
    
    Returns:
        ChatOpenAI model instance or None if initialization fails
    """
    global _cached_llm
    if _cached_llm:
        return _cached_llm
    
    # Try to get API key from Streamlit secrets first, then environment
    api_key = None
    if _IN_STREAMLIT_CONTEXT and _st_module:
        try:
            api_key = _st_module.secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OpenAI API Key not found. AI Agent disabled.")
        return None
    
    try:
        model = ChatOpenAI(
            model="gpt-4o-2024-11-20",
            temperature=0,
            api_key=api_key,
            max_retries=1,
            timeout=45
        )
        _cached_llm = model
        logger.info("ChatOpenAI model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {e}")
        return None


def get_agent_executor() -> Optional[ChatOpenAI]:
    """
    Create the agent executor by binding tools to the LLM.
    
    Returns:
        LLM with bound tools or None if binding fails
    """
    global _cached_agent_executor
    if _cached_agent_executor:
        return _cached_agent_executor
    
    llm = get_llm()
    if llm is None:
        return None
    
    if not available_tools:
        logger.error("No tools available for binding")
        return None
    
    try:
        llm_with_tools = llm.bind_tools(list(available_tools.values()))
        logger.info(f"LLM bound with {len(available_tools)} tools")
        _cached_agent_executor = llm_with_tools
        return llm_with_tools
    except Exception as e:
        logger.error(f"Failed to bind tools to LLM: {e}")
        return None


def call_agent(state: AgentState) -> Dict[str, Any]:
    """
    Invoke the LLM agent with the current message history.
    
    Args:
        state: Current agent state containing message history
        
    Returns:
        Dictionary containing the agent's response message
    """
    logger.info("Executing call_agent node")
    agent_executor = get_agent_executor()
    if agent_executor is None:
        return {"messages": [SystemMessage(content="LLM Error: Agent not configured.")]}
    
    messages = state.get("messages", [])
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    
    if not valid_messages:
        logger.warning("No valid messages found")
        return {}
    
    # Clean up orphaned tool calls
    for i, msg in enumerate(valid_messages):
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            needed_tool_call_ids = {
                tc.get('id') if isinstance(tc, dict) else tc.id 
                for tc in msg.tool_calls if tc
            }
            
            # Check for matching tool responses
            for j in range(i + 1, len(valid_messages)):
                if isinstance(valid_messages[j], ToolMessage):
                    tool_id = getattr(valid_messages[j], 'tool_call_id', None)
                    if tool_id in needed_tool_call_ids:
                        needed_tool_call_ids.remove(tool_id)
            
            # Remove messages with unmatched tool calls
            if needed_tool_call_ids:
                logger.warning(f"Removing AI message with unmatched tool calls: {needed_tool_call_ids}")
                valid_messages = valid_messages[:i] + valid_messages[i+1:]
                break
    
    try:
        response = agent_executor.invoke(valid_messages)
        logger.info("Agent response received successfully")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return {"messages": [SystemMessage(content=f"Error during LLM communication: {e}")]}


def prepare_tool_run(state: AgentState) -> Dict[str, Any]:
    """
    Extract the first tool call request from the last AI message.
    
    Args:
        state: Current agent state
        
    Returns:
        Dictionary containing the tool invocation request
    """
    logger.info("Executing prepare_tool_run node")
    messages = state.get("messages", [])
    tool_request = None
    
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
            tool_calls = last_message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]  # Process first call
                
                # Handle different tool call formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id")
                elif hasattr(tool_call, 'name') and hasattr(tool_call, 'args') and hasattr(tool_call, 'id'):
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_call_id = tool_call.id
                else:
                    logger.error(f"Unrecognized tool_call structure: {tool_call}")
                    error_msg = ToolMessage(
                        content="Internal Error: Invalid tool call structure received from LLM.",
                        tool_call_id="error_invalid_structure"
                    )
                    return {"messages": [error_msg], "tool_invocation_request": None}

                if tool_name and tool_call_id:
                    if not isinstance(tool_args, dict):
                        logger.warning(f"Converting tool args for '{tool_name}' to dict")
                        try:
                            tool_args = dict(tool_args)
                        except:
                            tool_args = {}
                    
                    logger.info(f"Preparing tool run: '{tool_name}' with ID: {tool_call_id}")
                    tool_request = ToolInvocationRequest(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
    
    return {"tool_invocation_request": tool_request}


def execute_tool(state: AgentState) -> Dict[str, Any]:
    """
    Execute the tool specified in the tool invocation request.
    
    Args:
        state: Current agent state containing tool request
        
    Returns:
        Dictionary containing tool message and UI updates
    """
    logger.info("Executing execute_tool node")
    request: Optional[ToolInvocationRequest] = state.get("tool_invocation_request")

    # Validate request
    if not request or not isinstance(request, dict) or not request.get("tool_name") or not request.get("tool_call_id"):
        logger.warning("Invalid or missing tool request")
        error_tool_call_id = request.get("tool_call_id") if request else "error_missing_request_id"
        tool_msg = ToolMessage(
            content="Internal Error: Invalid tool request received by execute_tool node.",
            tool_call_id=error_tool_call_id
        )
        return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}

    tool_name = request["tool_name"]
    tool_args = request.get("tool_args", {})
    tool_call_id = request["tool_call_id"]
    
    logger.info(f"Executing tool '{tool_name}' with ID: {tool_call_id}")

    # Handle get_image_info specifically
    if tool_name == 'get_image_info':
        if tool_name in available_tools:
            try:
                tool_message_content = available_tools[tool_name].invoke(tool_args or {})
                tool_msg = ToolMessage(content=str(tool_message_content), tool_call_id=tool_call_id)
                logger.info(f"Tool '{tool_name}' executed successfully")
                return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
            except Exception as e:
                logger.error(f"Error executing '{tool_name}': {e}")
                tool_msg = ToolMessage(content=f"Execution Error: {str(e)}", tool_call_id=tool_call_id)
                return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
        else:
            logger.error(f"Tool schema '{tool_name}' not found")
            tool_msg = ToolMessage(content=f"Error: Tool schema '{tool_name}' not found.", tool_call_id=tool_call_id)
            return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}

    # Handle other tools using the _execute_impl helper
    tool_impl_func = tool_implementations.get(tool_name)
    if tool_impl_func:
        try:
            # Check if image is needed by inspecting function signature
            sig = inspect.signature(tool_impl_func)
            needs_image = "input_image" in sig.parameters
            
            # Execute via helper function
            msg_str, ui_updates = _execute_impl(tool_impl_func, tool_name, needs_image, tool_args)
            
            tool_msg = ToolMessage(content=msg_str, tool_call_id=tool_call_id)
            logger.info(f"Tool '{tool_name}' executed successfully")
            
            return {
                "messages": [tool_msg],
                "tool_invocation_request": None,
                "pending_ui_updates": ui_updates
            }
        except NameError:
            logger.error("Tool execution helper not found")
            tool_msg = ToolMessage(content="Internal Error: Tool execution helper missing.", tool_call_id=tool_call_id)
            return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            tool_msg = ToolMessage(content=f"Internal Execution Error: {str(e)}", tool_call_id=tool_call_id)
            return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
    else:
        logger.error(f"Tool implementation for '{tool_name}' not found")
        tool_msg = ToolMessage(content=f"Error: Tool '{tool_name}' is not implemented.", tool_call_id=tool_call_id)
        return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}


def update_app_state(state: AgentState) -> Dict[str, Any]:
    """
    Apply pending UI updates to Streamlit session state.
    
    Args:
        state: Current agent state containing pending UI updates
        
    Returns:
        Dictionary clearing the pending updates
    """
    logger.info("Executing update_app_state node")
    pending_ui_updates = state.get("pending_ui_updates")
    app_state_updated = False

    if pending_ui_updates and isinstance(pending_ui_updates, dict):
        logger.info("Applying pending UI updates to Streamlit session state")
        if _IN_STREAMLIT_CONTEXT and _st_module:
            try:
                for key, value in pending_ui_updates.items():
                    if hasattr(_st_module, 'session_state') and key in _st_module.session_state:
                        current_value = _st_module.session_state[key]
                        if current_value != value:
                            _st_module.session_state[key] = value
                            logger.debug(f"Updated session_state['{key}']")
                            app_state_updated = True
                    else:
                        logger.warning(f"Ignoring UI update for key '{key}' - not found in session_state")
            except Exception as e:
                logger.error(f"Error applying UI updates: {e}")
                try:
                    _st_module.warning("Failed to apply some UI updates.")
                except:
                    pass
        else:
            logger.warning("Cannot apply UI updates - not in Streamlit context")
    elif pending_ui_updates is not None:
        logger.error(f"Invalid pending_ui_updates type: {type(pending_ui_updates)}")

    if app_state_updated:
        logger.info("Streamlit app state updated successfully")
    
    return {"pending_ui_updates": None}


def route_after_agent(state: AgentState) -> Literal["prepare_tool_run", END]:
    """
    Route after agent execution based on presence of tool calls.
    
    Args:
        state: Current agent state
        
    Returns:
        Next step in the workflow
    """
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_message = messages[-1]
    has_tool_calls = False
    
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', None)
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            first_call = tool_calls[0]
            if isinstance(first_call, dict):
                has_tool_calls = bool(first_call.get("name") and first_call.get("id"))
            elif hasattr(first_call, 'name') and hasattr(first_call, 'id'):
                has_tool_calls = True
    
    return "prepare_tool_run" if has_tool_calls else END


def route_after_tool_prep(state: AgentState) -> Literal["execute_tool", "agent"]:
    """
    Route after tool preparation based on presence of tool request.
    
    Args:
        state: Current agent state
        
    Returns:
        Next step in the workflow
    """
    pending_request = state.get("tool_invocation_request")
    if pending_request and isinstance(pending_request, dict) and pending_request.get("tool_call_id"):
        return "execute_tool"
    else:
        return "agent"


def build_graph() -> Optional[StateGraph]:
    """
    Construct and compile the LangGraph agent workflow.
    
    Returns:
        Compiled graph instance or None if build fails
    """
    if not _DEPENDENCIES_LOADED:
        logger.critical("Cannot build graph - dependencies failed")
        return None
    
    logger.info("Building LangGraph workflow")
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("prepare_tool_run", prepare_tool_run)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("update_app_state", update_app_state)

    # Define edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", 
        route_after_agent, 
        {"prepare_tool_run": "prepare_tool_run", END: END}
    )
    workflow.add_conditional_edges(
        "prepare_tool_run", 
        route_after_tool_prep, 
        {"execute_tool": "execute_tool", "agent": "agent"}
    )
    workflow.add_edge("execute_tool", "update_app_state")
    workflow.add_edge("update_app_state", "agent")

    memory = MemorySaver()
    try:
        graph = workflow.compile(checkpointer=memory)
        logger.info("LangGraph workflow compiled successfully")
        return graph
    except Exception as e:
        logger.critical(f"Failed to compile LangGraph workflow: {e}")
        if _IN_STREAMLIT_CONTEXT and _st_module:
            try:
                _st_module.error(f"Failed to compile LangGraph workflow: {e}")
            except:
                pass
        return None


# --- Global Compiled Graph Instance ---
compiled_graph = build_graph()
if compiled_graph:
    logger.info("Global compiled_graph instance created successfully")
else:
    logger.error("Global compiled_graph is None - AI Agent will be unavailable")


# --- Direct Execution Block ---
if __name__ == "__main__":
    """Direct execution for testing purposes."""
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s'
        )
    
    logger.info(f"Running {Path(__file__).name} directly for testing")
    
    if compiled_graph:
        logger.info("Graph compiled successfully")
        try:
            compiled_graph.get_graph().print_ascii()
        except Exception as e:
            logger.warning(f"Could not print graph structure: {e}")

        # Basic invocation test
        logger.info("Running basic invocation test")
        if not _IN_STREAMLIT_CONTEXT:
            # Set up mock Streamlit session state
            class MockSessionState(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.__dict__ = self
            
            mock_st = type('MockStreamlit', (), {'session_state': MockSessionState()})
            mock_st.session_state.update({
                'processed_image': Image.new("RGB", (100, 80), "orange"),
                'brightness_slider': 0,
                'contrast_slider': 1.0,
                'rotation_slider': 0,
                'binarize_thresh_slider': 128,
                'apply_binarization_cb': False,
                'zoom_x': 0,
                'zoom_y': 0,
                'zoom_w': 100,
                'zoom_h': 100
            })

            # Make mock available to imported modules
            if 'state.session_state_manager' in sys.modules:
                sys.modules['state.session_state_manager'].st = mock_st
            if 'agent.tools' in sys.modules:
                sys.modules['agent.tools']._st_module = mock_st
                sys.modules['agent.tools']._IN_STREAMLIT_CONTEXT_TOOLS = True
            
            _st_module = mock_st
            _IN_STREAMLIT_CONTEXT = True

        config = {"configurable": {"thread_id": "direct_test_thread"}}
        test_input = {"messages": [HumanMessage(content="Make the image much brighter, factor 50")]}

        try:
            logger.info("Invoking graph with test input")
            final_state = None
            for step in compiled_graph.stream(test_input, config):
                step_key = list(step.keys())[0]
                step_value = step[step_key]
                logger.info(f"Graph Step: {step_key}")
                final_state = step_value
            
            logger.info("Test invocation completed")
            if final_state:
                logger.info("Final state captured successfully")
            
            if _IN_STREAMLIT_CONTEXT and _st_module:
                logger.info("Mock Streamlit state updated")

        except Exception as e:
            logger.error(f"Graph invocation test failed: {e}")
    else:
        logger.error("Graph compilation failed - cannot run tests")
    
    logger.info("Finished direct test execution")