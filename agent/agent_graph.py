# streamlit_image_editor/agent/agent_graph.py
# Builds and compiles the LangGraph agent workflow, orchestrating node execution
# and interactions with Streamlit state.

# --- Standard Library Imports ---
import os
import sys
from pathlib import Path
import logging
from typing import Literal, Optional, Dict, Any, Tuple
import inspect # Needed for inspecting tool implementation signatures

# --- Path Setup (Add Project Root) ---
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        print(f"DEBUG (agent_graph.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (agent_graph.py): Failed during sys.path setup: {e}")

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
    _TOOLS_LOADED = True
    # Import the function to update Streamlit's global state (used by _execute_impl)
    from state.session_state_manager import update_processed_image
    _STATE_MANAGER_LOADED = True
    _DEPENDENCIES_LOADED = True
    print("DEBUG (agent_graph.py): Successfully imported agent dependencies.")
except ImportError as e:
    print(f"ERROR (agent_graph.py): Failed to import agent dependencies: {e}")
    # Define dummies
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def _get_current_image(): return None
    def _execute_impl(*args, **kwargs): return "Mock execution result", None
    def update_processed_image(img): print("[MOCK] update_processed_image")
except Exception as e:
    print(f"ERROR (agent_graph.py): Unexpected error during dependency import: {e}")
    # Define dummies
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def _get_current_image(): return None
    def _execute_impl(*args, **kwargs): return "Mock execution result", None
    def update_processed_image(img): print("[MOCK] update_processed_image")

# --- Streamlit Import (Conditional) ---
_IN_STREAMLIT_CONTEXT = False
_st_module = None
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
         _IN_STREAMLIT_CONTEXT = True
         _st_module = st
except (ImportError, RuntimeError): pass

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {log_level}")
logger.info(f"Dependencies Loaded: {_DEPENDENCIES_LOADED}")
logger.info(f"State Manager Loaded: {_STATE_MANAGER_LOADED}")
logger.info(f"Tools Loaded: {_TOOLS_LOADED}")
logger.info(f"Streamlit Context: {_IN_STREAMLIT_CONTEXT}")

# --- LLM and Agent Executor Configuration ---
_cached_llm = None
_cached_agent_executor = None

def get_llm():
    """Safely initializes and returns the ChatOpenAI model, using a simple cache."""
    global _cached_llm
    if _cached_llm: return _cached_llm
    api_key = None; source = "Not Found"
    if _IN_STREAMLIT_CONTEXT and _st_module:
        try: key = _st_module.secrets.get("OPENAI_API_KEY");
        except: key = None
        if key: api_key = key; source = "Streamlit Secrets"
    if not api_key: key = os.environ.get("OPENAI_API_KEY");
    if key: api_key = key; source = "Env Var"
    if not api_key:
        logger.error("OpenAI API Key not found. AI Agent disabled.")
        return None
    logger.info(f"OpenAI API Key loaded from: {source}")
    try:
        model = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0, api_key=api_key, max_retries=1, timeout=45)
        _cached_llm = model
        logger.info("ChatOpenAI model initialized.")
        return model
    except Exception as e: logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True); return None

def get_agent_executor():
    """Creates the agent executor by binding tools to the LLM, using a simple cache."""
    global _cached_agent_executor
    if _cached_agent_executor: return _cached_agent_executor
    llm = get_llm()
    if llm is None: return None
    if not available_tools: logger.error("No tools available for binding."); return None
    try:
        llm_with_tools = llm.bind_tools(list(available_tools.values()))
        logger.info(f"LLM bound with {len(available_tools)} tools: {list(available_tools.keys())}")
        _cached_agent_executor = llm_with_tools
        return llm_with_tools
    except Exception as e: logger.error(f"Failed to bind tools to LLM: {e}", exc_info=True); return None

# --- Graph Node Definitions ---

def call_agent(state: AgentState) -> Dict[str, Any]:
    """Node: Invokes the LLM agent with the current message history."""
    logger.info("Node: call_agent - Starting execution")
    agent_executor = get_agent_executor()
    if agent_executor is None: return {"messages": [SystemMessage(content="LLM Error: Agent not configured.")]}
    messages = state.get("messages", [])
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    if not valid_messages: logger.warning("call_agent: No valid messages found."); return {}
    
    # Check for any assistant messages with tool_calls that aren't followed by matching tool messages
    for i, msg in enumerate(valid_messages):
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            needed_tool_call_ids = {tc.get('id') if isinstance(tc, dict) else tc.id 
                                    for tc in msg.tool_calls if tc}
            
            # Check which tool calls are followed by responses
            for j in range(i + 1, len(valid_messages)):
                if isinstance(valid_messages[j], ToolMessage):
                    tool_id = getattr(valid_messages[j], 'tool_call_id', None)
                    if tool_id in needed_tool_call_ids:
                        needed_tool_call_ids.remove(tool_id)
            
            # If any tool calls aren't matched with responses, log and remove the assistant message
            if needed_tool_call_ids:
                logger.warning(f"Found AI message with unmatched tool calls: {needed_tool_call_ids}. Removing from history.")
                valid_messages = valid_messages[:i] + valid_messages[i+1:]
                break
    
    logger.debug(f"Invoking agent with {len(valid_messages)} valid messages.")
    try:
        response = agent_executor.invoke(valid_messages)
        logger.debug(f"Raw response from agent invoke: {response}")
        tool_calls_present = bool(getattr(response, 'tool_calls', None))
        logger.info(f"Agent response received: Type={type(response).__name__}, ToolCalls={tool_calls_present}")
        return {"messages": [response]}
    except Exception as e: 
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return {"messages": [SystemMessage(content=f"Error during LLM communication: {e}")]}

def prepare_tool_run(state: AgentState) -> Dict[str, Any]:
    """Node: Extracts the first tool call request from the last AI message."""
    logger.info("Node: prepare_tool_run - Starting execution")
    messages = state.get("messages", [])
    tool_request = None
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
            tool_calls = last_message.tool_calls
            if tool_calls:
                logger.debug(f"AIMessage tool_calls found: {tool_calls}")
                tool_call = tool_calls[0] # Process first call
                logger.debug(f"Processing tool_call: {tool_call}")
                if isinstance(tool_call, dict):
                    tool_name, tool_args, tool_call_id = tool_call.get("name"), tool_call.get("args", {}), tool_call.get("id")
                elif hasattr(tool_call, 'name') and hasattr(tool_call, 'args') and hasattr(tool_call, 'id'):
                    tool_name, tool_args, tool_call_id = tool_call.name, tool_call.args, tool_call.id
                else:
                    logger.error(f"Unrecognized tool_call structure: {tool_call} (Type: {type(tool_call)})")
                    tool_name, tool_args, tool_call_id = None, {}, None

                if tool_name and tool_call_id:
                    if not isinstance(tool_args, dict):
                         logger.warning(f"Tool call args for '{tool_name}' is not a dict ({type(tool_args)}), attempting conversion or using empty dict.")
                         try: tool_args = dict(tool_args)
                         except: tool_args = {}
                    logger.info(f"Preparing tool run: '{tool_name}', Args: {tool_args}, ID: {tool_call_id}")
                    tool_request = ToolInvocationRequest(tool_call_id=tool_call_id, tool_name=tool_name, tool_args=tool_args)
                else:
                    logger.error(f"Invalid tool_call structure from LLM (missing name or id): {tool_call}")
                    error_msg = ToolMessage(content="Internal Error: Invalid tool call structure received from LLM.", tool_call_id="error_invalid_structure")
                    return {"messages": [error_msg], "tool_invocation_request": None}
            else: logger.debug("prepare_tool_run: AIMessage has empty tool_calls list.")
        else: logger.debug(f"prepare_tool_run: Last message not AIMessage with tool calls (Type: {type(last_message).__name__}).")
    else: logger.warning("prepare_tool_run: No messages in state.")
    logger.debug(f"Returning from prepare_tool_run with tool_request: {tool_request}")
    return {"tool_invocation_request": tool_request}

# --- Updated execute_tool Node ---
def execute_tool(state: AgentState) -> Dict[str, Any]:
    """
    Node: Executes the tool specified in 'tool_invocation_request'.
          Calls the _execute_impl helper from tools.py.
          Returns the ToolMessage and pending UI updates in the graph state.
          Image updates are handled *inside* _execute_impl.
    """
    logger.info("Node: execute_tool - Starting execution")
    request: Optional[ToolInvocationRequest] = state.get("tool_invocation_request")

    # 1. Validate the request
    if not request or not isinstance(request, dict) or not request.get("tool_name") or not request.get("tool_call_id"):
        logger.warning("execute_tool: Invalid or missing tool request. Skipping execution.")
        # Return a ToolMessage indicating the error, otherwise the agent might get stuck
        error_msg = "Internal Error: Invalid tool request received by execute_tool node."
        # Use a placeholder ID if the original is missing
        error_tool_call_id = request.get("tool_call_id") if request else "error_missing_request_id"
        tool_msg = ToolMessage(content=error_msg, tool_call_id=error_tool_call_id)
        return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}

    tool_name = request["tool_name"]
    tool_args = request.get("tool_args", {})
    tool_call_id = request["tool_call_id"]
    logger.info(f"Executing tool '{tool_name}' (ID: {tool_call_id}) with args: {tool_args}")

    # 2. Find the implementation function
    tool_impl_func = tool_implementations.get(tool_name)

    # Handle get_image_info specifically (doesn't use _execute_impl)
    if tool_name == 'get_image_info':
        if tool_name in available_tools:
             logger.info(f"Executing info tool '{tool_name}' directly via schema invoke.")
             try:
                 # Invoke the @tool decorated function directly
                 tool_message_content = available_tools[tool_name].invoke(tool_args or {})
                 tool_msg = ToolMessage(content=str(tool_message_content), tool_call_id=tool_call_id)
                 logger.info(f"Tool '{tool_name}' executed successfully. Result: {str(tool_message_content)[:100]}...")
                 # Info tool doesn't update UI
                 return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
             except Exception as e:
                  logger.error(f"Error executing '{tool_name}' via schema invoke: {e}", exc_info=True)
                  tool_msg = ToolMessage(content=f"Execution Error: {str(e)}", tool_call_id=tool_call_id)
                  return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
        else:
             logger.error(f"'{tool_name}' schema not found in available_tools.")
             tool_msg = ToolMessage(content=f"Error: Tool schema '{tool_name}' not found.", tool_call_id=tool_call_id)
             return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}

    # Handle other tools using the _execute_impl helper
    elif tool_impl_func:
        try:
            # Determine if image is needed by inspecting the implementation function's signature
            sig = inspect.signature(tool_impl_func)
            needs_image = "input_image" in sig.parameters
            logger.debug(f"Tool '{tool_name}' needs image: {needs_image}")

            # Call the _execute_impl helper (imported from tools.py)
            # It returns (msg_str, ui_updates) and handles image update internally
            msg_str, ui_updates = _execute_impl(tool_impl_func, tool_name, needs_image, tool_args)

            tool_msg = ToolMessage(content=msg_str, tool_call_id=tool_call_id)
            logger.info(f"Tool '{tool_name}' executed via helper. Result: {msg_str[:100]}...")
            logger.debug(f"Prepared ToolMessage for ID {tool_call_id}. UI updates: {ui_updates}")

            # Return ToolMessage and UI updates in the state dictionary
            return {
                "messages": [tool_msg],
                "tool_invocation_request": None, # Clear the request
                "pending_ui_updates": ui_updates # Pass UI updates (or None)
            }
        except NameError: # If _execute_impl was not imported correctly
             logger.error("_execute_impl helper not found or imported. Cannot execute tool logic.")
             tool_msg = ToolMessage(content="Internal Error: Tool execution helper missing.", tool_call_id=tool_call_id)
             return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
        except Exception as e:
             logger.error(f"Unexpected error preparing/calling tool impl for '{tool_name}': {e}", exc_info=True)
             tool_msg = ToolMessage(content=f"Internal Execution Error: {str(e)}", tool_call_id=tool_call_id)
             return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}

    else: # Tool implementation not found
        logger.error(f"Tool implementation for '{tool_name}' not found in tool_implementations map.")
        tool_msg = ToolMessage(content=f"Error: Tool '{tool_name}' is not implemented.", tool_call_id=tool_call_id)
        return {"messages": [tool_msg], "tool_invocation_request": None, "pending_ui_updates": None}
# --- End of Updated execute_tool Node ---

# MODIFIED Node: Applies ONLY pending UI updates to Streamlit state
def update_app_state(state: AgentState) -> Dict[str, Any]:
    """
    Node: Processes 'pending_ui_updates' from the graph state and applies
          them to the Streamlit session_state. Clears this key after processing.
          Image updates are handled directly by the tool implementation.
    """
    logger.info("Node: update_app_state - Starting execution")
    pending_ui_updates = state.get("pending_ui_updates")
    app_state_updated = False

    # Apply Pending UI Updates to Streamlit State
    if pending_ui_updates and isinstance(pending_ui_updates, dict):
        logger.info(f"Applying pending UI updates to Streamlit session state: {pending_ui_updates}")
        if _IN_STREAMLIT_CONTEXT and _st_module:
             try:
                 for key, value in pending_ui_updates.items():
                     if hasattr(_st_module, 'session_state') and key in _st_module.session_state:
                          current_value = _st_module.session_state[key]
                          if current_value != value:
                               _st_module.session_state[key] = value
                               logger.debug(f"Updated st.session_state['{key}'] from {current_value} to {value}")
                               app_state_updated = True
                          else:
                               logger.debug(f"Skipping UI update for key '{key}' - value already set to {value}")
                     else:
                          logger.warning(f"Ignoring UI update for key '{key}' - not found in st.session_state.")
             except Exception as e:
                  logger.error(f"Error applying UI updates to Streamlit state: {e}", exc_info=True)
                  try: _st_module.warning("Failed to apply some UI updates.")
                  except: pass
        else:
             logger.warning("Cannot apply UI updates - not in Streamlit context or st module unavailable.")
    elif pending_ui_updates is not None:
         logger.error(f"Cannot apply UI updates: 'pending_ui_updates' has invalid type {type(pending_ui_updates)}")

    # Clean up the pending_ui_updates key from the graph state
    update_dict = {"pending_ui_updates": None}
    if app_state_updated:
        logger.info("Streamlit app state updated.")
    else:
        logger.info("No Streamlit app state updates were applied in this step.")
    logger.debug(f"Clearing intermediate graph state keys: {list(update_dict.keys())}")
    return update_dict

# --- Graph Condition Functions ---

def route_after_agent(state: AgentState) -> Literal["prepare_tool_run", END]:
    """Checks the last message for tool calls to decide the next step."""
    logger.debug("Router: route_after_agent executing...")
    messages = state.get("messages", [])
    if not messages: logger.debug("Routing decision: END (no messages)"); return END
    last_message = messages[-1]
    logger.debug(f"Last message for routing: Type={type(last_message).__name__}, ToolCalls={getattr(last_message, 'tool_calls', 'N/A')}")
    has_tool_calls = False
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', None)
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            first_call = tool_calls[0]
            if isinstance(first_call, dict): has_tool_calls = bool(first_call.get("name") and first_call.get("id"))
            elif hasattr(first_call, 'name') and hasattr(first_call, 'id'): has_tool_calls = True
    if has_tool_calls: logger.debug("Routing decision: prepare_tool_run"); return "prepare_tool_run"
    else: logger.debug("Routing decision: END"); return END

def route_after_tool_prep(state: AgentState) -> Literal["execute_tool", "agent"]:
    """Checks if a tool invocation request is pending."""
    logger.debug("Router: route_after_tool_prep executing...")
    pending_request = state.get("tool_invocation_request")
    logger.debug(f"Tool invocation request in state: {pending_request}")
    if pending_request and isinstance(pending_request, dict) and pending_request.get("tool_call_id"):
         logger.debug("Routing decision: execute_tool"); return "execute_tool"
    else: logger.debug("Routing decision: agent"); return "agent"

# --- Graph Construction ---
def build_graph():
    """Constructs and compiles the LangGraph agent workflow."""
    if not _DEPENDENCIES_LOADED: logger.critical("Cannot build graph - dependencies failed."); return None
    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("prepare_tool_run", prepare_tool_run)
    workflow.add_node("execute_tool", execute_tool)         # Executes tool via _execute_impl
    workflow.add_node("update_app_state", update_app_state) # Applies only UI updates

    # Define Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent, {"prepare_tool_run": "prepare_tool_run", END: END})
    workflow.add_conditional_edges("prepare_tool_run", route_after_tool_prep, {"execute_tool": "execute_tool", "agent": "agent"})
    workflow.add_edge("execute_tool", "update_app_state") # Always update state after execution
    workflow.add_edge("update_app_state", "agent")        # Always go back to agent after state update

    memory = MemorySaver()
    try:
        graph = workflow.compile(checkpointer=memory)
        logger.info("LangGraph workflow compiled successfully.")
        return graph
    except Exception as e:
         message = f"Failed to compile LangGraph workflow: {e}"
         logger.critical(message, exc_info=True)
         if _IN_STREAMLIT_CONTEXT and _st_module:
             try: _st_module.error(message)
             except: pass
         return None

# --- Global Compiled Graph Instance ---
compiled_graph = build_graph()
if compiled_graph: logger.info("Global 'compiled_graph' instance created successfully.")
else: logger.error("Global 'compiled_graph' IS NONE due to build failure. AI Agent will be unavailable.")

# --- Direct Execution Block ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')
    logger.info(f"--- Running {Path(__file__).name} directly for testing ---")
    if compiled_graph:
        logger.info("Graph compiled. Structure:")
        try: compiled_graph.get_graph().print_ascii()
        except Exception as draw_e: logger.warning(f"Could not print graph structure: {draw_e}")

        # --- Basic Invocation Test ---
        logger.info("--- Running basic invocation test ---")
        if not _IN_STREAMLIT_CONTEXT:
            logger.info("Setting up mock Streamlit session state for testing.")
            class MockSessionState(dict):
                def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.__dict__ = self
            mock_st = type('MockStreamlit', (), {'session_state': MockSessionState()})
            mock_st.session_state['processed_image'] = Image.new("RGB", (100, 80), "orange") # Provide a mock image
            mock_st.session_state['brightness_slider'] = 0 # Add keys expected by UI updates
            mock_st.session_state['contrast_slider'] = 1.0
            mock_st.session_state['rotation_slider'] = 0
            mock_st.session_state['binarize_thresh_slider'] = 128
            mock_st.session_state['apply_binarization_cb'] = False
            mock_st.session_state['zoom_x'] = 0
            mock_st.session_state['zoom_y'] = 0
            mock_st.session_state['zoom_w'] = 100
            mock_st.session_state['zoom_h'] = 100

            # Make mock available to imported modules if needed
            if 'state.session_state_manager' in sys.modules: sys.modules['state.session_state_manager'].st = mock_st
            if 'agent.tools' in sys.modules: sys.modules['agent.tools']._st_module = mock_st; sys.modules['agent.tools']._IN_STREAMLIT_CONTEXT_TOOLS = True
            _st_module = mock_st
            _IN_STREAMLIT_CONTEXT = True

        config = {"configurable": {"thread_id": "direct_test_thread_final"}}
        test_input = {"messages": [HumanMessage(content="Make the image much brighter, say factor 50")]}
        # test_input = {"messages": [HumanMessage(content="What is the image size?")]}
        # test_input = {"messages": [HumanMessage(content="Hello!")]}

        try:
             logger.info(f"Invoking graph with input: {test_input}")
             final_state = None
             for step in compiled_graph.stream(test_input, config):
                 step_key = list(step.keys())[0]; step_value = step[step_key]
                 logger.info(f"--- Graph Step: {step_key} ---")
                 logger.info(f"  Full Step Output: {step_value}")
                 final_state = step_value
             logger.info(f"--- Test Invocation Final State ---")
             if final_state: logger.info(f"  Final State: {final_state}")
             else: logger.info("  No final state captured.")
             if _IN_STREAMLIT_CONTEXT and _st_module:
                 logger.info("--- Mock Streamlit State After Run ---")
                 logger.info(f"  st.session_state: {_st_module.session_state}")

        except Exception as test_e: logger.error(f"Graph invocation test failed: {test_e}", exc_info=True)
    else: logger.error("Graph compilation FAILED. Cannot run tests.")
    logger.info(f"--- Finished {Path(__file__).name} direct test ---")