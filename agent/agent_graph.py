# streamlit_image_editor/agent/agent_graph.py
# Builds and compiles the LangGraph agent workflow.

# --- Standard Library Imports ---
import os
import sys
from pathlib import Path
import logging
from typing import Literal, Optional, Dict, Any, Tuple

# --- Path Setup (Add Project Root) ---
# Ensures local modules can be imported when run directly or by Streamlit
try:
    _PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT_DIR))
        print(f"DEBUG (agent_graph.py): Added project root {_PROJECT_ROOT_DIR} to sys.path")
except Exception as e:
    print(f"ERROR (agent_graph.py): Failed during sys.path setup: {e}")

# --- Third-Party Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # For conversation memory
from langchain_openai import ChatOpenAI             # Example LLM provider
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage, SystemMessage
from PIL import Image                               # For type checking

# --- Local Application Imports ---
# Import necessary components from other local modules
_DEPENDENCIES_LOADED = False
try:
    from agent.graph_state import AgentState, ToolInvocationRequest
    from agent.tools import available_tools, tool_implementations
    # Import the state manager function needed within the graph node
    from state.session_state_manager import update_processed_image
    _DEPENDENCIES_LOADED = True
    print("DEBUG (agent_graph.py): Successfully imported agent dependencies.")
except ImportError as e:
    print(f"ERROR (agent_graph.py): Failed to import agent dependencies: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define dummies if import fails to allow basic structure loading
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def update_processed_image(img): print("[MOCK] update_processed_image called")

# --- Streamlit Import (Conditional) ---
# Detect if running within a Streamlit application context
_IN_STREAMLIT_CONTEXT = False
try:
    import streamlit as st
    if hasattr(st, 'secrets'): # A reasonable check for running context
         _IN_STREAMLIT_CONTEXT = True
except (ImportError, RuntimeError):
    pass # Fail silently if streamlit is not available or not running

# --- Logging Setup ---
# Configure logging for this module
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {log_level}")
logger.info(f"Dependencies Loaded: {_DEPENDENCIES_LOADED}")
logger.info(f"Streamlit Context: {_IN_STREAMLIT_CONTEXT}")

# --- LLM and Agent Executor Configuration (Cached) ---
# Simple in-memory cache for LLM and executor to avoid re-initialization on every run
_cached_llm = None
_cached_agent_executor = None

def get_llm():
    """Safely initializes and returns the ChatOpenAI model, using a simple cache."""
    global _cached_llm
    if _cached_llm:
        logger.debug("Returning cached LLM instance.")
        return _cached_llm

    # Determine API key source
    api_key = None
    source = "Not Found"
    if _IN_STREAMLIT_CONTEXT:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
            if key: api_key = key; source = "Streamlit Secrets"
        except Exception as e: logger.warning(f"Error accessing St secrets for OpenAI Key: {e}")
    if not api_key:
        key = os.environ.get("OPENAI_API_KEY")
        if key: api_key = key; source = "Env Var"

    # Handle missing key
    if not api_key:
        message = "OpenAI API Key not found (Checked St Secrets & Env Var). AI Agent cannot function."
        logger.error(message)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show St error: {streamlit_e}")
        return None

    logger.info(f"OpenAI API Key loaded from: {source}")

    # Initialize LLM
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini", # Or configure via env var/secrets
            temperature=0, # Low temperature for predictable tool use
            api_key=api_key,
            max_retries=2, # Add some resilience
            timeout=30 # Set a reasonable timeout
        )
        _cached_llm = model
        logger.info("ChatOpenAI model initialized successfully.")
        return model
    except Exception as e:
        message = f"Failed to initialize ChatOpenAI model: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show St error: {streamlit_e}")
        return None

def get_agent_executor():
    """Creates the agent executor by binding tools to the LLM, using a simple cache."""
    global _cached_agent_executor
    if _cached_agent_executor:
        logger.debug("Returning cached agent executor instance.")
        return _cached_agent_executor

    llm = get_llm()
    if llm is None: return None # Error already logged by get_llm

    if not available_tools: # Check if the tool dictionary imported correctly
        logger.error("No tools available for agent binding. Check agent/tools.py import and definitions.")
        return None

    try:
        # Bind the TOOL DEFINITIONS (@tool decorated functions) from tools.py
        llm_with_tools = llm.bind_tools(list(available_tools.values()))
        logger.info(f"LLM bound with {len(available_tools)} tools: {list(available_tools.keys())}")
        _cached_agent_executor = llm_with_tools
        return llm_with_tools
    except Exception as e:
        message = f"CRITICAL: Failed to bind tools to LLM. Check tool schemas in agent/tools.py. Error: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show St error: {streamlit_e}")
        return None

# --- Graph Node Definitions ---

def call_agent(state: AgentState) -> Dict[str, Any]:
    """Node: Invokes the LLM agent with the current message history."""
    logger.info("Node: call_agent - Starting execution")
    agent_executor = get_agent_executor()
    if agent_executor is None:
        logger.error("Agent executor unavailable in call_agent.")
        return {"messages": [SystemMessage(content="LLM Error: Agent not configured.")]}

    messages = state.get("messages", [])
    if not messages:
         logger.warning("call_agent invoked with empty message history. Skipping.")
         return {} # No change to state

    # Ensure messages are valid BaseMessage instances
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    if not valid_messages:
        logger.error("call_agent: No valid messages found. Cannot invoke LLM.")
        return {"messages": [SystemMessage(content="Internal Error: Invalid message history.")]}

    logger.debug(f"Invoking agent with {len(valid_messages)} valid messages.")
    try:
        response = agent_executor.invoke(valid_messages)
        logger.info(f"Agent response received: Type={type(response).__name__}, ToolCalls={bool(getattr(response, 'tool_calls', None))}")
        return {"messages": [response]} # Return list for add_messages reducer
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return {"messages": [SystemMessage(content=f"Error during LLM communication: {e}")]}

def prepare_tool_run(state: AgentState) -> Dict[str, Any]:
    """Node: Extracts the first tool call request from the last AI message."""
    logger.info("Node: prepare_tool_run - Starting execution")
    messages = state.get("messages", [])
    tool_request = None # Default to no request

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
            if last_message.tool_calls:
                # Process only the first tool call for now
                tool_call = last_message.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                if tool_name and tool_call_id:
                    logger.info(f"Preparing tool run: '{tool_name}', Args: {tool_args}, ID: {tool_call_id}")
                    tool_request = ToolInvocationRequest(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                else:
                    logger.error(f"Invalid tool_call structure from LLM: {tool_call}")
                    # Return an error message immediately, don't proceed to execute
                    error_msg = ToolMessage(content="Internal Error: Invalid tool call structure.", tool_call_id="error_no_id")
                    return {"messages": [error_msg], "tool_invocation_request": None}
            else:
                logger.debug("prepare_tool_run: AIMessage has empty tool_calls list.")
        else:
            logger.debug("prepare_tool_run: Last message not AIMessage with tool calls.")
    else:
        logger.warning("prepare_tool_run: No messages in state.")

    # Update the state with the prepared request (or None if no valid call)
    return {"tool_invocation_request": tool_request}


def execute_tool_and_update(state: AgentState) -> Dict[str, Any]:
    """
    Node: Executes the prepared tool implementation based on 'tool_invocation_request'.
          Fetches image from Streamlit state if needed.
          Updates Streamlit image state via state manager if image changes.
          Stores potential UI updates in 'pending_ui_updates'.
          Clears 'tool_invocation_request' and returns the ToolMessage result.
    """
    logger.info("Node: execute_tool_and_update - Starting execution")
    request: Optional[ToolInvocationRequest] = state.get("tool_invocation_request")

    # Prepare updates, clearing request and pending UI updates immediately
    updates_to_return: Dict[str, Any] = {
        "tool_invocation_request": None,
        "pending_ui_updates": None # Clear any previous pending UI updates
    }

    # Validate the request
    if not request or not isinstance(request, dict) or not request.get("tool_name") or not request.get("tool_call_id"):
        logger.warning("execute_tool_and_update: Invalid/missing tool request. Skipping.")
        return updates_to_return # Return cleaned state

    tool_name = request["tool_name"]
    tool_args = request.get("tool_args", {})
    tool_call_id = request["tool_call_id"]
    logger.info(f"Executing tool '{tool_name}' with ID '{tool_call_id}' and args: {tool_args}")

    # --- Initialize variables ---
    tool_message_content: str = f"Error: Tool implementation '{tool_name}' not found."
    new_image: Optional[Image.Image] = None
    ui_updates: Optional[Dict[str, Any]] = None
    current_image: Optional[Image.Image] = None
    error_occurred = False

    # --- Check if tool implementation exists ---
    tool_impl_func = tool_implementations.get(tool_name)
    needs_image = False
    is_info_tool = tool_name == 'get_image_info' # Handle info tool specifically

    if tool_impl_func:
        try:
            import inspect
            sig = inspect.signature(tool_impl_func)
            needs_image = "input_image" in sig.parameters
            logger.debug(f"Tool '{tool_name}' needs image: {needs_image}")
        except Exception as inspect_e:
            logger.warning(f"Could not inspect signature for {tool_name}: {inspect_e}")
    elif not is_info_tool:
        logger.error(f"Tool implementation for '{tool_name}' not found.")
        error_occurred = True
    # Else: it's the get_image_info tool, handled later

    # --- Fetch image from Streamlit state if needed ---
    if needs_image and not error_occurred:
        logger.debug(f"Fetching required image for '{tool_name}'...")
        if _IN_STREAMLIT_CONTEXT:
            try:
                current_image_obj = st.session_state.get('processed_image')
                if current_image_obj is None or not isinstance(current_image_obj, Image.Image):
                    tool_message_content = "Error: No valid image available to process."
                    logger.warning(tool_message_content)
                    error_occurred = True
                else:
                    current_image = current_image_obj.copy() # Use a copy
                    logger.info(f"Image fetched from Streamlit state for '{tool_name}'.")
            except Exception as e:
                tool_message_content = "Error: Failed accessing image from app state."
                logger.error(f"Error accessing st.session_state: {e}", exc_info=True)
                error_occurred = True
        else:
            tool_message_content = "Error: Cannot access image (Agent not in Streamlit context)."
            logger.warning(tool_message_content)
            error_occurred = True

    # --- Execute the tool implementation ---
    if not error_occurred:
        if tool_impl_func:
            logger.info(f"Executing implementation: {tool_impl_func.__name__}")
            try:
                impl_args = tool_args.copy()
                if needs_image:
                    if not current_image: raise ValueError("Internal Error: Image needed but not available.")
                    impl_args["input_image"] = current_image

                # Call the implementation function
                tool_result = tool_impl_func(**impl_args)

                # Process the result tuple: (str_result, Optional[Image], Optional[Dict])
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    tool_message_content, returned_image, returned_ui_updates = tool_result
                    if returned_image is not None:
                        if isinstance(returned_image, Image.Image): new_image = returned_image
                        else: logger.error(f"Tool '{tool_name}' returned invalid image type: {type(returned_image)}")
                    if isinstance(returned_ui_updates, dict): ui_updates = returned_ui_updates
                else: # Handle unexpected return format
                     logger.error(f"Tool '{tool_name}' impl returned unexpected format: {type(tool_result)}")
                     tool_message_content = f"Error: Tool '{tool_name}' internal error (bad return format)."
                     error_occurred = True

                logger.info(f"Tool '{tool_name}' executed. Result msg: {str(tool_message_content)[:100]}...")

            except Exception as e:
                logger.error(f"Error executing tool impl '{tool_name}': {e}", exc_info=True)
                tool_message_content = f"Execution Error: {str(e)}"
                error_occurred = True

        elif is_info_tool: # Special case: get_image_info
            logger.info(f"Executing special tool: {tool_name}")
            try:
                tool_message_content = available_tools[tool_name].invoke(tool_args)
            except Exception as e:
                logger.error(f"Error executing '{tool_name}': {e}", exc_info=True)
                tool_message_content = f"Execution Error: {str(e)}"
                error_occurred = True
        # else: Error handled by initial check

    # --- Update Streamlit Image State (if applicable) ---
    if new_image is not None and not error_occurred and _IN_STREAMLIT_CONTEXT:
        try:
            update_success = update_processed_image(new_image) # Use manager function
            if update_success: logger.info(f"Streamlit image state updated by '{tool_name}'.")
            else:
                 logger.warning(f"update_processed_image returned False for '{tool_name}'.")
                 tool_message_content += " (Warning: UI image update failed)"
        except Exception as state_e:
            logger.error(f"Failed updating Streamlit state after '{tool_name}': {state_e}", exc_info=True)
            tool_message_content += " (Warning: Error updating UI image)"

    # --- Prepare final ToolMessage and state updates ---
    tool_msg = ToolMessage(content=str(tool_message_content), tool_call_id=tool_call_id)
    logger.debug(f"Prepared ToolMessage for ID {tool_call_id}")

    updates_to_return["messages"] = [tool_msg] # Add message result
    if ui_updates and not error_occurred: # Add UI updates only if they exist and no error
        updates_to_return["pending_ui_updates"] = ui_updates
        logger.debug(f"Adding pending UI updates to graph state: {ui_updates}")

    return updates_to_return

# --- Graph Condition Functions ---

def route_after_agent(state: AgentState) -> Literal["prepare_tool_run", END]:
    """Checks the last message for tool calls to decide the next step."""
    logger.debug("Router: route_after_agent executing...")
    messages = state.get("messages", [])
    if not messages: return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None) and last_message.tool_calls:
        logger.debug("Routing decision: prepare_tool_run (tool calls found)")
        return "prepare_tool_run"
    logger.debug(f"Routing decision: END (last msg type: {type(last_message).__name__}, tool_calls: {bool(getattr(last_message, 'tool_calls', None))})")
    return END

def route_after_tool_prep(state: AgentState) -> Literal["execute_tool_and_update", "agent"]:
    """Checks if a tool invocation request is pending."""
    logger.debug("Router: route_after_tool_prep executing...")
    if state.get("tool_invocation_request"):
         logger.debug("Routing decision: execute_tool_and_update (request pending)")
         return "execute_tool_and_update"
    else:
         logger.debug("Routing decision: agent (no tool request pending)")
         return "agent"

# --- Graph Construction ---
def build_graph():
    """Constructs and compiles the LangGraph agent workflow."""
    if not _DEPENDENCIES_LOADED:
         logger.critical("Cannot build graph, dependencies failed.")
         return None

    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState) # Use the AgentState TypedDict

    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("prepare_tool_run", prepare_tool_run)
    workflow.add_node("execute_tool_and_update", execute_tool_and_update)

    # Define entry point
    workflow.add_edge(START, "agent")

    # Define conditional edges from 'agent'
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "prepare_tool_run": "prepare_tool_run",
            END: END # If no tool calls, end the execution
        }
    )
    # Define conditional edges from 'prepare_tool_run'
    workflow.add_conditional_edges(
         "prepare_tool_run",
         route_after_tool_prep,
         {
              "execute_tool_and_update": "execute_tool_and_update", # If request prepared, execute
              "agent": "agent" # If prep failed or no tool call, go back to agent
         }
    )

    # Always return to the agent after executing a tool to process the result
    workflow.add_edge("execute_tool_and_update", "agent")

    # Configure memory for conversation history persistence
    memory = MemorySaver()

    # Compile the graph with the checkpointer
    try:
        graph = workflow.compile(checkpointer=memory)
        logger.info("LangGraph workflow compiled successfully.")
        return graph
    except Exception as e:
         message = f"Failed to compile LangGraph workflow: {e}"
         logger.critical(message, exc_info=True)
         if _IN_STREAMLIT_CONTEXT:
             try: st.error(message)
             except Exception as streamlit_e: logger.warning(f"Failed to show St error: {streamlit_e}")
         return None # Indicate compilation failure

# --- Global Compiled Graph Instance ---
# Build the graph when the module is loaded.
# This makes it readily available for import in the Streamlit page.
compiled_graph = build_graph()

if compiled_graph:
    logger.info("Global 'compiled_graph' instance created successfully.")
else:
    logger.error("Global 'compiled_graph' instance IS NONE due to build failure. Agent will be unavailable.")

# --- Direct Execution Block (for basic testing) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Ensure DEBUG level for direct testing
    logger.info(f"--- Running {__file__} directly for testing ---")
    if compiled_graph:
        logger.info("Graph compiled successfully. Ready for testing (using mocks if core modules failed).")
        # Example Test Invocation (uncomment to run)
        # config = {"configurable": {"thread_id": "direct_test_thread_1"}}
        # test_input = {"messages": [HumanMessage(content="Make the image 50 brighter")]}
        # logger.info(f"Invoking graph with input: {test_input}")
        # try:
        #     for event in compiled_graph.stream(test_input, config, stream_mode="values"):
        #         logger.info(f"Graph Event: {event}")
        #     final_state = compiled_graph.get_state(config)
        #     logger.info(f"Final State: {final_state}")
        # except Exception as test_e:
        #     logger.error(f"Graph invocation test failed: {test_e}", exc_info=True)
    else:
        logger.error("Graph compilation FAILED during module load. Cannot run tests.")
    logger.info(f"--- Finished {__file__} direct test ---")