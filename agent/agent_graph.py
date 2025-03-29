#agent/agent_graph.py
# --- Standard Library Imports ---
import os
import sys
from pathlib import Path
import logging
from typing import Literal, Optional, Dict, Any, Tuple

# --- Path Setup (Add Project Root) ---
try:
    # Project root is the parent directory of the 'agent' directory
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
from PIL import Image # Needed for isinstance checks

# --- Local Application Imports (Use Paths Relative to Project Root) ---
_DEPENDENCIES_LOADED = False
try:
    # Imports are now relative to the project root added to sys.path
    from agent.graph_state import AgentState, ToolInvocationRequest
    from agent.tools import available_tools, tool_implementations
    # Import the state manager for updating the image
    from state.session_state_manager import update_processed_image
    _DEPENDENCIES_LOADED = True
    print("DEBUG (agent_graph.py): Successfully imported agent dependencies.")
except ImportError as e:
    print(f"ERROR (agent_graph.py): Failed to import agent dependencies using absolute paths: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define dummies if import fails to allow basic structure loading
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def update_processed_image(img): print("[MOCK] update_processed_image called")

# --- Streamlit Import (Conditional) ---
_IN_STREAMLIT_CONTEXT = False
try:
    import streamlit as st
    # Check for a streamlit-specific attribute to confirm context
    if hasattr(st, 'secrets'):
         _IN_STREAMLIT_CONTEXT = True
except (ImportError, RuntimeError):
    pass # Fail silently if streamlit is not available or not running

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {log_level}")
logger.info(f"Dependencies Loaded: {_DEPENDENCIES_LOADED}")
logger.info(f"Streamlit Context: {_IN_STREAMLIT_CONTEXT}")

# --- LLM and Agent Executor Configuration (Cached) ---
_cached_llm = None
def get_llm():
    """Safely initializes and returns the ChatOpenAI model, using a simple cache."""
    global _cached_llm
    if _cached_llm: return _cached_llm

    api_key = None; source = "Not Found"
    if _IN_STREAMLIT_CONTEXT:
        try:
            key = st.secrets.get("OPENAI_API_KEY") # Use .get for safety
            if key: api_key = key; source = "Streamlit Secrets"
        except Exception as e: logger.warning(f"Error accessing St secrets for OpenAI Key: {e}")
    if not api_key:
        key = os.environ.get("OPENAI_API_KEY")
        if key: api_key = key; source = "Env Var"

    if not api_key:
        message = "OpenAI API Key not found (Checked Streamlit Secrets & OPENAI_API_KEY env var). AI Agent cannot function."
        logger.error(message)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

    logger.info(f"OpenAI API Key loaded from: {source}")
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini", # Consider making model name configurable
            temperature=0,
            api_key=api_key,
            max_retries=2
        )
        _cached_llm = model
        return model
    except Exception as e:
        message = f"Failed to initialize ChatOpenAI model: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

_cached_agent_executor = None
def get_agent_executor():
    """Creates the agent executor by binding tools to the LLM, using a simple cache."""
    global _cached_agent_executor
    if _cached_agent_executor: return _cached_agent_executor

    llm = get_llm()
    if llm is None: return None # Error logged previously
    if not available_tools:
        logger.error("No tools available for agent binding. Check tools.py.")
        return None

    try:
        # Bind the TOOL DEFINITIONS (schemas from @tool)
        llm_with_tools = llm.bind_tools(list(available_tools.values()))
        logger.info(f"LLM bound with {len(available_tools)} tools: {list(available_tools.keys())}")
        _cached_agent_executor = llm_with_tools
        return llm_with_tools
    except Exception as e:
        message = f"CRITICAL: Failed to bind tools to LLM. Check tool definitions/schemas in agent/tools.py. Error: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

# --- Graph Node Definitions ---

def call_agent(state: AgentState) -> Dict[str, Any]:
    """Node: Invokes the LLM agent with the current message history."""
    logger.info("Node: call_agent - Starting execution")
    agent_executor = get_agent_executor()
    if agent_executor is None:
        logger.error("Agent executor unavailable in call_agent.")
        # Return error message to be added to state
        return {"messages": [SystemMessage(content="LLM Error: Agent executor not configured or failed to initialize.")]}

    messages = state.get("messages", [])
    if not messages:
         logger.warning("call_agent invoked with empty message history. Skipping LLM call.")
         return {} # Return empty dict - state doesn't change

    # Filter out any potential None or non-BaseMessage entries
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    if len(valid_messages) != len(messages):
         logger.warning(f"Filtered out {len(messages) - len(valid_messages)} invalid entries from messages list.")

    if not valid_messages:
        logger.error("call_agent: No valid messages found after filtering. Cannot invoke LLM.")
        return {"messages": [SystemMessage(content="Internal Error: No valid messages to send to LLM.")]}

    logger.debug(f"Invoking agent with {len(valid_messages)} valid messages. Last: {valid_messages[-1].pretty_repr()}")
    try:
        # Invoke the LLM bound with tool definitions
        response = agent_executor.invoke(valid_messages)
        logger.info(f"Agent response received: Type={type(response).__name__}, ID={getattr(response, 'id', 'N/A')}, ToolCalls={bool(getattr(response, 'tool_calls', None))}")
        # Return the response to be added to the 'messages' list by the AgentState reducer
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        # Return a SystemMessage indicating the error
        return {"messages": [SystemMessage(content=f"Error during model invocation: {e}")]}


def prepare_tool_run(state: AgentState) -> Dict[str, Any]:
    """Node: Extracts tool call details from the last AI message."""
    logger.info("Node: prepare_tool_run - Starting execution")
    messages = state.get("messages", [])
    if not messages:
        logger.warning("prepare_tool_run: No messages in state.")
        return {"tool_invocation_request": None} # No request if no messages

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if not isinstance(last_message, AIMessage) or not getattr(last_message, 'tool_calls', None):
        logger.debug("prepare_tool_run: Last message is not an AIMessage with tool calls. Clearing any pending request.")
        return {"tool_invocation_request": None}

    # Process the first tool call (can be extended for parallel calls)
    if not last_message.tool_calls: # Should be caught by getattr check, but double-check
        logger.warning("prepare_tool_run: AIMessage has tool_calls attribute but it's empty.")
        return {"tool_invocation_request": None}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")

    if not tool_name or not tool_call_id:
         logger.error(f"Invalid tool_call structure received from LLM: {tool_call}")
         # Create an error message to send back
         tool_msg = ToolMessage(content="Internal Error: Invalid tool call structure received from LLM.", tool_call_id="error_no_id")
         # Add error message and clear request (important!)
         return {"messages": [tool_msg], "tool_invocation_request": None}

    logger.info(f"Preparing to run tool: '{tool_name}', Args: {tool_args}, ID: {tool_call_id}")

    # Create the request object (without the image)
    request = ToolInvocationRequest(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args
    )
    logger.debug(f"Tool request prepared successfully for '{tool_name}'.")
    # Return the request to be stored in the state
    return {"tool_invocation_request": request}


def execute_tool_and_update(state: AgentState) -> Dict[str, Any]:
    """Node: Fetches image (if needed), executes the prepared tool implementation, updates state, and returns the ToolMessage result."""
    logger.info("Node: execute_tool_and_update - Starting execution")
    request: Optional[ToolInvocationRequest] = state.get("tool_invocation_request")

    # *** CRUCIAL: Clean up the request from the state immediately ***
    # This prevents re-execution if the graph loops or retries.
    # We'll add the result message later.
    updates_to_return: Dict[str, Any] = {"tool_invocation_request": None}

    # Validate the request fetched from the state
    if not request or not isinstance(request, dict) or not request.get("tool_name") or not request.get("tool_call_id"):
        logger.warning("execute_tool_and_update: Invalid or missing tool request in state. Skipping execution.")
        # Return only the cleanup update
        return updates_to_return

    tool_name = request["tool_name"]
    tool_args = request.get("tool_args", {})
    tool_call_id = request["tool_call_id"]
    logger.info(f"Executing tool '{tool_name}' with ID '{tool_call_id}' and args: {tool_args}")

    tool_message_content: str = f"Error: Tool implementation for '{tool_name}' not found."
    new_image: Optional[Image.Image] = None # Track if image was modified
    current_image: Optional[Image.Image] = None # To hold the fetched image if needed
    error_occurred = False

    # --- Check if tool implementation exists and needs an image ---
    needs_image = False
    tool_impl_func = tool_implementations.get(tool_name)

    if tool_impl_func:
        # Inspect the implementation function's signature to see if it expects 'input_image'
        try:
            import inspect
            sig = inspect.signature(tool_impl_func)
            if "input_image" in sig.parameters:
                needs_image = True
        except Exception as inspect_e:
            logger.warning(f"Could not inspect signature for {tool_name}: {inspect_e}. Assuming image not needed.")
    elif tool_name == 'get_image_info' and tool_name in available_tools:
        # Special case: get_image_info logic is in the @tool function
        needs_image = False # The tool itself handles state access
    else:
        logger.error(f"Tool implementation for '{tool_name}' not found in tool_implementations dictionary.")
        error_occurred = True
        # Keep default error message

    # --- Fetch image from Streamlit state if needed ---
    if needs_image and not error_occurred:
        logger.debug(f"Tool '{tool_name}' requires input image. Attempting to fetch from Streamlit state...")
        if _IN_STREAMLIT_CONTEXT:
            try:
                current_image_obj = st.session_state.get('processed_image')
                if current_image_obj is None:
                    tool_message_content = "Error: No image loaded in the editor to process."
                    logger.warning(tool_message_content)
                    error_occurred = True
                elif isinstance(current_image_obj, Image.Image):
                    # Make a copy to avoid modifying the state object directly during processing
                    current_image = current_image_obj.copy()
                    logger.info(f"Image (mode={current_image.mode}, size={current_image.size}) fetched and copied from Streamlit state.")
                else:
                    tool_message_content = f"Error: Object in state 'processed_image' is not a PIL Image (Type: {type(current_image_obj)})."
                    logger.error(tool_message_content)
                    error_occurred = True
            except Exception as e:
                logger.error(f"Error accessing st.session_state['processed_image']: {e}", exc_info=True)
                tool_message_content = "Error: Failed to access image from application state."
                error_occurred = True
        else:
            # Cannot get image if not in Streamlit context
            tool_message_content = "Error: Cannot access image state (Agent not running within Streamlit)."
            logger.warning(tool_message_content)
            error_occurred = True
    elif not error_occurred:
        logger.debug(f"Tool '{tool_name}' does not require input image.")

    # --- Execute the tool only if no errors occurred so far ---
    if not error_occurred:
        if tool_impl_func: # Standard case with _impl function
            logger.info(f"Executing implementation: {tool_impl_func.__name__} for tool: {tool_name}")
            try:
                # Prepare arguments for the implementation function
                impl_args = tool_args.copy()
                if needs_image:
                    if current_image: # Should always be true if needs_image and no error
                        impl_args["input_image"] = current_image
                    else:
                        # This case indicates a logic error above
                        raise ValueError(f"Internal Error: Tool '{tool_name}' needs image, but 'current_image' is None.")

                # Execute the actual tool logic
                tool_result = tool_impl_func(**impl_args)

                # Process the result tuple: (result_string, optional_modified_image)
                if isinstance(tool_result, tuple) and len(tool_result) == 2:
                    tool_message_content, returned_image = tool_result
                    if returned_image is not None and isinstance(returned_image, Image.Image):
                         new_image = returned_image # Store valid image for potential state update
                    elif returned_image is not None:
                         logger.error(f"Tool {tool_name} impl returned invalid image type: {type(returned_image)}")
                         tool_message_content = f"Error: Tool '{tool_name}' returned invalid image object."
                         # Don't store the invalid image
                elif isinstance(tool_result, str): # Handle tools returning only string (shouldn't happen for _impl funcs)
                    logger.warning(f"Tool {tool_name} impl returned only string, expected (str, Image|None).")
                    tool_message_content = tool_result
                    new_image = None
                else:
                     logger.error(f"Tool {tool_name} impl returned unexpected format: {type(tool_result)}")
                     tool_message_content = f"Error: Tool '{tool_name}' implementation returned unexpected format."
                     new_image = None

                logger.info(f"Tool '{tool_name}' executed. Result preview: {str(tool_message_content)[:100]}...")

                # --- Update Streamlit State (ONLY if in context AND image changed) ---
                if new_image is not None and _IN_STREAMLIT_CONTEXT:
                    try:
                        # Use the imported state manager function
                        update_success = update_processed_image(new_image)
                        if update_success:
                            logger.info(f"Streamlit session_state.processed_image updated by tool '{tool_name}'.")
                            # Optionally, update corresponding UI widgets if needed (e.g., sliders)
                            # This requires access to st.session_state here
                            # Example: if tool_name == 'adjust_brightness': st.session_state.brightness_slider = tool_args.get('factor')
                        else:
                            logger.warning(f"update_processed_image returned False for tool '{tool_name}'. State might not be updated.")
                            tool_message_content += " (Warning: Failed to update application state)"
                    except Exception as state_e:
                        logger.error(f"Failed to update Streamlit state after tool '{tool_name}': {state_e}", exc_info=True)
                        tool_message_content += " (Warning: Failed to update application UI state)"

            except Exception as e:
                logger.error(f"Error executing tool implementation '{tool_name}': {e}", exc_info=True)
                tool_message_content = f"Execution Error during {tool_name}: {str(e)}"
                new_image = None # Ensure image isn't updated on error
                error_occurred = True

        elif tool_name == 'get_image_info': # Special case handled by @tool function
             logger.info(f"Executing tool directly (info only): {tool_name}")
             try:
                 # Invoke the @tool function directly (it handles state access)
                 tool_message_content = available_tools[tool_name].invoke(tool_args) # Pass args just in case
             except Exception as e:
                 logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                 tool_message_content = f"Execution Error for {tool_name}: {str(e)}"
                 error_occurred = True
        # else: The 'implementation not found' error remains

    # --- Prepare ToolMessage and update graph state ---
    # Always add a ToolMessage, whether success or error, linked to the original call
    tool_msg = ToolMessage(content=str(tool_message_content), tool_call_id=tool_call_id)
    logger.debug(f"Prepared ToolMessage for ID {tool_call_id}: {tool_msg.content[:100]}...")

    # Return the message to be added to the state's message list, along with the request cleanup
    updates_to_return["messages"] = [tool_msg]

    return updates_to_return


# --- Graph Condition Functions ---

def route_after_agent(state: AgentState) -> Literal["prepare_tool_run", END]:
    """Checks the last message for tool calls to decide the next step."""
    logger.debug("Router: route_after_agent executing...")
    messages = state.get("messages", [])
    if not messages:
        logger.debug("Routing decision: END (no messages)")
        return END

    last_message = messages[-1]
    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
         # Ensure tool_calls is not empty
         if last_message.tool_calls:
             logger.debug("Routing decision: prepare_tool_run (tool calls found)")
             return "prepare_tool_run"
         else:
             logger.debug("Routing decision: END (AIMessage has empty tool_calls list)")
             return END
    else:
         logger.debug(f"Routing decision: END (last message type: {type(last_message).__name__}, has tool_calls: {hasattr(last_message, 'tool_calls')})")
         return END

def route_after_tool_prep(state: AgentState) -> Literal["execute_tool_and_update", "agent"]:
    """Checks if a tool request was successfully prepared."""
    logger.debug("Router: route_after_tool_prep executing...")
    # Check if the tool_invocation_request key exists and is not None/empty
    if state.get("tool_invocation_request"):
         logger.debug("Routing decision: execute_tool_and_update (request found)")
         return "execute_tool_and_update"
    else:
         # If no request, it means prepare_tool_run found no tool call initially,
         # or encountered an error (and should have added an error ToolMessage).
         # Go back to the agent to process the state (which might contain the error msg or just the previous AI msg).
         logger.debug("Routing decision: agent (no tool request prepared or cleared due to error)")
         return "agent"

# --- Graph Construction ---
def build_graph():
    """Constructs and compiles the LangGraph agent workflow."""
    if not _DEPENDENCIES_LOADED:
         logger.critical("Cannot build graph, agent dependencies failed to load during import.")
         return None # Prevent graph building if core components are missing

    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState) # Use the correct state definition

    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("prepare_tool_run", prepare_tool_run)
    workflow.add_node("execute_tool_and_update", execute_tool_and_update)

    # Define entry point
    workflow.add_edge(START, "agent")

    # Define conditional edges
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "prepare_tool_run": "prepare_tool_run",
            END: END
        }
    )
    workflow.add_conditional_edges(
         "prepare_tool_run",
         route_after_tool_prep,
         {
              "execute_tool_and_update": "execute_tool_and_update",
              "agent": "agent" # Loop back if prep failed or no tool call
         }
    )

    # Always return to the agent after executing a tool to process the result
    workflow.add_edge("execute_tool_and_update", "agent")

    # Configure memory for persistence (allows chat history)
    memory = MemorySaver()

    # Compile the graph
    try:
        # Add interrupt_before to potentially pause before executing tools if needed for debugging
        # graph = workflow.compile(checkpointer=memory, interrupt_before=["execute_tool_and_update"])
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
# Attempt to build the graph when the module is loaded.
compiled_graph = build_graph()

if compiled_graph:
    logger.info("Global 'compiled_graph' instance created successfully.")
else:
    logger.error("Global 'compiled_graph' instance IS NONE due to build failure. Agent will be unavailable.")


# --- Direct Execution Block (for testing graph structure if needed) ---
if __name__ == "__ma# --- Standard Library Imports ---
import os
import sys
from pathlib import Path
import logging
from typing import Literal, Optional, Dict, Any, Tuple

# --- Path Setup (Add Project Root) ---
try:
    # Project root is the parent directory of the 'agent' directory
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
from PIL import Image # Needed for isinstance checks

# --- Local Application Imports (Use Paths Relative to Project Root) ---
_DEPENDENCIES_LOADED = False
try:
    # Imports are now relative to the project root added to sys.path
    from agent.graph_state import AgentState, ToolInvocationRequest
    from agent.tools import available_tools, tool_implementations
    # Import the state manager for updating the image
    from state.session_state_manager import update_processed_image
    _DEPENDENCIES_LOADED = True
    print("DEBUG (agent_graph.py): Successfully imported agent dependencies.")
except ImportError as e:
    print(f"ERROR (agent_graph.py): Failed to import agent dependencies using absolute paths: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define dummies if import fails to allow basic structure loading
    class AgentState(dict): pass
    class ToolInvocationRequest(dict): pass
    available_tools = {}
    tool_implementations = {}
    def update_processed_image(img): print("[MOCK] update_processed_image called")

# --- Streamlit Import (Conditional) ---
_IN_STREAMLIT_CONTEXT = False
try:
    import streamlit as st
    # Check for a streamlit-specific attribute to confirm context
    if hasattr(st, 'secrets'):
         _IN_STREAMLIT_CONTEXT = True
except (ImportError, RuntimeError):
    pass # Fail silently if streamlit is not available or not running

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {log_level}")
logger.info(f"Dependencies Loaded: {_DEPENDENCIES_LOADED}")
logger.info(f"Streamlit Context: {_IN_STREAMLIT_CONTEXT}")

# --- LLM and Agent Executor Configuration (Cached) ---
_cached_llm = None
def get_llm():
    """Safely initializes and returns the ChatOpenAI model, using a simple cache."""
    global _cached_llm
    if _cached_llm: return _cached_llm

    api_key = None; source = "Not Found"
    if _IN_STREAMLIT_CONTEXT:
        try:
            key = st.secrets.get("OPENAI_API_KEY") # Use .get for safety
            if key: api_key = key; source = "Streamlit Secrets"
        except Exception as e: logger.warning(f"Error accessing St secrets for OpenAI Key: {e}")
    if not api_key:
        key = os.environ.get("OPENAI_API_KEY")
        if key: api_key = key; source = "Env Var"

    if not api_key:
        message = "OpenAI API Key not found (Checked Streamlit Secrets & OPENAI_API_KEY env var). AI Agent cannot function."
        logger.error(message)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

    logger.info(f"OpenAI API Key loaded from: {source}")
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini", # Consider making model name configurable
            temperature=0,
            api_key=api_key,
            max_retries=2
        )
        _cached_llm = model
        return model
    except Exception as e:
        message = f"Failed to initialize ChatOpenAI model: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

_cached_agent_executor = None
def get_agent_executor():
    """Creates the agent executor by binding tools to the LLM, using a simple cache."""
    global _cached_agent_executor
    if _cached_agent_executor: return _cached_agent_executor

    llm = get_llm()
    if llm is None: return None # Error logged previously
    if not available_tools:
        logger.error("No tools available for agent binding. Check tools.py.")
        return None

    try:
        # Bind the TOOL DEFINITIONS (schemas from @tool)
        llm_with_tools = llm.bind_tools(list(available_tools.values()))
        logger.info(f"LLM bound with {len(available_tools)} tools: {list(available_tools.keys())}")
        _cached_agent_executor = llm_with_tools
        return llm_with_tools
    except Exception as e:
        message = f"CRITICAL: Failed to bind tools to LLM. Check tool definitions/schemas in agent/tools.py. Error: {e}"
        logger.error(message, exc_info=True)
        if _IN_STREAMLIT_CONTEXT:
            try: st.error(message)
            except Exception as streamlit_e: logger.warning(f"Failed to show Streamlit error: {streamlit_e}")
        return None

# --- Graph Node Definitions ---

def call_agent(state: AgentState) -> Dict[str, Any]:
    """Node: Invokes the LLM agent with the current message history."""
    logger.info("Node: call_agent - Starting execution")
    agent_executor = get_agent_executor()
    if agent_executor is None:
        logger.error("Agent executor unavailable in call_agent.")
        # Return error message to be added to state
        return {"messages": [SystemMessage(content="LLM Error: Agent executor not configured or failed to initialize.")]}

    messages = state.get("messages", [])
    if not messages:
         logger.warning("call_agent invoked with empty message history. Skipping LLM call.")
         return {} # Return empty dict - state doesn't change

    # Filter out any potential None or non-BaseMessage entries
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    if len(valid_messages) != len(messages):
         logger.warning(f"Filtered out {len(messages) - len(valid_messages)} invalid entries from messages list.")

    if not valid_messages:
        logger.error("call_agent: No valid messages found after filtering. Cannot invoke LLM.")
        return {"messages": [SystemMessage(content="Internal Error: No valid messages to send to LLM.")]}

    logger.debug(f"Invoking agent with {len(valid_messages)} valid messages. Last: {valid_messages[-1].pretty_repr()}")
    try:
        # Invoke the LLM bound with tool definitions
        response = agent_executor.invoke(valid_messages)
        logger.info(f"Agent response received: Type={type(response).__name__}, ID={getattr(response, 'id', 'N/A')}, ToolCalls={bool(getattr(response, 'tool_calls', None))}")
        # Return the response to be added to the 'messages' list by the AgentState reducer
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        # Return a SystemMessage indicating the error
        return {"messages": [SystemMessage(content=f"Error during model invocation: {e}")]}


def prepare_tool_run(state: AgentState) -> Dict[str, Any]:
    """Node: Extracts tool call details from the last AI message."""
    logger.info("Node: prepare_tool_run - Starting execution")
    messages = state.get("messages", [])
    if not messages:
        logger.warning("prepare_tool_run: No messages in state.")
        return {"tool_invocation_request": None} # No request if no messages

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if not isinstance(last_message, AIMessage) or not getattr(last_message, 'tool_calls', None):
        logger.debug("prepare_tool_run: Last message is not an AIMessage with tool calls. Clearing any pending request.")
        return {"tool_invocation_request": None}

    # Process the first tool call (can be extended for parallel calls)
    if not last_message.tool_calls: # Should be caught by getattr check, but double-check
        logger.warning("prepare_tool_run: AIMessage has tool_calls attribute but it's empty.")
        return {"tool_invocation_request": None}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")

    if not tool_name or not tool_call_id:
         logger.error(f"Invalid tool_call structure received from LLM: {tool_call}")
         # Create an error message to send back
         tool_msg = ToolMessage(content="Internal Error: Invalid tool call structure received from LLM.", tool_call_id="error_no_id")
         # Add error message and clear request (important!)
         return {"messages": [tool_msg], "tool_invocation_request": None}

    logger.info(f"Preparing to run tool: '{tool_name}', Args: {tool_args}, ID: {tool_call_id}")

    # Create the request object (without the image)
    request = ToolInvocationRequest(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args
    )
    logger.debug(f"Tool request prepared successfully for '{tool_name}'.")
    # Return the request to be stored in the state
    return {"tool_invocation_request": request}


def execute_tool_and_update(state: AgentState) -> Dict[str, Any]:
    """Node: Fetches image (if needed), executes the prepared tool implementation, updates state, and returns the ToolMessage result."""
    logger.info("Node: execute_tool_and_update - Starting execution")
    request: Optional[ToolInvocationRequest] = state.get("tool_invocation_request")

    # *** CRUCIAL: Clean up the request from the state immediately ***
    # This prevents re-execution if the graph loops or retries.
    # We'll add the result message later.
    updates_to_return: Dict[str, Any] = {"tool_invocation_request": None}

    # Validate the request fetched from the state
    if not request or not isinstance(request, dict) or not request.get("tool_name") or not request.get("tool_call_id"):
        logger.warning("execute_tool_and_update: Invalid or missing tool request in state. Skipping execution.")
        # Return only the cleanup update
        return updates_to_return

    tool_name = request["tool_name"]
    tool_args = request.get("tool_args", {})
    tool_call_id = request["tool_call_id"]
    logger.info(f"Executing tool '{tool_name}' with ID '{tool_call_id}' and args: {tool_args}")

    tool_message_content: str = f"Error: Tool implementation for '{tool_name}' not found."
    new_image: Optional[Image.Image] = None # Track if image was modified
    current_image: Optional[Image.Image] = None # To hold the fetched image if needed
    error_occurred = False

    # --- Check if tool implementation exists and needs an image ---
    needs_image = False
    tool_impl_func = tool_implementations.get(tool_name)

    if tool_impl_func:
        # Inspect the implementation function's signature to see if it expects 'input_image'
        try:
            import inspect
            sig = inspect.signature(tool_impl_func)
            if "input_image" in sig.parameters:
                needs_image = True
        except Exception as inspect_e:
            logger.warning(f"Could not inspect signature for {tool_name}: {inspect_e}. Assuming image not needed.")
    elif tool_name == 'get_image_info' and tool_name in available_tools:
        # Special case: get_image_info logic is in the @tool function
        needs_image = False # The tool itself handles state access
    else:
        logger.error(f"Tool implementation for '{tool_name}' not found in tool_implementations dictionary.")
        error_occurred = True
        # Keep default error message

    # --- Fetch image from Streamlit state if needed ---
    if needs_image and not error_occurred:
        logger.debug(f"Tool '{tool_name}' requires input image. Attempting to fetch from Streamlit state...")
        if _IN_STREAMLIT_CONTEXT:
            try:
                current_image_obj = st.session_state.get('processed_image')
                if current_image_obj is None:
                    tool_message_content = "Error: No image loaded in the editor to process."
                    logger.warning(tool_message_content)
                    error_occurred = True
                elif isinstance(current_image_obj, Image.Image):
                    # Make a copy to avoid modifying the state object directly during processing
                    current_image = current_image_obj.copy()
                    logger.info(f"Image (mode={current_image.mode}, size={current_image.size}) fetched and copied from Streamlit state.")
                else:
                    tool_message_content = f"Error: Object in state 'processed_image' is not a PIL Image (Type: {type(current_image_obj)})."
                    logger.error(tool_message_content)
                    error_occurred = True
            except Exception as e:
                logger.error(f"Error accessing st.session_state['processed_image']: {e}", exc_info=True)
                tool_message_content = "Error: Failed to access image from application state."
                error_occurred = True
        else:
            # Cannot get image if not in Streamlit context
            tool_message_content = "Error: Cannot access image state (Agent not running within Streamlit)."
            logger.warning(tool_message_content)
            error_occurred = True
    elif not error_occurred:
        logger.debug(f"Tool '{tool_name}' does not require input image.")

    # --- Execute the tool only if no errors occurred so far ---
    if not error_occurred:
        if tool_impl_func: # Standard case with _impl function
            logger.info(f"Executing implementation: {tool_impl_func.__name__} for tool: {tool_name}")
            try:
                # Prepare arguments for the implementation function
                impl_args = tool_args.copy()
                if needs_image:
                    if current_image: # Should always be true if needs_image and no error
                        impl_args["input_image"] = current_image
                    else:
                        # This case indicates a logic error above
                        raise ValueError(f"Internal Error: Tool '{tool_name}' needs image, but 'current_image' is None.")

                # Execute the actual tool logic
                tool_result = tool_impl_func(**impl_args)

                # Process the result tuple: (result_string, optional_modified_image)
                if isinstance(tool_result, tuple) and len(tool_result) == 2:
                    tool_message_content, returned_image = tool_result
                    if returned_image is not None and isinstance(returned_image, Image.Image):
                         new_image = returned_image # Store valid image for potential state update
                    elif returned_image is not None:
                         logger.error(f"Tool {tool_name} impl returned invalid image type: {type(returned_image)}")
                         tool_message_content = f"Error: Tool '{tool_name}' returned invalid image object."
                         # Don't store the invalid image
                elif isinstance(tool_result, str): # Handle tools returning only string (shouldn't happen for _impl funcs)
                    logger.warning(f"Tool {tool_name} impl returned only string, expected (str, Image|None).")
                    tool_message_content = tool_result
                    new_image = None
                else:
                     logger.error(f"Tool {tool_name} impl returned unexpected format: {type(tool_result)}")
                     tool_message_content = f"Error: Tool '{tool_name}' implementation returned unexpected format."
                     new_image = None

                logger.info(f"Tool '{tool_name}' executed. Result preview: {str(tool_message_content)[:100]}...")

                # --- Update Streamlit State (ONLY if in context AND image changed) ---
                if new_image is not None and _IN_STREAMLIT_CONTEXT:
                    try:
                        # Use the imported state manager function
                        update_success = update_processed_image(new_image)
                        if update_success:
                            logger.info(f"Streamlit session_state.processed_image updated by tool '{tool_name}'.")
                            # Optionally, update corresponding UI widgets if needed (e.g., sliders)
                            # This requires access to st.session_state here
                            # Example: if tool_name == 'adjust_brightness': st.session_state.brightness_slider = tool_args.get('factor')
                        else:
                            logger.warning(f"update_processed_image returned False for tool '{tool_name}'. State might not be updated.")
                            tool_message_content += " (Warning: Failed to update application state)"
                    except Exception as state_e:
                        logger.error(f"Failed to update Streamlit state after tool '{tool_name}': {state_e}", exc_info=True)
                        tool_message_content += " (Warning: Failed to update application UI state)"

            except Exception as e:
                logger.error(f"Error executing tool implementation '{tool_name}': {e}", exc_info=True)
                tool_message_content = f"Execution Error during {tool_name}: {str(e)}"
                new_image = None # Ensure image isn't updated on error
                error_occurred = True

        elif tool_name == 'get_image_info': # Special case handled by @tool function
             logger.info(f"Executing tool directly (info only): {tool_name}")
             try:
                 # Invoke the @tool function directly (it handles state access)
                 tool_message_content = available_tools[tool_name].invoke(tool_args) # Pass args just in case
             except Exception as e:
                 logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                 tool_message_content = f"Execution Error for {tool_name}: {str(e)}"
                 error_occurred = True
        # else: The 'implementation not found' error remains

    # --- Prepare ToolMessage and update graph state ---
    # Always add a ToolMessage, whether success or error, linked to the original call
    tool_msg = ToolMessage(content=str(tool_message_content), tool_call_id=tool_call_id)
    logger.debug(f"Prepared ToolMessage for ID {tool_call_id}: {tool_msg.content[:100]}...")

    # Return the message to be added to the state's message list, along with the request cleanup
    updates_to_return["messages"] = [tool_msg]

    return updates_to_return


# --- Graph Condition Functions ---

def route_after_agent(state: AgentState) -> Literal["prepare_tool_run", END]:
    """Checks the last message for tool calls to decide the next step."""
    logger.debug("Router: route_after_agent executing...")
    messages = state.get("messages", [])
    if not messages:
        logger.debug("Routing decision: END (no messages)")
        return END

    last_message = messages[-1]
    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
         # Ensure tool_calls is not empty
         if last_message.tool_calls:
             logger.debug("Routing decision: prepare_tool_run (tool calls found)")
             return "prepare_tool_run"
         else:
             logger.debug("Routing decision: END (AIMessage has empty tool_calls list)")
             return END
    else:
         logger.debug(f"Routing decision: END (last message type: {type(last_message).__name__}, has tool_calls: {hasattr(last_message, 'tool_calls')})")
         return END

def route_after_tool_prep(state: AgentState) -> Literal["execute_tool_and_update", "agent"]:
    """Checks if a tool request was successfully prepared."""
    logger.debug("Router: route_after_tool_prep executing...")
    # Check if the tool_invocation_request key exists and is not None/empty
    if state.get("tool_invocation_request"):
         logger.debug("Routing decision: execute_tool_and_update (request found)")
         return "execute_tool_and_update"
    else:
         # If no request, it means prepare_tool_run found no tool call initially,
         # or encountered an error (and should have added an error ToolMessage).
         # Go back to the agent to process the state (which might contain the error msg or just the previous AI msg).
         logger.debug("Routing decision: agent (no tool request prepared or cleared due to error)")
         return "agent"

# --- Graph Construction ---
def build_graph():
    """Constructs and compiles the LangGraph agent workflow."""
    if not _DEPENDENCIES_LOADED:
         logger.critical("Cannot build graph, agent dependencies failed to load during import.")
         return None # Prevent graph building if core components are missing

    logger.info("Building LangGraph workflow...")
    workflow = StateGraph(AgentState) # Use the correct state definition

    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("prepare_tool_run", prepare_tool_run)
    workflow.add_node("execute_tool_and_update", execute_tool_and_update)

    # Define entry point
    workflow.add_edge(START, "agent")

    # Define conditional edges
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "prepare_tool_run": "prepare_tool_run",
            END: END
        }
    )
    workflow.add_conditional_edges(
         "prepare_tool_run",
         route_after_tool_prep,
         {
              "execute_tool_and_update": "execute_tool_and_update",
              "agent": "agent" # Loop back if prep failed or no tool call
         }
    )

    # Always return to the agent after executing a tool to process the result
    workflow.add_edge("execute_tool_and_update", "agent")

    # Configure memory for persistence (allows chat history)
    memory = MemorySaver()

    # Compile the graph
    try:
        # Add interrupt_before to potentially pause before executing tools if needed for debugging
        # graph = workflow.compile(checkpointer=memory, interrupt_before=["execute_tool_and_update"])
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
# Attempt to build the graph when the module is loaded.
compiled_graph = build_graph()

if compiled_graph:
    logger.info("Global 'compiled_graph' instance created successfully.")
else:
    logger.error("Global 'compiled_graph' instance IS NONE due to build failure. Agent will be unavailable.")


# --- Direct Execution Block (for testing graph structure if needed) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for direct testing
    logger.info(f"--- Running {__file__} directly for testing ---")
    if compiled_graph:
        logger.info("Graph compiled successfully. Ready for testing.")
        # Example test (requires mocks in tools.py to work standalone):
        # config = {"configurable": {"thread_id": "direct_test_thread"}}
        # test_input = {"messages": [HumanMessage(content="Make the image brighter by 20")]}
        # logger.info(f"Invoking graph with input: {test_input}")
        # try:
        #     for event in compiled_graph.stream(test_input, config, stream_mode="values"):
        #         logger.info(f"Graph Event: {event}")
        # except Exception as test_e:
        #     logger.error(f"Graph invocation test failed: {test_e}", exc_info=True)
    else:
        logger.error("Graph compilation FAILED during module load. Cannot run tests.")
    logger.info(f"--- Finished {__file__} direct test ---")
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for direct testing
    logger.info(f"--- Running {__file__} directly for testing ---")
    if compiled_graph:
        logger.info("Graph compiled successfully. Ready for testing.")
        # Example test (requires mocks in tools.py to work standalone):
        # config = {"configurable": {"thread_id": "direct_test_thread"}}
        # test_input = {"messages": [HumanMessage(content="Make the image brighter by 20")]}
        # logger.info(f"Invoking graph with input: {test_input}")
        # try:
        #     for event in compiled_graph.stream(test_input, config, stream_mode="values"):
        #         logger.info(f"Graph Event: {event}")
        # except Exception as test_e:
        #     logger.error(f"Graph invocation test failed: {test_e}", exc_info=True)
    else:
        logger.error("Graph compilation FAILED during module load. Cannot run tests.")
    logger.info(f"--- Finished {__file__} direct test ---")