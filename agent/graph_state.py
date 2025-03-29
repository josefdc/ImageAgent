# streamlit_image_editor/agent/graph_state.py
# Defines the state structure for the LangGraph agent.

from typing import List, Optional, Tuple, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# State for passing tool execution details between nodes
class ToolInvocationRequest(TypedDict, total=False):
    """Holds the details needed to execute a specific tool call."""
    tool_call_id: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]

# Main state for the graph
class AgentState(TypedDict):
    """The overall state of the agent conversation and pending actions."""
    # Chat history managed by LangGraph's add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]

    # Pending tool request details (cleared after execution attempt)
    tool_invocation_request: Optional[ToolInvocationRequest]

    # --- NUEVO: Actualizaciones de UI pendientes ---
    # Diccionario para almacenar {widget_key: new_value} a aplicar en Streamlit
    pending_ui_updates: Optional[Dict[str, Any]]