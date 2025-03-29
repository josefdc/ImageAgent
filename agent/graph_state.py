# agent/graph_state.py
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
    # Use Annotated with add_messages reducer for proper chat history management.
    # This ensures ToolMessages are added correctly after AIMessages with tool_calls.
    messages: Annotated[List[BaseMessage], add_messages]
    # Store the pending tool request separately to avoid interfering with message history reducer
    tool_invocation_request: Optional[ToolInvocationRequest]  # Contains only name, args, id