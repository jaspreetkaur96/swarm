from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    """
    Defines a model for an artificial agent with attributes for its name, model
    type, instructions, available functions, tool choice, and parallel tool call
    capability. This class is likely used in a larger system for agent-based
    interactions or simulations.

    Attributes:
        name (str): Initialized with a default value of "Agent".
        model (str): Initialized with the value "gpt-4o".
        instructions (Union[str, Callable[[], str]]): Initialized with the string
            "You are a helpful agent.". It can hold either a string or a callable
            function that returns a string.
        functions (List[AgentFunction]): Initialized as an empty list. It appears
            to store a collection of AgentFunction objects, which are not defined
            in this code snippet.
        tool_choice (str): Initialized as None. This suggests that it is an optional
            choice that can be set to a specific string value, indicating the
            selected tool.
        parallel_tool_calls (bool): Set to True by default, indicating that parallel
            tool calls are enabled for the agent.

    """
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    """
    Represents a response to a user query, containing a list of messages, an
    optional agent, and a dictionary of context variables.

    Attributes:
        messages (List): Initialized with an empty list. It appears to be a
            collection of messages, likely related to a conversation or user interaction.
        agent (Optional[Agent]): Initialized with a value of None. This suggests
            that the agent is an optional component of the response, and its
            presence or absence may depend on the specific requirements of the application.
        context_variables (dict): Initialized as an empty dictionary. It stores
            variables that are relevant to the current conversation context.

    """
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Defines a data model for representing the outcome of an operation or process.
    It contains three attributes: `value` (a string), `agent` (an optional instance
    of the `Agent` class), and `context_variables` (a dictionary of context-specific
    variables).

    Attributes:
        value (str): Initialized with an empty string.
        agent (Optional[Agent]): Initialized with a value of None. This means it
            can be either an instance of the `Agent` class or None.
        context_variables (dict): Initialized as an empty dictionary. It stores
            key-value pairs of context variables, which are likely used to provide
            additional information about the result.

    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
