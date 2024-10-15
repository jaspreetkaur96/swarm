# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    """
    Orchestrates conversations with AI agents, allowing for the execution of custom
    functions and tool calls within the conversation. It handles chat completion,
    tool calls, and function results, providing a flexible framework for complex
    interactions.

    Attributes:
        client (OpenAI|None): Initialized in the `__init__` method. It is used to
            interact with the OpenAI API.

    """
    def __init__(self, client=None):
        """
        Initializes a Swarm object with an optional OpenAI client instance. If no
        client is provided, it defaults to creating a new instance of OpenAI.

        Args:
            client (OpenAI | None): Defaulted to None.

        """
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        """
        Constructs a chat completion request by combining user history, system
        instructions, and context variables, then sends it to a chat client for processing.

        Args:
            agent (Agent): Used to retrieve instructions from its `instructions`
                method or attribute.
            history (List): Represented by a list of messages, where each message
                is a dictionary containing a "role" and a "content".
            context_variables (dict): Converted to a `defaultdict` with a default
                value of an empty string. This allows it to handle missing keys.
            model_override (str): Optional, allowing the user to override the
                default model with a specific model.
            stream (bool): Used to control whether the chat completion is streamed
                or not.
            debug (bool): Used to control the display of debug messages, likely
                print statements, within the function.

        Returns:
            ChatCompletionMessage: An object representing the completion of a chat.

        """
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Result:
        """
        Handles the result of a function call, returning a Result object based on
        the type of the result. It supports Result, Agent, and other types,
        attempting to convert them to strings if necessary and logging errors if
        conversion fails.

        Args:
            result (Result | Agent | Any): Matched against different types using
                a pattern matching statement, allowing for various types of results
                from function calls.
            debug (bool): Used to control debug printing. It is likely used to
                enable or disable the printing of debug messages, with `True`
                indicating that debug messages should be printed and `False`
                indicating that they should not.

        Returns:
            Result: An object containing a 'value' and an optional 'agent' attribute.

        """
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        """
        Processes a list of tool calls, executing corresponding functions from a
        given list of agent functions and returning a response with results and
        potentially updated context variables.

        Args:
            tool_calls (List[ChatCompletionMessageToolCall]): Expected to contain
                a list of tool calls.
            functions (List[AgentFunction]): Containing a list of AgentFunction objects.
            context_variables (dict): Used to store variables that are shared
                across multiple function calls. It is referenced by the special
                key `__CTX_VARS_NAME__` in the function code.
            debug (bool): Used to enable or disable debug printing within the function.

        Returns:
            Response: A composite object containing a list of messages, an optional
            agent, and an updated context.

        """
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        """
        Simulates a conversation between an agent and a chat model, streaming
        intermediate results and executing tool calls as needed, until a maximum
        number of turns is reached or tool calls are exhausted.

        Args:
            agent (Agent): Assigned to the `active_agent` variable. It represents
                an active participant in the conversation and is used to interact
                with the chat completion model.
            messages (List): Filled with the conversation history.
            context_variables (dict): Defaulted to an empty dictionary. It is used
                to store and update context information throughout the conversation.
            model_override (str): Optional. It allows for the specification of a
                custom model to be used for chat completion.
            debug (bool): Used to enable or disable debug printing. When `debug`
                is True, debug messages are printed.
            max_turns (int): Optional. It specifies the maximum number of turns
                to be executed. If not provided, the function will execute indefinitely.
            execute_tools (bool): Set to True by default. It controls whether to
                execute tool calls in the completion messages.

        Yields:
            Dict[str,Any]|str: A stream of messages, including the start and end
            of each turn, tool calls, and the final response.

        """
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        """
        Simulates a conversation between an agent and a model, processing messages
        in a loop until a maximum number of turns is reached or a specific condition
        is met, handling tool calls and updating context variables as necessary.

        Args:
            agent (Agent): Required for the function to execute. It is used to
                interact with a conversational model or AI system.
            messages (List): Expected to contain a list of messages.
            context_variables (dict): Optional. It represents a dictionary of
                variables that are used to provide additional context to the conversation.
            model_override (str): Optional. It allows the caller to override the
                model used in the chat completion process.
            stream (bool): Set to True by default, it determines whether to run
                the function in stream mode or not. If True, the function calls
                `run_and_stream` instead of executing the main while loop.
            debug (bool): Used to control the level of debugging information printed
                during the execution of the function.
            max_turns (int): Defaulted to infinity, meaning the loop will run
                indefinitely unless a finite number is provided as the argument.
            execute_tools (bool): True by default. It determines whether to execute
                tool calls or not. If `execute_tools` is False, tool calls are
                ignored, and messages are appended to the history without execution.

        Returns:
            Response: A dictionary containing the following keys:
            
            - `messages`: a list of messages exchanged during the conversation
            - `agent`: the current active agent
            - `context_variables`: a dictionary of variables used in the conversation

        """
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
