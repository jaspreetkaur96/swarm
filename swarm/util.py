import inspect
from datetime import datetime


def debug_print(debug: bool, *args: str) -> None:
    """
    Prints a timestamped message in a specific color scheme, but only if the `debug`
    flag is `True`. It takes a variable number of string arguments, which are
    joined into a single message before being printed.

    Args:
        debug (bool): Used to determine whether to print debug messages. When
            `debug` is `False`, the function returns immediately without printing
            any message.
        *args (str): List of positional arguments

    """
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    """
    Merges nested dictionaries, concatenating string values and recursively merging
    nested dictionaries. It takes two arguments: `target` and `source`, where
    `source` is merged into `target`.

    Args:
        target (Dict[str | int | dict | None, str | int | dict | None].): Used to
            accumulate merged fields from the `source` parameter. It is expected
            to be a dictionary, but its initial value is not specified in the
            function definition.
        source (Dict[str | int | List[str] | Dict[str, int]]): Expected to have
            key-value pairs where the key is a string and the value can be a string,
            a dictionary, a list of strings, or an integer.

    """
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    """
    Updates a dictionary `final_response` by merging it with a delta `delta`. It
    removes the "role" key from `delta` and then merges the remaining keys into
    `final_response`. It also handles tool calls by merging the first tool call
    into the corresponding list in `final_response`.

    Args:
        final_response (dict): Modified by the function to merge data from the
            `delta` parameter into it.
        delta (dict): Used to merge changes into the `final_response` dictionary.
            It appears to be a collection of updates, possibly from a previous
            response. The `delta` dictionary is modified within the function,
            removing the "role" key.

    """
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a given function's metadata into a JSON-compatible dictionary. It
    extracts the function's name, description, and parameters, including their
    types and whether they are required.

    Args:
        func (Callable[[ typing.Callable ], dict]): Used to represent a function
            whose signature is to be converted to JSON format. It is expected to
            be a callable object, such as a function or method.

    Returns:
        dict: A JSON representation of a function.

    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
