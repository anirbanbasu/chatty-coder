# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various utility functions used in the project."""

import os
from typing import Any
import constants


def parse_env(
    var_name: str,
    default_value: str | None = None,
    type_cast=str,
    convert_to_list=False,
    list_split_char=constants.SPACE_STRING,
) -> Any | list[Any]:
    """
    Parse the environment variable and return the value.

    Args:
        var_name (str): The name of the environment variable.
        default_value (str | None): The default value to use if the environment variable is not set. Defaults to None.
        type_cast (str): The type to cast the value to.
        convert_to_list (bool): Whether to convert the value to a list.
        list_split_char (str): The character to split the list on.

    Returns:
        (Any | list[Any]) The parsed value, either as a single value or a list. The type of the returned single
        value or individual elements in the list depends on the supplied type_cast parameter.
    """
    if os.getenv(var_name) is None and default_value is None:
        raise ValueError(
            f"Environment variable {var_name} does not exist and a default value has not been provided."
        )
    parsed_value = None
    if type_cast is bool:
        parsed_value = (
            os.getenv(var_name, default_value).lower() in constants.TRUE_VALUES_LIST
        )
    else:
        parsed_value = os.getenv(var_name, default_value)

    value: Any | list[Any] = (
        type_cast(parsed_value)
        if not convert_to_list
        else [type_cast(v) for v in parsed_value.split(list_split_char)]
    )
    return value


def get_terminal_size(fallback=(100, 25)) -> tuple[int, int]:
    """
    Get the terminal size.
    See: https://granitosaur.us/getting-terminal-size.html

    Args:
        fallback (tuple[int, int]): The fallback size to use if the terminal size cannot be determined.

    Returns:
        tuple[int, int]: The terminal size as a tuple of (columns, rows).
    """
    for i in range(0, 3):
        try:
            columns, rows = os.get_terminal_size(i)
        except OSError:
            continue
        break
    else:  # set default if the loop completes which means all failed
        columns, rows = fallback
    return columns, rows


def check_list_subset(list_a: list[Any], list_b: list[Any]) -> list[Any]:
    """
    Check if the elements of list_a forms a set that is a subset of the set formed by the elements of list_a.
    This includes the cases where list_a is an empty list, and list_a contains all the elements of list_b.

    Args:
        list_a (list[Any]): The first list.
        list_b (list[Any]): The second list.

    Returns:
        list[Any]: The distinct elements of list_a that are not in list_b. This should be an empty list if list_a
        is a subset of list_b.
    """
    s1 = set(list_a)
    s2 = set(list_b)
    return list(s1 - s2)


def remove_markdown_codeblock_backticks(text: str) -> str:
    """
    Remove all occurrences of the Markdown code block backticks from the text.

    Args:
        text (str): The text to strip the backticks from.

    Returns:
        str: The text with the backticks stripped.
    """
    return text.replace(
        constants.MARKDOWN_CODE_BLOCK_THREE_BACKTICKS, constants.EMPTY_STRING
    )
