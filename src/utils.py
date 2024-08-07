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
    self,
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
