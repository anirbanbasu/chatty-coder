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

"""Stuff common to all the workflows."""

from llama_index.core.workflow import (
    Event,
)

from llama_index.core.bridge.pydantic import BaseModel, Field

from constants import EMPTY_STRING


class TestCase(BaseModel):
    inputs: str = Field(
        default=EMPTY_STRING,
        description="Space separated list of inputs to the test case.",
    )
    outputs: str = Field(
        default=EMPTY_STRING,
        description="Expected outputs of the test case.",
    )


# Generic Events
class WorkflowStatusEvent(Event):
    """
    Event to update the status of the workflow.

    Fields:
        msg (str): The message to display.
        total_steps (int): Optional total number of steps, defaults to zero.
        finished_steps (int): Optional number of steps finished, defaults to zero.
    """

    msg: str
    total_steps: int = 0
    finished_steps: int = 0


class WorkflowStatusWithPartialSolution(Event):
    """
    Event to update the status of the workflow with a partial solution.

    Fields:
        msg (str): The message to display.
        partial_solution (str): The partial solution.
        partial_pseudo_code (str): The partial pseudo code.
        partial_code (str): The partial code.
        total_steps (int): Optional total number of steps, defaults to zero.
        finished_steps (int): Optional number of steps finished, defaults to zero.
    """

    msg: str
    partial_solution: str
    partial_pseudo_code: str
    partial_code: str
    total_steps: int = 0
    finished_steps: int = 0
