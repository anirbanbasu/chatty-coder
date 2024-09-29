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

from typing import Any, Optional
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.prompts import PromptTemplate

from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool

from constants import CHAR_ENCODING_UTF8
from workflows.common import WorkflowStatusEvent

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class DraftSolutionResultEvent(Event):
    """
    Event to store the draft solution.

    Fields:
        reasoning (str): The reasoning behind the solution.
        pseudocode (str): The pseudocode for the solution.
        code (str): The code for the solution.
    """

    reasoning: str
    pseudocode: str
    code: str


class SelfReflectiveCoderWorkflow(Workflow):
    PROMPT_TEMPLATE_DRAFT_SOLVE = PromptTemplate(
        "You are a world-class Python programmer."
        "You write concise and well-documented code following the PEP8 style guide. Please respond with a Python 3 solution to the given problem below."
        "\nFirst, output a detailed `reasoning` through the problem and conceptualise a solution. Whenever possible, add a time and a space complexity analysis for your solution."
        "Then, output a complete `pseudocode` in Pascal to implement your concept solution."
        "Finally, output a well-documented and working Python 3 `code` for your solution. Do not use external libraries."
        "Make your Python code executable from the command line. Your code must accept all its inputs from `sys.stdin` and print the final output to `sys.stdout`."
        "Please format your response as a JSON dictionary, using `reasoning`, `pseudocode`, and `code` as keys."
        "Please format the values of these JSON keys as Markdown."
        "\n\n[BEGIN PROBLEM]\n{problem}\n[END PROBLEM]"
    )

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        draft_solve_prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.llm = llm
        self.tools = tools
        self.draft_solve_prompt = (
            draft_solve_prompt
            or SelfReflectiveCoderWorkflow.PROMPT_TEMPLATE_DRAFT_SOLVE
        )
        self._total_steps = 0
        self._finished_steps = 0

    @step
    async def draft_solve(self, ctx: Context, ev: StartEvent) -> StopEvent:
        # Draft the solution
        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg="Drafting the solution...",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        response: DraftSolutionResultEvent = await self.llm.astructured_predict(
            output_cls=DraftSolutionResultEvent,
            prompt=self.draft_solve_prompt,
            problem=ev.problem,
        )
        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg="Obtained a solution...",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        # ic(response)
        return StopEvent(
            result=response.model_dump_json(serialize_as_any=True).encode(
                encoding=CHAR_ENCODING_UTF8
            )
        )

    @step
    async def retrieve(self, ctx: Context, ev: Event) -> Event:
        # Draft the solution
        pass

    @step
    async def solve(self, ctx: Context, ev: Event) -> Event | StopEvent:
        # Draft the solution
        pass

    @step
    async def evaluate(self, ctx: Context, ev: Event) -> Event:
        # Draft the solution
        pass
