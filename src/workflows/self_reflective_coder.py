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

from typing import Any, List, Optional
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
)

from llama_index.core.workflow.events import (
    InputRequiredEvent,
    HumanResponseEvent,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.prompts import PromptTemplate

from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.base.llms.types import CompletionResponse

import dirtyjson as json

from code_executor import CodeExecutor
from constants import (
    CHAR_ENCODING_UTF8,
    EMPTY_STRING,
    EXECUTOR_MESSAGE__FAILED,
    EXECUTOR_MESSAGE__PASSED,
    PYDANTIC_MODEL__CODE_OUTPUT__CODE,
    PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE,
)
from utils import remove_markdown_codeblock_backticks
from workflows.common import CodeExecutionResult, TestCase, WorkflowStatusEvent

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


class CodeExecutionResultEvent(Event):
    """
    Event to store the code execution result.

    Fields:
        results (List[CodeExecutionResult]): The result of the code execution.
    """

    results: List[CodeExecutionResult]


class CompetitiveCoderWorkflow(Workflow):
    """
    # Competitive coder

    A workflow implementation of the research paper at: https://arxiv.org/abs/2404.10952v1. The Implementation of the code is based on: https://princeton-nlp.github.io/USACOBench/

    **Caveats**
     - Not known yet.
    """

    CTX_KEY_PROBLEM = "problem"
    CTX_KEY_TEST_CASES = "test_cases"
    CTX_RUNTIME_LIMIT = "runtime_limit"
    CTX_DRAFT_SOLUTION = "draft_solution"

    PROMPT_TEMPLATE_DRAFT_SOLVE = PromptTemplate(
        "You are a world-class Python programmer. Please respond with a Python 3 solution to the given problem below."
        "\nFirst, output a detailed `reasoning` through the problem and conceptualise a solution. Whenever possible, add a time and a space complexity analysis for your solution. "
        "Then, output a complete `pseudocode` in Pascal to implement your concept solution. "
        "Finally, output a well-documented and working Python 3 `code` for your solution. Do not use external libraries."
        # "\nPlease ALWAYS output a JSON dictionary exactly as follows:"
        # "\n{"
        # '\n\t"reasoning": "the reasoning you generate",'
        # '\n\t"pseudocode": "the pseudocode you generate",'
        # '\n\t"code": "the code you generate",'
        # "\n}\n"
        "\nPlease ONLY output a correct JSON dictionary with only the following keys: `reasoning`, `pseudocode`, and `code`. Please DO NOT output anything else."
        "Implement your Python code as a function named `solve`. The function must accept `args` as a list of string arguments. It must convert each argument to the correct type as necessary. The function must return the result. "
        # Stop outputting a generator function.
        "The function must `return` a result, not `yield` intermediate results."
        # "\nPlease format your response as a JSON dictionary, using `reasoning`, `pseudocode`, and `code` as keys. "
        # The following instruction about Markdown formatting is necessary to make the output look good on Gradio.
        # "Please format the values of these JSON keys as Markdown."
        # "Please ensure that the line breaks in the values of these JSON keys are correct for renderring as Markdown. However, do not format the `pseudocode` or `code` as Markdown."
        "\n\n[BEGIN PROBLEM]\n{problem}\n[END PROBLEM]"
    )

    PROMPT_TEMPLATE_SOLVE_FIX_TEST_RESULTS = PromptTemplate(
        "You are a world-class Python programmer. You write concise and well-documented code following the PEP8 style guide. "
        "As a response to the given problem below, you have generated a Python 3 code solution as shown below."
        "\nHowever, your solution failed to pass the test cases given below. "
        "Please review the test results and generate an improved solution. "
        "\nFor your improved solution, first, output a detailed `reasoning` through the problem and conceptualise a solution. Whenever possible, add a time and a space complexity analysis for your solution. "
        "Then, output a complete `pseudocode` in Pascal to implement your concept solution. "
        "Finally, output a well-documented and working Python 3 `code` for your solution. Do not use external libraries."
        # "\nPlease ALWAYS output a JSON dictionary exactly as follows:"
        # "\n{"
        # '\n\t"reasoning": "the reasoning you generate",'
        # '\n\t"pseudocode": "the pseudocode you generate",'
        # '\n\t"code": "the code you generate",'
        # "\n}\n"
        "\nPlease ONLY output a correct JSON dictionary with only the following keys: `reasoning`, `pseudocode`, and `code`. Please DO NOT output anything else."
        "Implement your Python code as a function named `solve`. The function must accept `args` as a list of string arguments. It must convert each argument to the correct type as necessary. The function must return the result. "
        # Stop outputting a generator function.
        "The function must `return` a result, not `yield` intermediate results."
        # The following instruction about Markdown formatting is necessary to make the output look good on Gradio.
        # "Please format the values of these JSON keys as Markdown."
        "\n\n[BEGIN PROBLEM]\n{problem}\n[END PROBLEM]"
        "\n\n[BEGIN DRAFT SOLUTION]\n{draft_solution}\n[END DRAFT SOLUTION]"
        "\n\n[BEGIN TEST RESULTS]\n{test_results}\n[END TEST RESULTS]"
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
        self._code_executor = CodeExecutor()
        self._total_steps = 0
        self._finished_steps = 0

    @step
    async def intercept_human_intervention(
        self, ctx: Context, ev: HumanResponseEvent
    ) -> StopEvent:
        draft_solution: DraftSolutionResultEvent = await ctx.get(
            CompetitiveCoderWorkflow.CTX_DRAFT_SOLUTION
        )
        self._total_steps += 1
        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Obtained a response from the human:\n{ev.response}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        return StopEvent(
            result=draft_solution.model_dump_json(serialize_as_any=True).encode(
                encoding=CHAR_ENCODING_UTF8
            )
        )

    @step
    async def draft_solve(
        self, ctx: Context, ev: StartEvent
    ) -> DraftSolutionResultEvent:
        # Draft the solution
        if not hasattr(ev, CompetitiveCoderWorkflow.CTX_KEY_PROBLEM):
            raise ValueError("The problem statement is missing. Try again!")

        await ctx.set(CompetitiveCoderWorkflow.CTX_KEY_PROBLEM, ev.problem)

        if hasattr(ev, CompetitiveCoderWorkflow.CTX_KEY_TEST_CASES):
            await ctx.set(CompetitiveCoderWorkflow.CTX_KEY_TEST_CASES, ev.test_cases)
        else:
            await ctx.set(CompetitiveCoderWorkflow.CTX_KEY_TEST_CASES, [])

        await ctx.set(CompetitiveCoderWorkflow.CTX_RUNTIME_LIMIT, ev.runtime_limit)

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Drafting the solution to the problem:\n{ev.problem}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        # # TODO: Prefer acomplete to astructured_predict?
        raw_response: CompletionResponse = await self.llm.acomplete(
            prompt=CompetitiveCoderWorkflow.PROMPT_TEMPLATE_DRAFT_SOLVE.format(
                problem=ev.problem
            ),
        )
        # ic(raw_response)
        response: DraftSolutionResultEvent = DraftSolutionResultEvent.model_validate(
            json.loads(raw_response.text), strict=False
        )
        # response: DraftSolutionResultEvent = (
        #     DraftSolutionResultEvent.model_validate_strings(
        #         raw_response.text, strict=False
        #     )
        # )
        # response: DraftSolutionResultEvent = DraftSolutionResultEvent(
        #     reasoning="The Test Reasoning",
        #     pseudocode="Hello",
        #     code="```\nprint('Hello, World!')\n```",
        # )
        if hasattr(response, PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE):
            response.pseudocode = remove_markdown_codeblock_backticks(
                response.pseudocode
            )
        if hasattr(response, PYDANTIC_MODEL__CODE_OUTPUT__CODE):
            response.code = remove_markdown_codeblock_backticks(response.code)
        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Obtained a draft solution along the following reasoning:\n{response.reasoning}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        ctx.write_event_to_stream(response)
        return response

    @step
    async def retrieve(self, ctx: Context, ev: Event) -> Event:
        # Draft the solution
        pass

    @step
    async def solve(
        self, ctx: Context, ev: CodeExecutionResultEvent
    ) -> Event | StopEvent:
        # Review and improve the solution
        draft_solution: DraftSolutionResultEvent = await ctx.get(
            CompetitiveCoderWorkflow.CTX_DRAFT_SOLUTION
        )
        problem = await ctx.get(CompetitiveCoderWorkflow.CTX_KEY_PROBLEM)
        test_results = EMPTY_STRING
        for i, result in enumerate(ev.results):
            if result.expected_outputs != result.actual_outputs:
                # Do not test cases that passed.
                test_results += f"[BEGIN TEST CASE {i + 1}]\nInputs:\n{result.inputs}\nExpected outputs:\n{result.expected_outputs}\nActual outputs:\n{result.actual_outputs}\n[END TEST CASE {i + 1}]\n"

        if len(test_results) > 0:
            self._total_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"Attempting to improve the solution to the problem:\n{problem}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            raw_response: CompletionResponse = await self.llm.acomplete(
                prompt=CompetitiveCoderWorkflow.PROMPT_TEMPLATE_SOLVE_FIX_TEST_RESULTS.format(
                    problem=problem,
                    draft_solution=draft_solution.code,
                    test_results=test_results,
                ),
            )
            response: DraftSolutionResultEvent = (
                DraftSolutionResultEvent.model_validate(
                    json.loads(raw_response.text), strict=False
                )
            )
            if hasattr(response, PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE):
                response.pseudocode = remove_markdown_codeblock_backticks(
                    response.pseudocode
                )
            if hasattr(response, PYDANTIC_MODEL__CODE_OUTPUT__CODE):
                response.code = remove_markdown_codeblock_backticks(response.code)
            self._finished_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"Obtained an improved solution with the following code:\n{response.code}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            ctx.write_event_to_stream(response)
        else:
            response = draft_solution
        return StopEvent(
            result=response.model_dump_json(serialize_as_any=True).encode(
                encoding=CHAR_ENCODING_UTF8
            )
        )

    @step
    async def evaluate_with_test_cases(
        self, ctx: Context, ev: DraftSolutionResultEvent
    ) -> CodeExecutionResultEvent | InputRequiredEvent:
        result: List[CodeExecutionResult] = []
        # Evaluate a solution
        test_cases: List[TestCase] = await ctx.get(
            CompetitiveCoderWorkflow.CTX_KEY_TEST_CASES
        )
        at_least_one_test_failed: bool = False
        await ctx.set(CompetitiveCoderWorkflow.CTX_DRAFT_SOLUTION, ev)
        runtime_limit: int = await ctx.get(CompetitiveCoderWorkflow.CTX_RUNTIME_LIMIT)
        if not test_cases or len(test_cases) == 0:
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg="No test cases provided. Skipping evaluation.",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
        else:
            self._total_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"Evaluating the solution with the provided {len(test_cases)} test case{'s' if len(test_cases)>1 else EMPTY_STRING}.",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            for i, test_case_dict in enumerate(test_cases):
                test_case = TestCase.model_validate(test_case_dict)
                actual_outputs = self._code_executor.check_correctness(
                    program=ev.code,
                    input_data=test_case.inputs,
                    expected_output=test_case.expected_outputs,
                    timeout=runtime_limit,
                )
                result.append(
                    CodeExecutionResult(
                        inputs=test_case.inputs,
                        expected_outputs=test_case.expected_outputs,
                        actual_outputs=actual_outputs,
                    )
                )
                test_passed: bool = test_case.expected_outputs == actual_outputs
                at_least_one_test_failed = at_least_one_test_failed or not test_passed
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"Evaluated test case {i + 1}:\n\t{test_case}\nResult:\n\t{EXECUTOR_MESSAGE__PASSED if test_passed else EXECUTOR_MESSAGE__FAILED}",
                        total_steps=self._total_steps,
                        finished_steps=self._finished_steps,
                    )
                )
            self._finished_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"Finished running {len(test_cases)} test case{'s' if len(test_cases)>1 else EMPTY_STRING}.",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
        if len(result) > 0 and at_least_one_test_failed:
            # Update the draft solution
            return CodeExecutionResultEvent(results=result)

        # return StopEvent(
        #     result=ev.model_dump_json(serialize_as_any=True).encode(
        #         encoding=CHAR_ENCODING_UTF8
        #     )
        # )
        return InputRequiredEvent(
            prefix=EMPTY_STRING,
            msg=ev.reasoning,
        )
