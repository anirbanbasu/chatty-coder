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


from typing import Any
from llama_index.core.tools import FunctionTool
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event

from pydantic import BaseModel
from typing_extensions import TypedDict

from llama_index.core.program import FunctionCallingProgram

from code_executor import CodeExecutor

import constants

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class TestCase(TypedDict):
    inputs: str
    outputs: str


class CoderAgentOutput(BaseModel):
    reasoning: str
    pseudocode: str
    code: str


class InputEvent(Event):
    input: list[ChatMessage]
    test_cases: list[TestCase] = None


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class CoderAgent(Workflow):
    def __init__(
        self,
        llm: FunctionCallingLLM,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.tools = [FunctionTool.from_defaults(self.evaluate)]

        self.pydantic_code_generator = FunctionCallingProgram.from_defaults(
            output_cls=CoderAgentOutput,
            llm=self.llm,
            verbose=True,
            prompt_template_str="{input}",
            tool_choice=self.tools[0],
            allow_parallel_tool_calls=True,
        )

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.sources = []

    @step()
    async def prepare_chat_history(self, ev: StartEvent) -> InputEvent:
        # clear sources
        self.sources = []

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        # get chat history
        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step()
    async def handle_llm_input(self, ev: InputEvent) -> ToolCallEvent | StopEvent:
        ic(ev)
        chat_history = ev.input

        ic(chat_history[-1].content)
        pydantic_call = self.pydantic_code_generator(input=chat_history[-1].content)
        ic(pydantic_call)
        response = await self.llm.achat_with_tools(
            self.tools, chat_history=chat_history
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            return StopEvent(result={"response": response, "sources": [*self.sources]})
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step()
    async def handle_tool_calls(self, ev: ToolCallEvent) -> InputEvent:
        ic(ev)
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        for msg in tool_msgs:
            self.memory.put(msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    def format_example(self, row):
        question = row["description"]
        answer = row["solution"]
        result = f"""<problem>
            {question}
            </problem>
            <solution>
            {answer}
            </solution>"""
        return result

    def format_test_cases(self, test_cases: list[TestCase]) -> str:
        """
        Format the test cases for the agent.

        Args:
            test_cases: The test cases to be formatted.

        Returns:
            str: The formatted test cases.
        """
        test_cases_str = "\n".join(
            [
                f"<test id={i}>\n{test_case}\n</test>"
                for i, test_case in enumerate(test_cases)
            ]
        )
        return f"<TestCases>\n{test_cases_str}\n<TestCases>"

    def evaluate(self, code: str, test_cases: list[TestCase] = None) -> str:
        """
        Evaluate the correctness of the code.

        Args:
            code: The code to be evaluated.

        Returns:
            str: The result of the evaluation.
        """
        evaluator = CodeExecutor()
        # try:
        #     # Use PythonREPL to sanitise the code
        #     # See: https://api.python.langchain.com/en/latest/utilities/langchain_experimental.utilities.python.PythonREPL.html
        #     sanitised_code = CodeExecutor.sanitise_code(code)
        # except Exception as e:
        #     # If there was an error extracting the code, return an error message as state.
        #     return e
        num_test_cases = len(test_cases) if test_cases is not None else 0
        if num_test_cases == 0:
            return {constants.AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES}
        succeeded = 0
        test_results = []
        for test_case in test_cases:
            input_data = test_case[constants.TEST_CASE__KEY_INPUTS]
            expected_output = test_case[constants.TEST_CASE__KEY_OUTPUTS]
            test_result = evaluator.check_correctness(
                code,
                input_data,
                expected_output,
            )
            test_results.append(test_result)
            if test_result == constants.EXECUTOR_MESSAGE__PASSED:
                succeeded += 1
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        if pass_rate == 1:
            return constants.AGENT_NODE__EVALUATE_STATUS_SUCCESS

        responses = "\n".join(
            [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
        )
        response = f"Incorrect submission. Please review the failures reported below and respond with updated code.\nPass rate: {pass_rate}\nResults:\n{responses}"

        return response
