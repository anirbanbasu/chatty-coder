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


from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.schema.output_parser import StrOutputParser

from langchain_core.tools import BaseTool

from code_executor import CodeExecutor
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START

import sqlite3
import constants

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class TestCase(TypedDict):
    inputs: str
    outputs: str


class AgentState(TypedDict):
    # Candidate solution to the problem.
    candidate: AIMessage
    # Few shot examples.
    examples: str
    # Append-only chat memory so the agent can try to recover from initial mistakes.
    messages: Annotated[list[AnyMessage], add_messages]
    # These test cases are used for testing the solution.
    test_cases: list[TestCase]
    runtime_limit: int
    status: str
    draft: bool = False


class CoderInput(BaseModel):
    challenge: str = Field(..., description="The coding challenge.")
    examples: str = Field(
        ...,
        description="Examples of similar challenges and their solutions.",
        default=constants.EMPTY_STRING,
    )


class CoderOutput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the conceptual solution.")
    pseudocode: str = Field(..., description="Pseudocode for the solution.")
    code: str = Field(..., description="Python code implementation for the solution.")


class CoderTool(BaseTool):
    name = "coder_tool"
    description = "Generate Python code to solve the given problem."
    args_schema = CoderOutput

    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(CoderOutput)

    def _run(self, challenge: str, examples: str) -> CoderOutput:
        """Run the tool"""
        messages = [
            ("system", constants.ENV_VAR_VALUE__LLM_CODER_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
        chain = (
            ChatPromptTemplate.from_messages(messages=messages)
            | self._llm
            | StrOutputParser()
        )
        return chain.invoke({"input": challenge, "examples": examples})


class MultiAgentOrchestrator:
    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self._llm = llm
        self._prompt = prompt
        self._runnable_solver = self._prompt | self._llm.bind_tools([CoderOutput])
        self._runnable_draft_solver = self._prompt | self._llm.bind_tools([CoderOutput])
        self._evaluator = CodeExecutor()
        # self._retriever = BM25Retriever.from_texts(
        #     [self.format_example(row) for row in train_ds]
        # )

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

    def retrieve_examples(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        Retrieve examples of similar problems.

        Args:
            state: The state of the agent.
            config: The configuration of the agent.

        Returns:
            dict: The updated state of the agent.
        """
        # The number of examples to retrieve. Defaults to 2, if unspecified.
        top_k = config["configurable"].get("k") or 2
        # Find the solution by the draft solver
        ai_message: AIMessage = state[constants.AGENT_STATE__KEY_CANDIDATE]
        if not ai_message.tool_calls:
            # We err here. To make more robust, you could loop back
            raise ValueError("Draft agent did not produce a valid code block")
        code = ai_message.tool_calls[0]["args"][
            constants.PYDANTIC_MODEL__CODE_OUTPUT__CODE
        ]
        examples_str = "\n".join(
            [doc.page_content for doc in self._retriever.invoke(code)[:top_k]]
        )
        examples_str = f"""
            Here are some example solutions of similar problems.
            <Examples>
            {examples_str}
            <Examples>
            Approach this new question with similar sophistication."""

        return {constants.AGENT_STATE__KEY_EXAMPLES: examples_str}

    def solve(self, state: AgentState) -> dict:
        """
        Solve the problem presented in the `messages` key of the state.

        Args:
            state: The state of the agent.

        Returns:
            dict: The updated state of the agent.
        """
        # Get the inputs for the solver
        inputs = {
            constants.AGENT_STATE__KEY_MESSAGES: state[
                constants.AGENT_STATE__KEY_MESSAGES
            ]
        }
        # Have we been presented with examples?
        has_examples = bool(state.get(constants.AGENT_STATE__KEY_EXAMPLES))
        ic(state)
        # If `draft`` is requested in the state then output a candidate solution
        output_key = (
            constants.AGENT_STATE__KEY_CANDIDATE
            if state[constants.AGENT_STATE__KEY_DRAFT]
            else constants.AGENT_STATE__KEY_MESSAGES
        )
        if has_examples:
            output_key = constants.AGENT_STATE__KEY_MESSAGES
            # Retrieve examples to solve the problem
            inputs[constants.AGENT_STATE__KEY_EXAMPLES] = state[
                constants.AGENT_STATE__KEY_EXAMPLES
            ]
        response = (
            # Use the draft solver only if the `draft` flag is set in the state
            self._runnable_draft_solver.invoke(inputs)
            if state[constants.AGENT_STATE__KEY_DRAFT] is True
            else self._runnable_solver.invoke(inputs)
        )
        # FIXME: Why do we need this? `OllamaFunctions`, for example, does not output `content`.
        # if not response.content:
        #     return {
        #         output_key: AIMessage(
        #             content="I'll need to think about this step by step."
        #         )
        #     }
        return (
            {
                output_key: [response],
                constants.AGENT_STATE__KEY_DRAFT: (False),
            }
            if state[constants.AGENT_STATE__KEY_DRAFT]
            else {output_key: [response]}
        )

    def draft_solve(self, state: AgentState) -> dict:
        state[constants.AGENT_STATE__KEY_DRAFT] = True
        return self.solve(state)

    def format_tool_message(self, response: str, ai_message: AIMessage) -> ToolMessage:
        """
        Format the response as a tool message specifying the tool call ID.

        Args:
            response: The response to be formatted.
            ai_message: The AIMessage object containing the tool call ID.

        Returns:
            ToolMessage: The formatted tool message.
        """
        # This is the node we will add to the agent graph.
        # Most tool-calling APIs require that the `ToolMessage` contain the ID
        # of the tool call that generated the message.
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0][constants.AGENT_TOOL_CALL__ID],
        )

    def evaluate(self, state: AgentState) -> dict:
        """
        Evaluate the correctness of the code.

        Args:
            state: The state of the agent.

        Returns:
            dict: The updated state of the agent.
        """
        test_cases = state[constants.AGENT_STATE__KEY_TEST_CASES]
        # Extract the `AIMessage` that is expected to contain the code from the last tool call.
        ai_message: AIMessage = state[constants.AGENT_STATE__KEY_MESSAGES][-1]
        if not ai_message.tool_calls:
            # If there was no tool call, add a `HumanMessage` to prompt the agent to generate code.
            return {
                constants.AGENT_STATE__KEY_MESSAGES: [
                    HumanMessage(
                        content="No code submitted. Please try again using the correct Python code."
                    )
                ]
            }
        try:
            # Extract the code from the tool call.
            code: str = ai_message.tool_calls[0][constants.AGENT_TOOL_CALL__ARGS][
                constants.PYDANTIC_MODEL__CODE_OUTPUT__CODE
            ]
            # Use PythonREPL to sanitise the code
            # See: https://api.python.langchain.com/en/latest/utilities/langchain_experimental.utilities.python.PythonREPL.html
            code = CodeExecutor.sanitise_code(code)
        except Exception as e:
            # If there was an error extracting the code, return an error message as state.
            return {
                constants.AGENT_STATE__KEY_MESSAGES: [
                    self.format_tool_message(repr(e), ai_message)
                ]
            }
        num_test_cases = len(test_cases) if test_cases is not None else 0
        if num_test_cases == 0:
            return {
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES
            }
        succeeded = 0
        test_results = []
        # TODO: Enable parallel execution of test cases.
        for test_case in test_cases:
            input_data = test_case[constants.TEST_CASE__KEY_INPUTS]
            expected_output = test_case[constants.TEST_CASE__KEY_OUTPUTS]
            test_result = self._evaluator.check_correctness(
                code,
                input_data,
                expected_output,
                state[constants.AGENT_STATE__KEY_RUNTIME_LIMIT],
            )
            test_results.append(test_result)
            if test_result == constants.EXECUTOR_MESSAGE__PASSED:
                succeeded += 1
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        if pass_rate == 1:
            return {
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_SUCCESS
            }

        responses = "\n".join(
            [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
        )
        response = f"Incorrect submission. Please review the failures reported below and respond with updated code.\nPass rate: {pass_rate}\nResults:\n{responses}"

        return {
            constants.AGENT_STATE__KEY_MESSAGES: [
                self.format_tool_message(response, ai_message)
            ]
        }

    def build_agent_graph(self):
        builder = StateGraph(AgentState)
        # builder.add_node("draft", self.draft_solve)
        # builder.add_node("retrieve", self.retrieve_examples)
        builder.add_node("solve", self.solve)
        builder.add_node("evaluate", self.evaluate)
        # Add connectivity
        builder.add_edge(START, "solve")
        # builder.add_edge("draft", "retrieve")
        # builder.add_edge("retrieve", "solve")
        builder.add_edge("solve", "evaluate")

        def control_edge(state: AgentState):
            if (
                state.get(constants.AGENT_STATE__KEY_STATUS)
                == constants.AGENT_NODE__EVALUATE_STATUS_SUCCESS
                or state.get(constants.AGENT_STATE__KEY_STATUS)
                == constants.AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES
            ):
                return END
            return "solve"

        builder.add_conditional_edges(
            "evaluate", control_edge, {END: END, "solve": "solve"}
        )
        builder.add_edge("solve", END)
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn=connection)
        self.agent_graph = builder.compile(checkpointer=checkpointer, debug=False)
