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

# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated
from langchain_core.tools import BaseTool

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig


import json
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


class CoderOutput(BaseModel):
    """Useful to output the solution to the given coding problem in a structured format."""

    reasoning: str = Field(..., description="Reasoning for the conceptual solution.")
    pseudocode: str = Field(..., description="Pseudocode for the solution.")
    code: str = Field(..., description="Python code implementation for the solution.")


# TODO: This is unnecessary? See: https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling/
class CoderOutputTool(BaseTool):
    name = CoderOutput.__name__
    description = "Useful to output the solution to the given coding problem in a structured format."
    args_schema = CoderOutput

    def _run(self, inputs: dict) -> dict:
        # Is there anything to do here?
        return inputs


class AgentOrchestrator:
    agent_graph = None

    def __init__(self, llm: BaseChatModel, solver_prompt: ChatPromptTemplate):
        """
        Initialize the agent orchestrator.

        Args:
            llm: The base chat model (LLM) to be used by the agents.
            solver_prompt: The prompt to be used by the solver agent.
        """
        self._llm = llm
        self._solver_agent = solver_prompt | self._llm.bind_tools([CoderOutput])
        # This is NOT an agent and it should not be.
        self._evaluator = CodeExecutor()
        # self._retriever = BM25Retriever.from_texts(
        #     [self.format_example(row) for row in train_ds]
        # )
        connection = sqlite3.connect(
            ":memory:", check_same_thread=False, autocommit=True
        )
        self._checkpointer = SqliteSaver(conn=connection)

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
                f"<test id='{i}'>\n{test_case}\n</test>"
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
            raise ValueError("Solver agent did not produce a valid code block")
        code = ai_message.tool_calls[0][constants.AGENT_TOOL_CALL__ARGS][
            constants.PYDANTIC_MODEL__CODE_OUTPUT__CODE
        ]
        examples_str = "\n".join(
            [doc.page_content for doc in self._retriever.invoke(code)[:top_k]]
        )
        examples_str = f"""
            Here are some example solutions of similar problems.
            [BEGIN EXAMPLES]
            {examples_str}
            [END EXAMPLES]
            Approach the given problem with similar sophistication."""

        return {constants.AGENT_STATE__KEY_EXAMPLES: examples_str}

    def solve(self, state: AgentState) -> dict:
        """
        Solve the problem presented through the `messages` key of the state.

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
        # If `draft`` is requested in the state then output a candidate solution
        output_key = (
            constants.AGENT_STATE__KEY_CANDIDATE
            if state[constants.AGENT_STATE__KEY_DRAFT]
            else constants.AGENT_STATE__KEY_MESSAGES
        )
        if has_examples:
            # Retrieve examples to solve the problem
            inputs[constants.AGENT_STATE__KEY_EXAMPLES] = state[
                constants.AGENT_STATE__KEY_EXAMPLES
            ]
        else:
            inputs[constants.AGENT_STATE__KEY_EXAMPLES] = constants.EMPTY_STRING

        response = self._solver_agent.invoke(inputs)
        ic(response)
        return (
            {
                output_key: [response],
                constants.AGENT_STATE__KEY_DRAFT: False,
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_IN_PROGRESS,
            }
            if state[constants.AGENT_STATE__KEY_DRAFT]
            else {
                output_key: [response],
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_IN_PROGRESS,
            }
        )

    def draft_solve(self, state: AgentState) -> dict:
        state[constants.AGENT_STATE__KEY_DRAFT] = True
        return self.solve(state)

    def extract_dictionary_from_ai_message(
        self, ai_message: AIMessage, match_tool_call_name: str = None
    ) -> dict:
        """
        Extract the dictionary from the contents or the last tool call arguments of an AI message.

        Args:
            ai_message: The AI message to extract the dictionary from.
            match_tool_call_name: (Optional) The name of the last tool call to match.

        Returns:
            dict: The extracted dictionary.
        """
        # FIXME: Why are there more than one tool calls, one to a function (which one?) and another for the Pydantic model?
        result: dict = None
        ic(ai_message)
        if ai_message.tool_calls:
            if match_tool_call_name is None or (
                match_tool_call_name is not None
                and ai_message.tool_calls[-1][constants.AGENT_TOOL_CALL__NAME]
                == match_tool_call_name
            ):
                result = ai_message.tool_calls[-1][constants.AGENT_TOOL_CALL__ARGS]
        else:
            result = json.loads(ai_message.content)
        return result

    def format_as_tool_message(
        self, response: str, ai_message: AIMessage
    ) -> ToolMessage:
        """
        Format the response as a tool message specifying the tool call ID.

        Args:
            response: The response to be formatted.
            ai_message: The AIMessage object containing the tool call ID.

        Returns:
            ToolMessage: The formatted tool message.
        """
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0][constants.AGENT_TOOL_CALL__ID],
        )

    def evaluate(self, state: AgentState) -> dict:
        """
        Evaluate the correctness of the code. This node does not call the LLM.

        Args:
            state: The state of the agent.

        Returns:
            dict: The updated state of the agent.
        """
        test_cases = state[constants.AGENT_STATE__KEY_TEST_CASES]
        # Extract the `AIMessage` that is expected to contain the code from the last call to the solver that was NOT to generate a candidate solution.
        ai_message: AIMessage = state[constants.AGENT_STATE__KEY_MESSAGES][-1]
        # ai_message is a list of dictionaries.
        solution: dict = self.extract_dictionary_from_ai_message(ai_message=ai_message)
        # Extract the code from the solution.
        code: str = solution[constants.PYDANTIC_MODEL__CODE_OUTPUT__CODE]
        if not code:
            return {
                constants.AGENT_STATE__KEY_MESSAGES: [
                    self.format_as_tool_message(
                        response="No code was generated. Please generate correct Python code.",
                        ai_message=ai_message,
                    )
                ],
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_ERROR,
            }
        num_test_cases = len(test_cases) if test_cases is not None else 0
        if num_test_cases == 0:
            return {
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES
            }
        try:
            # Use PythonREPL to sanitise the code
            # See: https://api.python.langchain.com/en/latest/utilities/langchain_experimental.utilities.python.PythonREPL.html
            # FIXME: Why not use PythonREPL to run the code?
            code = CodeExecutor.sanitise_code(code)
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
                [f"<test id='{i}'>\n{r}\n</test>" for i, r in enumerate(test_results)]
            )
            response = f"Incorrect submission. Please review the failures reported below and respond with updated code.\nPass rate: {pass_rate}\nResults:\n{responses}"

            return {
                constants.AGENT_STATE__KEY_MESSAGES: [
                    self.format_tool_message(response, ai_message)
                ],
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_ERROR,
            }
        except Exception as e:
            # If there was an error extracting the code, return an error message as state.
            return {
                constants.AGENT_STATE__KEY_MESSAGES: [
                    # self.format_tool_message(repr(e), ai_message)
                    self.format_as_tool_message(
                        response=repr(e),
                        ai_message=ai_message,
                    )
                ],
                constants.AGENT_STATE__KEY_STATUS: constants.AGENT_NODE__EVALUATE_STATUS_ERROR,
            }

    def build_agent_graph(self):
        """
        Build the agent graph for the agent orchestrator.
        Upon success, the agent graph is available as the `agent_graph` parameter.
        """
        builder = StateGraph(AgentState)
        # builder.add_node("draft", self.draft_solve)
        # builder.add_node("retrieve", self.retrieve_examples)
        builder.add_node(
            node=constants.AGENT_STATE_GRAPH_NODE__SOLVE, action=self.solve
        )
        builder.add_node(
            node=constants.AGENT_STATE_GRAPH_NODE__EVALUATE,
            action=self.evaluate,
        )
        # Add connectivity
        builder.add_edge(
            start_key=START, end_key=constants.AGENT_STATE_GRAPH_NODE__SOLVE
        )
        # builder.add_edge("draft", "retrieve")
        # builder.add_edge("retrieve", "solve")
        builder.add_edge(
            start_key=constants.AGENT_STATE_GRAPH_NODE__SOLVE,
            end_key=constants.AGENT_STATE_GRAPH_NODE__EVALUATE,
        )

        def decide_evaluation_result(state: AgentState):
            # Note that this decision node is preceeded by the "evaluate" node.
            if (
                state.get(constants.AGENT_STATE__KEY_STATUS)
                == constants.AGENT_NODE__EVALUATE_STATUS_SUCCESS
                or state.get(constants.AGENT_STATE__KEY_STATUS)
                == constants.AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES
            ):
                return END
            return constants.AGENT_STATE_GRAPH_NODE__SOLVE

        builder.add_conditional_edges(
            source=constants.AGENT_STATE_GRAPH_NODE__EVALUATE,
            path=decide_evaluation_result,
            path_map={
                END: END,
                constants.AGENT_STATE_GRAPH_NODE__SOLVE: constants.AGENT_STATE_GRAPH_NODE__SOLVE,
            },
        )
        builder.add_edge(start_key=constants.AGENT_STATE_GRAPH_NODE__SOLVE, end_key=END)
        self.agent_graph = builder.compile(checkpointer=self._checkpointer, debug=False)