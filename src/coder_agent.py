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

import os
import zipfile

import datasets
import requests

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import BM25Retriever

from code_executor import CodeExecutor
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START

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


class codeOutput(BaseModel):
    """Pydantic model with reasoning, pseudocode and Python code
    that represent the solution to the given problem."""

    reasoning: str = Field(..., description="Conceptual solution.")
    pseudocode: str = Field(..., description="Detailed pseudocode in English.")
    code: str = Field(..., description="Valid solution to the problem in Python code.")


class CoderAgent:
    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        # Retrieve the USA Computing Olympiad dataset
        usaco_url = "https://storage.googleapis.com/benchmarks-artifacts/usaco/usaco_sampled_with_tests.zip"
        zip_path = "usaco.zip"
        extract_path = "usaco_datasets"

        response = requests.get(usaco_url)
        with open(zip_path, "wb") as file:
            file.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        os.remove(zip_path)

        ds = datasets.load_from_disk(
            os.path.join(extract_path, "usaco_v3_sampled_with_tests")
        )

        # We will test our agent on index 0 (the same as above).
        # Later, we will test on index 2 (the first 'silver difficulty' question)
        test_indices = [0, 2]
        train_ds = [row for i, row in enumerate(ds) if i not in test_indices]
        # test_ds = [row for i, row in enumerate(ds) if i in test_indices]

        self._llm = llm
        self._prompt = prompt
        self._runnable_solver = self._prompt | self._llm.bind_tools([codeOutput])
        self._runnable_draft_solver = self._prompt.partial(
            examples=constants.EMPTY_STRING
        ) | self._llm.bind_tools([codeOutput])
        self._evaluator = CodeExecutor()
        self._retriever = BM25Retriever.from_texts(
            [self.format_example(row) for row in train_ds]
        )

    def format_example(self, row):
        question = row["description"]
        answer = row["solution"]
        return f"""<problem>
            {question}
            </problem>
            <solution>
            {answer}
            </solution>"""

    def retrieve_examples(self, state: AgentState, config: RunnableConfig):
        top_k = config["configurable"].get("k") or 2
        ai_message: AIMessage = state["candidate"]
        if not ai_message.tool_calls:
            # We err here. To make more robust, you could loop back
            raise ValueError("Draft agent did not produce a valid code block")
        code = ai_message.tool_calls[0]["args"]["code"]
        examples_str = "\n".join(
            [doc.page_content for doc in self._retriever.invoke(code)[:top_k]]
        )
        examples_str = f"""
            You previously solved the following problems in this competition:
            <Examples>
            {examples_str}
            <Examples>
            Approach this new question with similar sophistication."""
        return {"examples": examples_str}

    def solve(self, state: AgentState, draft: bool = False) -> dict:
        # The agent only can see the "messages" and will ignore the test info
        # return {"messages": [self.runnable.invoke({"messages": state["messages"]})]}
        inputs = {"messages": state["messages"]}
        has_examples = bool(state.get("examples"))
        output_key = "candidate"  # Used in the draft node
        if has_examples:
            output_key = "messages"
            # Used in the solve node
            inputs["examples"] = state["examples"]
        response = (
            self._runnable_draft_solver.invoke(inputs)
            if draft is True
            else self._runnable_solver.invoke(inputs)
        )
        if not response.content:
            return {
                output_key: AIMessage(
                    content="I'll need to think about this step by step."
                )
            }
        return {output_key: response}

    def draft_solve(self, state: AgentState) -> dict:
        return self.solve(state, draft=True)

    # This is the node we will add to the graph.
    # Most tool-calling APIs require that the `ToolMessage` contain the ID
    # of the
    def format_tool_message(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content=response + "\nMake all fixes using the codeOutput tool.",
            tool_call_id=ai_message.tool_calls[0]["id"],
        )

    def evaluate(self, state: AgentState):
        test_cases = state["test_cases"]
        ai_message: AIMessage = state["messages"][-1]
        if not ai_message.tool_calls:
            return {
                "messages": [
                    HumanMessage(
                        content="No code submitted. Please try again using the correct Python code."
                    )
                ]
            }
        try:
            code = ai_message.tool_calls[0]["args"]["code"]
        except Exception as e:
            return {"messages": [self.format_tool_message(repr(e), ai_message)]}
        num_test_cases = len(test_cases)
        succeeded = 0
        test_results = []
        # TODO: Multiprocess
        for test_case in test_cases:
            input_data = test_case["inputs"]
            expected_output = test_case["outputs"]
            test_result = self._evaluator.check_correctness(
                code, input_data, expected_output, state["runtime_limit"]
            )
            test_results.append(test_result)
            if test_result == constants.EXECUTOR_MESSAGE__PASSED:
                succeeded += 1
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        if pass_rate == 1:
            return {"status": "success"}

        responses = "\n".join(
            [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
        )
        response = f"Incorrect submission. Please respond with updated code.\nPass rate: {succeeded}/{num_test_cases}\nResults:\n{responses}"
        formatted_message = self.format_tool_message(response, ai_message)
        return {"messages": [formatted_message]}

    def build_agent_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("draft", self.draft_solve)
        builder.add_edge(START, "draft")
        builder.add_node("retrieve", self.retrieve_examples)
        builder.add_node("solve", self.solve)
        builder.add_node("evaluate", self.evaluate)
        # Add connectivity
        builder.add_edge("draft", "retrieve")
        builder.add_edge("retrieve", "solve")
        builder.add_edge("solve", "evaluate")

        def control_edge(state: AgentState):
            if state.get("status") == "success":
                return END
            return "solve"

        builder.add_conditional_edges(
            "evaluate", control_edge, {END: END, "solve": "solve"}
        )
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        self.agent_graph = builder.compile(checkpointer=checkpointer, debug=True)
