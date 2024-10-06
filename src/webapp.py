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
"""The main web application module for the Gradio app."""

from typing import List
from dotenv import load_dotenv

from engine import ChattyCoderEngine
from workflows.common import TestCase, WorkflowStatusEvent
from utils import parse_env
from llama_index.core.workflow.events import InputRequiredEvent

from workflows.self_reflective_coder import DraftSolutionResultEvent

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI

import gradio as gr
import constants
from constants import (
    ELLIPSIS,
    EMPTY_STRING,
    FAKE_STRING,
    PYDANTIC_MODEL__CODE_OUTPUT__CODE,
    PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE,
    PYDANTIC_MODEL__CODE_OUTPUT__REASONING,
    TEST_CASE__KEY_EXPECTED_OUTPUTS,
    TEST_CASE__KEY_INPUTS,
    EnvironmentVariables,
)


class GradioApp:
    """The main Gradio app class."""

    def __init__(self):
        """Default constructor for the Gradio app."""
        # Load the environment variables
        ic(load_dotenv())
        # Set Gradio server host and port
        self._gradio_host: str = parse_env(
            constants.ENV_VAR_NAME__GRADIO_SERVER_HOST,
            default_value=constants.ENV_VAR_VALUE__GRADIO_SERVER_HOST,
        )
        self._gradio_port: int = parse_env(
            constants.ENV_VAR_NAME__GRADIO_SERVER_PORT,
            default_value=constants.ENV_VAR_VALUE__GRADIO_SERVER_PORT,
            type_cast=int,
        )
        ic(self._gradio_host, self._gradio_port)
        # Setup LLM provider and LLM configuration
        self.set_llm_provider()

        self.workflow_engine = ChattyCoderEngine(llm=self._llm)
        self.agent_task_pending = False

    def set_llm_provider(self, provider: str | None = None):
        """Set the LLM provider for the application."""
        if provider is not None:
            self._llm_provider = provider
        else:
            # Setup LLM provider and LLM configuration
            self._llm_provider = parse_env(
                EnvironmentVariables.KEY__LLM_PROVIDER,
                default_value=EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA,
            )

        if self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA:
            self._llm = Ollama(
                base_url=parse_env(
                    EnvironmentVariables.KEY__LLM_OLLAMA_URL,
                    default_value=EnvironmentVariables.VALUE__LLM_OLLAMA_URL,
                ),
                # Increase the timeout to 180 seconds to allow for longer queries on slower computers.
                request_timeout=180.0,
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_OLLAMA_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_OLLAMA_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
                # JSON mode is not required because the LLM will be only sometimes instructed to output JSON.
                # json_mode=True,
                additional_kwargs={
                    "top_p": parse_env(
                        EnvironmentVariables.KEY__LLM_TOP_P,
                        default_value=EnvironmentVariables.VALUE__LLM_TOP_P,
                        type_cast=float,
                    ),
                    "top_k": parse_env(
                        EnvironmentVariables.KEY__LLM_TOP_K,
                        default_value=EnvironmentVariables.VALUE__LLM_TOP_K,
                        type_cast=int,
                    ),
                    "repeat_penalty": parse_env(
                        EnvironmentVariables.KEY__LLM_REPEAT_PENALTY,
                        default_value=EnvironmentVariables.VALUE__LLM_REPEAT_PENALTY,
                        type_cast=float,
                    ),
                    "seed": parse_env(
                        EnvironmentVariables.KEY__LLM_SEED,
                        default_value=EnvironmentVariables.VALUE__LLM_SEED,
                        type_cast=int,
                    ),
                },
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_GROQ:
            self._llm = Groq(
                api_key=parse_env(
                    EnvironmentVariables.KEY__GROQ_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_GROQ_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_GROQ_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_ANTHROPIC:
            self._llm = Anthropic(
                api_key=parse_env(
                    EnvironmentVariables.KEY__ANTHROPIC_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_ANTHROPIC_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_ANTHROPIC_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_COHERE:
            self._llm = Cohere(
                api_key=parse_env(
                    EnvironmentVariables.KEY__COHERE_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_COHERE_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_COHERE_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_OPENAI:
            self._llm = OpenAI(
                api_key=parse_env(
                    EnvironmentVariables.KEY__OPENAI_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_OPENAI_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_OPENAI_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self._llm_provider}")

        ic(
            self._llm_provider,
            self._llm.model,
            self._llm.temperature,
        )

    async def find_solution(
        self,
        user_question: str,
        runtime_limit: float,
        test_cases: List[TestCase] = None,
        chat_history: List[ChatMessage] = [],
        agent_status=gr.Progress(),
    ):
        """
        Generator function to find a coding solution to the user question.

        Args:
            user_question (str): The question asked by the user.
            runtime_limit (int): The runtime limit for the code execution.
            test_cases (List[TestCase], optional): The list of test cases to evaluate the code against. Defaults to None.

        Yields:
            str: The reasoning behind the solution.
            str: The pseudocode for the solution.
            str: The Python code for the solution.
            List[ChatMessage]: The chat history.
        """

        if self.agent_task_pending:
            gr.Warning(
                "ch@tty cod:r is busy, please wait for the current task to complete!"
            )
            return
        if (
            user_question is not None
            and user_question != EMPTY_STRING
            and self.agent_task_pending is False
        ):
            # Stream events and results
            self.agent_task_pending = True
            chat_history = []
            generator = self.workflow_engine.run(
                problem=user_question,
                test_cases=test_cases,
                runtime_limit=runtime_limit,
            )
            async for (
                done,
                finished_steps,
                total_steps,
                streamed_object,
            ) in generator:
                if done:
                    agent_status(progress=None)
                    self.agent_task_pending = False
                    yield [
                        streamed_object[PYDANTIC_MODEL__CODE_OUTPUT__REASONING],
                        streamed_object[PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE],
                        streamed_object[PYDANTIC_MODEL__CODE_OUTPUT__CODE],
                        gr.update(),
                    ]
                else:
                    if isinstance(streamed_object, InputRequiredEvent):
                        # Hide the progress bar?
                        agent_status(progress=None)
                        chat_history.append(
                            ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=str(streamed_object.msg),
                            ).dict()
                        )
                        yield [
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(value=chat_history),
                        ]
                    elif isinstance(streamed_object, WorkflowStatusEvent):
                        status_msg = streamed_object.msg
                        status = (
                            str(status_msg)[:125] + ELLIPSIS
                            if len(str(status_msg)) > 125
                            else str(status_msg)
                        )
                        agent_status(
                            progress=(finished_steps, total_steps), desc=status
                        )
                    elif isinstance(streamed_object, DraftSolutionResultEvent):
                        yield [
                            streamed_object.reasoning,
                            streamed_object.pseudocode,
                            streamed_object.code,
                            chat_history,
                        ]

    async def hitl_response(self, response: str, chat_history):
        """
        Handle the human-in-the-loop response.

        Args:
            response (str): The response from the human.
            chat_history (List[ChatMessage]): The chat history.
        """
        self.workflow_engine.send_human_response(response)
        chat_history.append(ChatMessage(role=MessageRole.USER, content=response).dict())
        # ic(chat_history)
        return [
            gr.update(value=EMPTY_STRING),
            gr.update(value=chat_history),
        ]

    def add_test_case(
        self, test_cases: list[TestCase] | None, test_case_in: str, test_case_out: str
    ) -> list[TestCase]:
        """
        Add a test case to the list of test cases.

        Args:
            test_cases (list[TestCase]): The list of test cases to add the new test case to.
            test_case_in (str): The input for the test case.
            test_case_out (str): The expected output for the test case.

        Returns:
            list[TestCase]: The updated list of test cases.
        """
        if test_cases is None:
            test_cases = []
        test_cases.append(
            {
                TEST_CASE__KEY_INPUTS: test_case_in.strip(),
                TEST_CASE__KEY_EXPECTED_OUTPUTS: test_case_out.strip(),
            }
        )
        return test_cases

    def delete_test_case(
        self, test_cases: list[TestCase], test_case_id: int
    ) -> list[TestCase]:
        """
        Delete a test case from the list of test cases.

        Args:
            test_cases (list[TestCase]): The list of test cases to delete the test case from.
            test_case_id (int): The ID of the test case to delete.

        Returns:
            list[TestCase]: The updated list of test cases.
        """
        if test_cases is not None and test_case_id in range(0, len(test_cases)):
            test_cases.pop(test_case_id)
        return test_cases

    def construct_interface(self):
        """Construct the Gradio user interface and make it available through the `interface` property of this class."""
        with gr.Blocks(
            # See theming guide at https://www.gradio.app/guides/theming-guide
            # theme="gstaff/xkcd",
            # theme="derekzen/stardust",
            # theme="bethecloud/storj_theme",
            theme="abidlabs/Lime",
            fill_width=True,
            fill_height=True,
            css=constants.CSS__GRADIO_APP,
            analytics_enabled=False,
            # Delete the cache content every day that is older than a day
            delete_cache=(86400, 86400),
        ) as self.interface:
            gr.set_static_paths(paths=[constants.PROJECT_LOGO_PATH])
            with gr.Row(elem_id="ui_header", equal_height=True):
                with gr.Column(scale=10):
                    gr.HTML(
                        """
                            <img
                                width="384"
                                height="96"
                                style="filter: invert(0.5);"
                                alt="chatty-coder logo"
                                src="https://raw.githubusercontent.com/anirbanbasu/chatty-coder/master/assets/logo-embed.svg" />
                        """,
                        # "/file={constants.PROJECT_LOGO_PATH}"
                        # "https://raw.githubusercontent.com/anirbanbasu/chatty-coder/master/assets/logo-embed.svg"
                    )
                with gr.Column(scale=3):
                    btn_theme_toggle = gr.Button("Toggle dark mode")
            with gr.Row(elem_id="ui_main"):
                with gr.Column(elem_id="ui_main_left"):
                    with gr.Group():
                        chk_show_user_input_preview = gr.Checkbox(
                            value=False,
                            label="Preview question (Markdown formatted)",
                        )
                        input_user_question = gr.TextArea(
                            label="Question (in Markdown format)",
                            placeholder="Enter the question for which you want a coding solution.",
                            lines=5,
                            elem_id="user_question",
                            show_copy_button=True,
                        )
                        user_input_preview = gr.Markdown(visible=False)
                        btn_code = gr.Button(
                            value="Let's code!",
                            variant="primary",
                        )
                    with gr.Group():
                        chatbot_hitl = gr.Chatbot(
                            label="Human-in-the-loop chat",
                            layout="bubble",
                            type="messages",
                            bubble_full_width=False,
                            # placeholder="The AI model will initiate a chat when necessary.",
                        )
                        input_hitl_response = gr.Textbox(
                            label="Human message",
                            show_label=False,
                            placeholder="Enter a response to the last message from the AI model.",
                        )
                    with gr.Accordion(label="Code evaluation", open=False):
                        gr.Markdown(
                            "Provide test cases to evaluate the code. _These test cases will not be sent to the AI model_."
                        )
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                input_test_cases_in = gr.Textbox(
                                    label="Test case input (from stdin)", scale=1
                                )
                                input_test_cases_out = gr.Textbox(
                                    label="Test case output (in stdout)", scale=1
                                )
                                btn_add_test_case = gr.Button("Add", scale=1)
                            list_test_cases = gr.JSON(label="Test cases")
                            with gr.Row(equal_height=True):
                                test_case_id_to_delete = gr.Number(
                                    label="Test case ID to delete",
                                    minimum=0,
                                    scale=2,
                                )
                                btn_delete_test_case = gr.Button(
                                    "Delete", variant="stop", scale=1
                                )
                            slider_runtime_limit = gr.Slider(
                                label="Runtime limit (in seconds)",
                                minimum=5,
                                maximum=35,
                                step=1,
                                value=10,
                            )
                with gr.Column(elem_id="ui_main_right"):
                    with gr.Accordion(
                        label=f"Generated solution using {self._llm_provider} LLM (model: {self._llm.model})",
                        open=True,
                    ):
                        with gr.Group():
                            output_reasoning = gr.Markdown(
                                label="Reasoning",
                                show_label=True,
                                line_breaks=True,
                            )
                            output_pseudocode = gr.Code(
                                label="Pseudocode",
                                show_label=True,
                                interactive=False,
                            )
                            output_code = gr.Code(
                                label="Python code",
                                show_label=True,
                                language="python",
                                interactive=False,
                            )
            # Button actions
            btn_theme_toggle.click(
                None,
                js=constants.JS__DARK_MODE_TOGGLE,
                api_name=False,
            )

            btn_code.click(
                self.find_solution,
                inputs=[
                    input_user_question,
                    slider_runtime_limit,
                    list_test_cases,
                    chatbot_hitl,
                ],
                outputs=[
                    output_reasoning,
                    output_pseudocode,
                    output_code,
                    chatbot_hitl,
                ],
                api_name="get_coding_solution",
                # queue=False,
            )

            btn_add_test_case.click(
                self.add_test_case,
                inputs=[
                    list_test_cases,
                    input_test_cases_in,
                    input_test_cases_out,
                ],
                outputs=[list_test_cases],
                api_name=False,
            )
            btn_delete_test_case.click(
                self.delete_test_case,
                inputs=[list_test_cases, test_case_id_to_delete],
                outputs=[list_test_cases],
                api_name=False,
            )

            # Automatically generate preview the user input as Markdown
            input_user_question.change(
                lambda text: text,
                inputs=[input_user_question],
                outputs=[user_input_preview],
                api_name=False,
            )

            input_hitl_response.submit(
                self.hitl_response,
                inputs=[input_hitl_response, chatbot_hitl],
                outputs=[input_hitl_response, chatbot_hitl],
                api_name="human_response",
                # show_progress=False,
                # queue=False,
            )

            chk_show_user_input_preview.change(
                fn=lambda checked: (
                    (
                        gr.update(visible=False),
                        gr.update(visible=True),
                    )
                    if checked
                    else (
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )
                ),
                inputs=[chk_show_user_input_preview],
                outputs=[input_user_question, user_input_preview],
                api_name=False,
            )

    def run(self):
        """Run the Gradio app by launching a server."""
        self.construct_interface()
        allowed_static_file_paths = [
            constants.PROJECT_LOGO_PATH,
        ]
        self.interface.queue().launch(
            server_name=self._gradio_host,
            server_port=self._gradio_port,
            show_api=True,
            show_error=True,
            allowed_paths=allowed_static_file_paths,
            enable_monitoring=True,
        )


if __name__ == "__main__":
    app = GradioApp()
    app.run()
