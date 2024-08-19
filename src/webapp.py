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

from dotenv import load_dotenv
import uuid

from coder_agent import CoderOutput, MultiAgentDirectedGraph, TestCase
from utils import parse_env

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import AnyMessage


# from langchain_groq.chat_models import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

import gradio as gr
import constants


class GradioApp:
    """The main Gradio app class."""

    _gr_state_user_input_text = gr.State(constants.EMPTY_STRING)
    _llm: BaseChatModel = None

    def __init__(self):
        """Default constructor for the Gradio app."""
        ic(load_dotenv())
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
        self._llm_provider = parse_env(
            constants.ENV_VAR_NAME__LLM_PROVIDER,
            default_value=constants.ENV_VAR_VALUE__LLM_PROVIDER,
        )
        if self._llm_provider == "Ollama":
            # ChatOllama does not support structured outputs: https://python.langchain.com/v0.2/docs/integrations/chat/ollama/
            # Yet, the API docs seems to suggest that it does: https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html#langchain_community.chat_models.ollama.ChatOllama.with_structured_output
            self._llm = ChatOllama(
                base_url=parse_env(
                    constants.ENV_VAR_NAME__LLM_OLLAMA_URL,
                    default_value=constants.ENV_VAR_VALUE__LLM_OLLAMA_URL,
                ),
                model=parse_env(
                    constants.ENV_VAR_NAME__LLM_OLLAMA_MODEL,
                    default_value=constants.ENV_VAR_VALUE__LLM_OLLAMA_MODEL,
                ),
                temperature=parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    default_value=constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
                top_p=parse_env(
                    constants.ENV_VAR_NAME__LLM_TOP_P,
                    default_value=constants.ENV_VAR_VALUE__LLM_TOP_P,
                    type_cast=float,
                ),
                top_k=parse_env(
                    constants.ENV_VAR_NAME__LLM_TOP_K,
                    default_value=constants.ENV_VAR_VALUE__LLM_TOP_K,
                    type_cast=int,
                ),
                repeat_penalty=parse_env(
                    constants.ENV_VAR_NAME__LLM_REPEAT_PENALTY,
                    default_value=constants.ENV_VAR_VALUE__LLM_REPEAT_PENALTY,
                    type_cast=float,
                ),
                seed=parse_env(
                    constants.ENV_VAR_NAME__LLM_SEED,
                    default_value=constants.ENV_VAR_VALUE__LLM_SEED,
                    type_cast=int,
                ),
                format="json",
            )
        elif self._llm_provider == "Groq":
            self._llm = ChatGroq(
                api_key=parse_env(constants.ENV_VAR_NAME__LLM_GROQ_API_KEY),
                model=parse_env(
                    constants.ENV_VAR_NAME__LLM_GROQ_MODEL,
                    default_value=constants.ENV_VAR_VALUE__LLM_GROQ_MODEL,
                ),
                temperature=parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    default_value=constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
                # model_kwargs={
                #     "top_p": parse_env(
                #         constants.ENV_VAR_NAME__LLM_TOP_P,
                #         default_value=constants.ENV_VAR_VALUE__LLM_TOP_P,
                #         type_cast=float,
                #     ),
                #     "top_k": parse_env(
                #         constants.ENV_VAR_NAME__LLM_TOP_K,
                #         default_value=constants.ENV_VAR_VALUE__LLM_TOP_K,
                #         type_cast=int,
                #     ),
                #     "repeat_penalty": parse_env(
                #         constants.ENV_VAR_NAME__LLM_REPEAT_PENALTY,
                #         default_value=constants.ENV_VAR_VALUE__LLM_REPEAT_PENALTY,
                #         type_cast=float,
                #     ),
                #     "seed": parse_env(
                #         constants.ENV_VAR_NAME__LLM_SEED,
                #         default_value=constants.ENV_VAR_VALUE__LLM_SEED,
                #         type_cast=int,
                #     ),
                # },
                # Streaming is not compatible with JSON
                # streaming=False,
                # JSON response is not compatible with tool calling
                # response_format={"type": "json_object"},
            )
        elif self._llm_provider == "Anthropic":
            self._llm = ChatAnthropic(
                api_key=parse_env(constants.ENV_VAR_NAME__LLM_ANTHROPIC_API_KEY),
                model=parse_env(
                    constants.ENV_VAR_NAME__LLM_ANTHROPIC_MODEL,
                    default_value=constants.ENV_VAR_VALUE__LLM_ANTHROPIC_MODEL,
                ),
                temperature=parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    default_value=constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == "Cohere":
            self._llm = ChatCohere(
                cohere_api_key=parse_env(constants.ENV_VAR_NAME__LLM_COHERE_API_KEY),
                model=parse_env(
                    constants.ENV_VAR_NAME__LLM_COHERE_MODEL,
                    default_value=constants.ENV_VAR_VALUE__LLM_COHERE_MODEL,
                ),
                temperature=parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    default_value=constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == "Open AI":
            self._llm = ChatOpenAI(
                api_key=parse_env(constants.ENV_VAR_NAME__LLM_OPENAI_API_KEY),
                model=parse_env(
                    constants.ENV_VAR_NAME__LLM_OPENAI_MODEL,
                    default_value=constants.ENV_VAR_VALUE__LLM_OPENAI_MODEL,
                ),
                temperature=parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    default_value=constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self._llm_provider}")

    def find_solution(
        self,
        user_question: str,
        runtime_limit: int,
        test_cases: list[TestCase] = None,
    ):
        """
        Generator function to find a coding solution to the user question.

        Args:
            user_question (str): The question asked by the user.
            runtime_limit (int): The runtime limit for the code execution.
            test_cases (list[TestCase], optional): The list of test cases to evaluate the code against. Defaults to None.

        Yields:
            str: The reasoning behind the solution.
            str: The pseudocode for the solution.
            str: The Python code for the solution.
        """
        prompt = ChatPromptTemplate(
            input_variables=[constants.PROMPT_TEMPLATE__VAR_MESSAGES],
            input_types={constants.PROMPT_TEMPLATE__VAR_MESSAGES: AnyMessage},
            partial_variables={
                constants.PROMPT_TEMPLATE__PARTIAL_VAR__EXAMPLES: constants.EMPTY_STRING
            },
            messages=[
                SystemMessagePromptTemplate.from_template(
                    template=parse_env(
                        constants.ENV_VAR_NAME__LLM_CODER_SYSTEM_PROMPT,
                        constants.ENV_VAR_VALUE__LLM_CODER_SYSTEM_PROMPT,
                    )
                ),
                MessagesPlaceholder(
                    variable_name=constants.PROMPT_TEMPLATE__VAR_MESSAGES
                ),
            ],
        )
        coder_agent = MultiAgentDirectedGraph(llm=self._llm, solver_prompt=prompt)
        coder_agent.build_agent_graph()
        config = {"configurable": {"thread_id": uuid.uuid4().hex, "k": 3}}
        graph_input = {
            constants.AGENT_STATE__KEY_MESSAGES: HumanMessage(content=user_question),
            constants.AGENT_STATE__KEY_RUNTIME_LIMIT: runtime_limit,
            constants.AGENT_STATE__KEY_TEST_CASES: test_cases,
        }
        result_iterator = coder_agent.agent_graph.stream(
            input=graph_input,
            config=config,
        )
        for result in result_iterator:
            if constants.AGENT_STATE_GRAPH_NODE__SOLVE in result:
                response: AIMessage = result[constants.AGENT_STATE_GRAPH_NODE__SOLVE][
                    constants.AGENT_STATE__KEY_MESSAGES
                ][-1]
                # FIXME: Why are there more than one tool calls to the same tool?
                if (
                    response.tool_calls[-1][constants.AGENT_TOOL_CALL__NAME]
                    == CoderOutput.__name__
                ):
                    tool_call_args = response.tool_calls[-1][
                        constants.AGENT_TOOL_CALL__ARGS
                    ]
                    # coder_output: CoderOutput = CoderOutput.parse_json(tool_call_args)
                    yield [
                        tool_call_args[
                            constants.PYDANTIC_MODEL__CODE_OUTPUT__REASONING
                        ],
                        tool_call_args[
                            constants.PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE
                        ],
                        tool_call_args[constants.PYDANTIC_MODEL__CODE_OUTPUT__CODE],
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
                constants.TEST_CASE__KEY_INPUTS: test_case_in,
                constants.TEST_CASE__KEY_OUTPUTS: test_case_out,
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
                        f"""
                            <img
                                width="384"
                                height="96"
                                style="filter: invert(0.5);"
                                alt="chatty-coder logo"
                                src="/file={constants.PROJECT_LOGO_PATH}" />
                        """,
                        # "/file=assets/logo-embed.svg" or "file/assets/logo-embed.svg"?
                        # "https://raw.githubusercontent.com/anirbanbasu/chatty-coder/master/assets/logo-embed.svg"
                    )
                with gr.Column(scale=3):
                    btn_theme_toggle = gr.Button("Toggle dark mode")
            with gr.Row(elem_id="ui_main"):
                with gr.Column(elem_id="ui_main_left"):
                    # chatbot_conversation_history = gr.Chatbot(
                    #     label="{cha@tty cod:r}",
                    #     type="messages",
                    #     layout="bubble",
                    #     placeholder="Your conversation with AI will appear here...",
                    # )
                    with gr.Accordion(
                        label=f"{self._llm_provider} LLM configuration", open=False
                    ):
                        gr.JSON(value=self._llm.to_json(), show_label=False)
                    chk_show_user_input_preview = gr.Checkbox(
                        value=False,
                        label="Preview question (Markdown formatted)",
                    )
                    input_user_question = gr.TextArea(
                        label="Question (in Markdown)",
                        placeholder="Enter the coding question that you want to ask...",
                        lines=5,
                        elem_id="user_question",
                    )
                    user_input_preview = gr.Markdown(visible=False)
                    btn_code = gr.Button(
                        value="Let's code!",
                        variant="primary",
                    )
                    with gr.Accordion(label="Code evaluation", open=False):
                        gr.Markdown(
                            "Provide test cases to evaluate the code. _These test cases will not be sent to the AI model_."
                        )
                        with gr.Row(equal_height=True):
                            input_test_cases_in = gr.Textbox(
                                label="Test case input", scale=1
                            )
                            input_test_cases_out = gr.Textbox(
                                label="Test case output", scale=1
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
                        label="Generated solution",
                        open=True,
                    ):
                        output_reasoning = gr.Markdown(
                            label="Reasoning",
                            show_label=True,
                            line_breaks=True,
                        )
                        output_pseudocode = gr.Code(label="Pseudocode", show_label=True)
                        output_code = gr.Code(
                            label="Python code",
                            show_label=True,
                            language="python",
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
                ],
                outputs=[
                    output_reasoning,
                    output_pseudocode,
                    output_code,
                ],
                api_name="get_coding_solution",
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
        ic(allowed_static_file_paths)
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
