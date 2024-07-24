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

import os
from dotenv import load_dotenv
from typing import Any

from coder_agent import CoderAgent

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

# from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# from langchain_groq.chat_models import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import gradio as gr
import constants


gr.set_static_paths(paths=["assets/logo-embed.svg"])


class GradioApp:
    """The main Gradio app class."""

    _gr_state_user_input_text = gr.State(constants.EMPTY_STRING)

    def __init__(self):
        """Default constructor for the Gradio app."""
        ic(load_dotenv())
        self._gradio_host: str = self.parse_env(
            constants.ENV_VAR_NAME__GRADIO_SERVER_HOST,
            default_value=constants.ENV_VAR_VALUE__GRADIO_SERVER_HOST,
        )
        self._gradio_port: int = self.parse_env(
            constants.ENV_VAR_NAME__GRADIO_SERVER_PORT,
            default_value=constants.ENV_VAR_VALUE__GRADIO_SERVER_PORT,
            type_cast=int,
        )
        ic(self._gradio_host, self._gradio_port)
        self._llm_provider = self.parse_env(
            constants.ENV_VAR_NAME__LLM_PROVIDER,
            default_value=constants.ENV_VAR_VALUE__LLM_PROVIDER,
        )
        if self._llm_provider == "Ollama":
            self._llm = OllamaFunctions(
                base_url=self.parse_env(
                    constants.ENV_VAR_NAME__LLM_OLLAMA_URL,
                    constants.ENV_VAR_VALUE__LLM_OLLAMA_URL,
                ),
                model=self.parse_env(
                    constants.ENV_VAR_NAME__LLM_OLLAMA_MODEL,
                    constants.ENV_VAR_VALUE__LLM_OLLAMA_MODEL,
                ),
                temperature=self.parse_env(
                    constants.ENV_VAR_NAME__LLM_TEMPERATURE,
                    constants.ENV_VAR_VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
                format="json",
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self._llm_provider}")
        ic(self._llm_provider, self._llm)

    def parse_env(
        self,
        var_name: str,
        default_value: str = None,
        type_cast=str,
        convert_to_list=False,
        list_split_char=constants.SPACE_STRING,
    ) -> Any | list[Any]:
        """
        Parse the environment variable and return the value.

        Args:
            var_name (str): The name of the environment variable.
            default_value (str): The default value to use if the environment variable is not set.
            type_cast (str): The type to cast the value to.
            convert_to_list (bool): Whether to convert the value to a list.
            list_split_char (str): The character to split the list on.

        Returns:
            (Any | list[Any]) The parsed value, either as a single value or a list. The type of the returned single
            value or individual elements in the list depends on the supplied type_cast parameter.
        """
        if var_name not in os.environ and default_value is None:
            raise ValueError(f"Environment variable {var_name} does not exist.")
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

    def find_solution(self, user_question: str) -> list[str, str, str]:
        """
        Find a coding solution to the user question.

        Args:
            user_question (str): The user question to provide a solution for.

        Returns:
            list[str, str, str]: A list containing the reasoning, pseudocode, and code for the solution.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.parse_env(
                        constants.ENV_VAR_NAME__LLM_SYSTEM_PROMPT,
                        constants.ENV_VAR_VALUE__LLM_SYSTEM_PROMPT,
                    )
                ),
                HumanMessage(content=user_question),
            ]
        )
        coder_agent = CoderAgent(llm=self._llm, prompt=prompt)
        coder_agent.build_agent_graph()
        config = {"configurable": {"thread_id": "1", "k": 3}}
        result_iterator = coder_agent.agent_graph.stream(
            input={
                "messages": prompt.messages,
                "runtime_limit": 10,
            },
            config=config,
        )
        json_dict = next(result_iterator)
        ic(json_dict)
        ai_message: AIMessage = json_dict["solve"]["messages"][-1]
        if not ai_message.tool_calls:
            raise ValueError("Coding agent did not produce a valid code block")
        return [
            ai_message.tool_calls[0]["args"]["reasoning"],
            ai_message.tool_calls[0]["args"]["pseudocode"],
            ai_message.tool_calls[0]["args"]["code"],
        ]

    def construct_interface(self):
        """Construct the Gradio user interface and make it available through the `interface` property of this class."""
        with gr.Blocks(css=constants.CSS__GRADIO_APP) as self.interface:
            gr.HTML(
                """
                    <img
                        width="384"
                        height="96"
                        style="filter: invert(0.5);"
                        alt="chatty-coder logo"
                        src="https://raw.githubusercontent.com/anirbanbasu/chatty-coder/master/assets/logo-embed.svg" />
                """,
                elem_id="ui_header",
            )
            with gr.Row(elem_id="ui_main"):
                with gr.Column(elem_id="ui_main_left"):
                    gr.Markdown("# Coding challenge")
                    with gr.Tab(label="The coding question"):
                        input_user_question = gr.TextArea(
                            label="Question (in Markdown)",
                            placeholder="Enter the coding question that you want to ask...",
                            lines=10,
                            elem_id="user_question",
                        )
                    with gr.Tab(label="Question preview"):
                        user_input_preview = gr.Markdown()
                    btn_ask = gr.Button(
                        value="Ask",
                        elem_id="btn_ask",
                    )
                with gr.Column(elem_id="ui_main_right"):
                    gr.Markdown(
                        f"""# Solution
                        Using `{self._llm.model}@{self._llm_provider}`"""
                    )
                    output_reasoning = gr.Markdown(
                        label="Reasoning",
                        show_label=True,
                        line_breaks=True,
                    )
                    output_pseudocode = gr.Markdown(
                        label="Pseudocode",
                        show_label=True,
                        line_breaks=True,
                    )
                    output_code = gr.Code(
                        label="Code",
                        show_label=True,
                        language="python",
                    )
            # Button actions
            btn_ask.click(
                self.find_solution,
                inputs=[input_user_question],
                outputs=[output_reasoning, output_pseudocode, output_code],
                api_name="get_coding_solution",
            )

            # Automatically generate preview the user input as Markdown
            input_user_question.change(
                lambda text: text,
                inputs=[input_user_question],
                outputs=[user_input_preview],
                api_name=False,
            )

    def run(self):
        """Run the Gradio app by launching a server."""
        self.construct_interface()
        self.interface.launch(
            server_name=self._gradio_host,
            server_port=self._gradio_port,
            show_api=True,
            show_error=True,
        )


if __name__ == "__main__":
    app = GradioApp()
    app.run()
