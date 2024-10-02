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

"""Various constants used in the project."""

EMPTY_STRING = ""
ELLIPSIS = "..."
EMPTY_DICT = {}
EMPTY_JSON_DICT = "{}"
SPACE_STRING = " "
COLON_STRING = ":"
FAKE_STRING = "fake"
MARKDOWN_CODE_BLOCK_THREE_BACKTICKS = "```"
BOOLEAN_AND = "AND"
BOOLEAN_OR = "OR"
BOOLEAN_NOT = "NOT"
BOOLEAN_XOR = "XOR"

TRUE_VALUES_LIST = ["true", "yes", "t", "y", "on"]

PROJECT_GIT_REPO_URL = "https://github.com/anirbanbasu/chatty-coder"
PROJECT_GIT_REPO_LABEL = "GitHub repository"
PROJECT_NAME = "chatty-coder"
PROJECT_LOGO_PATH = "assets/logo-embed.svg"

HTTP_TARGET_BLANK = "_blank"

ISO639SET1_LANGUAGE_ENGLISH = "en"

CHAR_ENCODING_UTF8 = "utf-8"

# Use custom formatter for coloured logs, see: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# See formatting attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes
LOG_FORMAT = "%(levelname)s:\t%(message)s (%(module)s/%(filename)s::%(funcName)s:L%(lineno)d@%(asctime)s) [P%(process)d:T%(thread)d]"

EXECUTOR_MESSAGE__TIMEDOUT = "Timed out"
EXECUTOR_MESSAGE__PASSED = "Passed"
EXECUTOR_MESSAGE__FAILED = "Failed"
EXECUTOR_MESSAGE__WRONG_ANSWER = "Wrong answer"
EXECUTOR_MESSAGE__NO_RESULTS = "No result returned"

AGENT_STATE_GRAPH_NODE__DRAFT_SOLVE = "draft_solve"
AGENT_STATE_GRAPH_NODE__DRAFT_REVIEW = "draft_review"
AGENT_STATE_GRAPH_NODE__SOLVE = "solve"
AGENT_STATE_GRAPH_NODE__EVALUATE = "evaluate"


AGENT_STATE__KEY_CANDIDATE = "candidate"
AGENT_STATE__KEY_EXAMPLES = "examples"
AGENT_STATE__KEY_MESSAGES = "messages"
AGENT_STATE__KEY_TEST_CASES = "test_cases"
AGENT_STATE__KEY_RUNTIME_LIMIT = "runtime_limit"
AGENT_STATE__KEY_STATUS = "status"
AGENT_STATE__KEY_DRAFT = "draft"

CHATBOT__ROLE_USER = "user"
CHATBOT__ROLE_AGENT = "assistant"
CHATBOT__ROLE_SYSTEM = "system"
CHATBOT__ROLE_LABEL = "role"
CHATBOT__MESSAGE_CONTENT = "content"

AGENT_NODE__EVALUATE_STATUS_IN_PROGRESS = "in-progress"
AGENT_NODE__EVALUATE_STATUS_SUCCESS = "success"
AGENT_NODE__EVALUATE_STATUS_ERROR = "error"
AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES = "no test cases"

AGENT_TOOL_CALL__ARGS = "args"
AGENT_TOOL_CALL__ID = "id"
AGENT_TOOL_CALL__NAME = "name"

TEST_CASE__KEY_INPUTS = "inputs"
TEST_CASE__KEY_EXPECTED_OUTPUTS = "expected_outputs"

PYDANTIC_MODEL__CODE_OUTPUT__REASONING = "reasoning"
PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE = "pseudocode"
PYDANTIC_MODEL__CODE_OUTPUT__CODE = "code"
PYDANTIC_MODEL__CODE_OUTPUT__SUMMARY = "summary"

PROMPT_TEMPLATE__VAR_MESSAGES = AGENT_STATE__KEY_MESSAGES
PROMPT_TEMPLATE__PARTIAL_VAR__EXAMPLES = "examples"

# Environment variables
ENV_VAR_NAME__GRADIO_SERVER_HOST = "GRADIO__SERVER_HOST"
ENV_VAR_VALUE__GRADIO_SERVER_HOST = "localhost"
ENV_VAR_NAME__GRADIO_SERVER_PORT = "GRADIO__SERVER_PORT"
ENV_VAR_VALUE__GRADIO_SERVER_PORT = "7860"

ENV_VAR_NAME__LLM_CODER_SYSTEM_PROMPT = "LLM__CODER_SYSTEM_PROMPT"
ENV_VAR_VALUE__LLM_CODER_SYSTEM_PROMPT = """
You are a world-class Python programmer. You write concise and well-documented code following the PEP8 style guide. Please respond with a Python 3 solution to the given problem below.

First, output a detailed `reasoning` through the problem and conceptualise a solution. Whenever possible, add a time and a space complexity analysis for your solution.
Then, output a complete `pseudocode` in Pascal to implement your concept solution.
Finally, output a well-documented and working Python 3 `code` for your solution. Do not use external libraries.
Make your Python code executable from the command line. Your code must accept all its inputs from `sys.stdin` and print the final output to `sys.stdout`.
Please format your response as a JSON dictionary, using `reasoning`, `pseudocode`, and `code` as keys.

{examples}

The problem is given below.
"""


class EnvironmentVariables:
    KEY__LLM_PROVIDER = "LLM__PROVIDER"

    VALUE__LLM_PROVIDER_OLLAMA = "Ollama"
    VALUE__LLM_PROVIDER_GROQ = "Groq"
    VALUE__LLM_PROVIDER_ANTHROPIC = "Anthropic"
    VALUE__LLM_PROVIDER_COHERE = "Cohere"
    VALUE__LLM_PROVIDER_OPENAI = "Open AI"

    ALL_SUPPORTED_LLM_PROVIDERS = [
        VALUE__LLM_PROVIDER_OLLAMA,
        VALUE__LLM_PROVIDER_GROQ,
        VALUE__LLM_PROVIDER_ANTHROPIC,
        VALUE__LLM_PROVIDER_COHERE,
        VALUE__LLM_PROVIDER_OPENAI,
    ]

    KEY__SUPPORTED_LLM_PROVIDERS = "SUPPORTED_LLM_PROVIDERS"
    VALUE__SUPPORTED_LLM_PROVIDERS = COLON_STRING.join(ALL_SUPPORTED_LLM_PROVIDERS)

    KEY__LLM_TEMPERATURE = "LLM__TEMPERATURE"

    VALUE__LLM_TEMPERATURE = "0.4"
    KEY__LLM_TOP_P = "LLM__TOP_P"
    VALUE__LLM_TOP_P = "0.4"
    KEY__LLM_TOP_K = "LLM__TOP_K"
    VALUE__LLM_TOP_K = "40"
    KEY__LLM_REPEAT_PENALTY = "LLM__REPEAT_PENALTY"
    VALUE__LLM_REPEAT_PENALTY = "1.1"
    KEY__LLM_SEED = "LLM__SEED"
    VALUE__LLM_SEED = "1"

    KEY__LLM_GROQ_MODEL = "LLM__GROQ_MODEL"
    VALUE__LLM_GROQ_MODEL = "llama3-groq-8b-8192-tool-use-preview"

    KEY__LLM_ANTHROPIC_MODEL = "LLM__ANTHROPIC_MODEL"
    VALUE__LLM_ANTHROPIC_MODEL = "claude-3-opus-20240229"

    KEY__LLM_COHERE_MODEL = "LLM__COHERE_MODEL"
    VALUE__LLM_COHERE_MODEL = "command-r-plus"

    KEY__LLM_OPENAI_MODEL = "LLM__OPENAI_MODEL"
    VALUE__LLM_OPENAI_MODEL = "gpt-4o-mini"

    KEY__LLM_OLLAMA_URL = "LLM__OLLAMA_URL"
    VALUE__LLM_OLLAMA_URL = "http://localhost:11434"
    KEY__LLM_OLLAMA_MODEL = "LLM__OLLAMA_MODEL"
    VALUE__LLM_OLLAMA_MODEL = "mistral-nemo"

    KEY__ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    KEY__COHERE_API_KEY = "COHERE_API_KEY"
    KEY__GROQ_API_KEY = "GROQ_API_KEY"
    KEY__OPENAI_API_KEY = "OPENAI_API_KEY"


CSS__GRADIO_APP = """
    #ui_header * {
        margin-top: auto;
        margin-bottom: auto;
    }
    #ui_header_text {
        text-align: right
    }
"""

JS__DARK_MODE_TOGGLE = """
    () => {
        document.body.classList.toggle('dark');
        document.querySelector('gradio-app').style.background = 'var(--body-background-fill)';
    }
"""
