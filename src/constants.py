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
EMPTY_DICT = {}
EMPTY_JSON_DICT = "{}"
SPACE_STRING = " "
BOOLEAN_AND = "AND"
BOOLEAN_OR = "OR"
BOOLEAN_NOT = "NOT"
BOOLEAN_XOR = "XOR"

TRUE_VALUES_LIST = ["true", "yes", "t", "y", "on"]

PROJECT_GIT_REPO_URL = "https://github.com/anirbanbasu/chatty-coder"
PROJECT_GIT_REPO_LABEL = "GitHub repository"
PROJECT_NAME = "chatty-coder"

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

AGENT_STATE__KEY_CANDIDATE = "candidate"
AGENT_STATE__KEY_EXAMPLES = "examples"
AGENT_STATE__KEY_MESSAGES = "messages"
AGENT_STATE__KEY_TEST_CASES = "test_cases"
AGENT_STATE__KEY_RUNTIME_LIMIT = "runtime_limit"
AGENT_STATE__KEY_STATUS = "status"
AGENT_STATE__KEY_DRAFT = "draft"

AGENT_NODE__EVALUATE_STATUS_SUCCESS = "success"
AGENT_NODE__EVALUATE_STATUS_NO_TEST_CASES = "no test cases"

AGENT_TOOL_CALL__ARGS = "args"
AGENT_TOOL_CALL__ID = "id"
AGENT_TOOL_CALL__NAME_CODE_OUTPUT = "codeOutput"

TEST_CASE__KEY_INPUTS = "inputs"
TEST_CASE__KEY_OUTPUTS = "outputs"

PYDANTIC_MODEL__CODE_OUTPUT__REASONING = "reasoning"
PYDANTIC_MODEL__CODE_OUTPUT__PSEUDOCODE = "pseudocode"
PYDANTIC_MODEL__CODE_OUTPUT__CODE = "code"

PROMPT_TEMPLATE__VAR_MESSAGES = AGENT_STATE__KEY_MESSAGES
PROMPT_TEMPLATE__PARTIAL_VAR__EXAMPLES = "examples"

# Environment variables
ENV_VAR_NAME__GRADIO_SERVER_HOST = "GRADIO__SERVER_HOST"
ENV_VAR_VALUE__GRADIO_SERVER_HOST = "localhost"
ENV_VAR_NAME__GRADIO_SERVER_PORT = "GRADIO__SERVER_PORT"
ENV_VAR_VALUE__GRADIO_SERVER_PORT = "7860"

ENV_VAR_NAME__LLM_PROVIDER = "LLM__PROVIDER"
ENV_VAR_VALUE__LLM_PROVIDER = "Ollama"

ENV_VAR_NAME__LLM_OPENAI_API_KEY = "LLM__OPENAI_API_KEY"
ENV_VAR_NAME__LLM_OPENAI_MODEL = "LLM__OPENAI_MODEL"
ENV_VAR_VALUE__LLM_OPENAI_MODEL = "gpt-4o-mini"
ENV_VAR_NAME__LLM_ANTHROPIC_API_KEY = "LLM__ANTHROPIC_API_KEY"
ENV_VAR_NAME__LLM_ANTHROPIC_MODEL = "LLM__ANTHROPIC_MODEL"
ENV_VAR_VALUE__LLM_ANTHROPIC_MODEL = "claude-3-opus-20240229"
ENV_VAR_NAME__LLM_COHERE_API_KEY = "LLM__COHERE_API_KEY"
ENV_VAR_NAME__LLM_COHERE_MODEL = "LLM__COHERE_MODEL"
ENV_VAR_VALUE__LLM_COHERE_MODEL = "command-r-plus"
ENV_VAR_NAME__LLM_GROQ_API_KEY = "LLM__GROQ_API_KEY"
ENV_VAR_NAME__LLM_GROQ_MODEL = "LLM__GROQ_MODEL"
ENV_VAR_VALUE__LLM_GROQ_MODEL = "llama3-groq-70b-8192-tool-use-preview"
ENV_VAR_NAME__LLM_OLLAMA_URL = "LLM__OLLAMA_URL"
ENV_VAR_VALUE__LLM_OLLAMA_URL = "http://localhost:11434"
ENV_VAR_NAME__LLM_OLLAMA_MODEL = "LLM__OLLAMA_MODEL"
ENV_VAR_VALUE__LLM_OLLAMA_MODEL = "llama3"
ENV_VAR_NAME__LLM_LLAMAFILE_URL = "LLM__LLAMAFILE_URL"
ENV_VAR_VALUE__LLM_LLAMAFILE_URL = "http://localhost:8080"
ENV_VAR_NAME__LLM_TEMPERATURE = "LLM__TEMPERATURE"
ENV_VAR_VALUE__LLM_TEMPERATURE = "0.4"
ENV_VAR_NAME__LLM_TOP_P = "LLM__TOP_P"
ENV_VAR_VALUE__LLM_TOP_P = "0.4"
ENV_VAR_NAME__LLM_TOP_K = "LLM__TOP_K"
ENV_VAR_VALUE__LLM_TOP_K = "40"
ENV_VAR_NAME__LLM_REPEAT_PENALTY = "LLM__REPEAT_PENALTY"
ENV_VAR_VALUE__LLM_REPEAT_PENALTY = "1.1"
ENV_VAR_NAME__LLM_SEED = "LLM__SEED"
ENV_VAR_VALUE__LLM_SEED = "1"
ENV_VAR_NAME__LLM_SYSTEM_PROMPT = "LLM__SYSTEM_PROMPT"
ENV_VAR_VALUE__LLM_CODER_SYSTEM_PROMPT = """
You are a world-class competitive Python programmer. You write concise and well documented Python only code. You follow the PEP8 style guide.
Please respond with a Python 3 solution to the problem below.
First, reason through the problem and conceptualise a solution. Output this reasoning in Markdown format.
Then write a detailed pseudocode to uncover any potential logical errors or omissions in your reasoning.
Wherever relevant, the pseudocode must also be accompanied by a time and a space complexity estimation. Output the pseudocode in Markdown format.
Finally output the working Python code for your solution, ensuring to fix any errors uncovered while writing the pseudocode.
Do not use external libraries.
You may be provided with examples, some of which may be in languages other than Python.
{examples}
"""

ENV_VAR_VALUE__LLM_CODER_SYSTEM_PROMPT = """
You are a world-class Python programmer. You write concise and well-documented code following the PEP8 style guide.

Please respond with a Python 3 solution to the problem below.
First, output a reasoning through the problem and conceptualise a solution. Whenever possible, add a time and a space complexity analysis for your solution.
Then, output a pseudocode in Pascal to implement your concept solution.
Then, output the working Python 3 code for your solution. Do not use external libraries. Your code must be able to accept inputs from `sys.stdin` and write the final output to `sys.stdout` (or, to `sys.stderr` in case of errors).
Finally, output a one sentence summary describing what your solution does, as if you are explaining your solution to the human user.

Optional examples of similar problems and solutions (may not be in Python):
{examples}

Given problem:
"""


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
                                // document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)';
                                document.querySelector('gradio-app').style.background = 'var(--body-background-fill)';
                            }
                        """
