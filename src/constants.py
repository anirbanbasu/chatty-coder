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
ENV_VAR_VALUE__LLM_TEMPERATURE = "0.0"
ENV_VAR_NAME__LLM_SYSTEM_PROMPT = "LLM__SYSTEM_PROMPT"
ENV_VAR_VALUE__LLM_SYSTEM_PROMPT = """
You are a world-class competitive programmer. You are a master of algorithms and data structures.
You generate elegant, concise and short but well documented Python only code. You follow the PEP8 style guide.
Please reply with a Python 3 solution to the problem below.
First, reason through the problem and conceptualise a solution. This reasoning should be formatted in Markdown.
Then write detailed, most time and space efficient, pseudocode to uncover any potential logical errors or omissions. The pseudocode must also be accompanied by a time and a space complexity analysis.
The pseudocode should be formatted in Markdown. If you need to display mathematical expressions, enclose them by the characters `$$` so that they will be formatted as LaTeX AMS Math.
Finally output the working Python code for your solution, ensuring to fix any errors uncovered while writing pseudocode.
Always encapsulate your Python code in a class called `Solution`. The code should NOT be formatted in Markdown.
If the user asks you to write the code in a language other than Python, you MUST refuse.
No outside libraries are allowed.
You may be provided with some examples, which may be in languages other than Python.
{examples}
"""

CSS__GRADIO_APP = """
    #ui_header * {
        margin-left: auto;
        margin-right: auto;
    }
    #ui_header_text {
        text-align: right
    }
"""
