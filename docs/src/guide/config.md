# Configuration through environment variables

The app configuration can be controlled using environment variables. You can specify those variables using a `.env` file, for example. If you are containerising the app then you must update the `.env.docker` before creating the Docker image.

The following is a list of configuration variables. All environment variables must be specified as quoted strings, such as `ENV_KEY = "value"`. Type casting will be performed by the app as necessary. Some of the variables are required or conditionally required.

## Gradio server settings

Notice that host and port variables listed below are not the same as the corresponding ones mentioned in [Gradio documentation](https://www.gradio.app/guides/environment-variables), which will be ignored.

- `GRADIO__SERVER_HOST`: The host name to which the Gradio server should bind to. Default value: `localhost`. Set to `0.0.0.0` in `.env.docker`.
- `GRADIO__SERVER_PORT`: The port number on the `GRADIO__SERVER_HOST`, which the Gradio server should bind on. Default value: `7860`. Parsed as non-string type: `int`.

## General large language model (LLM) settings

- `LLM__PROVIDER`: The provider for the language language models. Default value: `Ollama`. Possible values are: `Open AI`, `Cohere`, `Groq` and `Ollama`.
- `LLM__TEMPERATURE`: The temperature value for the LLM to be used. Default value: `0.0`. The value should be a decimal number between `0` and `1` except if the provider is Open AI, for which the value can be between `0` and `2`. Parsed as non-string type: `float`.

## Language model specific settings

- `LLM__OPENAI_API_KEY`: (**required** only if language model provider is Open AI) A valid Open AI API key.
- `LLM__OPENAI_MODEL`: The Open AI model you want to use. Default value: `gpt-4o-mini`.
- `LLM__COHERE_API_KEY`: (**required** only if language model provider is Cohere) A valid Cohere API key.
- `LLM__COHERE_MODEL`: The Cohere model you want to use. Default value: `command-r-plus`.
- `LLM__GROQ_API_KEY`: (**required** only if language model provider is Groq) A valid Groq API key.
- `LLM__GROQ_MODEL`: The Groq model you want to use. Default value: `llama3-groq-70b-8192-tool-use-preview`.
- `LLM__OLLAMA_URL`: URL for a running Ollama instance. Default value: `http://localhost:11434`. Set to `http://host.docker.internal:11434` in `.env.docker`. (Learn about [Docker networking](https://docs.docker.com/desktop/networking/) and set it correctly as needed.)
- `LLM__OLLAMA_MODEL`: The Ollama model you want to use. Default value: `llama3`.
