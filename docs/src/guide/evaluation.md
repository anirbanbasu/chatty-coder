# Performance evaluation using Langfuse

If you wish to use [Langfuse](https://langfuse.com/) for performance evaluation then set `EVAL_USE_LANGFUSE = "True"` in the `.env` file followed by `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PRIVATE_KEY`, `LANGFUSE_URL`. You can set these up by running Langfuse self-hosted or by signing up and signing in to [the Langfuse cloud](https://cloud.langfuse.com/).

**Note** that for [cloud deployment(s)](cloud-deployments.md), unless you create your separate cloud deployment by yourself, LangFuse evaluation is enabled by default and cannot be turned off. This is meant for evaluation purposes of this project. Hence, all information about your interactions with the chatbot will be available to the maintainer(s) of this project. _For local, Docker or cloud deployments that you create, LangFuse is **not** enabled by default. Even when enabled, by yourself, LangFuse traces will be available to you, and not the maintainer(s) of this project_.
