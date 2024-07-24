<p align="center">
  <img width="384" height="96" src="https://raw.githubusercontent.com/anirbanbasu/chatty-coder/master/assets/logo-embed.svg" alt="chatty-coder logo" style="filter: invert(0.5)">
</p>

The **chatty-coder** is an agent-based human-in-the-loop computer program code generator. It is based on the idea introduced in the 2024 paper _Can Language Models Solve Olympiad Programming?_[^1], which uses a combination of _self-reflection_ and _retrieval over episodic knowledge_ with _human-in-the-loop_ collaboration to improve code generation responses of language models to coding challenges from the USA Computing Olympiad (USACO). The chatty-coder resembles the functionality of the experimental LangGraph-based implementation, called [Competitive Programming](https://langchain-ai.github.io/langgraph/tutorials/usaco/usaco/), of the aforementioned paper. The chatty-coder project has a Gradio-based user interface and an associated FastAPI-based REST API.

[^1]: Shi, Q., Tang, M., Narasimhan, K. and Yao, S., 2024. Can Language Models Solve Olympiad Programming? arXiv preprint [arXiv:2404.10952](https://arxiv.org/abs/2404.10952).
