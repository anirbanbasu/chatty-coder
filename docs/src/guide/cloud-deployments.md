# Cloud deployment

There is a public deployment of this app available through a Hugging Face Spaces at [https://huggingface.co/spaces/xtremebytes/TLDRLC](https://huggingface.co/spaces/xtremebytes/TLDRLC). Note that this deployment is experimental and bugs are quite likely. There may be additional problems due to any restrictions on the Hugging Face Spaces infrastructure, which will not manifest in a local deployment.

For a cloud deployment, you have to use Open AI or Cohere. By default, graph, index and documents will be stored in memory with no disk persistence. If you want persistence with the deployment with a cloud deployment, you must use [Neo4j Aura DB](https://neo4j.com/cloud/platform/aura-graph-database/) and a remotely hosted Redis (e.g., [Redis on Render](https://docs.render.com/redis)). Alternatively, you can use local or on-premises hosted Ollama, Neo4j and Redis by exposing those services with TCP (or, TLS) tunnels publicly through [ngrok](https://ngrok.com/).
