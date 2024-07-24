# Run locally

## The non-containerised way

Once you have installed the dependencies mentioned above in your Python virtual environment, to run the chatbot, execute `python src/app.py`. If the web server starts up without any problem, the web application user interface will be available at the host and port specified by the environment variables `GRADIO__SERVER_HOST` and `GRADIO__SERVER_PORT` respectively, assuming that the server can bind to the specified port on the specified host. If you do not specify these variables, they will default to `localhost` and `7860` respectively, which means your web application will be available at [http://localhost:7860](http://localhost:7860]).

## Containerised (Docker)

Following the [creation of the container](container.md), you can run the app using `docker container start chatty-coder-container` the app will be accessible on your Docker host, for example as [http://localhost:7860](http://localhost:7860) -- assuming that nothing else on host is blocking port 7860 when the container starts.

The Docker container has to depend on external LLM provider, graph database, document and index storage. If any of these, such as `Ollama`, is running on the Docker host then you should use the host name for the service as `host.docker.internal` or `gateway.docker.internal`. See [the networking documentation of Docker desktop](https://docs.docker.com/desktop/networking/) for details.
