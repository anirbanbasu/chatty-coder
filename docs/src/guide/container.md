# Container (Docker)
You can run the app in a Docker container. To do so, you have to build its image (which we name as `chatty-coder` although you can choose any other name) and run an instance (which we name as `chatty-coder-container` but you can also pick a name of your choice) of that image, as follows.

```zsh
docker build -f local.dockerfile -t chatty-coder .
docker create -p 7860:7860/tcp --name chatty-coder-container chatty-coder
```
