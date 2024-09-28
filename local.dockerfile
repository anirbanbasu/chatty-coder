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

# Pull Python on Debian image
FROM python:3.12.5-slim-bookworm AS build

# Upgrade and install basic packages
RUN apt-get update && apt-get -y upgrade && apt-get -y install build-essential curl

# Create a non-root user
RUN useradd -m -u 1000 app_builder

ENV HOME="/home/app_builder"

USER app_builder
# Set the working directory in the container
WORKDIR ${HOME}/app

# Install the Rust toolchain, which is required by some Python packages during their building processes
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="${HOME}/.cargo/bin:${PATH}"

# Copy the requirements file, which will not be retained in the final image
COPY ./requirements.txt ./requirements.txt

# Setup a virtual environment
ENV VIRTUAL_ENV="${HOME}/app/venv"
RUN python -m venv ${VIRTUAL_ENV}
RUN ${VIRTUAL_ENV}/bin/python -m ensurepip
RUN ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -U pip

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install and build dependencies
RUN ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -U -r requirements.txt

# [Multi-stage build](https://docs.docker.com/build/building/multi-stage/) to reduce the size of the final image to about 30% of the original size!
FROM python:3.12.5-slim-bookworm

# Create a non-root user. The home directory is used for caching stuff, e.g., from Hugging Face Transformers
RUN useradd -m -u 1000 app_user

# Switch to the non-root user
ENV HOME="/home/app_user"
USER app_user
# Set the working directory in the container
WORKDIR ${HOME}/app

# Copy only the Python virtual environment from the build image
COPY --from=build /home/app_builder/app/venv ./venv

# Copy the project files
COPY ./LICENSE ./.env.docker ./
# Although the example .env file is copied here, use -v to load a different .env file from the Docker host, if necessary.
RUN mv .env.docker .env
COPY ./src/*.py ./src/
# Copy the required assets
COPY ./assets/logo-embed.svg ./assets/

# Expose the port to conect
EXPOSE 7860
# Run the application
ENTRYPOINT [ "/home/app_user/app/venv/bin/python", "src/webapp.py" ]
