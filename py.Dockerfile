# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update

COPY . /usr/local/minigrid/
WORKDIR /usr/local/minigrid/

RUN pip install .[testing] --no-cache-dir
