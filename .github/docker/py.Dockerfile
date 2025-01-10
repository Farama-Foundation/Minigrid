# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
ARG GYMNASIUM_VERSION
ARG NUMPY_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    xvfb

COPY ../.. /usr/local/minigrid/
WORKDIR /usr/local/minigrid/

RUN pip install .[wfc,testing] gymnasium$GYMNASIUM_VERSION numpy$NUMPY_VERSION --no-cache-dir

ENTRYPOINT ["/usr/local/minigrid/.github/docker/docker_entrypoint"]