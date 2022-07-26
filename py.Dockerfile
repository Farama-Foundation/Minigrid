# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION
RUN apt-get -y update

COPY . /usr/local/gym_minigrid/
WORKDIR /usr/local/gym_minigrid/

RUN pip install . && pip install -r test_requirements.txt
