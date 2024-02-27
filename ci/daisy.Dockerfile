# syntax=docker/dockerfile:1
#FROM python:3.11-slim
FROM tensorflow/tensorflow:2.15.0-gpu
ARG gpu=no

WORKDIR /app
COPY ../LICENSE.txt .
COPY ../pyproject.toml .
COPY ../src ./src
COPY ../tests ./tests


RUN pip3 install .

EXPOSE 8000/tcp
EXPOSE 8001/tcp
EXPOSE 8002/tcp

# CMD [ "python3", "src/trafficlight_service/trafficlight_service.py", "--mappingPath", "src/trafficlight_service/mapping.json", "--debug", "True"]