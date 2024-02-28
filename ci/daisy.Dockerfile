# syntax=docker/dockerfile:1

FROM python:3.11-slim AS daisy-base

WORKDIR /app
EXPOSE 8000/tcp
EXPOSE 8001/tcp
EXPOSE 8002/tcp

COPY ../LICENSE.txt .
COPY ../pyproject.toml .
COPY ../src ./src
COPY ../tests ./tests

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir .


FROM daisy-base as daisy-gpu

RUN pip3 install --no-cache-dir .[cuda]