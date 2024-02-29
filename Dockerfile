# syntax=docker/dockerfile:1


# Setup of Depedencies:
FROM python:3.11-slim AS setup

WORKDIR /app

# setup of venv
RUN python3 -m venv ./venv
ENV PATH=/app/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pip-tools virtualenv

# installation of (base) dependencies into venv
COPY pyproject.toml .
RUN pip-compile --output-file=requirements.txt pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt


# Building of Base Image:
FROM python:3.11-slim AS daisy-base

WORKDIR /app
EXPOSE 8000-8003

# import of built venv
COPY --from=setup /app/venv /app/venv
ENV PATH=/app/venv/bin:$PATH

# installtion/build
COPY . .
RUN pip3 install --no-cache-dir -e .


# Building of GPU Image:
FROM base as build-gpu
RUN pip3 install --no-cache-dir -e .[cuda]

# Building of CPU Image:
FROM base as build