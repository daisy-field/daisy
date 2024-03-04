# syntax=docker/dockerfile:1
ARG BUILD_VERSION=cpu


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

# Installation of additional (GPU) Depedencies:
FROM setup AS setup-gpu
RUN pip-compile --extra cuda --output-file=requirements.txt pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt


# Setup of Base Image:
FROM python:3.11-slim AS base
WORKDIR /app
EXPOSE 8000-8003
# installtion of additional (non-python/venv) dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y tshark \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists

# Import of CPU Dependencies:
FROM base as base-cpu
# import of built venv
COPY --from=setup /app/venv /app/venv

# Import of GPU Dependencies:
FROM base AS base-gpu
COPY --from=setup-gpu /app/venv /app/venv


# Build of Daisy Image:
FROM base-${BUILD_VERSION} AS daisy-build
ENV PATH=/app/venv/bin:$PATH
COPY . .
RUN pip3 install --no-cache-dir -e .
