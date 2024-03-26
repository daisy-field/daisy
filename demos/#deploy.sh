#!/bin/bash

git pull
git submodule update --init --recursive

docker compose pull
docker compose up
