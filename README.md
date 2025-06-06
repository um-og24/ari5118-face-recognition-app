# ARI5118 - Face Recognition and Benchmarking System

## Overview
This repository contains the Python implementation of a face recognition and benchmarking system, as described in the ARI5118 Assignment paper.

## Setup
Linux or WSL Run: `setup.sh`
This script will get the environment ready and set up for you.

## Dependencies
Install via: `pip install -r requirements.txt`

## Execution
Run the start script: `start.sh`

or execute the individual components seperately:

- `start.sh --monitoring`
- `start.sh --backend`
- `start.sh --frontend`

Access the interface at `http://localhost:8501`.


## Docker Image

If `docker-compose.yml` file is available, Then:-

Build the image locally: `docker-compose up --build -d`

Else:-

Login (make sure Docker Desktop is running): `docker login`

Build and push onto Docker hub: `docker build -t owengauci24/ari5118-face-recognition-app:latest . --push`

, or to make it platform independent: `docker buildx build --platform linux/amd64,linux/arm64 -t owengauci24/ari5118-face-recognition-app:latest . --push`

Download a copy: `docker pull owengauci24/ari5118-face-recognition-app:latest`

Run the image: `docker run -it -p 8506:8506 owengauci24/ari5118-face-recognition-app:latest`

