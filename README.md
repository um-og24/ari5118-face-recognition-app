# ARI5118 - Face Recognition and Benchmarking System

## Overview
This repository contains the Python implementation of a face recognition and benchmarking system, as described in the ARI5118 Assignment paper.

## Setup
This script will get the environment ready and set up for you.

WARNING: Runs only on Linux or WSL and uses Python3.10

Run: `./setup.sh -y`


## Dependencies
Install via: `pip install -r requirements.txt`

## Execution
Run the start script: `./start.sh` or `./start.sh --all`

or execute the individual components seperately:

- `./start.sh --monitoring`
- `./start.sh --backend`
- `./start.sh --frontend`

Access the interface at `http://localhost:8501`.


## Docker Image

If `docker-compose.yml` file is available, Then:-

Build the image locally: `docker-compose up --build -d`

Else:-

Login (make sure Docker Desktop is running): `docker login`

Build and push onto Docker hub: `docker build -t owengauci24/ari5118-face-recognition-app:latest . --push`

Download a copy: `docker pull owengauci24/ari5118-face-recognition-app:latest`

Run the image: `docker run -it -p 8501:8501 owengauci24/ari5118-face-recognition-app:latest`

