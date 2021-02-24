#!/usr/bin/env bash

PORT=${1:-1337}

~/anaconda3/bin/gunicorn manager_gunicorn:app --bind 0.0.0.0:${PORT} --worker-class sanic.worker.GunicornWorker -w 4 --threads 8 --timeout 240
