#!/bin/bash
export PORT=${PORT:-8080}
gunicorn --bind 0.0.0.0:$PORT app.app:server