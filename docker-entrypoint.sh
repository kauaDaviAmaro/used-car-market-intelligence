#!/bin/bash
set -e

if [ ! -f "/app/models/price_predictor_v1.pkl" ] && [ ! -f "/app/models/price_predictor_v4.pkl" ]; then
    echo "Model not found. Running pipeline to generate model..."
    python pipeline.py pipeline --skip-checks
fi

exec "$@"

