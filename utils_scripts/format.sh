#!/bin/bash
set -e  # Exit on first failure

python -m black metassl
python -m isort metassl -rc
