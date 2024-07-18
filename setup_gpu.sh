#!/bin/bash

# setup.cpu.sh

# Set the name for the virtual environment
ENV_NAME="venv"

# Create a virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Update pip without using cache
pip install --no-cache-dir --upgrade pip

# Install required packages without using cache
pip install --no-cache-dir datasets==2.18.0
pip install --no-cache-dir pyarrow==15.0.2
pip install --no-cache-dir tokenizers==0.15.2
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir transformers==4.39.3
pip install --no-cache-dir triton==2.2.0
pip install --no-cache-dir python-igraph
pip install --no-cache-dir tensorboard matplotlib
pip install --no-cache-dir accelerate==0.29.3
pip install --no-cache-dir einops==0.7.0

echo "Installation complete!"
echo "To activate this environment in the future, run:"
echo "source $ENV_NAME/bin/activate"