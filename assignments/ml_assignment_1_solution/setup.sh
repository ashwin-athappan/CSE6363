#!/bin/bash

# Define environment name and Python version
ENV_NAME="myenv"
PYTHON_VERSION="3.13"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
else
    echo "Conda environment $ENV_NAME already exists."
fi

# Ensure conda is initialized
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

echo "Conda environment $ENV_NAME is now active."

#conda deactivate
#conda remove --name myenv --all -y