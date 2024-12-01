#!/bin/bash

# Update system packages
echo "Updating system packages..."

# Install sudo if it's not already installed
if ! command -v sudo &> /dev/null
then
    echo "sudo not found, installing..."
    apt-get update && apt-get install -y sudo
fi

sudo apt-get update -y

# Install pip if not installed
if ! command -v pip &> /dev/null; then
    echo "pip not found, installing..."
    sudo apt-get install -y python3-pip
else
    echo "pip is already installed."
fi

# Install Git LFS if not installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found, installing..."
    sudo apt-get install -y git-lfs
    git lfs install
else
    echo "Git LFS is already installed."
fi

# Pull Git LFS data
echo "Pulling Git LFS data..."
git lfs pull

# Install Python requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
