#!/bin/bash

# Check if Git is installed
echo "Checking for Git..."
if ! command -v git &> /dev/null; then
    echo "Git could not be found! Exiting..."
    exit 1
fi
echo "Git is installed."

# Install Git LFS (without root access)
echo "Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
git lfs install

# Pull Git LFS data (this assumes LFS pointers are already in the repo)
echo "Pulling Git LFS data..."
git lfs pull

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt