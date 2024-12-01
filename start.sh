#!/bin/bash

# Update system packages
echo "Updating system packages..."
apt-get update

# Install Git (ensure it's installed before installing Git LFS)
echo "Installing Git..."
apt-get install -y git

# Download and install Git LFS manually (without needing sudo)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get update && apt-get install -y git-lfs

# Ensure Git LFS is initialized
git lfs install

# Pull Git LFS files
git lfs pull

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

