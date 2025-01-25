#!/bin/bash

# Create data directory
mkdir -p data
cd data

# Download and extract files showing progress
echo "Downloading annotations..."
wget --progress=bar:force "https://www.dropbox.com/scl/fi/hdcsmc7l688427k5dvslk/annotations.zip?rlkey=nh7gh6t16980nt82kd61ad2lz" -O annotations.zip

echo "Downloading videos..."
wget --progress=bar:force "https://www.dropbox.com/scl/fi/w81mdnkual8l4xdi9500q/videos.zip?rlkey=gqgwotuwyfmtb8igfn3pp181w&st=6ctlppvr&dl=1" -O videos.zip

echo "Extracting files..."
unzip -q videos.zip
unzip -q annotations.zip

echo "Setup complete!"

# Clean up zip files
rm videos.zip annotations.zip