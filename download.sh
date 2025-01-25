#!/bin/bash
mkdir data 
cd data
# Download files with progress bar
wget --progress=bar:force https://www.dropbox.com/scl/fi/hdcsmc7l688427k5dvslk/annotations.zip?rlkey=nh7gh6t16980nt82kd61ad2lz -O annotations.zip && \
wget --progress=bar:force https://www.dropbox.com/scl/fi/w81mdnkual8l4xdi9500q/videos.zip?rlkey=gqgwotuwyfmtb8igfn3pp181w&st=6ctlppvr&dl=1 -O videos.zip && \
unzip videos.zip && \
unzip annotations.zip