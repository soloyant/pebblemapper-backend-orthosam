#!/bin/bash
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
#
# Build the pm-orthosam environment with the exact versions the adapter was
# tested with. Needs conda on PATH, or CONDA_EXE pointing at conda.exe (Git
# Bash on Windows). For a machine without an NVIDIA GPU, replace the cu121
# index with https://download.pytorch.org/whl/cpu and drop the +cu121 tags.
set -ex
CONDA="${CONDA_EXE:-conda}"
"$CONDA" create -y -n pm-orthosam -c conda-forge python=3.12 pip
PIP="$CONDA run -n pm-orthosam --no-capture-output python -m pip"
$PIP install --upgrade pip
$PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
$PIP install segment-anything==1.0 "numpy<2.4" opencv-python-headless tifffile scikit-image scipy pandas psutil tqdm requests matplotlib pillow rasterio
$PIP list
echo ENV_DONE
