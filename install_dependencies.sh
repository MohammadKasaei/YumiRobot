#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install numpy and matplotlib
pip install numpy matplotlib

# Install pybullet
pip install pybullet

# Install torch torchvision
pip install torch torchvision 

# Install Segment anything 
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install open-cv and other basic packages
pip install opencv-python pycocotools numpy matplotlib onnxruntime onnx 


# Install open3d and scikit-image
pip install open3d scikit-image

echo "All dependencies installed successfully!"
