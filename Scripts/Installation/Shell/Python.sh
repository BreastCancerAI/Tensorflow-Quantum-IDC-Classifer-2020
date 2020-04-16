#!/bin/bash

read -p "? This script will install Python dependencies. Are you ready (y/n)? " cmsg

if [ "$cmsg" = "Y" -o "$cmsg" = "y" ]; then
    echo "- Python dependencies installing"
    pip3 install --user opencv-python
    pip3 install --user matplotlib
    pip3 install --user numpy
    pip3 install --user scipy
    pip3 install --user Pillow
    pip3 install --user jsonpickle
    pip3 install --user scikit-learn
    pip3 install --user scikit-image
    echo "- Python dependencies installed";
    exit 0
else
    echo "- Python dependencies installation terminated";
    exit 1
fi