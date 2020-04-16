#!/bin/bash

FMSG="- COVID-19 AI Quantum Tensorflow installation terminated"

read -p "? This script will install the IDC QNN, Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma on your device. Are you ready (y/n)? " cmsg

if [ "$cmsg" = "Y" -o "$cmsg" = "y" ]; then

    echo "- Installing IDC QNN, Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma"

    sh Scripts/Installation/Shell/Python.sh
    if [ $? -ne 0 ]; then
        echo $FMSG;
        exit
    fi

    sh Scripts/Installation/Shell/TF-Quantum.sh
    if [ $? -ne 0 ]; then
        echo $FMSG;
        exit
    fi

    echo "!IDC QNN, Leveraging Quantum MNIST to detect Invasive Ductal Carcinoma installation complete!"

else
    echo $FMSG;
    exit
fi