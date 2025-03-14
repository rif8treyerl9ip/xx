#!/bin/bash

set -eu

clear && g++ -o tensorrt_program src/c/run_resnet50.cc \
    -I/usr/local/cuda/include \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/include/opencv4 \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -lnvinfer -lcudart -lcuda -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc && ./tensorrt_program

