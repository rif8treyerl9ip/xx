#!/bin/bash

set -eu

clear && g++ -o tensorrt_program src/c/run_resnet50.cc \
    -I/usr/local/cuda/include \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/include/opencv4 \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -lnvinfer -lcudart -lcuda -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc && ./tensorrt_program



# inference with opencv
- [x] install torch, onnx
- [x] build
- [x] convert torch to onnx(py)
- [x] convert onnx to trt(py)
- [x] load resnet50.engine
- [x] install opencv
- [x] load image with opencv
- [x] run 


- 8.6.3 & 10.5.0
- plugin 
  - implicit batch vs explicit batch
    - パフォーマンス
  - 使えるオブジェクトの選択肢
    - パフォーマンス
- v2 -> v3に挙げる方法など

