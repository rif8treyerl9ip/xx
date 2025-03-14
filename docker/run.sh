#!/bin/bash
# docker run --rm -it --gpus all \
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  tensorrt:with_opencv \
  bash
  # tensorrt:latest \

docker commit a5451fabbdfc tensorrt:with_opencv
