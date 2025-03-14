#!/bin/bash
set -e

echo "=== プラグインをビルド中 ==="
g++ -std=c++11 \
    -fPIC \
    -shared \
    -o libsimple_identity_plugin.so src/c/simple_identity_plugin.cc \
    -I/usr/local/cuda-12.4/targets/x86_64-linux/include \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -lcudart \
    -lnvparsers \
    -lnvinfer \
    -lnvinfer_plugin

# echo "=== プラグインのシンボルを確認 ==="
# nm -D libsimple_identity_plugin.so | grep -E 'Plugin|register|init'

echo "=== テストプログラムをビルド中 ==="
g++ -std=c++11 -o trt_plugin_test src/c/main.cc \
    -I/usr/local/cuda-12.4/targets/x86_64-linux/include \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/x86_64-linux-gnu \
    -lcudart \
    -lnvparsers \
    -lnvinfer \
    -lnvinfer_plugin \
    -ldl

echo "=== テストプログラムを実行 ==="
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./trt_plugin_test