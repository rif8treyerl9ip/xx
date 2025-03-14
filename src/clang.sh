#!/bin/bash

set -eu

rm -f tmp.cc && clang-format src/c/run_resnet50.cc > tmp.cc && mv tmp.cc src/c/run_resnet50.cc
rm -f tmp.cc && clang-format src/c/main.cc > tmp.cc && mv tmp.cc src/c/main.cc
rm -f tmp.cc && clang-format src/c/simple_identity_plugin.cc > tmp.cc && mv tmp.cc src/c/simple_identity_plugin.cc
