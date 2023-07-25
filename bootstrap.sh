#!/usr/bin/env bash

set -e

cd third_party/maskrcnn-benchmark
python3 setup.py build_ext develop
cd ../..
