#!/bin/sh

mkdir -p build_gcc

cd build_gcc
cmake -DCMAKE_CXX_FLAGS="-mavx" -DCMAKE_BUILD_TYPE=Release ..
make -j3
