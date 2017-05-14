#!/bin/sh

mkdir -p build_gcc

cd build_gcc
cmake -DUSE_AVX2=ON -DCMAKE_BUILD_TYPE=Release ..
make -j3
