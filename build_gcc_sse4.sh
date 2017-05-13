#!/bin/sh

mkdir -p build_gcc

cd build_gcc
cmake -DCMAKE_CXX_FLAGS="-msse4.1" -DCMAKE_BUILD_TYPE=Release ..
make -j3
