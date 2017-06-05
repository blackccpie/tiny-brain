#!/bin/bash

mkdir -p build_emcc

cd build_emcc
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake -DUSE_EMSCRIPTEN=ON -DCMAKE_BUILD_TYPE=Release ..
make -j3 install
