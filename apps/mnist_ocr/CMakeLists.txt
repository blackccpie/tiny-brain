#The MIT License
#
#Copyright (c) 2017-2017 Albert Murienne
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

cmake_minimum_required (VERSION 3.2)
project (mnist_ocr)

if (USE_EMSCRIPTEN)

    # NO_EXIT_RUNTIME is here to have std::cout available event after main has finished
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} --bind -s WASM=1  -s NO_EXIT_RUNTIME=1 -O3 --preload-file ${CMAKE_SOURCE_DIR}/data@/")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")

    # no WebAssembly multi-threading support for now
    add_definitions( -DCNN_SINGLE_THREAD )

    set(extra_sources ocr_wrapper.cpp ocr_embind.cpp)

else ()

    if (CIMG_FOUND)
        set(extra_link_libs X11)
	endif ()

    set(extra_sources main.cpp)

endif ()

set (sources_list
ocr.cpp
${extra_sources}
)

add_executable(mnist_ocr ${sources_list} ${headers_list})

target_link_libraries(mnist_ocr
    pthread
    ${extra_link_libs}
)

install(
    TARGETS mnist_ocr
    DESTINATION mnist_ocr
)

if (USE_EMSCRIPTEN)
    install(
        FILES
        ${CMAKE_SOURCE_DIR}/bin/mnist_ocr.wasm
        ${CMAKE_SOURCE_DIR}/bin/mnist_ocr.data
        ${CMAKE_SOURCE_DIR}/data/ocr/images/123456.png
        index.html
        run.sh
    	DESTINATION mnist_ocr
    )
endif ()

# TODO : still no clue about why cotire and emscripten seem incompatible
if (NOT USE_EMSCRIPTEN)
    cotire(mnist_ocr)
endif ()
