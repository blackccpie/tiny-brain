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
project (mnist_sign)

if (USE_EMSCRIPTEN)

# TODO : ALLOW_MEMORY_GROWTH prevents some optimizations, try to define new total memory size
# NO_EXIT_RUNTIME is here to have std::cout available event after main has finished
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --bind -s WASM=1  -s NO_EXIT_RUNTIME=1 -s ALLOW_MEMORY_GROWTH=1 -O3")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --emrun")

else ()

    if (CIMG_FOUND)
        set(extra_link_libs X11)
	endif ()

endif ()

set (sources_list
main.cpp
)

add_executable(mnist_sign ${sources_list} ${headers_list})

target_link_libraries(mnist_sign
    pthread
    ${extra_link_libs}
)

install(
    TARGETS mnist_sign
    DESTINATION mnist_sign
)

if (USE_EMSCRIPTEN)
    install(
        FILES
        ${CMAKE_SOURCE_DIR}/bin/mnist_sign.wasm
        #${CMAKE_SOURCE_DIR}/bin/mnist_sign.data
        index.html
        run.sh
    	DESTINATION mnist_sign
    )
endif ()

cotire(mnist_sign)
