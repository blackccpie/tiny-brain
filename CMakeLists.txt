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

cmake_minimum_required (VERSION 3.5)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_SOURCE_DIR}/cmake/cotire.cmake)

if (USE_EMSCRIPTEN)

    # NOTHING TO DO YET

else ()

	include(CheckCXXCompilerFlag)
	check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
	check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
	check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
	check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

	# set Streaming SIMD Extension (SSE) instructions
	if (USE_SSE AND COMPILER_HAS_SSE_FLAG)
    	add_definitions(-DCNN_USE_SSE)
    	set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
	endif (USE_SSE AND COMPILER_HAS_SSE_FLAG)
	# set Advanced Vector Extensions (AVX)
	if (USE_AVX AND COMPILER_HAS_AVX_FLAG)
    	add_definitions(-DCNN_USE_AVX)
    	set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
	endif (USE_AVX AND COMPILER_HAS_AVX_FLAG)
	# set Advanced Vector Extensions 2 (AVX2)
	if (USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
    	add_definitions(-DCNN_USE_AVX2)
    	set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
	endif (USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

endif ()

# Set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

include_directories(
    "${CMAKE_SOURCE_DIR}/tiny-dnn"
	"${CMAKE_SOURCE_DIR}/tiny-dnn/third_party"
)

add_subdirectory(apps)
