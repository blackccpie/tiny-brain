/*
The MIT License

Copyright (c) 2017-2017 Albert Murienne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/em_asm.h>
using namespace emscripten;

#include "ocr_wrapper.h"

// Binding code
EMSCRIPTEN_BINDINGS(mnist_ocr)
{
    register_vector<size_t>("VectorSizeT");
    register_vector<uint8_t>("VectorUInt8");

    class_<ocr_wrapper>( "ocr_wrapper" )
        .constructor()
        .function( "process", &ocr_wrapper::process )
        .function( "cropped_numbers", &ocr_wrapper::cropped_numbers )
        .function( "cropped_size", &ocr_wrapper::cropped_size )
        .function( "reco_string", &ocr_wrapper::reco_string );
}

int main()
{
    EM_ASM( allReady() );
}
