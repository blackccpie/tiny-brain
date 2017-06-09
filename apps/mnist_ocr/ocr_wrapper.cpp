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

#include "ocr_wrapper.h"
#include "ocr.h"

#include "tiny_brain/tinymage.h"

#include <emscripten/bind.h>
using namespace emscripten;

class ocr_wrapper::ocr_wrapper_impl
{
public:
    ocr_wrapper_impl() {}
    void process( const tinymage<float>& img )
    {
        m_ocr.process( img );
    }
private:
    ocr_helper m_ocr;
};

ocr_wrapper::ocr_wrapper()
    : m_pimpl( std::make_unique<ocr_wrapper_impl>() )
{
}

ocr_wrapper::~ocr_wrapper()
{
}

void ocr_wrapper::process( emscripten::val image, emscripten::val onComplete )
{
    auto ptr = reinterpret_cast<uint8_t*>( image["ptr"].as<int>() );
    auto size = input["size"].as<int>();

    std::cout << "ocr_wrapper::process - " << size << " bytes @" << reinterpret_cast<int>( ptr ) << std::endl;

    tinymage<float> img;
    img.load( "./ocr/images/123456.png" );
    m_pimpl->process( img );
}
