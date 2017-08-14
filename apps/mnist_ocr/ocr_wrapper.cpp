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

#include "tiny_brain/tinydigit.h"

#include <emscripten/bind.h>
using namespace emscripten;

class ocr_wrapper::ocr_wrapper_impl
{
public:
    ocr_wrapper_impl() : m_digit_ocr( tinydigit_base::model::kaggle ) {}
    void process( const tinymage<float>& img )
    {
        m_digit_ocr.process( img );
    }
    const tinymage<float>& cropped_numbers()
    {
        return m_digit_ocr.cropped_numbers();
    }
    std::string reco_string()
    {
        return m_digit_ocr.reco_string();
    }
private:
    tinydigit<0,2,2> m_digit_ocr;
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
    auto sx = image["sizeX"].as<int>();
    auto sy = image["sizeY"].as<int>();

    std::cout << "ocr_wrapper::process - " << sx << "x" << sy << " bytes @" << reinterpret_cast<int>( ptr ) << std::endl;

    tinymage<float> img( sx, sy );

    // image pointer is RGBA formatted
    size_t i=0;
    img.apply( [&]( float& val ) {
        val =  ptr[i] * 0.2126f + ptr[i+1] * 0.7152f + ptr[i+2] * 0.0722f;
        i+=4;
    });

    //img.load( "./ocr/images/123456.png" );
    m_pimpl->process( img );

    onComplete();
}

std::vector<size_t> ocr_wrapper::cropped_size()
{
    auto& cropped = m_pimpl->cropped_numbers();
    std::cout << "ocr_wrapper::cropped_size - " << cropped.width() << "x" << cropped.height() << std::endl;
    return { { cropped.width(), cropped.height() } };
}

std::vector<uint8_t> ocr_wrapper::cropped_numbers()
{
    auto output = m_pimpl->cropped_numbers().convert<uint8_t>();
    return std::vector<uint8_t>( output.data(), output.data() + output.size() );
}

std::string ocr_wrapper::reco_string()
{
    return m_pimpl->reco_string();
}
