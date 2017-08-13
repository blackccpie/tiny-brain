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

#ifdef __EMSCRIPTEN__
	#include <emscripten.h>
	#include <emscripten/bind.h>
	#include <emscripten/em_asm.h>
	using namespace emscripten;
#endif

#include "tiny_brain/tinydigit.h"
#include "tiny_brain/tinysign.h"

#include <sstream>
#include <iostream>

#ifdef __EMSCRIPTEN__
class digits_sign_detector
{
public:
    digits_sign_detector( size_t sx, size_t sy ) : m_img( sx, sy ), m_sign_helper( sx, sy )
        { std::cout << "digits_sign_detector::digits_sign_detector - " << sx << "x" << sy << std::endl; }

	/************************ LOCATE ************************/

	void locate( emscripten::val image )
    {
        auto ptr = reinterpret_cast<uint8_t*>( image["ptr"].as<int>() );
        auto sx = image["sizeX"].as<int>();
        auto sy = image["sizeY"].as<int>();

        std::cout << "digits_sign_detector::process - " << sx << "x" << sy << " bytes @" << reinterpret_cast<int>( ptr ) << std::endl;

        // image pointer is RGBA formatted
        size_t i=0;
        m_img.apply( [&]( float& val ) {
            val =  ptr[i] * 0.2126f + ptr[i+1] * 0.7152f + ptr[i+2] * 0.0722f;
            i+=4;
        });

		m_sign_helper.locate( m_img );
    }
    std::vector<size_t> get_sign_bounds()
    {
		const auto& sign_bounds = m_sign_helper.get_sign_bounds();
        auto w = sign_bounds[2] - sign_bounds[0];
        auto h = sign_bounds[3] - sign_bounds[1];
        return { sign_bounds[0], sign_bounds[1], w, h };
    }
    std::vector<uint8_t> get_sign_thresh()
    {
        auto thresh_sign = m_sign_helper.get_sign_thresh().convert<uint8_t>();
        return { thresh_sign.data(), thresh_sign.data() + thresh_sign.size() };
    }

	/************************ EXTRACT ************************/

	void extract()
    {
		const auto& sign_bounds = m_sign_helper.get_sign_bounds();
		m_sign_helper.extract( m_img, sign_bounds );
	}
	std::vector<size_t> get_sign_warp_size()
    {
		const auto& warp_sign = m_sign_helper.get_sign_warp();
		return { warp_sign.width(), warp_sign.height() };
	}
	std::vector<uint8_t> get_sign_warp()
    {
		auto warp_sign = m_sign_helper.get_sign_warp().convert<uint8_t>();
        return { warp_sign.data(), warp_sign.data() + warp_sign.size() };
	}

	/************************ RECOGNIZE ************************/

	void recognize()
    {
		const auto& warp_sign = m_sign_helper.get_sign_warp();
		m_digit_ocr_helper.process( warp_sign );
	}
	std::vector<size_t> get_crop_size()
	{
		auto& cropped = m_digit_ocr_helper.cropped_numbers();
	    return { { cropped.width(), cropped.height() } };
	}
    std::vector<uint8_t> get_crop()
	{
		auto output = m_digit_ocr_helper.cropped_numbers().convert<uint8_t>();
		return std::vector<uint8_t>( output.data(), output.data() + output.size() );
	}
	std::string reco_string()
	{
		return m_digit_ocr_helper.reco_string();
	}

private:

	tinymage<float> m_img;

    tinysign m_sign_helper;
	tinydigit<4,0,0> m_digit_ocr_helper;
};

// Binding code
EMSCRIPTEN_BINDINGS(digits_sign_detector)
{
    register_vector<size_t>("VectorSizeT");
    register_vector<uint8_t>("VectorUInt8");

    class_<digits_sign_detector>( "digits_sign_detector" )
        .constructor<size_t,size_t>()
        .function( "locate", &digits_sign_detector::locate )
        .function( "get_sign_thresh", &digits_sign_detector::get_sign_thresh )
        .function( "get_sign_bounds", &digits_sign_detector::get_sign_bounds )
		.function( "extract", &digits_sign_detector::extract )
		.function( "get_sign_warp", &digits_sign_detector::get_sign_warp )
		.function( "get_sign_warp_size", &digits_sign_detector::get_sign_warp_size )
		.function( "recognize", &digits_sign_detector::recognize )
		.function( "get_crop", &digits_sign_detector::get_crop )
		.function( "get_crop_size", &digits_sign_detector::get_crop_size )
		.function( "reco_string", &digits_sign_detector::reco_string );
}

int main( int argc, char **argv )
{
    EM_ASM( allReady() );
}
#else

static const unsigned char green[] = { 0,255,0 };

int main( int argc, char **argv )
{
    tinymage<float> img;
    img.load( "../../data/ocr/images/3167-sign.png" );

	tinysign m_sign_helper( img.width(), img.height() );
	m_sign_helper.locate( img );

	auto sign_thresh = m_sign_helper.get_sign_thresh();
	auto sign_bounds = m_sign_helper.get_sign_bounds();

#ifdef USE_CIMG
	// no smarter way to copy from grayscale to color image :-(
    cimg_library::CImg<float> cimg_out( static_cast<int>( img.width() ), static_cast<int>( img.height() ), 1, 3 );
    cimg_forXYC(cimg_out,x,y,c) { cimg_out(x,y,c) = sign_thresh.c_at(x,y) > 0 ? 85.f : 0.f; }
	cimg_out.draw_rectangle( sign_bounds[0], sign_bounds[1], sign_bounds[2], sign_bounds[3], green, .25f );
    cimg_out.display();
#endif

	m_sign_helper.extract( img, sign_bounds );

	const auto& warped = m_sign_helper.get_sign_warp();

	tinydigit<4,0,0> digit_ocr_helper( tinydigit_base::model::caffe );
	digit_ocr_helper.process( warped );

    auto& cropped_numbers = digit_ocr_helper.cropped_numbers();
    cropped_numbers.display();

    std::cout << "INFERRED DIGITS ARE : " << digit_ocr_helper.reco_string() << std::endl;

	return 0;
}
#endif
