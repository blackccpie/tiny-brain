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
    void process( emscripten::val image )
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

		m_sign_helper.process( m_img );
    }
    std::vector<size_t> get_sign_bounds()
    {
		const auto& _bounds = m_sign_helper.get_sign_bounds();
        auto w = _bounds[2]-_bounds[0];
        auto h = _bounds[3]-_bounds[1];
        return { _bounds[0], _bounds[1], w, h };
    }
    std::vector<uint8_t> get_sign_thresh()
    {
        auto thresh_sign = m_sign_helper.get_sign_thresh().convert<uint8_t>();
        return { thresh_sign.data(), thresh_sign.data() + thresh_sign.size() };
    }

private:

	tinymage<float> m_img;

    tinysign m_sign_helper;
};

// Binding code
EMSCRIPTEN_BINDINGS(digits_sign_detector)
{
    register_vector<size_t>("VectorSizeT");
    register_vector<uint8_t>("VectorUInt8");

    class_<digits_sign_detector>( "digits_sign_detector" )
        .constructor<size_t,size_t>()
        .function( "process", &digits_sign_detector::process )
        .function( "get_sign_thresh", &digits_sign_detector::get_sign_thresh )
        .function( "get_sign_bounds", &digits_sign_detector::get_sign_bounds );
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
    img.load( "../sandbox/ocr_ex.png" );

	tinysign m_sign_helper( img.width(), img.height() );
	m_sign_helper.process( img );

	auto sign_thresh = m_sign_helper.get_sign_thresh();
	auto sign_bounds = m_sign_helper.get_sign_bounds();

	// no smarter way to copy from grayscale to color image :-(
    cimg_library::CImg<float> cimg_out( static_cast<int>( img.width() ), static_cast<int>( img.height() ), 1, 3 );
    cimg_forXYC(cimg_out,x,y,c) { cimg_out(x,y,c) = sign_thresh.c_at(x,y) > 0 ? 85.f : 0.f; }
	cimg_out.draw_rectangle( sign_bounds[0], sign_bounds[1], sign_bounds[2], sign_bounds[3], green, .25f );
    cimg_out.display();

	tinymage<float> cropped = img.get_crop( sign_bounds[0], sign_bounds[1], sign_bounds[2], sign_bounds[3] );

	auto thresh_cropped = cropped.get_auto_threshold();

	auto line_sums = thresh_cropped.line_sums();
	auto row_sums = thresh_cropped.row_sums();

	//line_sums.display();
	//row_sums.display();

	auto dline_sums = line_sums.get_dcolumn();
	auto drow_dums = row_sums.get_dline();

	//dline_sums.display();
	//drow_dums.display();

	size_t i0,i1,j0,j1;
	for ( i0=1; i0<drow_dums.width(); i0++ )
		if ( drow_dums[i0] == 0 )
			break;
	for ( i1=drow_dums.width()-2; i1>=0; i1-- )
		if ( drow_dums[i1] == 0 )
			break;
	for ( j0=1; j0<dline_sums.height(); j0++ )
		if ( dline_sums[j0] == 0 )
			break;
	for ( j1=dline_sums.height()-2; j1>=0; j1-- )
		if ( dline_sums[j1] == 0 )
			break;

	std::cout << i0 << " " << i1 << " " << j0 << " " << j1 << std::endl;

	thresh_cropped.display();

	auto w = cropped.width()-1;
	auto h = cropped.height()-1;

	//tinymage_types::quad_coord_t incoord{ { 21,0 },{ 531,19 },{ 523,256 },{ 0,235 } };
	tinymage_types::quad_coord_t incoord{ {i0,0U}, {w,j0}, {i1,h}, {0U,j1} };
	tinymage_types::quad_coord_t outcoord{ {0U,0U}, {w,0U}, {w,h}, {0U,h} };
	auto warped = cropped.get_warp( incoord, outcoord );
	warped.remove_border( 2 );
	warped.display();

	tinydigit digit_ocr;
	digit_ocr.process( warped );

    auto& cropped_numbers = digit_ocr.cropped_numbers();
    cropped_numbers.display();

    std::cout << "INFERRED DIGITS ARE : " << digit_ocr.reco_string() << std::endl;

	return 0;
}
#endif
