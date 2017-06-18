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

#include "tiny_brain/tinymage.h"

#include <sstream>
#include <iostream>
#include <map>
#include <set>

/* blobs.detect()
 *
 *  Given image dimensions and a raw string of grayscale pixels, detects blobs
 *  in the "image" Uses two-pass connected component algorithm described here:
 *  http://en.wikipedia.org/wiki/Blob_extraction#Two-pass (Jan 2011).
 */
template<typename T>
std::map<size_t,std::vector<size_t>> blob_detect( const tinymage<T> image )
{
    std::map<size_t,std::set<size_t>> groups;

    /*
    * Pass one: provisionally label each non-background cell with a label.
    */

    // an array to hold the labels
    std::vector<size_t> labels( image.size() );
    std::set<size_t> groupset;

    auto w = image.width();
    auto h = image.height();

    size_t blobs = 0;

    tinymage_forXY( image, x, y )
    {
        // offset in the image for a given (x, y) pixel
        auto off = ( y * w ) + x;

        if ( image.c_at(x,y) < 1 ) // TODO : parameters?
        {
            // dark pixel means it's part of the background
            labels[off] = 0;
        }
        else
        {
            // light pixel means it's part of a blob

            size_t label;

            if ( y > 0 && labels[off - w] > 0 && x > 0 && labels[off - 1] > 0 )
            {
                // pixels up and left are both known blobs
                label = labels[off - w];

                if( label != labels[off - 1] )
                {
                    // associate the two labels
                    groups[label].insert( labels[off - 1] );
                    groups[labels[off - 1]].insert( label );

                    // unify the sets - make sure they all have the same items

                    groupset = groups[label];

                    for( auto groupiter = groupset.begin(); groupiter != groupset.end(); groupiter++ )
                        groups[labels[off - 1]].insert( *groupiter );

                    groupset = groups[labels[off - 1]];

                    for( auto groupiter = groupset.begin(); groupiter != groupset.end(); groupiter++ )
                        groups[label].insert( *groupiter );
                }
            }
            else if ( y > 0 && labels[off - w] > 0 )
            {
                // pixel one row up is a known blob
                label = labels[off - w];
            }
            else if ( x > 0 && labels[off - 1] > 0 )
            {
                // pixel to the left is a known blob
                label = labels[off - 1];
            }
            else
            {
                // a new blob!
                blobs++;
                label = blobs;

                groups[label].insert( label );
            }

            labels[off] = label;
        }
    }

   /*
    * Pass two: merge labels of connected components, collect bboxes along the way.
    */

    std::map<size_t,std::vector<size_t>> bounds;

    for ( auto y = size_t(0); y < h; y++ )
    {
        for ( auto x = size_t(0); x < w; x++ )
        {
            // offset in the string for a given (x, y) pixel
            auto off = ( y * w ) + x;

            if ( labels[off] > 0 )
            {
                size_t label = *(groups[labels[off]].begin());

                if ( bounds.find( label ) == bounds.end() )
                    bounds[label] = { x, y, x, y, 1 };
                else
                {
                    auto& _bounds = bounds[label];

                    _bounds[0] = std::min( x, _bounds[0] );
                    _bounds[1] = std::min( y, _bounds[1] );
                    _bounds[2] = std::max( x, _bounds[2] );
                    _bounds[3] = std::max( y, _bounds[3] );
                    _bounds[4] += 1;
                }
            }
        }
    }

   return bounds;
}

#ifdef __EMSCRIPTEN__
class blob_detector
{
public:
    blob_detector( size_t sx, size_t sy ) : m_img( sx, sy ), m_output( sx, sy )
        { std::cout << "blob_detector::blob_detector - " << sx << "x" << sy << std::endl; }
    void process( emscripten::val image )
    {
        auto ptr = reinterpret_cast<uint8_t*>( image["ptr"].as<int>() );
        auto sx = image["sizeX"].as<int>();
        auto sy = image["sizeY"].as<int>();

        std::cout << "blob_detector::process - " << sx << "x" << sy << " bytes @" << reinterpret_cast<int>( ptr ) << std::endl;

        // image pointer is RGBA formatted
        size_t i=0;
        m_img.apply( [&]( float& val ) {
            val =  ptr[i] * 0.2126f + ptr[i+1] * 0.7152f + ptr[i+2] * 0.0722f;
            i+=4;
        });

		m_img.auto_threshold();

		m_blobs = blob_detect( m_img );
        m_filtered_blobs.clear();

		for ( const auto& _blob : m_blobs )
		{
    		auto w = _blob.second[2]-_blob.second[0];
    		auto h = _blob.second[3]-_blob.second[1];
    		auto aspect_ratio = static_cast<float>(w)/h;
    		auto fill_ratio = static_cast<float>(_blob.second[4])/(w*h);

    		if ( _blob.second[4] < 2500 || aspect_ratio < 1.25f || fill_ratio < 0.5f )
        		continue;

    		//auto cropped = img.get_crop(_blob.second[0],_blob.second[1],_blob.second[2],_blob.second[3]);

            m_filtered_blobs.emplace_back( std::vector<size_t>{ _blob.second[0], _blob.second[1], _blob.second[2], _blob.second[3], _blob.second[4] } );
		}

        for ( const auto& _blob : m_filtered_blobs )
        {
            std::stringstream ss;
            for ( const auto& _coord : _blob )
            {
                ss << _coord << " ";
            }
            std::cout << "-> blob! " << " (" << ss.str() << ")" << std::endl;
        }
    }
    std::vector<size_t> get_blob()
    {
        auto& _blob = m_filtered_blobs.front();
        auto w = _blob[2]-_blob[0];
        auto h = _blob[3]-_blob[1];
        return { _blob[0], _blob[1], w, h };
    }
    std::vector<uint8_t> get_thresh()
    {
        m_output = m_img.convert<uint8_t>();
        return { m_output.data(), m_output.data() + m_output.size() };
    }

private:

    tinymage<float> m_img;
    tinymage<uint8_t> m_output;

    using blobs_t = std::map<size_t,std::vector<size_t>>;
    blobs_t m_blobs;
    using filt_blobs_t = std::vector<std::vector<size_t>>;
    filt_blobs_t m_filtered_blobs;
};

// Binding code
EMSCRIPTEN_BINDINGS(blob)
{
    register_vector<size_t>("VectorSizeT");
    register_vector<uint8_t>("VectorUInt8");

    class_<blob_detector>( "blob_detector" )
        .constructor<size_t,size_t>()
        .function( "process", &blob_detector::process )
        .function( "get_thresh", &blob_detector::get_thresh )
        .function( "get_blob", &blob_detector::get_blob );
}

int main( int argc, char **argv )
{
    EM_ASM( allReady() );
}
#else
int main( int argc, char **argv )
{
    tinymage<float> img;
    img.load( "../sandbox/ocr_ex.png" );
    img.auto_threshold();
    img.display();

    auto bounds = blob_detect( img );
    std::vector<std::vector<size_t>> filtered_bounds;

    for ( const auto& _bounds : bounds )
    {
        auto w = _bounds.second[2]-_bounds.second[0];
        auto h = _bounds.second[3]-_bounds.second[1];
        auto aspect_ratio = static_cast<float>(w)/h;
        auto fill_ratio = static_cast<float>(_bounds.second[4])/(w*h);

        if ( _bounds.second[4] < 2500 || aspect_ratio < 1.25f || fill_ratio < 0.5f )
            continue;

        //auto cropped = img.get_crop(_bounds.second[0],_bounds.second[1],_bounds.second[2],_bounds.second[3]);

        filtered_bounds.emplace_back( std::vector<size_t>{ _bounds.second[0], _bounds.second[1], _bounds.second[2], _bounds.second[3], _bounds.second[4] } );
    }

    for ( const auto& _bounds : filtered_bounds )
    {
        std::stringstream ss;
        for ( const auto& _coord : _bounds )
        {
            ss << _coord << " ";
        }
        std::cout << "-> blob! " << " (" << ss.str() << ")" << std::endl;
    }

    return 0;
}
#endif
