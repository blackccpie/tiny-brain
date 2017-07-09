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

#pragma once

#include "tiny_brain/tinymage.h"

#include <map>
#include <set>

// 'sign' localization helper class
// NOTE : what I call 'sign' is meant to be a white rectangular paper sheet with digits written on it
class tinysign
{
public:
    tinysign( size_t sx, size_t sy ) : m_input( sx, sy ) {}

    void locate( const tinymage<float>& img_in )
    {
		m_input = img_in.get_auto_threshold();
        m_input.display();

		m_bounds = _blob_detect( m_input );
        m_filtered_bounds.clear();

		for ( const auto& _bounds : m_bounds )
		{
    		auto w = _bounds.second[2]-_bounds.second[0];
    		auto h = _bounds.second[3]-_bounds.second[1];
    		auto aspect_ratio = static_cast<float>(w)/h;
    		auto fill_ratio = static_cast<float>(_bounds.second[4])/(w*h);

    		if ( _bounds.second[4] < 2500 || aspect_ratio < 1.25f || fill_ratio < 0.5f )
        		continue;

    		//auto cropped = img.get_crop(_blob.second[0],_blob.second[1],_blob.second[2],_blob.second[3]);

            m_filtered_bounds.emplace_back( std::vector<size_t>{ _bounds.second[0], _bounds.second[1], _bounds.second[2], _bounds.second[3], _bounds.second[4] } );
		}

        for ( const auto& _bounds : m_filtered_bounds )
        {
            std::stringstream ss;
            for ( const auto& _coord : _bounds )
            {
                ss << _coord << " ";
            }
            std::cout << "-> blob! " << " (" << ss.str() << ")" << std::endl;
        }
    }
    const std::vector<size_t>& get_sign_bounds()
    {
        return m_filtered_bounds.front();
    }
    const tinymage<float>& get_sign_thresh()
    {
        return m_input; // TODO:  really usefull to keep thresholded image?
    }

    void extract( const tinymage<float>& img_in, const std::vector<size_t>& sign_bounds )
    {
        tinymage<float> cropped = img_in.get_crop( sign_bounds[0], sign_bounds[1], sign_bounds[2], sign_bounds[3] );

    	auto thresh_cropped = cropped.get_auto_threshold();

    	auto line_sums = thresh_cropped.line_sums();
    	auto row_sums = thresh_cropped.row_sums();

    	//line_sums.display();
    	//row_sums.display();

    	auto dline_sums = line_sums.get_dcolumn();
    	auto drow_dums = row_sums.get_dline();

    	//dline_sums.display();
    	//drow_dums.display();

    	int i0,i1,j0,j1;
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

    	//std::cout << i0 << " " << i1 << " " << j0 << " " << j1 << std::endl;

    	thresh_cropped.display();

    	auto w = cropped.width()-1;
    	auto h = cropped.height()-1;

    	//tinymage_types::quad_coord_t incoord{ { 21,0 },{ 531,19 },{ 523,256 },{ 0,235 } };
    	tinymage_types::quad_coord_t incoord{ {i0,0U}, {w,j0}, {i1,h}, {0U,j1} };
    	tinymage_types::quad_coord_t outcoord{ {0U,0U}, {w,0U}, {w,h}, {0U,h} };
    	m_warped = cropped.get_warp( incoord, outcoord );
    	m_warped.remove_border( 2 );
    	m_warped.display();
    }
    const tinymage<float>& get_sign_warp()
    {
        return m_warped;
    }

private:

    tinymage<float> m_input;
    tinymage<float> m_warped;

    using bounds_t = std::map<size_t,std::vector<size_t>>;
    bounds_t m_bounds;
    using filt_bounds_t = std::vector<std::vector<size_t>>;
    filt_bounds_t m_filtered_bounds;

private:

     // Given image dimensions and a raw string of grayscale pixels, detects blobs
     // in the "image" Uses two-pass connected component algorithm described here:
     // http://en.wikipedia.org/wiki/Blob_extraction#Two-pass (Jan 2011).
    template<typename T>
    std::map<size_t,std::vector<size_t>> _blob_detect( const tinymage<T> image )
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
};
