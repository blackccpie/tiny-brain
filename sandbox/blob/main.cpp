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

#include "tiny_brain/tinymage.h"

#include "tiny_dnn/tiny_dnn.h"

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
                    bounds[label][0] = std::min( x, bounds[label][0] );
                    bounds[label][1] = std::min( y, bounds[label][1] );
                    bounds[label][2] = std::max( x, bounds[label][2] );
                    bounds[label][3] = std::max( y, bounds[label][3] );
                    bounds[label][4] += 1;
                }
            }
        }
    }

   return bounds;
}

int main( int argc, char **argv )
{
    tinymage<float> img;
    img.load( "../sandbox/ocr_ex.png" );
    img.auto_threshold();
    img.display();

    auto bounds = blob_detect( img );

    for ( const auto& _bounds : bounds )
    {
        if ( _bounds.second[4] < 100 )
            continue;

        std::stringstream ss;
        for ( const auto& _coord : _bounds.second )
        {
            ss << _coord << " ";
        }
        std::cout << "label " << _bounds.first << " (" << ss.str() << ")" << std::endl;
    }

    return 0;
}
