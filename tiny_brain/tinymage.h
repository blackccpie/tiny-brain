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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#ifdef USE_CIMG
    #include <CImg.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_INLINE
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_INLINE
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_INLINE
#include "stb/stb_image_resize.h"

#define tinymage_for1(bound,i) for (std::size_t i = 0UL; i<bound; ++i)
#define tinymage_forX(img,x) tinymage_for1( img.width(), x )
#define tinymage_forY(img,y) tinymage_for1( img.height(), y )
#define tinymage_forXY(img,x,y) tinymage_forY(img,y) tinymage_forX(img,x)

template<typename T=float>
class tinymage final : private std::vector<T>
{
    // any other type of tinymage is a friend.
    template <typename U>
    friend class tinymage;

	// TODO : factorize
    template<typename U>
    using tinymage_if_uchar = std::enable_if_t<std::is_same<unsigned char, U>::value, tinymage<U>>;
    template<typename U>
    using tinymage_if_float = std::enable_if_t<std::is_same<float, U>::value, tinymage<U>>;

    using std::vector<T>::at;
    using std::vector<T>::assign;
    using std::vector<T>::size;

    using std::vector<T>::begin;
    using std::vector<T>::end;

public:

    using std::vector<T>::data;

    tinymage() : m_width{0}, m_height{0} {}
    tinymage( std::size_t sx, std::size_t sy, T val = 0 ) : std::vector<T>( sx*sy, val ), m_width{sx}, m_height{sy} {}
    tinymage( uint8_t* buf, std::size_t sx, std::size_t sy, std::size_t bpp ) : m_width{sx}, m_height{sy}
    {
        assert( bpp == sizeof(T) );
        assign( buf, buf + sx*sy );
    }

    std::size_t size() const { return m_width * m_height; }

    bool load( const std::string& img_path )
    {
        int width, height, bpp;
        auto gray_image = stbi_load( img_path.c_str(), &width, &height, &bpp, 1); // force grayscale at image load
        if ( gray_image == nullptr )
            return false;
        assert( bpp == sizeof(T) );
        m_width = static_cast<std::size_t>( width );
        m_height = static_cast<std::size_t>( height );
        assign( gray_image, gray_image + m_width*m_height );
        stbi_image_free( gray_image );
        return true;
    }

    bool save_png( const std::string& img_path )
    {
        return stbi_write_png( img_path.c_str(), m_width, m_height, 1, data(), m_width*sizeof(T));
    }

    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }

    T& operator[]( std::size_t i )
    {
        return at( i );
    }

    T& at( std::size_t x, std::size_t y )
    {
        return at( x + m_width*y );
    }

    const T& c_at( std::size_t x, std::size_t y ) const
    {
        return at( x + m_width*y );
    }

    template<typename R>
    tinymage<R> convert() const
    {
        tinymage<R> output( m_width, m_height, 0 );
        auto output_iter = output.begin();

        std::for_each( begin(), end(), [&]( const T& val )
            {
                *output_iter = static_cast<R>( val );
                output_iter++;
            });

        return output;
    }

    void normalize( T min, T max )
    {
        assert( max > min );

        auto cur_min = *std::min_element( begin(), end() );
        auto cur_max = *std::max_element( begin(), end() );

        assert( cur_max > cur_min );

        double cur_dyn = cur_max - cur_min;
        double out_dyn = max - min;

        std::for_each( begin(), end(), [&]( T& val )
            {
                val = min + ( out_dyn * ( val - cur_min ) / cur_dyn );
            });
    }

    // TODO : factorize with normalize
    tinymage<T> get_normalize( T min, T max )
    {
        assert( max > min );

        auto cur_min = *std::min_element( begin(), end() );
        auto cur_max = *std::max_element( begin(), end() );

        assert( cur_max > cur_min );

        double cur_dyn = cur_max - cur_min;
        double out_dyn = max - min;

        tinymage<T> output( *this );

        std::for_each( output.begin(), output.end(), [&]( T& val )
            {
                val = min + ( out_dyn * ( val - cur_min ) / cur_dyn );
            });

        return output;
    }

    T mean() const
    {
        double sum;

        std::for_each( begin(), end(), [&]( const T& val )
            {
                sum += val;
            });

        return static_cast<T>( sum / ( m_width * m_height ) );
    }

    void threshold( T thresh )
    {
        std::for_each( begin(), end(), [&]( T& val )
            {
                val = val > thresh ? m_one : m_zero;
            });
    }

    tinymage<T> get_crop(   std::size_t startx,
                            std::size_t starty,
                            std::size_t stopx,
                            std::size_t stopy ) const
    {
        assert( stopx >= startx );
        assert( stopy >= starty );

        assert( stopx-startx <= m_width );
        assert( stopy-starty <= m_height );

        tinymage<T> output( stopx-startx, stopy-starty );

        for ( auto yin = starty,yout = std::size_t(0); yin<stopy; yin++, yout++ )
            for ( auto xin = startx,xout = std::size_t(0); xin<stopx; xin++, xout++ )
                output.at(xout,yout) = c_at(xin,yin);

        return output;
    }

    void crop(  std::size_t startx,
                std::size_t starty,
                std::size_t stopx,
                std::size_t stopy )
    {
        *this = get_crop( startx, starty, stopx, stopy );
    }

    tinymage<T> get_columns(    std::size_t startx,
                                std::size_t stopx ) const
    {
        return get_crop( startx, 0, stopx, m_height );
    }

    tinymage<T> get_lines(  std::size_t starty,
                            std::size_t stopy ) const
    {
        return get_crop( 0, starty, m_width, stopy );
    }

    void canvas_resize( std::size_t nsx, std::size_t nsy, float centering_x = 0.5f, float centering_y = 0.5f )
    {
        *this = get_canvas_resize( nsx, nsy, centering_x, centering_y );
    }

    tinymage<T> get_canvas_resize( std::size_t nsx, std::size_t nsy, float centering_x = 0.5f, float centering_y = 0.5f  )
    {
        // Only default dirichlet condition is managed for now

        assert( nsx > m_width && nsy > m_height );
        assert( centering_x <= 1.f && centering_x >= 0.f );
        assert( centering_y <= 1.f && centering_y >= 0.f );

		tinymage<T> output( nsx, nsy, 0 );

        const int32_t   xc{ static_cast<int32_t>( centering_x * ( nsx - m_width ) )},
                        yc{ static_cast<int32_t>( centering_y * ( nsy - m_height ) )};

        tinymage_forXY( (*this), x, y )
        {
            output.at( x + xc, y + yc ) = c_at( x, y );
        }

        return output;
    }

    void resize( std::size_t nsx, std::size_t nsy )
    {
        *this = get_resize( nsx, nsy );
    }

    template<typename U = T>
    tinymage_if_uchar<U> get_resize( std::size_t nsx, std::size_t nsy )
    {
        tinymage<T> output( nsx, nsy );
        stbir_resize_uint8( data(), m_width, m_height, 0, output.data(), nsx, nsy, 0, 1 );
        return output;
    }

    template<typename U = T>
    tinymage_if_float<U> get_resize( std::size_t nsx, std::size_t nsy )
    {
        tinymage<T> output( nsx, nsy );
        stbir_resize_float( data(), m_width, m_height, 0, output.data(), nsx, nsy, 0, 1 );
        return output;
    }

    template<typename U = T>
    tinymage_if_float<U> line_sums() const
    {
        tinymage<T> output( 1, m_height, 0.f );

        std::size_t line_index{0},line_index_out{0},width{m_width};

        std::for_each( begin(), end(), [&]( const T& val )
            {
                output.at(0,line_index_out) += val;
                if ( ++line_index == width )
                {
                    line_index = 0;
                    line_index_out++;
                }
            });

        return output;
    }

    template<typename U = T>
    tinymage_if_float<U> row_sums() const
    {
        tinymage<T> output( m_width, 1, 0.f );

        std::size_t row_index{0};

        std::for_each( begin(), end(), [&]( const T& val )
            {
                output.at(row_index,0) += val;
                if ( ++row_index == output.width() )
                    row_index = 0;
            });

        return output;
    }

    template<std::size_t nb_bins>
    std::array<std::size_t,nb_bins> get_histogram()
    {
        auto min = *std::min_element( begin(), end() );
        auto max = *std::max_element( begin(), end() );

        std::array<std::size_t,nb_bins> hist;
        std::for_each( begin(), end(), [&]( const T& val )
            {
                ++hist[val==max?nb_bins-1:static_cast<std::size_t>((val - min)*nb_bins/(max - min))];
            });

        return hist;
    }

    void auto_threshold()
    {
        // One of the many autothreshold IJ implementations:
        // https://imagej.nih.gov/ij/developer/source/ij/process/AutoThresholder.java.html
        int thresh = _default_isodata<256>( get_histogram<256>() );
        threshold( static_cast<T>( thresh ) );
    }

    // returns [0...255] clamped image
    template<typename U = T>
    tinymage_if_uchar<U> get_sobel()
    {
        tinymage<T> output( m_width, m_height );

    	T sum;
        float sumX, sumY;

        // Sobel Matrices Horizontal
        const float GX[3][3] = {    {   1,  0,  -1  },
                                    {   2,  0,  -2  },
                                    {   1,  0,  -1  } };
        // Sobel Matrices Vertical
    	const float GY[3][3] = {    {   1,  2,  1   },
                                    {   0,  0,  0   },
                                    {   -1, -2, -1  } };

    	/*Edge detection */

        tinymage_forXY((*this),x,y)
        {
			sumX	= 0;
			sumY	= 0;

			/*Image Boundaries*/
			if( y == 0 || y == m_height - 1 )
				sum = 0;
			else if( x == 0 || x == m_width - 1 )
				sum = 0;
			else
			{
				/*Convolution for X*/
				for( auto i = -1; i < 2; i++ )
				{
					for( auto j = -1; j < 2; j++ )
					{
						sumX = sumX + GX[j+1][i+1] * c_at(x+j,y+i);
					}
				}

				/*Convolution for Y*/
				for( auto i = -1; i < 2; i++ )
				{
					for( auto j = -1; j < 2; j++ )
					{
						sumY = sumY + GY[j+1][i+1] * c_at(x+j,y+i);
					}
				}

				/*Edge strength*/
				sum = static_cast<T>( std::sqrt( sumX*sumX + sumY*sumY ) );
                // sum = std::abs(sumX) + std::abs(sumY);
			}

            // clamp range to [0,255]
    	    output.at(x,y) = sum % 255;
		}

        return output;
    }

    tinymage<T> get_rotate( float angle, T pad_val = 0 )
    {
        tinymage<T> output( m_width, m_height, pad_val );

        // define the center of the image, which is the center of rotation.
        auto vertical_center = std::floor( m_width / 2 );
        auto horizontal_center = std::floor( m_height / 2 );

        // loop through each pixel of the new image, select the new vertical
        // and horizontal positions, and interpolate the image to make the change.
        tinymage_forXY( output, x, y )
        {
            // figure out how rotated we want the image.
            auto _rad = angle * m_pi / 180.;
            auto vertical_position = std::cos( _rad ) *
                ( x - vertical_center ) + std::sin( _rad ) * ( y - horizontal_center )
                + vertical_center;
            auto horizontal_position = -std::sin(_rad) *
                ( x - vertical_center ) + std::cos( _rad ) * ( y - horizontal_center )
                + horizontal_center;

            // figure out the four locations (and then, four pixels)
            // that we must interpolate from the original image.
            auto top = std::floor( vertical_position );
            auto bottom = top + 1;
            auto left = std::floor( horizontal_position );
            auto right = left + 1;

            // check if any of the four locations are invalid. If they are,
            // skip interpolating this pixel. Otherwise, interpolate the
            // pixel according to the dimensions set above and set the
            // resulting pixel.
            if ( top >= 0 && bottom < m_width && left >= 0 && right < m_height )
            {
                output.at( x, y ) = _bilinear_interpolation(top, bottom,
                    left, right, horizontal_position, vertical_position );
            }
        }

        return output;
    }

    void display() const
    {
#ifdef USE_CIMG
        cimg_library::CImg<T> cimg( data(), m_width, m_height, 1, 1, true/*shared*/ );
        cimg.display();
#else
        // NOT IMPLEMENTED YET
#endif
    }

private:
    std::size_t m_width;
    std::size_t m_height;

    constexpr static T m_zero{ 0 };
    constexpr static T m_one{ 1 };

    constexpr static double m_pi{ std::acos( -1. ) };

private:

    template<std::size_t length>
    int _default_isodata( const std::array<std::size_t,length>& data )
    {
        std::array<std::size_t,length> data2;
        std::size_t mode=0, maxCount=0;
        for ( auto i=std::size_t(0); i<length; i++ ) {
            data2[i] = std::round<std::size_t>( data[i] );
            if ( data2[i]>maxCount ) {
                maxCount = data2[i];
                mode = i;
            }
        }
        std::size_t maxCount2 = 0;
        for ( auto i=std::size_t(0); i<length; i++ ) {
            if ((data2[i]>maxCount2) && (i!=mode))
                maxCount2 = data2[i];
        }
        auto hmax = maxCount;
        if ( (hmax>(maxCount2*2)) && (maxCount2!=0) ) {
            hmax = (int)(maxCount2 * 1.5);
            data2[mode] = hmax;
        }
        return _isodata<length>( data2 );
    }

    template<std::size_t length>
    int _isodata( std::array<std::size_t,length>& data )
    {
        // This is the original ImageJ IsoData implementation, here for backward compatibility.
        int maxValue = length - 1;
        double result, sum1, sum2, sum3, sum4;
        auto count0 = data[0];
        data[0] = 0; //set to zero so erased areas aren't included
        auto countMax = data[maxValue];
        data[maxValue] = 0;
        auto min = 0;
        while ( (data[min]==0) && (min<maxValue) )
            min++;
        auto max = maxValue;
        while ((data[max]==0) && (max>0))
            max--;
        if (min>=max) {
            data[0]= count0; data[maxValue]=countMax;
            return length/2;
        }
        auto movingIndex = min;
        do {
            sum1=sum2=sum3=sum4=0.0;
            for (int i=min; i<=movingIndex; i++) {
                sum1 += (double)i*data[i];
                sum2 += data[i];
            }
            for (int i=(movingIndex+1); i<=max; i++) {
                sum3 += (double)i*data[i];
                sum4 += data[i];
            }
            result = (sum1/sum2 + sum3/sum4)/2.0;
            movingIndex++;
        } while ((movingIndex+1)<=result && movingIndex<max-1);
        data[0]= count0; data[maxValue]=countMax;
        return std::round<int>(result);
    }

    T _bilinear_interpolation(  int top, int bottom, int left, int right,
                                float horizontal_position, float vertical_position )
    {
        // figure out "how far" the output pixel being considered is between *_left and *_right.
        auto horizontal_progress = horizontal_position - left;
        auto vertical_progress = vertical_position - top;

        // combine top_left and top_right into one large, horizontal block.
        auto top_block = c_at( top, left ) + horizontal_progress * ( c_at( top, right ) - c_at( top, left ) );

        // combine bottom_left and bottom_right into one large, horizontal block.
        auto bottom_block = c_at( bottom, left ) + horizontal_progress * ( c_at( bottom, right ) - c_at( bottom, left ) );

        // combine the top_block and bottom_block using vertical interpolation and return as the resulting pixel.
        return static_cast<T>( top_block + vertical_progress * ( bottom_block - top_block ) );
    }

 private:

     template <typename F>
     friend tinymage<F> operator-( const F& val, const tinymage<F>& t );
};

template <typename T>
inline tinymage<T> operator-( const T& val, const tinymage<T>& t )
{
    tinymage<T> output( t );
    std::for_each( output.begin(), output.end(), [&]( T& tval )
        {
            tval = val - tval;
        });
    return output;
}
