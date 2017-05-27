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
#include <cassert>
#include <vector>

#ifdef USE_CIMG
    #include <CImg.h>
#endif

#define tinymage_for1(bound,i) for (std::size_t i = 0UL; i<bound; ++i)
#define tinymage_forX(img,x) tinymage_for1( img.width(), x )
#define tinymage_forY(img,y) tinymage_for1( img.height(), y )

template<typename T=float>
class tinymage final : private std::vector<T>
{
    using std::vector<T>::at;
    using std::vector<T>::assign;
    using std::vector<T>::data;

public:
    using std::vector<T>::begin;
    using std::vector<T>::end;

public:
    tinymage() : m_width{0}, m_height{0} {}
    tinymage( std::size_t sx, std::size_t sy, T val = 0 ) : std::vector<T>( sx*sy, val ), m_width{sx}, m_height{sy} {}
    tinymage( uint8_t* buf, std::size_t sx, std::size_t sy, std::size_t bpp ) : m_width{sx}, m_height{sy}
    {
        assert( bpp == sizeof(T) );
        assign( buf, buf + sx*sy );
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
        tinymage<R> output( m_width, m_height, 0.f );
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
        double cur_dyn = cur_max - cur_min;
        double out_dyn = max - min;

        std::for_each( begin(), end(), [&]( T& val )
            {
                val = min + ( out_dyn * ( val - cur_min ) / cur_dyn );
            });
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
                val = std::max( val, thresh );
            });
    }

    tinymage<T> get_crop(   std::size_t startx,
                            std::size_t starty,
                            std::size_t stopx,
                            std::size_t stopy ) const
    {
        assert( stopx >= startx );
        assert( stopy >= starty );

        tinymage<T> output( stopx-startx, stopy-starty );
        for ( auto yin = starty,yout = 0UL; yin<stopy; yin++, yout++ )
            for ( auto xin = startx,xout = 0UL; xin<stopx; xin++, xout++ )
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

    tinymage<T> line_sums() const
    {
        tinymage<T> output( 1, m_height, 0.f );

        std::size_t line_index{0},line_index_out{0};

        std::for_each( begin(), end(), [&]( const T& val )
            {
                output.at(0,line_index_out) += val;
                if ( ++line_index == output.width() )
                    line_index_out++;
            });

        return output;
    }

    tinymage<T> row_sums() const
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

    void display()
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
};
