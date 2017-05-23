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

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

inline void csv_parse_mnist( const std::string& csv_file, std::vector<label_t>& labels, std::vector<vec_t>& images )
{
    auto sample_sizeX{0}, sample_sizeY{0};

    std::ifstream data_in( csv_file );

    if ( data_in.bad() || data_in.fail() )
      throw nn_error( "failed to open file:" + csv_file );

    std::string line;

    // get first ignored line : labels etc...
    std::getline( data_in, line );

    while ( std::getline( data_in, line ) )
    {
        std::stringstream ss{ line };

        // Read the target values from the line:
        std::vector<std::uint8_t> _vals;
        while ( !ss.eof() )
        {
            std::uint16_t val;
            if ( !(ss >> val).fail() )
                _vals.push_back( static_cast<std::uint8_t>( val ) );
        }

        int digit = _vals.at(0);
        size_t input_size = _vals.size()-1;
        size_t square_size = std::sqrt( input_size );

        labels.emplace_back( digit );
        images.emplace_back( input_size, 0 );

        auto& dst = images.back();

        for (size_t y = 0; y < square_size; y++)
          for (size_t x = 0; x < square_size; x++)
            dst[square_size * y + x] = _vals[y * square_size + x + 1] / float_t(255);
    }

    std::cout << "csv_parse_mnist - successfully loaded " << labels.size() << " samples" << std::endl;
}

}
