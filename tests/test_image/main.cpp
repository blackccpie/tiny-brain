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

#include <iostream>

int main( int argc, char **argv )
{
    tinymage<float> img;
    img.load( "../data/ocr/123456.png" );
    img.display();

    tinymage<unsigned char> work = img.convert<unsigned char>().get_sobel();
    work.display();

    work.auto_threshold();
    work.display();

    tinymage<unsigned char> crop = work.get_crop( 88, 185, 380, 275 );
    crop.display();

    tinymage<float> row_sums( work.convert<float>().row_sums() );
    row_sums.display();

    tinymage<float> line_sums( work.convert<float>().line_sums() );
    line_sums.display();

    work.resize( 100, 100 );
    work.display();

    work.save_png( "test.png" );

    return 0;
}
