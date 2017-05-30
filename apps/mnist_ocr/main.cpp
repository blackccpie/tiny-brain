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

#include "ocr.h"

#include <iostream>

int main( int argc, char **argv )
{
#ifdef __EMSCRIPTEN__
    tinymage<float> img;
#else
    tinymage<float> img;
    img.load( "../data/ocr/123456.png" );
    img.display();
#endif

    using namespace tiny_dnn;
    network<sequential> nn;
    nn.load( "kaggle-mnist-model" );

    ocr_helper ocr_h( nn );
    ocr_h.process( img );

    //auto& cropped_numbers = ocr_h.cropped_numbers();
    //cropped_numbers.display();

    std::cout << "INFERRED DIGITS ARE : " << ocr_h.reco_string() << std::endl;

    return 0;
}
