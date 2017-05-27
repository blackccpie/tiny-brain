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

#include "tiny_dnn/tiny_dnn.h"

template <typename T>
class tinymage;

class ocr_helper
{
public:

    struct reco
    {
        size_t position;
        size_t value;
        float confidence;
    };

public:
    ocr_helper( tiny_dnn::network<tiny_dnn::sequential>& net_manager );
    virtual ~ocr_helper(); // Implement (with an empty body) where pimpl is complete

    void process( const tinymage<float>& img );

    const tinymage<float>& cropped_numbers();

    const std::vector<reco>& recognitions();
    std::string reco_string();

private:

    class ocr_helper_impl;
    std::unique_ptr<ocr_helper_impl> m_pimpl;
};
