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

#include <array>
#include <utility>

class tinyutils
{
public:

    template<size_t A>
    static constexpr std::array<float, 2*A+1> make_symetric_sequence() noexcept
    {
        return sequence_build( sequence_add<-static_cast<int>(A)>( std::make_integer_sequence<int,2*A+1>{} ) );
    }

private:

    template<typename T, T... I>
    inline static constexpr std::array< float, sizeof...(I) >  sequence_build( std::integer_sequence<T, I...> ) noexcept
    {
       return std::array<float, sizeof...(I) > {{ I... }};
    }

    template<int N, typename T, T... I>
    inline static constexpr std::integer_sequence<T, N+I...>
    sequence_add( std::integer_sequence<T, I...> ) noexcept
    {
        return {};
    }
};
