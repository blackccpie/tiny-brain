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

#include <unordered_map>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

    struct adamax : public stateful_optimizer<2> {
      adamax()
        : alpha(float_t(0.002f)),
          b1(float_t(0.9f)),
          b2(float_t(0.999f)),
          b1_t(b1),
          eps(float_t(1e-8f)) {}

      void update(const vec_t &dW, vec_t &W, bool parallelize) {
        vec_t &mt = get<0>(W);
        vec_t &vt = get<1>(W);

        b1_t *= b1;

        for_i(parallelize, static_cast<int>(W.size()), [&](int i) {
          mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
          vt[i] = std::max( b2 * vt[i], std::abs( dW[i] ) );

          W[i] -= alpha * ( mt[i] / (float_t(1) - b1_t)) /
                  ( vt[i] + eps);
        });
      }

      float_t alpha;  // learning rate
      float_t b1;     // decay term
      float_t b2;     // decay term
      float_t b1_t;   // decay term power t

     private:
      float_t eps;  // constant value to avoid zero-division
    };

} // namespace tiny_dnn
