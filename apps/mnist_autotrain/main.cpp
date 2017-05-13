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

#include "tiny_dnn/tiny_dnn.h"

#include <iostream>

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

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


static void construct_net(  network<sequential> &nn, core::backend_t backend_type )
{
    // construct nets
    //
    // C : convolution
    // S : sub-sampling
    // F : fully connected
    // clang-format off
    nn  << conv( 32, 32, 5, 1, 12 ) << relu() // C1, 1@32x32-in, 12@28x28-out
        << max_pool( 28, 28, 12, 2 ) // S2, 12@28x28-in, 12@14x14-out
        << conv( 14, 14, 5, 12, 25 ) << relu() // C3, 12@14x14-in, 25@10x10-out
        << max_pool( 10, 10, 25, 2 ) // S4, 25@10x10-in, 25@5x5-out
        << fc( 625, 180 ) << relu() // F5, 625-in, 180-out
        << dropout( 180, 0.5f )
        << fc( 180, 100 ) << relu() // F6, 180-in, 100-out
        << dropout( 100, 0.5f )
        << fc( 100, 10 ) << softmax_layer(10); // F7, 100-in, 10-out
    // clang-format on
}

static void train_mnist(    const std::string &data_dir_path,
                            double learning_rate,
                            const int n_train_epochs,
                            const int n_minibatch,
                            core::backend_t backend_type )
{
    // specify loss-function and learning strategy
    network<sequential> nn;
    adamax optimizer;

    construct_net( nn, backend_type );

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels( data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images( data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2 );
    parse_mnist_labels( data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images( data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2 );

    std::cout << "start training" << std::endl;

    progress_display disp( train_images.size() );
    timer t;

    // What is this for?
    //optimizer.alpha *= std::min( tiny_dnn::float_t(4),
    //    static_cast<tiny_dnn::float_t>( sqrt( n_minibatch ) * learning_rate ) );

    int epoch = 1;

    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
            << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = nn.test( test_images, test_labels );
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart( train_images.size() );
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<cross_entropy_multiclass>( optimizer, train_images, train_labels, n_minibatch,
        n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch );

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test( test_images, test_labels ).print_detail( std::cout );
    // save network model & trained weights
    nn.save( "kaggle-mnist-model" );
}

static core::backend_t parse_backend_name( const std::string &name )
{
    const std::array<const std::string, 5> names =
    {
        "internal", "nnpack", "libdnn", "avx", "opencl",
    };
    for ( size_t i = 0; i < names.size(); ++i )
    {
        if ( name.compare(names[i]) == 0 )
            return static_cast<core::backend_t>(i);
    }
    return core::default_engine();
}

static void usage( const char *argv0 )
{
    std::cout   << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
                << " --learning_rate 1"
                << " --epochs 30"
                << " --minibatch_size 128"
                << " --backend_type internal" << std::endl;
}

int main( int argc, char **argv )
{
    double learning_rate         = 1;
    int epochs                   = 30;
    std::string data_path        = "";
    int minibatch_size           = 128;
    core::backend_t backend_type = core::default_engine();

    if ( argc == 2 )
    {
        std::string argname( argv[1] );
        if ( argname == "--help" || argname == "-h" )
        {
            usage( argv[0] );
            return 0;
        }
    }
    for ( int count = 1; count + 1 < argc; count += 2 )
    {
        std::string argname(argv[count]);
        if ( argname == "--learning_rate" )
        {
            learning_rate = atof(argv[count + 1]);
        }
        else if ( argname == "--epochs" )
        {
            epochs = atoi(argv[count + 1]);
        }
        else if ( argname == "--minibatch_size" )
        {
            minibatch_size = atoi( argv[count + 1] );
        }
        else if ( argname == "--backend_type" )
        {
            backend_type = parse_backend_name( argv[count + 1] );
        }
        else if ( argname == "--data_path" )
        {
            data_path = std::string( argv[count + 1] );
        }
        else
        {
            std::cerr << "Invalid parameter specified - \"" << argname << "\"" << std::endl;
            usage( argv[0] );
            return -1;
        }
    }
    if ( data_path == "" )
    {
        std::cerr << "Data path not specified." << std::endl;
        usage( argv[0] );
        return -1;
    }
    if ( learning_rate <= 0 )
    {
        std::cerr << "Invalid learning rate. The learning rate must be greater than 0." << std::endl;
        return -1;
    }
    if ( epochs <= 0 )
    {
        std::cerr << "Invalid number of epochs. The number of epochs must be greater than 0." << std::endl;
        return -1;
    }
    if ( minibatch_size <= 0 || minibatch_size > 60000 )
    {
        std::cerr << "Invalid minibatch size. The minibatch size must be greater than 0 and less than dataset size (60000)." << std::endl;
        return -1;
    }
    std::cout   << "Running with the following parameters:" << std::endl
                << "Data path: " << data_path << std::endl
                << "Learning rate: " << learning_rate << std::endl
                << "Minibatch size: " << minibatch_size << std::endl
                << "Number of epochs: " << epochs << std::endl
                << "Backend type: " << backend_type << std::endl
                << std::endl;
    try
    {
        train_mnist( data_path, learning_rate, epochs, minibatch_size, backend_type );
    }
    catch( tiny_dnn::nn_error &err )
    {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
    return 0;
}
