/**
 * Summary:
 *        Autonomous car controller to follow test track with front camera and
 * CNN. fixed point and multi-threading version
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include <webots/camera.h>
#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots/vehicle/driver.h>

#define FIXED_POINT_FRACTIONAL_BITS 16

#define CONV 0
#define POOL 1
#define LINEAR 2

const float INPUT_MEANS[] = {75.5759, 80.3124, 82.3455};
const float INPUT_STDS[] = {41.0884, 50.9617, 55.9314};
const float TARGET_MEANS[] = {6.3157e-04, 5.1197e+01};
const float TARGET_STDS[] = {0.0729, 16.7403};

typedef int64_t fixed_point_t;

using namespace std;

float fixedpt_to_float(fixed_point_t input) {
  return (float)input / (float)(1 << FIXED_POINT_FRACTIONAL_BITS);
}

fixed_point_t float_to_fixedpt(float input) {
  return (fixed_point_t)(round(input * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

fixed_point_t fixed_mul(fixed_point_t a, fixed_point_t b) {
  fixed_point_t result;
  int64_t temp;

  temp = (int64_t)a * (int64_t)b;

  temp += (1 << (FIXED_POINT_FRACTIONAL_BITS - 1)); // round

  result = temp >> FIXED_POINT_FRACTIONAL_BITS; // divide by fractional bits

  return result;
}

class Neuron {
public:
  vector<fixed_point_t> out_weights;
  fixed_point_t bias;

  int nb_weights;

  Neuron(){};

  Neuron(int input_size) {
    for (int i = 0; i < input_size; i++) {
      out_weights.push_back(0);
    }
    bias = 0;

    nb_weights = out_weights.size();
  };

  ~Neuron(){};
};

class Kernel {
public:
  int nb_weights;
  int k_size;
  int k_depth;

  vector<fixed_point_t> weights;
  fixed_point_t bias;

  Kernel(){};
  Kernel(int size, int depth) {
    for (int i = 0; i < size * size * depth; i++) {
      weights.push_back(0);
    }
    bias = 0;

    nb_weights = weights.size();
    k_size = size;
    k_depth = depth;
  };

  ~Kernel(){};
};

class LinearLayer {
public:
  int nb_neurons;
  vector<Neuron> neurons;

  LinearLayer(){};

  LinearLayer(int s, int input_size) {
    nb_neurons = s;
    for (int i = 0; i < s; i++) {
      neurons.push_back(Neuron(input_size));
    }
  };

  ~LinearLayer(){};
};

class ConvLayer {
public:
  int nb_kernels;
  int kernel_size;
  vector<Kernel> kernels;

  int in_channels;
  int in_width;
  int in_height;
  int out_width;
  int out_height;

  ConvLayer(){};
  ConvLayer(int k_size, int inChan, int inH, int inW, int outChan, int outH,
            int outW) {

    in_channels = inChan;
    in_width = inW;
    in_height = inH;

    out_width = outW;
    out_height = outH;
    nb_kernels = outChan;
    kernel_size = k_size;
    for (int i = 0; i < outChan; i++) {
      kernels.push_back(Kernel(k_size, inChan));
    }
  };

  ~ConvLayer(){};
};

class MaxPoolLayer {
public:
  int filter_size;
  int in_channels;
  int in_width;
  int in_height;

  MaxPoolLayer(){};
  MaxPoolLayer(int f_size, int inChan, int inH, int inW) {
    filter_size = f_size;
    in_channels = inChan;
    in_height = inH;
    in_width = inW;
  };

  ~MaxPoolLayer(){};
};

class Network {
public:
  vector<ConvLayer> conv_layers;
  vector<MaxPoolLayer> pool_layers;
  vector<LinearLayer> linear_layers;

  vector<int> layer_types;
  vector<int> layers;

  int nb_layers;

  Network() {
    conv_layers.push_back(ConvLayer(3, 3, 80, 320, 16, 78, 318));
    conv_layers.push_back(ConvLayer(3, 16, 39, 159, 32, 37, 157));
    conv_layers.push_back(ConvLayer(3, 32, 18, 78, 64, 16, 76));

    pool_layers.push_back(MaxPoolLayer(2, 16, 78, 318));
    pool_layers.push_back(MaxPoolLayer(2, 32, 37, 157));
    pool_layers.push_back(MaxPoolLayer(2, 64, 16, 76));

    linear_layers.push_back(LinearLayer(500, 19456));
    linear_layers.push_back(LinearLayer(2, 500));

    layer_types = {CONV, POOL, CONV, POOL, CONV, POOL, LINEAR, LINEAR};
    layers = {0, 0, 1, 1, 2, 2, 0, 1};

    nb_layers = layers.size();
  };

  ~Network(){};
};

Network net;

fixed_point_t elu(fixed_point_t x) {

  fixed_point_t output = 0;

  float x1 = fixedpt_to_float(x);

  output = float_to_fixedpt((x1 > 0) * x1 + (x1 <= 0) * ((float)exp(x1) - 1));

  return output;
}

vector<fixed_point_t> forward_prop(vector<fixed_point_t> input) {

  ConvLayer current_conv_layer;
  MaxPoolLayer current_pool_layer;
  LinearLayer current_lin_layer;

  vector<fixed_point_t> x;
  vector<fixed_point_t> prevx;

  for (int i = 0; i < net.nb_layers; i++) {

    int layer_id = net.layers[i];
    int layer_type = net.layer_types[i];

    prevx.clear();
    if (i == 0)
      prevx = input;
    else
      prevx = x;
    x.clear();

    if (layer_type == CONV) {
      switch (layer_id) {
      case 0:
        omp_set_num_threads(16);
        break;
      case 2:
        omp_set_num_threads(32);
        break;
      case 4:
        omp_set_num_threads(64);
        break;
      }

      current_conv_layer = net.conv_layers[layer_id];
      int ih = current_conv_layer.in_height;
      int iw = current_conv_layer.in_width;
      int ks = current_conv_layer.kernel_size;

#pragma omp parallel
      {
        vector<fixed_point_t> x_private;
#pragma omp for nowait schedule(static)
        for (int nc = 0; nc < current_conv_layer.nb_kernels; nc++) {
          Kernel current_kernel = current_conv_layer.kernels[nc];
          for (int nh = 0; nh < current_conv_layer.out_height; nh++) {
            for (int nw = 0; nw < current_conv_layer.out_width; nw++) {
              fixed_point_t sum = 0;
              for (int kc = 0; kc < current_conv_layer.in_channels; kc++) {
                for (int kh = 0; kh < ks; kh++) {
                  for (int kw = 0; kw < ks; kw++) {
                    const int prevIdx =
                        kc * ih * iw + (kh + nh) * iw + (kw + nw);
                    const int kernelIdx = kc * ks * ks + kh * ks + kw;
                    sum += fixed_mul(prevx[prevIdx],
                                     current_kernel.weights[kernelIdx]);
                  }
                }
              }
              sum += current_kernel.bias;

              x_private.push_back(elu(sum));
            }
          }
        }
#pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
          x.insert(x.end(), x_private.begin(), x_private.end());
        }
      }

    }

    else if (layer_type == POOL) {

      current_pool_layer = net.pool_layers[layer_id];

      int out_channels = current_pool_layer.in_channels;
      int out_height =
          floor(current_pool_layer.in_height / current_pool_layer.filter_size);
      int out_width =
          floor(current_pool_layer.in_width / current_pool_layer.filter_size);

      for (int nc = 0; nc < out_channels; nc++) {
        for (int nh = 0; nh < out_height; nh++) {
          for (int nw = 0; nw < out_width; nw++) {
            fixed_point_t result = 0;

            for (int ph = 0; ph < current_pool_layer.filter_size; ph++) {
              for (int pw = 0; pw < current_pool_layer.filter_size; pw++) {
                const int inY = nh * current_pool_layer.filter_size + ph;
                const int inX = nw * current_pool_layer.filter_size + pw;
                if (inY >= 0 && inY < current_pool_layer.in_height &&
                    inX >= 0 && inX < current_pool_layer.in_width) {
                  const int prevIdx = nc * current_pool_layer.in_height *
                                          current_pool_layer.in_width +
                                      inY * current_pool_layer.in_width + inX;
                  if (ph == 0 && pw == 0) {
                    result = prevx[prevIdx];
                  } else if (result < prevx[prevIdx]) {
                    result = prevx[prevIdx];
                  }
                }
              }
            }

            x.push_back(result);
          }
        }
      }
    }

    else if (layer_type == LINEAR) {
      omp_set_num_threads(6);
#pragma omp parallel
      {
        vector<fixed_point_t> x_private;
#pragma omp for nowait schedule(static)
        for (int j = 0; j < net.linear_layers[layer_id].nb_neurons; j++) {

          fixed_point_t sum = 0;

          for (int k = 0; k < net.linear_layers[layer_id].neurons[j].nb_weights;
               k++) {
            sum +=
                fixed_mul(net.linear_layers[layer_id].neurons[j].out_weights[k],
                          prevx[k]);
          }

          if (i == net.nb_layers - 1) {
            x_private.push_back(sum +
                                net.linear_layers[layer_id].neurons[j].bias);
          } else {
            x_private.push_back(
                elu(sum + net.linear_layers[layer_id].neurons[j].bias));
          }
        }
#pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
          x.insert(x.end(), x_private.begin(), x_private.end());
        }
      }
    }
  }

  return x;
}

void load_weights() {
  string line;
  float tmp;
  fixed_point_t tmpFixed;
  fstream wfile("model/weights.txt");
  fstream bfile("model/biases.txt");

  for (int i = 0; i < net.nb_layers; i++) {
    if (net.layer_types[i] == CONV) {
      int layer_id = net.layers[i];
      for (int j = 0; j < net.conv_layers[layer_id].nb_kernels; j++) {
        for (int k = 0; k < net.conv_layers[layer_id].kernels[j].nb_weights;
             k++) {
          getline(wfile, line);
          stringstream ss(line);
          ss >> tmp;
          tmpFixed = float_to_fixedpt(tmp);
          net.conv_layers[layer_id].kernels[j].weights[k] = tmpFixed;
        }
        getline(bfile, line);
        stringstream ss(line);
        ss >> tmp;
        tmpFixed = float_to_fixedpt(tmp);
        net.conv_layers[layer_id].kernels[j].bias = tmpFixed;
      }
    }

    else if (net.layer_types[i] == LINEAR) {
      int layer_id = net.layers[i];
      for (int j = 0; j < net.linear_layers[layer_id].nb_neurons; j++) {
        for (int k = 0; k < net.linear_layers[layer_id].neurons[j].nb_weights;
             k++) {
          getline(wfile, line);
          stringstream ss(line);
          ss >> tmp;
          tmpFixed = float_to_fixedpt(tmp);
          net.linear_layers[layer_id].neurons[j].out_weights[k] = tmpFixed;
        }
        getline(bfile, line);
        stringstream ss(line);
        ss >> tmp;
        tmpFixed = float_to_fixedpt(tmp);
        net.linear_layers[layer_id].neurons[j].bias = tmpFixed;
      }
    }
  }

  bfile.close();
  wfile.close();
}

int main(void) {
  // create the driver instance.
  wbu_driver_init();

  // get the time step of the current world.
  int timeStep = (int)wb_robot_get_basic_time_step();

  // enable camera
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timeStep);

  // load CNN parameters
  load_weights();
  printf("Weights and biases successfully loaded !\n");

  // int exit;
  int count = 0;
  double sim_time = 0;
  double real_time = 0;

  struct timeval sim_start;
  gettimeofday(&sim_start, NULL);

  wb_robot_step(timeStep);
 
  do {
    if (wb_robot_step_begin(timeStep) == -1)
      break;

    int input_size = 3 * 80 * 320;
    vector<fixed_point_t> inputFixed;

    // get camera image
    const unsigned char *input_char = wb_camera_get_image(camera);

    // normalize input image
    for (int i = 0; i < input_size / 3; i++) {
      int idx = 4 * ((floor(i / 320)) * (320) + (i % 320)) + 2;
      float px_value = input_char[idx];
      inputFixed.push_back(
          float_to_fixedpt((px_value - INPUT_MEANS[0]) / INPUT_STDS[0]));
    }
    for (int i = input_size / 3; i < 2 * input_size / 3; i++) {
      int idx = 4 * ((floor((i - input_size / 3) / 320)) * (320) +
                     ((i - input_size / 3) % 320)) +
                1;
      float px_value = input_char[idx];
      inputFixed.push_back(
          float_to_fixedpt((px_value - INPUT_MEANS[1]) / INPUT_STDS[1]));
    }
    for (int i = 2 * input_size / 3; i < input_size; i++) {
      int idx = 4 * ((floor((i - 2 * input_size / 3) / 320)) * (320) +
                     ((i - 2 * input_size / 3) % 320));
      float px_value = input_char[idx];
      inputFixed.push_back(
          float_to_fixedpt((px_value - INPUT_MEANS[2]) / INPUT_STDS[2]));
    }

    // compute forward propagation from input image
    vector<fixed_point_t> result;
    result = forward_prop(inputFixed);

    float steering =
        fixedpt_to_float(result[0]) * TARGET_STDS[0] + TARGET_MEANS[0];
    float speed =
        fixedpt_to_float(result[1]) * TARGET_STDS[1] + TARGET_MEANS[1];

    // time informations
    struct timeval now;
    gettimeofday(&now, NULL);

    sim_time = wb_robot_get_time();
    real_time = (now.tv_sec - sim_start.tv_sec) +
                (now.tv_usec - sim_start.tv_usec) * 1e-6;

    count++;
    if (int(sim_time) % 20 <= 0.01) {
      printf("real_time = %f\n", real_time);
      printf("sim time = %f, sim ratio = %f\n", sim_time, sim_time / real_time);
      printf("mean time = %f\n", real_time / count);
    }

    // quit simulation after 60s (optional)
    if (sim_time > 60) {
      wb_supervisor_simulation_quit(0);
    }
    
    // Apply steering and speed values to car
    wbu_driver_set_cruising_speed(speed);
    wbu_driver_set_steering_angle(steering * 1.7);

  } while (wb_robot_step_end() != -1);

  wbu_driver_cleanup();
  return 0;
}
