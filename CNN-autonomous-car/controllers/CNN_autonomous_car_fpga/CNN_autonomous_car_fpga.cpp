/**
 * Summary:
 *        Autonomous car controller to follow test track with front camera and
 * CNN. FPGA version
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include <webots/camera.h>
#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots/vehicle/driver.h>

#include "ConvNetwork.h"
#include <MaxSLiCInterface.h>

#define FIXED_POINT_FRACTIONAL_BITS 16

#define CONV 0
#define POOL 1
#define LINEAR 2

const float INPUT_MEANS[] = {75.5759, 80.3124, 82.3455};
const float INPUT_STDS[] = {41.0884, 50.9617, 55.9314};
const float TARGET_MEANS[] = {6.3157e-04, 5.1197e+01};
const float TARGET_STDS[] = {0.0729, 16.7403};

typedef int32_t fixed_point_t;

using namespace std;

static max_file_t *max_file;
static max_engine_t *max_engine;

static vector<fixed_point_t> weights;
static vector<double> bias;

static vector<int> layer_types = {CONV, POOL, CONV,   POOL,
                                  CONV, POOL, LINEAR, LINEAR};
static vector<int> layers = {0, 0, 1, 1, 2, 2, 0, 1};
static vector<int> conv_layers = {16, 32, 64};
static vector<int> linear_layers_neurons = {500, 2};
static vector<int> linear_layers_weights = {19456, 500};

float fixedpt_to_float(fixed_point_t input) {
  return (float)input / (float)(1 << FIXED_POINT_FRACTIONAL_BITS);
}

fixed_point_t float_to_fixedpt(float input) {
  return (fixed_point_t)(round(input * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

void load_weights() {
  string line;
  float tmp;
  fixed_point_t tmpFixed;
  fstream wfile("model/weights.txt");
  fstream bfile("model/biases.txt");

  while (getline(wfile, line)) {
    stringstream ss(line);
    ss >> tmp;
    tmpFixed = float_to_fixedpt(tmp);
    weights.push_back(tmpFixed);
    // weights.push_back(0);
  }

  while (getline(bfile, line)) {
    stringstream ss(line);
    ss >> tmp;
    bias.push_back((double)tmp);
    // bias.push_back(0.);
  }

  /*for (int i = 0; i < layer_types.size(); i++) {
    if (layer_types[i] == CONV) {
      int layer_id = layers[i];
      for (int j = 0; j < conv_layers[layer_id]; j++) {
        for (int k = 0; k < 9; k++) {
          getline(wfile, line);
          stringstream ss(line);
          ss >> tmp;
          tmpFixed = float_to_fixedpt(tmp);
                  weights.push_back(tmpFixed);
                  //weights.push_back(0);
        }
        getline(bfile, line);
        stringstream ss(line);
        ss >> tmp;
        //tmpFixed = float_to_fixedpt(tmp);
                bias.push_back((double)tmp);
      }
    }

    else if (layer_types[i] == LINEAR) {
      int layer_id = layers[i];
      for (int j = 0; j < linear_layers_neurons[layer_id]; j++) {
        for (int k = 0; k < linear_layers_weights[layer_id]; k++) {
          getline(wfile, line);
          stringstream ss(line);
          ss >> tmp;
          tmpFixed = float_to_fixedpt(tmp);
                  weights.push_back(tmpFixed);
                  //weights.push_back(0);
        }
        getline(bfile, line);
        stringstream ss(line);
        ss >> tmp;
        //tmpFixed = float_to_fixedpt(tmp);
                bias.push_back((double)tmp);
      }
    }
  }*/

  bfile.close();
  wfile.close();
}

vector<fixed_point_t> transpose_weights(vector<fixed_point_t> linearWeights,
                                        int layerId) {
  vector<fixed_point_t> transposedWeights;

  int ic[] = {ConvNetwork_IC_CONV1, ConvNetwork_IC_CONV2, ConvNetwork_IC_CONV3};
  int oc[] = {ConvNetwork_OC_CONV1, ConvNetwork_OC_CONV2, ConvNetwork_OC_CONV3};

  for (int l = 0; l < ic[layerId]; l++) {
    for (int b = 0; b < oc[layerId]; b++) {
      for (int k = 0; k < ConvNetwork_K_SIZE * ConvNetwork_K_SIZE; k++) {
        transposedWeights.push_back(
            linearWeights[b * ic[layerId] * ConvNetwork_K_SIZE *
                              ConvNetwork_K_SIZE +
                          l * ConvNetwork_K_SIZE * ConvNetwork_K_SIZE + k]);
      }
    }
  }

  return transposedWeights;
}

void dfe_init() {
  load_weights();

  printf("Loading engine!\n");
  max_file = ConvNetwork_init();
  max_engine = max_load(max_file, "*");
  printf("Engine loaded!\n");

  printf("Loading network parameters into DFE memory.\n");

  int weightSize1 = (ConvNetwork_OC_CONV1 * ConvNetwork_IC_CONV1) *
                    pow(ConvNetwork_K_SIZE, 2);
  int weightSize2 = (ConvNetwork_OC_CONV2 * ConvNetwork_IC_CONV2) *
                    pow(ConvNetwork_K_SIZE, 2);
  int weightSize3 = (ConvNetwork_OC_CONV3 * ConvNetwork_IC_CONV3) *
                    pow(ConvNetwork_K_SIZE, 2);
  int weightSize4 =
      ConvNetwork_VS_LINEAR1 * ConvNetwork_IS_LINEAR1 * ConvNetwork_OS_LINEAR1;
  int weightSize5 =
      ConvNetwork_VS_LINEAR2 * ConvNetwork_IS_LINEAR2 * ConvNetwork_OS_LINEAR2;

  auto wFirst = weights.begin();
  auto wLast1 = weights.begin() + weightSize1;
  auto wLast2 = weights.begin() + weightSize1 + weightSize2;
  auto wLast3 = weights.begin() + weightSize1 + weightSize2 + weightSize3;
  auto wLast4 =
      weights.begin() + weightSize1 + weightSize2 + weightSize3 + weightSize4;
  auto wLast5 = weights.begin() + weightSize1 + weightSize2 + weightSize3 +
                weightSize4 + weightSize5;
  vector<fixed_point_t> fWeights1(wFirst, wLast1);
  vector<fixed_point_t> fWeights2(wLast1, wLast2);
  vector<fixed_point_t> fWeights3(wLast2, wLast3);
  vector<fixed_point_t> fWeights4(wLast3, wLast4);
  vector<fixed_point_t> fWeights5(wLast4, wLast5);

  fWeights1 = transpose_weights(fWeights1, 0);
  fWeights2 = transpose_weights(fWeights2, 1);
  fWeights3 = transpose_weights(fWeights3, 2);
  for (int p = 0; p < ConvNetwork_PAD_LINEAR1; p++)
    fWeights4.push_back(0);
  for (int p = 0; p < ConvNetwork_PAD_LINEAR2; p++)
    fWeights5.push_back(0);
  fWeights1.insert(fWeights1.end(), fWeights2.begin(), fWeights2.end());
  fWeights1.insert(fWeights1.end(), fWeights3.begin(), fWeights3.end());
  fWeights1.insert(fWeights1.end(), fWeights4.begin(), fWeights4.end());
  fWeights1.insert(fWeights1.end(), fWeights5.begin(), fWeights5.end());

  // LMEM data must be a multiple of burst size (192 bytes)
  int paddingToAdd =
      (max_get_burst_size(max_file, NULL) -
       (fWeights1.size() * 4 % max_get_burst_size(max_file, NULL))) /
      4;
  for (int i = 0; i < paddingToAdd; i++)
    fWeights1.push_back(0);

  printf("nb weights to transfer = %d\n", fWeights1.size());

  ConvNetwork_writeLMem_actions_t memoryActions;
  memoryActions.param_start = 0;
  memoryActions.param_size = fWeights1.size();
  memoryActions.instream_fromcpu = fWeights1.data();

  /*max_actions_t * actions = max_actions_init(max_file, "default");
  for(int i = 0; i < 17; i++)
          max_set_mem_double(actions, "CONVOLUTION_LAYER1", "biasMem", i, 0.);
  for(int i = 0; i < 33; i++)
          max_set_mem_double(actions, "CONVOLUTION_LAYER2", "biasMem", i, 0.);
  for(int i = 0; i < 65; i++)
          max_set_mem_double(actions, "CONVOLUTION_LAYER3",
  "replicated_mem_1_biasMem", i, 0.);*/

  ConvNetwork_writeLMem_run(max_engine, &memoryActions);
  printf("Memory loaded.\n");
}

vector<fixed_point_t> network_evaluation(vector<fixed_point_t> input) {

  int biasSize1 = ConvNetwork_OC_CONV1;
  int biasSize2 = ConvNetwork_OC_CONV2;
  int biasSize3 = ConvNetwork_OC_CONV3;
  int biasSize4 = ConvNetwork_OS_LINEAR1;
  int biasSize5 = ConvNetwork_OS_LINEAR2;
  auto bFirst = bias.begin();
  auto bLast1 = bias.begin() + biasSize1;
  auto bLast2 = bias.begin() + biasSize1 + biasSize2;
  auto bLast3 = bias.begin() + biasSize1 + biasSize2 + biasSize3;
  auto bLast4 = bias.begin() + biasSize1 + biasSize2 + biasSize3 + biasSize4;
  auto bLast5 =
      bias.begin() + biasSize1 + biasSize2 + biasSize3 + biasSize4 + biasSize5;
  vector<double> fBias1(bFirst, bLast1);
  vector<double> fBias2(bLast1, bLast2);
  vector<double> fBias3(bLast2, bLast3);
  vector<double> fBias4(bLast3, bLast4);
  vector<double> fBias5(bLast4, bLast5);

  printf("nb bias conv layer 1 = % d\n ", fBias1.size());
  printf("nb bias conv layer 2 = % d\n ", fBias2.size());
  printf("nb bias conv layer 3 = %d\n", fBias3.size());
  printf("nb bias linear layer 1 = %d\n", fBias4.size());
  printf("nb bias linear layer 2 = %d\n", fBias5.size());

  int outputSize = ConvNetwork_OS_LINEAR2 + ConvNetwork_OUT_PAD;

  // printf("Output size = %d\n", ConvNetwork_OS_LINEAR2);
  vector<fixed_point_t> output(outputSize);

  ConvNetwork_actions_t actions;

  actions.instream_inputfromcpu = input.data();
  actions.inmem_CONVOLUTION_LAYER1_biasMem = fBias1.data();
  actions.inmem_CONVOLUTION_LAYER2_biasMem = fBias2.data();
  actions.inmem_CONVOLUTION_LAYER3_biasMem = fBias3.data();
  actions.inmem_LINEAR_LAYER1_biasMem = fBias4.data();
  actions.inmem_LINEAR_LAYER2_biasMem = fBias5.data();
  actions.outstream_outputtocpu = (fixed_point_t *)output.data();

  struct timeval start;
  gettimeofday(&start, NULL);

  printf("Running DFE.\n");
  ConvNetwork_run(max_engine, &actions);

  struct timeval end;
  gettimeofday(&end, NULL);
  float acc_time =
      (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
  printf("Run time = %f\n", acc_time);

  return output;
}

int main() {
  // load DFE with max file and LMEM content
  dfe_init();

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

    int input_size =
        ConvNetwork_IC_CONV1 * ConvNetwork_HT_CONV1 * ConvNetwork_WT_CONV1;
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
    vector<fixed_point_t> result = network_evaluation(inputFixed);

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
