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

#define VERBOSE 1

const float INPUT_MEANS[] = {75.5759, 80.3124, 82.3455};
const float INPUT_STDS[] = {41.0884, 50.9617, 55.9314};
const float TARGET_MEANS[] = {6.3157e-04, 5.1197e+01};
const float TARGET_STDS[] = {0.0729, 16.7403};

typedef int32_t fixed_point_t;

using namespace std;

// DFE
static max_file_t *maxFile;
static max_engine_t *maxEngine;

// Network data
static vector<fixed_point_t> weights;
static vector<double> bias;

static vector<fixed_point_t> inputFixed;
static int cnnOutputSize = ConvNetwork_OS_LINEAR2 + ConvNetwork_OUT_PAD;
static vector<fixed_point_t> cnnOutput(cnnOutputSize);

// Time
static double simTime = 0;
static double realTime = 0;
static double stepTime = 0;
static double beginEndTime = 0;
static double imgTime = 0;
static double netAccTime = 0;
static double imgAccTime = 0;

WbDeviceTag camera;

double elapsed_time(struct timeval start, struct timeval end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

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

  bfile.close();
  wfile.close();
}

vector<fixed_point_t> transpose_weights(vector<fixed_point_t> linearWeights,
                                        int layerId) {
  vector<fixed_point_t> transposedWeights;

  const int ic[] = {ConvNetwork_IC_CONV1, ConvNetwork_IC_CONV2,
                    ConvNetwork_IC_CONV3};
  const int oc[] = {ConvNetwork_OC_CONV1, ConvNetwork_OC_CONV2,
                    ConvNetwork_OC_CONV3};

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
  printf("Reading text files for network parameters.\n");
  load_weights();

  printf("Loading engine!\n");
  maxFile = ConvNetwork_init();
  maxEngine = max_load(maxFile, "*");
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
      (max_get_burst_size(maxFile, NULL) -
       (fWeights1.size() * 4 % max_get_burst_size(maxFile, NULL))) /
      4;
  for (int i = 0; i < paddingToAdd; i++)
    fWeights1.push_back(0);

  printf("nb weights to transfer = %ld\n", fWeights1.size());

  ConvNetwork_writeLMem_actions_t memoryActions;
  memoryActions.param_start = 0;
  memoryActions.param_size = fWeights1.size();
  memoryActions.instream_fromcpu = fWeights1.data();

  ConvNetwork_writeLMem_run(maxEngine, &memoryActions);
  printf("Memory loaded.\n");
}

void process_new_image() {
  int inputSize =
      ConvNetwork_IC_CONV1 * ConvNetwork_HT_CONV1 * ConvNetwork_WT_CONV1;

  // get camera image
  const unsigned char *inputChar = wb_camera_get_image(camera);
  inputFixed.clear();
  // normalize input image
  for (int i = 0; i < inputSize / 3; i++) {
    int idx = 4 * ((floor(i / 320)) * (320) + (i % 320)) + 2;
    float pxValue = inputChar[idx];
    inputFixed.push_back(
        float_to_fixedpt((pxValue - INPUT_MEANS[0]) / INPUT_STDS[0]));
  }
  for (int i = inputSize / 3; i < 2 * inputSize / 3; i++) {
    int idx = 4 * ((floor((i - inputSize / 3) / 320)) * (320) +
                   ((i - inputSize / 3) % 320)) +
              1;
    float pxValue = inputChar[idx];
    inputFixed.push_back(
        float_to_fixedpt((pxValue - INPUT_MEANS[1]) / INPUT_STDS[1]));
  }
  for (int i = 2 * inputSize / 3; i < inputSize; i++) {
    int idx = 4 * ((floor((i - 2 * inputSize / 3) / 320)) * (320) +
                   ((i - 2 * inputSize / 3) % 320));
    float pxValue = inputChar[idx];
    inputFixed.push_back(
        float_to_fixedpt((pxValue - INPUT_MEANS[2]) / INPUT_STDS[2]));
  }
}

void set_dfe_actions(ConvNetwork_actions_t *actions) {
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

  actions->instream_inputfromcpu = inputFixed.data();
  actions->inmem_CONVOLUTION_LAYER1_biasMem = fBias1.data();
  actions->inmem_CONVOLUTION_LAYER2_biasMem = fBias2.data();
  actions->inmem_CONVOLUTION_LAYER3_biasMem = fBias3.data();
  actions->inmem_LINEAR_LAYER1_biasMem = fBias4.data();
  actions->inmem_LINEAR_LAYER2_biasMem = fBias5.data();
  actions->outstream_outputtocpu = (fixed_point_t *)cnnOutput.data();
}

void display_logs(const double *carPosition, int stepCount, int period) {
  if (stepCount % period == 0 || stepCount == 1) {
    printf("### Step number %d ###\n", stepCount);
    printf("Car position = %f %f %f\n", carPosition[0], carPosition[1],
           carPosition[2]);
    printf("Real elapsed time = %f\n", realTime);
    printf("Simulation elapsed time = %f\n", simTime);
    printf("Mean simulation speed ratio = %f\n", simTime / realTime);
    printf("Mean step duration (real time) = %f\n\n", realTime / stepCount);
  }
}

int main(int argc, const char *argv[]) {
  // Get mode argument to choose parallelization process to apply
  if (argc != 2) {
    fprintf(
        stderr,
        "WARNING: Exactly one argument must be passed to the controller.\n");
    return 1;
  }

  if (argv[1] && !argv[1][0]) {
    fprintf(stderr, "WARNING: Mode argument cannot be empty.\n");
    return 1;
  }

  int parallelMode;
  stringstream s(argv[1]);
  s >> parallelMode;

  // load DFE with max file and LMEM content
  dfe_init();
  max_run_t *cnnRun;

  // create the driver instance.
  wbu_driver_init();
  WbNodeRef car = wb_supervisor_node_get_from_def("AUTONOMOUS_CAR");

  // get the time step of the current world.
  int timeStep = (int)wb_robot_get_basic_time_step();

  // enable camera
  camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timeStep);

  // time variables
  struct timeval simStart;
  struct timeval stepStart;
  struct timeval stepEnd;
  struct timeval imgEnd;
  gettimeofday(&simStart, NULL);
  gettimeofday(&stepEnd, NULL);

  int stepCount = 0;

  switch (parallelMode) {
  /****** MODE 0 => (CNN + IMAGE PROCESSING) // WEBOTS RENDERING ******/
  case 0:
    // mandatory initial step
    wb_robot_step(timeStep);

    do {
      const double *carPosition = wb_supervisor_node_get_position(car);

      if (wb_robot_step_begin(timeStep) == -1)
        break;

      gettimeofday(&stepStart, NULL);

      // get and normalize new image
      process_new_image();

      gettimeofday(&imgEnd, NULL);

      ConvNetwork_actions_t actions;
      set_dfe_actions(&actions);

      struct timeval start;
      gettimeofday(&start, NULL);

      // perform DFE inference
      ConvNetwork_run(maxEngine, &actions);

      struct timeval end;
      gettimeofday(&end, NULL);
      float cnn_time = elapsed_time(start, end);
      // printf("Run time = %f\n", cnn_time);

      float steering =
          fixedpt_to_float(cnnOutput[0]) * TARGET_STDS[0] + TARGET_MEANS[0];
      float speed =
          fixedpt_to_float(cnnOutput[1]) * TARGET_STDS[1] + TARGET_MEANS[1];

      // printf("steering = %f\n", steering);
      // printf("speed = %f\n\n", speed);

      // apply steering and speed values to car
      wbu_driver_set_cruising_speed(speed);
      wbu_driver_set_steering_angle(steering * 1.7);

      // time information
      gettimeofday(&stepEnd, NULL);
      simTime = wb_robot_get_time();
      realTime = elapsed_time(simStart, stepEnd);
      imgTime = elapsed_time(stepStart, imgEnd);
      // printf("imgTime = %f\n\n", imgTime);
      stepTime = elapsed_time(stepStart, stepEnd);
      // printf("stepTime = %f\n\n", stepTime);
      beginEndTime = elapsed_time(stepEnd, stepStart);
      // printf("step_begin + step_end time = %f\n\n", beginEndTime);
      netAccTime += stepTime - imgTime;
      imgAccTime += imgTime;

      stepCount++;

#if VERBOSE
      display_logs(carPosition, stepCount, 300);
      if (stepCount % 300 == 0 || stepCount == 1) {
        printf("mean network time = %f\n\n", netAccTime / stepCount);
        printf("mean img time = %f\n\n", imgAccTime / stepCount);
      }
#endif

      // quit simulation after 120s (optional)
      if (simTime > 120) {
        wb_supervisor_simulation_quit(0);
      }
    } while (wb_robot_step_end() != -1);
    break;

  /****** MODE 1 =>  (CNN + WEBOTS RENDERING) // IMAGE PROCESSING ******/
  case 1:
    while (wb_robot_step(timeStep) != -1) {
      // no CNN inference at first step
      if (stepCount == 0)
        process_new_image();
      else {
        ConvNetwork_actions_t actions;
        set_dfe_actions(&actions);

        struct timeval start;
        gettimeofday(&start, NULL);

        // DFE inference parallelized with image processing
        cnnRun = ConvNetwork_run_nonblock(maxEngine, &actions);
        process_new_image();
        max_wait(cnnRun);

        struct timeval end;
        gettimeofday(&end, NULL);
        // float cnn_time = elapsed_time(start, end);
        // printf("Run time = %f\n", cnn_time);

        float steering =
            fixedpt_to_float(cnnOutput[0]) * TARGET_STDS[0] + TARGET_MEANS[0];
        float speed =
            fixedpt_to_float(cnnOutput[1]) * TARGET_STDS[1] + TARGET_MEANS[1];

        // printf("steering = %f\n", steering);
        // printf("speed = %f\n\n", speed);

        // apply steering and speed values to car
        wbu_driver_set_cruising_speed(speed);
        wbu_driver_set_steering_angle(steering * 1.7);
      }

      // time information
      gettimeofday(&stepEnd, NULL);
      simTime = wb_robot_get_time();
      realTime = elapsed_time(simStart, stepEnd);

      stepCount++;

#if VERBOSE
      const double *carPosition = wb_supervisor_node_get_position(car);
      display_logs(carPosition, stepCount, 300);
#endif

      // quit simulation after 120s (optional)
      if (simTime > 120) {
        wb_supervisor_simulation_quit(0);
      }
    }
    break;

  /****** MODE 2 => (WEBOTS RENDERING + IMAGE PROCESSING) // CNN ******/
  case 2:
    while (wb_robot_step(timeStep) != -1) {
      // get and normalize new image
      process_new_image();

      if (stepCount != 0) {
        // wait for DFE to finish before applying driving values
        max_wait(cnnRun);

        float steering =
            fixedpt_to_float(cnnOutput[0]) * TARGET_STDS[0] + TARGET_MEANS[0];
        float speed =
            fixedpt_to_float(cnnOutput[1]) * TARGET_STDS[1] + TARGET_MEANS[1];

        // printf("steering = %f\n", steering);
        // printf("speed = %f\n\n", speed);

        // apply steering and speed values to car
        wbu_driver_set_cruising_speed(speed);
        wbu_driver_set_steering_angle(steering * 1.7);
      }

      ConvNetwork_actions_t actions;
      set_dfe_actions(&actions);
      cnnRun = ConvNetwork_run_nonblock(maxEngine, &actions);

      // time information
      gettimeofday(&stepEnd, NULL);
      simTime = wb_robot_get_time();
      realTime = elapsed_time(simStart, stepEnd);

      stepCount++;

#if VERBOSE
      const double *carPosition = wb_supervisor_node_get_position(car);
      display_logs(carPosition, stepCount, 300);
#endif

      // quit simulation after 120s (optional)
      if (simTime > 120) {
        wb_supervisor_simulation_quit(0);
      }
    }
    break;

  default:
    fprintf(stderr,
            "This parallelization mode does not exist. Use either mode '0',  "
            "'1' or '2'.\n");
    return 1;
    break;
  }

  wbu_driver_cleanup();
  return 0;
}
