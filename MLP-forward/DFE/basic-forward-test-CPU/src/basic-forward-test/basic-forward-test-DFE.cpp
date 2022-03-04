/**
 * Summary:
 *        Test speed of forward propagation on DFE
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tuple>
#include <vector>

#include "../basic-forward-test/mnist/include/mnist/mnist_reader.hpp"
#include "ForwardProp.h"
#include <MaxSLiCInterface.h>

#define BATCH_SIZE 16

using namespace std;
using PropResult = tuple<vector<vector<float>>, vector<vector<float>>>;

const vector<int> netDimensions = {ForwardProp_SIZE_LAYER_0,
                                   ForwardProp_SIZE_LAYER_1,
                                   ForwardProp_SIZE_LAYER_2};

template <typename T> vector<T> flatten(const vector<vector<T>> &orig) {
  vector<T> ret;
  for (const auto &v : orig)
    ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

vector<vector<float>> convert_labels(vector<u_int8_t> labels) {
  vector<vector<float>> converted_labels;

  for (int i = 0; i < labels.size(); i++) {
    vector<float> current_label;
    for (int j = 0; j < 10; j++) {
      if (j == labels[i]) {
        current_label.push_back(0.9);
      } else {
        current_label.push_back(0.0);
      }
    }
    current_label.shrink_to_fit();
    converted_labels.push_back(move(current_label));
  }
  return converted_labels;
}

float compute_mu(vector<vector<float>> vec) {
  float mu;
  float vecsize1 = (float)vec.size();
  float vecsize2 = (float)vec[0].size();

  for (int i = 0; i < vecsize1; i++) {
    for (int j = 0; j < vecsize2; j++) {
      mu += vec[i][j];
    }
  }
  mu = mu / vecsize1 / vecsize2;
  return mu;
}

float compute_std(vector<vector<float>> vec, float mu) {
  float std;
  float vecsize1 = (float)vec.size();
  float vecsize2 = (float)vec[0].size();

  for (int i = 0; i < vecsize1; i++) {
    for (int j = 0; j < vecsize2; j++) {
      std += pow(vec[i][j] - mu, 2);
    }
  }
  std = sqrt(std / (vecsize1 * (vecsize2 - 1)));
  return std;
}

vector<vector<float>> process_images(vector<vector<u_int8_t>> images_init,
                                     int desired_nb_images) {
  vector<vector<u_int8_t>>::const_iterator first = images_init.begin();
  vector<vector<u_int8_t>>::const_iterator last =
      images_init.begin() + desired_nb_images;
  vector<vector<u_int8_t>> train_input_init(first, last);

  int nb_images = desired_nb_images;
  int train_input_size = train_input_init[0].size();

  vector<vector<float>> train_input;
  for (int i = 0; i < nb_images; i++) {
    vector<float> train_input_single(train_input_init[i].begin(),
                                     train_input_init[i].end());
    train_input_single.shrink_to_fit();
    train_input.push_back(move(train_input_single));
  }

  float mu = compute_mu(train_input);
  float std = compute_std(train_input, mu);
  for (int i = 0; i < nb_images; i++) {
    for (int j = 0; j < train_input_size; j++) {
      train_input[i][j] -= mu;
      train_input[i][j] /= std;
    }
  }

  return train_input;
}

vector<vector<float>> process_targets(vector<u_int8_t> targets_init,
                                      int desired_nb_images) {
  vector<u_int8_t>::const_iterator first = targets_init.begin();
  vector<u_int8_t>::const_iterator last =
      targets_init.begin() + desired_nb_images;
  vector<u_int8_t> train_target_init(first, last);

  vector<vector<float>> converted_train_target =
      convert_labels(train_target_init); // Convert to hot one labels

  return converted_train_target;
}

vector<vector<float>> load_weights() {

  vector<vector<float>> allWeights;
  string line;
  float tmp;
  fstream wfile("model/weights.txt");

  for (int i = 0; i < netDimensions.size() - 1; i++) {
    vector<float> weights;
    for (int j = 0; j < netDimensions[i] * netDimensions[i + 1]; j++) {
      getline(wfile, line);
      stringstream ss(line);
      ss >> tmp;
      weights.push_back(tmp);
    }
    allWeights.push_back(weights);
  }

  wfile.close();

  return allWeights;
}

vector<vector<float>> load_biases(int batch_size) {

  vector<vector<float>> allBiases;

  string line;
  float tmp;
  fstream bfile("model/biases.txt");

  for (int i = 0; i < netDimensions.size() - 1; i++) {
    vector<float> bias;
    vector<float> biases;

    for (int j = 0; j < netDimensions[i + 1]; j++) {
      getline(bfile, line);
      stringstream ss(line);
      ss >> tmp;
      bias.push_back(tmp);
    }

    for (int j = 0; j < batch_size; j++) {
      biases.insert(biases.end(), bias.begin(), bias.end());
      // printf("biases %f\n", biases[j]);
    }

    allBiases.push_back(biases);
  }

  bfile.close();

  return allBiases;
}

PropResult forward_prop_dfe(vector<vector<float>> input, int batchSize) {
  max_file_t *max_file;
  max_engine_t *max_engine;

  max_file = ForwardProp_init();
  max_engine = max_load(max_file, "*");

  vector<float> flattenInput = flatten(input);
  vector<vector<float>> allWeights;
  vector<vector<float>> allBiases;

  int firstLayerSize = netDimensions[1];
  int secondLayerSize = netDimensions[2];

  allWeights = load_weights();
  allBiases = load_biases(batchSize);

  vector<float> s1(firstLayerSize * batchSize);
  vector<float> x1(firstLayerSize * batchSize);
  vector<float> s2(secondLayerSize * batchSize);
  vector<float> x2(secondLayerSize * batchSize);

  ForwardProp_actions_t actions;

  actions.instream_weights1 = allWeights[0].data();
  actions.instream_biases1 = allBiases[0].data();
  actions.instream_weights2 = allWeights[1].data();
  actions.instream_biases2 = allBiases[1].data();

  actions.instream_input = flattenInput.data();
  actions.outstream_s1 = (float *)s1.data();
  actions.outstream_x1 = (float *)x1.data();
  actions.outstream_s2 = (float *)s2.data();
  actions.outstream_x2 = (float *)x2.data();
  actions.routing_string = "x11 -> x1Fanout, x12 -> x1Fanout";

  actions.param_BS = batchSize;

  struct timeval start;
  gettimeofday(&start, NULL);
  ForwardProp_run(max_engine, &actions);
  struct timeval end;
  gettimeofday(&end, NULL);
  printf("%f seconds\n",
         (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);

  max_unload(max_engine);

  vector<vector<float>> s;
  vector<vector<float>> x;

  s.push_back(s1);
  s.push_back(s2);
  x.push_back(x1);
  x.push_back(x2);

  PropResult result;
  result = make_tuple(s, x);

  return result;
}

int verify_classification(vector<float> result) {
  int pred;
  pred = max_element(result.begin(), result.end()) - result.begin();
  return pred;
}

int main(void) {
  // Load MNIST dataset using external library
  // (https://github.com/wichtounet/mnist)
  const string &folder = "src/basic-forward-test/mnist";
  auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(folder);

  // Process input data and labels
  int desired_nb_images = BATCH_SIZE; // 60000 for full test set

  vector<vector<u_int8_t>> test_input_full = dataset.training_images;
  vector<u_int8_t> test_target_full = dataset.training_labels;

  vector<vector<float>> test_input =
      process_images(test_input_full, desired_nb_images);
  vector<vector<float>> test_target =
      process_targets(test_target_full, desired_nb_images);

  int nb_images = test_input.size();
  int test_input_size = test_input[0].size();
  int test_target_size = test_target[0].size();

  int nb_test_errors = 0;
  float perc_test_error = 0.;

  PropResult result;
  result = forward_prop_dfe(test_input, nb_images);
  for (int i = 0; i < nb_images; i++) {
    vector<float>::const_iterator first =
        get<1>(result)[1].begin() + i * (test_target_size);
    vector<float>::const_iterator last =
        get<1>(result)[1].begin() + (i + 1) * (test_target_size);
    vector<float> this_img_result(first, last);

    /*for(int j = 0; j < test_target_size; j++){
            printf("%d  %f\n",i*10+j, this_img_result[j]);
    }*/

    if (test_target[i][verify_classification(this_img_result)] < 0.5) {
      nb_test_errors++;
    }
  }

  perc_test_error = 100 * (float)nb_test_errors / (float)nb_images;
  printf("%% of test errors = %.1f%%\n", perc_test_error);

  return 0;
}
