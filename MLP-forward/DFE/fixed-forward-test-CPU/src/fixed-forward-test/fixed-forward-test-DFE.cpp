/**
 * Summary:
 *        Test speed of forward propagation on DFE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <tuple>
#include <iterator>
#include <random>
#include <algorithm>
#include <MaxSLiCInterface.h>

#include "../fixed-forward-test/mnist/include/mnist/mnist_reader.hpp"
#include "FixedForwardProp.h"

#define BATCH_SIZE 1000 // minimum 4 * numEngines

#define FIXED_POINT_FRACTIONAL_BITS 16
typedef int32_t fixed_point_t;

using namespace std;
using PropResult = tuple<vector<vector<fixed_point_t>>, vector<vector<fixed_point_t>>>;

// Network dimensions
const vector<int> netDimensions = {FixedForwardProp_SIZE_LAYER_0, FixedForwardProp_SIZE_LAYER_1, FixedForwardProp_SIZE_LAYER_2};

// Parallel input
const int inVecSize[] = {FixedForwardProp_IN_VEC_SIZE1, FixedForwardProp_IN_VEC_SIZE2};

// Parallel output
const int outVecSize[] = {FixedForwardProp_OUT_VEC_SIZE1, FixedForwardProp_OUT_VEC_SIZE2};


fixed_point_t float_to_fixedpt(float input)
{
    return (fixed_point_t)(round(input * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

fixed_point_t fixed_mul(fixed_point_t a, fixed_point_t b)
{
    fixed_point_t result;
    int32_t temp;

    temp = (int32_t)a * (int32_t)b;

    temp += (1 << (FIXED_POINT_FRACTIONAL_BITS - 1)); //round

    result = temp >> FIXED_POINT_FRACTIONAL_BITS; // divide by fractional bits

    return result;
}

template<typename T>
vector<T> flatten(const vector<vector<T>> &orig)
{
    vector<T> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());
    return ret;
}

vector<vector<float>> convert_labels(vector<u_int8_t> labels){
	vector<vector<float>> converted_labels;
	
	for(int i = 0; i<labels.size(); i++){
		vector<float> current_label;
		for(int j = 0; j<10; j++){
			if(j == labels[i]){
				current_label.push_back(0.9);
			}
			else{
				current_label.push_back(0.0);
			}
		}
  		current_label.shrink_to_fit();                          
  		converted_labels.push_back(move(current_label)); 
	}
	return converted_labels;
}

float compute_mu(vector<vector<float>> vec){
	float mu;
	float vecsize1 = (float)vec.size();
	float vecsize2 = (float)vec[0].size();

	for(int i = 0; i<vecsize1; i++){
		for(int j = 0; j<vecsize2; j++){
			mu += vec[i][j];
		}
	}
	mu = mu / vecsize1 / vecsize2;
	return mu;
}

float compute_std(vector<vector<float>> vec, float mu){
	float std;
	float vecsize1 = (float)vec.size();
	float vecsize2 = (float)vec[0].size();

	for(int i = 0; i<vecsize1; i++){
		for(int j = 0; j<vecsize2; j++){
			std += pow(vec[i][j]-mu,2);
		}
	}
	std = sqrt(std / (vecsize1 * (vecsize2-1)));
	return std;
}

float mu;
float stdev;

vector<vector<float>> process_images(vector<vector<u_int8_t>> images_init, int desired_nb_images){
	vector<vector<u_int8_t>>::const_iterator first = images_init.begin();
	vector<vector<u_int8_t>>::const_iterator last = images_init.begin() + desired_nb_images;
	vector<vector<u_int8_t>> train_input_init(first, last);

	int nb_images = desired_nb_images;
	int train_input_size = train_input_init[0].size();

    vector<vector<float>> train_input;
	for(int i = 0; i< nb_images;i++){
		vector<float> train_input_single(train_input_init[i].begin(), train_input_init[i].end());
		train_input_single.shrink_to_fit();                         
  		train_input.push_back(move(train_input_single));
	}
	
	mu = compute_mu(train_input);
	stdev = compute_std(train_input, mu);
	/*for(int i = 0; i<nb_images; i++){
		for(int j = 0; j<train_input_size; j++){
			train_input[i][j] -= mu;
			train_input[i][j] /= std;
		}
	}*/

	return train_input;
}

vector<vector<float>> process_targets(vector<u_int8_t> targets_init, int desired_nb_images){
	vector<u_int8_t>::const_iterator first = targets_init.begin();
	vector<u_int8_t>::const_iterator last = targets_init.begin() + desired_nb_images;
	vector<u_int8_t> train_target_init(first, last);

	vector<vector<float>> converted_train_target = convert_labels(train_target_init); // Convert to hot one labels
	
	return converted_train_target;
}


vector<vector<fixed_point_t>> load_weights(){
	vector<vector<fixed_point_t>> allWeights;
	string line;
	float tmp;
	fixed_point_t tmpFixed;
	fstream wfile("model/weights.txt");

	for(int i = 0; i < netDimensions.size()- 1; i++){
		vector<fixed_point_t> weights;
		for(int j = 0; j< netDimensions[i]*netDimensions[i+1]; j++){
			getline(wfile, line);
			stringstream ss(line);
			ss >> tmp;
			tmpFixed = float_to_fixedpt(tmp);
			weights.push_back(tmpFixed);
		}
		allWeights.push_back(weights);
	}

	wfile.close();

	return allWeights;
}

vector<vector<fixed_point_t>> load_biases(){

	vector<vector<fixed_point_t>> allBiases;
	string line;
	float tmp;
	fixed_point_t tmpFixed;
	fstream bfile("model/biases.txt");

	for(int i = 0; i < netDimensions.size()- 1; i++){
		vector<fixed_point_t> bias;

		for(int j = 0; j< netDimensions[i+1]; j++){
			getline(bfile,line);
			stringstream ss(line);
			ss >> tmp;
			tmpFixed = float_to_fixedpt(tmp);
			bias.push_back(tmpFixed);
		}

		// Padding for last layer
		if(i == netDimensions.size()- 2){
			bias.push_back(0);
			bias.push_back(0);
		}

		allBiases.push_back(bias);
	}

	bfile.close();

	return allBiases;
}

vector<vector<u_int8_t>> split_input(int nbStreams, vector<u_int8_t> input){
	vector<vector<u_int8_t>> splittedInput;

	int splittedInputSize = input.size()/nbStreams;

	for(int i = 0; i< nbStreams; i++){
		vector<u_int8_t>::const_iterator first = input.begin() + i * splittedInputSize ;
		vector<u_int8_t>::const_iterator last = input.begin() + (i+1) * splittedInputSize;
		vector<u_int8_t> inputPart(first, last);
		splittedInput.push_back(inputPart);
	}

	return splittedInput;
}

PropResult forward_prop_dfe(vector<vector<u_int8_t>> input, int batchSize){
	const int numEngines = 1;
	const int batchFractionSize = batchSize / numEngines;

	max_file_t *max_file;
	max_engine_t *max_engines[numEngines];
	max_file = FixedForwardProp_init();

	printf("Loading %d engine(s)!\n", numEngines);

	for(int i = 0; i < numEngines; i++){
		max_engines[i] = max_load(max_file, "*");
	}
	printf("%d engine(s) loaded!\n", numEngines);

	vector<u_int8_t> flattenInput = flatten(input);
	vector<vector<u_int8_t>> inputParts = split_input(numEngines, flattenInput);

	vector<vector<fixed_point_t>> allWeights = load_weights();
	vector<vector<fixed_point_t>> allBiases = load_biases();

	int zeroLayerSize = netDimensions[0];
	int firstLayerSize = netDimensions[1];
	int secondLayerSize = netDimensions[2];

	vector<vector<fixed_point_t>> s1(numEngines, vector<fixed_point_t>(firstLayerSize * batchSize / numEngines));
	vector<vector<fixed_point_t>> x1(numEngines, vector<fixed_point_t>(firstLayerSize * batchSize / numEngines));
	vector<vector<fixed_point_t>> s2(numEngines, vector<fixed_point_t>(secondLayerSize * batchSize / numEngines));
	vector<vector<fixed_point_t>> x2(numEngines, vector<fixed_point_t>(secondLayerSize * batchSize / numEngines));

	printf("Setting actions!\n");
	FixedForwardProp_actions_t actions[numEngines];
	for (int i = 0; i < numEngines; i++) {
		actions[i].instream_weights1 = allWeights[0].data();
		actions[i].instream_biases1 = allBiases[0].data();
		actions[i].instream_weights2 = allWeights[1].data();
		actions[i].instream_biases2 = allBiases[1].data();

		actions[i].instream_input = inputParts[i].data();
		actions[i].outstream_s1 = (fixed_point_t *)s1[i].data();
		actions[i].outstream_x1 = (fixed_point_t *)x1[i].data();
		actions[i].outstream_s2 = (fixed_point_t *)s2[i].data();
		actions[i].outstream_x2 = (fixed_point_t *)x2[i].data();

		char* route = "x11 -> x1Fanout, x12 -> x1Fanout";
		actions[i].routing_string = route;

		printf("%d %d %d %d %d\n", batchSize/numEngines, batchFractionSize, zeroLayerSize, firstLayerSize, secondLayerSize);
		actions[i].param_BS = batchFractionSize;
		printf("%d %d \n", float_to_fixedpt(mu), float_to_fixedpt(stdev));

		actions[i].param_MU = mu;
		actions[i].param_STD = stdev;
	}

	printf("Running on DFEs!\n");

	max_run_t *runs[numEngines];

	struct timeval start;
	gettimeofday(&start, NULL);
	for(int i = 0; i < numEngines; i++){
		runs[i] = FixedForwardProp_run_nonblock(max_engines[i], &actions[i]);
	}
	for(int i = 0; i < numEngines; i++){
		max_wait(runs[i]);
	}
	struct timeval end;
	gettimeofday(&end, NULL);
	printf ("%f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6);


	gettimeofday(&start, NULL);
	for(int i = 0; i < numEngines; i++){
		runs[i] = FixedForwardProp_run_nonblock(max_engines[i], &actions[i]);
	}
	for(int i = 0; i < numEngines; i++){
		max_wait(runs[i]);
	}
	gettimeofday(&end, NULL);
	printf ("%f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6);


	/*gettimeofday(&start, NULL);
	for(int i = 0; i < numEngines; i++){
		runs[i] = ForwardProp_run_nonblock(max_engines[i], &actions[i]);
	}
	for(int i = 0; i < numEngines; i++){
		max_wait(runs[i]);
	}
	gettimeofday(&end, NULL);
	printf ("%f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6);*/

	printf("Unloading %d engine(s)!\n", numEngines);
	for(int i = 0; i < numEngines; i++){
		max_unload(max_engines[i]);
	}

	vector<vector<fixed_point_t>> s;
	vector<vector<fixed_point_t>> x;

	s.push_back(flatten(s1));
	s.push_back(flatten(s2));
	x.push_back(flatten(x1));
	x.push_back(flatten(x2));

	PropResult result;
	result = make_tuple(s,x);

	return result;
}


int verify_classification(vector<fixed_point_t> result){
	int pred;
	pred = max_element(result.begin(),result.end()) - result.begin();
	return pred;
}


int main(void)
{		
	// Load MNIST dataset using external library (https://github.com/wichtounet/mnist)
	const string& folder = "src/fixed-forward-test/mnist";
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(folder);

	// Process input data and labels
	int desired_nb_images = BATCH_SIZE; // 60000 for full test set

	vector<vector<u_int8_t>> test_input_full= dataset.training_images;
	vector<u_int8_t> test_target_full= dataset.training_labels;

	vector<vector<float>> float_test_input = process_images(test_input_full, desired_nb_images);
	vector<vector<float>> float_test_target = process_targets(test_target_full, desired_nb_images);

	vector<vector<u_int8_t>> test_input;
	vector<vector<fixed_point_t>> test_target;

	for(int i = 0; i < float_test_input.size(); i++){
		vector<u_int8_t> curr_in;
		for(int j = 0; j < float_test_input[0].size(); j++){
			curr_in.push_back((u_int8_t)(float_test_input[i][j]));
		}
		test_input.push_back(curr_in);
	}
	for(int i = 0; i < float_test_target.size(); i++){
		vector<fixed_point_t> curr_tar;
		for(int j = 0; j < float_test_target[0].size(); j++){
			curr_tar.push_back(float_to_fixedpt(float_test_target[i][j]));
		}
		test_target.push_back(curr_tar);
	}

	int nb_images = test_input.size();
	int test_input_size = test_input[0].size();
	int test_target_size = test_target[0].size();

	int nb_test_errors = 0;
	float perc_test_error = 0.;

	printf("MNIST is loaded !\n");

	PropResult result;
	result = forward_prop_dfe(test_input, nb_images);
	for (int i = 0; i< nb_images; i++){
		vector<fixed_point_t>::const_iterator first = get<1>(result)[1].begin() + i * (test_target_size);
		vector<fixed_point_t>::const_iterator last = get<1>(result)[1].begin() + (i+1) * (test_target_size);
		vector<fixed_point_t> this_img_result(first, last);

		/*for(int j = 0; j < test_target_size; j++){
			printf("%d  %f\n",i*10+j, (float)this_img_result[j]/ (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
		}*/

		if (test_target[i][verify_classification(this_img_result)] < 0.5){nb_test_errors++;}
	}
	
	perc_test_error = 100*(float)nb_test_errors/ (float)nb_images;
	printf("%% of test errors = %.1f%%\n", perc_test_error);	

	return 0;
}
