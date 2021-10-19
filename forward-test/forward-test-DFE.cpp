/**
 * Document: forward-test-DFE.cpp
 * Summary:
 *        Test speed of forward propagation on DFE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iterator>
#include <random>
#include <algorithm>
#include "mnist/include/mnist/mnist_reader.hpp"	

#include <MaxSLiCInterface.h>
#include "ForwardProp.h"

#define BATCH_SIZE 60000

using namespace std;
using PropResult = tuple<vector<vector<float>>, vector<vector<float>>>;

class Neuron{
	public:
		vector<float> out_weights;
		float bias;
		float x;
		float s;	

		vector<float> dw;
		float dbias;
		float dx;
		float ds;

		int nb_weights;

		Neuron(){};

		Neuron(int input_size){
			random_device rd;
    		default_random_engine e1(rd());	
			normal_distribution<float> normal(0.,1e-6);
			for (int i = 0; i < input_size; i++) { 
				out_weights.push_back(normal(e1));
				dw.push_back(0.);
			}
			bias = normal(e1);
			dbias = 0.;
			x = 0.;
			dx = 0.;
			s = 0.;
			ds = 0.;

			nb_weights = out_weights.size();
		};

		~Neuron(){};
};

class LinearLayer{
	public:
		int size;
		vector<Neuron> neurons;

		LinearLayer(){};

		LinearLayer(int s, int input_size){
			size = s;
			for (int i = 0; i < s; i++) {
				neurons.push_back(Neuron(input_size));
			}
		};

		~LinearLayer(){};

};

class Network{
	public:
		int nb_layers;
		vector<LinearLayer> layers;

		Network(){};

		Network(vector<LinearLayer> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		};

		void setLayers(vector<LinearLayer> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		}

		~Network(){};

};

Network net;

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
	
	float mu = compute_mu(train_input);
	float std = compute_std(train_input, mu);
	for(int i = 0; i<nb_images; i++){
		for(int j = 0; j<train_input_size; j++){
			train_input[i][j] -= mu;
			train_input[i][j] /= std;
		}
	}

	return train_input;
}

vector<vector<float>> process_targets(vector<u_int8_t> targets_init, int desired_nb_images){
	vector<u_int8_t>::const_iterator first = targets_init.begin();
	vector<u_int8_t>::const_iterator last = targets_init.begin() + desired_nb_images;
	vector<u_int8_t> train_target_init(first, last);

	vector<vector<float>> converted_train_target = convert_labels(train_target_init); // Convert to hot one labels
	
	return converted_train_target;
}

PropResult forward_prop_dfe(vector<vector<float>> input, int batch_size){
	max_file_t *max_file;
	max_engine_t *max_engine;

	max_file = ForwardProp_init();
	max_engine = max_load(max_file, "*");

	vector<float> flatten_input = flatten(input);
	vector<vector<float>> allWeights;
	vector<vector<float>> allBiases;

	for(int i = 0; i<net.nb_layers; i++){

		vector<vector<float>> weights;
		for(int j = 0; j<net.layers[i].size; j++){
			weights.push_back(net.layers[i].neurons[j].out_weights);
		}
		vector<float> flatten_weights = flatten(weights);
		allWeights.push_back(flatten_weights);

		vector<float> bias;
		vector<float> biases;
		for(int j = 0; j<net.layers[i].size; j++){
			bias.push_back(net.layers[i].neurons[j].bias);
		}
		for(int j = 0; j < batch_size; j++){
			biases.insert(biases.end(), bias.begin(), bias.end());
			//printf("biases %f\n", biases[j]);
		}

		allBiases.push_back(biases);

	}

	vector<float> s1(net.layers[0].size * batch_size);
	vector<float> x1(net.layers[0].size * batch_size);
	vector<float> s2(net.layers[1].size * batch_size);
	vector<float> x2(net.layers[1].size * batch_size);

	ForwardProp_actions_t actions;

	actions.instream_weights1 = allWeights[0].data();
	actions.instream_biases1 = allBiases[0].data();
	actions.instream_weights2 = allWeights[1].data();
	actions.instream_biases2 = allBiases[1].data();

	actions.instream_input = flatten_input.data();
	actions.outstream_s1 = (float *)s1.data();
	actions.outstream_x1 = (float *)x1.data();
	actions.outstream_s2 = (float *)s2.data();
	actions.outstream_x2 = (float *)x2.data();
	actions.routing_string = "x11 -> x1Fanout, x12 -> x1Fanout";

	actions.param_BS = batch_size;
	actions.param_LS0 = flatten_input.size() / batch_size;
	actions.param_LS1 = net.layers[0].size;
	actions.param_LS2 = net.layers[1].size;

	ForwardProp_run(max_engine, &actions);

	max_unload(max_engine);

	vector<vector<float>> s;
	vector<vector<float>> x;

	s.push_back(s1);
	s.push_back(s2);
	x.push_back(x1);
	x.push_back(x2);

	PropResult result;
	result = make_tuple(s,x);

	return result;
}


int verify_classification(vector<float> result){
	int pred;
	pred = max_element(result.begin(),result.end()) - result.begin();
	return pred;
}

void load_weights(){
	string line;
	float tmp;
	fstream wfile("../framework-dl-CPU/model/weights.txt");
	fstream bfile("../framework-dl-CPU/model/biases.txt");

	for(int i = 0; i < net.nb_layers; i++){
		for(int j = 0; j< net.layers[i].size; j++){
			for(int k = 0; k< net.layers[i].neurons[j].nb_weights; k++){
				getline(wfile, line);
				stringstream ss(line);
				ss >> tmp;
				net.layers[i].neurons[j].out_weights[k] = tmp;
			}
			getline(bfile,line);
			stringstream ss(line);
			ss >> tmp;
			net.layers[i].neurons[j].bias = tmp;	
		}
	}

	bfile.close();
	wfile.close();
}

int main(void)
{		
	// Load MNIST dataset using external library (https://github.com/wichtounet/mnist)
	const string& folder = "mnist";	
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(folder);

	// Process input data and labels
	int desired_nb_images = BATCH_SIZE; // 60000 for full test set

	vector<vector<u_int8_t>> test_input_full= dataset.training_images;
	vector<u_int8_t> test_target_full= dataset.training_labels;

	vector<vector<float>> test_input = process_images(test_input_full, desired_nb_images);
	vector<vector<float>> test_target = process_targets(test_target_full, desired_nb_images);

	int nb_images = test_input.size();
	int test_input_size = test_input[0].size();
	int test_target_size = test_target[0].size();

	int nb_test_errors = 0;
	float perc_test_error = 0.;

	// Network parameters
	int hidden = 50;
	vector<int> layer_size = {test_input_size, hidden, test_target_size};

	int nb_layers = layer_size.size() - 1;

	// Network creation
	vector<LinearLayer> net_layers;
	for(int i = 1; i< nb_layers+1;i++){
		net_layers.push_back(LinearLayer(layer_size[i],layer_size[i-1]));
	}
	net.setLayers(net_layers, nb_layers);
	
	load_weights();
	printf("Weights and biases successfully loaded !\n");

	PropResult result;
	result = forward_prop_dfe(test_input, nb_images);
	for (int i = 0; i< nb_images; i++){
		vector<float>::const_iterator first = get<1>(result)[1].begin() + i * test_target_size;
		vector<float>::const_iterator last = get<1>(result)[1].begin() + (i+1) * test_target_size;
		vector<float> this_img_result(first, last);
		if (test_target[i][verify_classification(this_img_result)] < 0.5){nb_test_errors++;}
	}
	
	perc_test_error = 100*(float)nb_test_errors/ (float)nb_images;
	printf("%% of test errors = %.1f%%\n", perc_test_error);	


	return 0;
}
