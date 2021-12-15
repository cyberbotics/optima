/**
 * Summary:
 *        Test speed of forward propagation on CPU
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iterator>
#include <random>
#include <algorithm>
#include "mnist/include/mnist/mnist_reader.hpp"	

#define BATCH_SIZE 10000

using namespace std;
using PropResult = tuple<vector<vector<float>>, vector<vector<float>>>;	

class Neuron{
	public:
		vector<float> out_weights;
		float bias;	

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
			dx = 0.;
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

float sigma(float x){	
	float output = 0.;
	
	//output = fmax(x,0.);
	output = tanh(x);

	return output;
}

PropResult forward_prop(vector<float> input){
	float sum_of_elems = 0.;
	Neuron current_neuron;
	vector<vector<float>> s;
	vector<vector<float>> x;

	for(int j = 0; j<net.nb_layers; j++){
		//printf("j = %d \n", j);
		vector<float> curr_s;
		vector<float> curr_x;

		for(int i = 0; i<net.layers[j].size; i++){
			sum_of_elems = 0.;
			current_neuron = net.layers[j].neurons[i];
			for(int l = 0; l<current_neuron.nb_weights; l++){
				if (j==0){ //input layer
					sum_of_elems += current_neuron.out_weights[l] * input[l];
				}
				else{ // hidden layers
					sum_of_elems += current_neuron.out_weights[l] * x[j-1][l];
				}
			}

			curr_s.push_back(sum_of_elems + current_neuron.bias);
			curr_x.push_back(sigma(sum_of_elems + current_neuron.bias));
		}

		s.push_back(curr_s);
		x.push_back(curr_x);
	}

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
	fstream wfile("model/weights.txt");
	fstream bfile("model/biases.txt");

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
	const string& folder = "src/mnist";	
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
	int hidden = 64;
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

	float acc_time = 0;
	struct timeval start;
	gettimeofday(&start, NULL);

	for (int i = 0; i< nb_images; i++){
		PropResult result;
		result = forward_prop(test_input[i]);
		
		if (test_target[i][verify_classification(get<1>(result)[1])] < 0.5){nb_test_errors++;}
	}

	struct timeval end;
	gettimeofday(&end, NULL);
	acc_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;

	printf ("%f seconds\n", acc_time);
	
	perc_test_error = 100*(float)nb_test_errors/ (float)nb_images;
	printf("%% of test errors = %.1f%%\n", perc_test_error);	


	return 0;
}
