/**
 * Document: DNN-Framework-naive.cpp
 * Summary:
 *       Naive implementation of a Neural Net Framework for future DFE implementation
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <random>
#include <algorithm>
#include "mnist/include/mnist/mnist_reader.hpp"	

using namespace std;

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
		};

		~Neuron(){};
};

class Linear{
	public:
		int size;
		vector<Neuron> neurons;

		Linear(){};

		Linear(int s, int input_size){
			size = s;
			for (int i = 0; i < s; i++) {
				neurons.push_back(Neuron(input_size));
			}
		};

		~Linear(){};

};

class Net{
	public:
		int nb_layers;
		vector<Linear> layers;

		Net(){};

		Net(vector<Linear> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		};

		void setLayers(vector<Linear> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		}

		~Net(){};

};

Net net;

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

float dsigma(float x){
	float output = 0.;

	//output = 4 * pow(exp(x) + exp(-x), -2);
	output = 1.-pow(tanh(x),2.0);

	return output;
}

float MSELoss(vector<float> input, vector<float> target){
	 
	float loss = 0.;
	for(int i = 0; i<input.size(); i++){
		loss += pow(input[i] - target[i], 2.);
	}
	return loss;
}

float MSEdLoss(float input, float target){
	float dloss = 0.;

	dloss = 2*input - 2*target;
	
	return dloss;
}

void forward_prop(vector<float> input){
	float sum_of_elems = 0.;

	for(int j = 0; j<net.nb_layers; j++){
		//vector<float> hidden_x;
		for(int i = 0; i<net.layers[j].size; i++){
			sum_of_elems = 0.;
			for(int l = 0; l<net.layers[j].neurons[i].out_weights.size(); l++){
				if (j==0){ //input layer
					sum_of_elems += net.layers[j].neurons[i].out_weights[l] * input[l];
				}
				else{ // hidden layers
					sum_of_elems += net.layers[j].neurons[i].out_weights[l] * net.layers[j-1].neurons[l].x;
				}
			}
		
			net.layers[j].neurons[i].s = sum_of_elems + net.layers[j].neurons[i].bias;
			net.layers[j].neurons[i].x = sigma(net.layers[j].neurons[i].s);
		}
	}
}

void back_prop(vector<float> input, vector<float> target){
	float sum_of_elems = 0.;

	for(int i=0;i<target.size();i++)
    {
        net.layers[net.nb_layers-1].neurons[i].dx = MSEdLoss(net.layers[net.nb_layers-1].neurons[i].x, target[i]);
		net.layers[net.nb_layers-1].neurons[i].ds = dsigma(net.layers[net.nb_layers-1].neurons[i].s)*net.layers[net.nb_layers-1].neurons[i].dx;
		net.layers[net.nb_layers-1].neurons[i].dbias += net.layers[net.nb_layers-1].neurons[i].ds; 
		for(int j = 0; j<net.layers[net.nb_layers-1].neurons[i].dw.size(); j++){
			net.layers[net.nb_layers-1].neurons[i].dw[j] += net.layers[net.nb_layers-2].neurons[j].x * net.layers[net.nb_layers-1].neurons[i].ds;
		}   
    }

	for(int i = (net.nb_layers-2); i>=0; i--){
		for(int j = 0; j<net.layers[i].size; j++){
			sum_of_elems = 0.;

			for(int k = 0; k<net.layers[i+1].size; k++){
				sum_of_elems += net.layers[i+1].neurons[k].out_weights[j] * net.layers[i+1].neurons[k].ds;
			}
			net.layers[i].neurons[j].dx = sum_of_elems;
			net.layers[i].neurons[j].ds = net.layers[i].neurons[j].dx * dsigma(net.layers[i].neurons[j].s);
			net.layers[i].neurons[j].dbias += net.layers[i].neurons[j].ds;
			for(int k = 0; k<net.layers[i].neurons[j].dw.size(); k++){
				if(i != 0){
					net.layers[i].neurons[j].dw[k] += net.layers[i-1].neurons[k].x * net.layers[i].neurons[j].ds;
				}
				else{
					net.layers[i].neurons[j].dw[k] += input[k] * net.layers[i].neurons[j].ds;
				}		
			}

		}
	}
}

void update_weights(float rate){
	for(int i = 0; i < net.nb_layers; i++){
		for(int j = 0; j< net.layers[i].size; j++){
			for(int k = 0; k< net.layers[i].neurons[j].out_weights.size(); k++){
				net.layers[i].neurons[j].out_weights[k] -= (rate * net.layers[i].neurons[j].dw[k]);
			}
			net.layers[i].neurons[j].bias -= (rate * net.layers[i].neurons[j].dbias);			
		}
	}
}

void reset_gradients(){
	for(int i = 0; i < net.nb_layers; i++){
		for(int j = 0; j< net.layers[i].size; j++){
			for(int k = 0; k< net.layers[i].neurons[j].dw.size(); k++){
				net.layers[i].neurons[j].dw[k] = 0.;
			}
			net.layers[i].neurons[j].dbias = 0.;			
		}
	}
}

int verify_classification(){
	vector<float> output_vector;
	int pred;
	for(int j = 0; j<net.layers[net.nb_layers-1].size; j++){
		output_vector.push_back(net.layers[net.nb_layers-1].neurons[j].x);
	}
	pred = max_element(output_vector.begin(),output_vector.end()) - output_vector.begin();

	return pred;
}

int main(void)
{	
	// Load MNIST dataset using external library (https://github.com/wichtounet/mnist)
	const string& folder = "src/mnist";	
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(folder);

	// Process input data and labels
	int desired_nb_images = 6000;

	vector<vector<u_int8_t>> train_input_full= dataset.training_images;
	vector<u_int8_t> train_target_full= dataset.training_labels;
	vector<vector<u_int8_t>> test_input_full= dataset.test_images;
	vector<u_int8_t> test_target_full= dataset.test_labels;

	vector<vector<float>> train_input = process_images(train_input_full, desired_nb_images);
	vector<vector<float>> train_target = process_targets(train_target_full, desired_nb_images);
	vector<vector<float>> test_input = process_images(test_input_full, desired_nb_images);
	vector<vector<float>> test_target = process_targets(test_target_full, desired_nb_images);

	int nb_images = train_input.size();
	int train_input_size = train_input[0].size();
	int train_target_size = train_target[0].size();

	// Training parameters
	int nb_epochs = 10;
	float learning_rate = 0.1 / (float)nb_images;
	float acc_loss = 0.;
	float prev_loss = 0.;
	int nb_train_errors = 0;
	int nb_test_errors = 0;
	float perc_train_error = 0.;
	float perc_test_error = 0.;

	// Net creation
	int hidden1 = 50;
	int nb_layers = 2;
	vector<Linear> net_layers;
	vector<int> layer_size;
	layer_size.insert(layer_size.end(), { train_input_size, hidden1, train_target_size} );
	for(int i = 1; i< nb_layers+1;i++){
		net_layers.push_back(Linear(layer_size[i],layer_size[i-1]));
	}
	net.setLayers(net_layers, nb_layers);

	
	// Training
	for(int e = 0; e < nb_epochs; e++){
		
		reset_gradients();
		nb_train_errors = 0;
		nb_test_errors = 0;
		prev_loss = acc_loss;
		acc_loss = 0.;

		printf("EPOCH %d\n", e);

		for (int i = 0 ; i<nb_images; i++){
			
			float fraction = (float)i/(float)nb_images;
			//printf("\r%% %.2f", fraction*100);

			forward_prop(train_input[i]);

			vector<float> output_vector;
			for(int j = 0; j<net.layers[net.nb_layers-1].size; j++){
				output_vector.push_back(net.layers[net.nb_layers-1].neurons[j].x);
			}
			if (train_target[i][verify_classification()] < 0.5){nb_train_errors++;}

			back_prop(train_input[i], train_target[i]);

			acc_loss += MSELoss(output_vector, train_target[i]);
			
		}

		update_weights(learning_rate);

		// Decreasing Learning rate if oscillation
		if(prev_loss < acc_loss){
			learning_rate *= 0.8;
		}
	
    	for (int i = 0; i< nb_images; i++){
			forward_prop(test_input[i]);
        	if (test_target[i][verify_classification()] < 0.5){nb_test_errors++;}
		}
		
		printf("\nLoss = %.6f \n", acc_loss);

		perc_train_error = 100*(float)nb_train_errors/ (float)nb_images;
		perc_test_error = 100*(float)nb_test_errors/ (float)nb_images;
		printf("%% of train errors = %.1f%%\n", perc_train_error);
		printf("%% of test errors = %.1f%%\n", perc_test_error);		
	}

	return 0;
}
