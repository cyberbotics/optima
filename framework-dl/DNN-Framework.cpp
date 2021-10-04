/**
 * Document: DNN-Framework.cpp
 * Summary:
 *        Neural Net Framework for future DFE implementation
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
			random_device rd{};
    		mt19937 generator{rd()};	
			normal_distribution<float> normal(0.,1e-6);
			for (int i = 0; i < input_size; i++) {
				out_weights.push_back(normal(generator));
				dw.push_back(0.);
			}
			bias = normal(generator);
			dbias = 0.;
			x = 0.;
			dx = 0.;
			s = 0.;
			ds = 0.;
		};

		~Neuron(){};
};

class Layer{
	public:
		int size;
		vector<Neuron> neurons;

		Layer(){};

		Layer(int s, int input_size){
			size = s;
			for (int i = 0; i < s; i++) {
				neurons.push_back(Neuron(input_size));
			}
		};

		~Layer(){};

};

class Net{
	public:
		int nb_layers;
		vector<Layer> layers;

		Net(){};

		Net(vector<Layer> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		};

		void setLayers(vector<Layer> ls, int size){
			nb_layers = size;
			for (int i = 0; i < size; i++) {
				layers.push_back(ls[i]);
			}
		}

		~Net(){};

};

Net net;

vector<vector<u_int8_t>> convert_labels(vector<u_int8_t> labels){
	vector<vector<u_int8_t>> converted_labels;
	
	for(int i = 0; i<labels.size(); i++){
		vector<u_int8_t> current_label;
		for(int j = 0; j<10; j++){
			if(j == labels[i]){
				current_label.push_back(1);
			}
			else{
				current_label.push_back(0);
			}
		}
  		current_label.shrink_to_fit();                          
  		converted_labels.push_back(move(current_label)); 
	}
	return converted_labels;
}

float compute_mu(vector<vector<float>> vec){
	float mu;
	for(int i = 0; i<vec.size(); i++){
		for(int j = 0; j<vec[0].size(); j++){
			mu += vec[i][j];
		}
	}
	mu = mu / (float)vec.size() / (float)vec[0].size();
	return mu;
}

float compute_std(vector<vector<float>> vec, float mu){
	float std;
	for(int i = 0; i<vec.size(); i++){
		for(int j = 0; j<vec[0].size(); j++){
			std += pow(vec[i][j]-mu,2);
		}
	}
	std = sqrt(std / ((float)vec.size() * (float)vec[0].size()-1));
	return std;
}

vector<vector<float>> process_images(vector<vector<u_int8_t>> images_init, int desired_nb_images){
	vector<vector<u_int8_t>>::const_iterator first = images_init.begin();
	vector<vector<u_int8_t>>::const_iterator last = images_init.begin() + desired_nb_images;
	vector<vector<u_int8_t>> train_input_init(first, last);

	int nb_images = train_input_init.size();
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

	vector<vector<u_int8_t>> converted_train_target = convert_labels(train_target_init); // Convert to hot one labels

	vector<vector<float>> train_target;
	for(int i = 0; i< train_target_init.size();i++){
		vector<float> train_target_single(converted_train_target[i].begin(), converted_train_target[i].end());
		train_target_single.shrink_to_fit();                         
  		train_target.push_back(move(train_target_single));
	}
	
	return train_target;
}

float sigma(float x){	
	float output;
	
	//output = fmax(x,0.);
	output = tanh(x);

	return output;
}

float dsigma(float x){
	float output;

	/*if(x <= 0.)
		output = 0.;
	else
		output = 1.;*/
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
	float dloss;

	dloss = 2*input - 2*target;
	
	return dloss;
}

void forward_prop(vector<float> input){
	vector<float> out;
	float sum_of_elems;
	Layer current_layer;
	Neuron current_neuron;

	for(int j = 0; j<net.nb_layers; j++){
		vector<float> hidden_x;
		current_layer = net.layers[j];
		if (j==0){ //input layer
			for(int i = 0; i<current_layer.size; i++){
				sum_of_elems = 0;
				current_neuron = current_layer.neurons[i];
				for(int l = 0; l<current_neuron.out_weights.size(); l++){
					sum_of_elems += current_neuron.out_weights[l] * input[l];
				}
				current_neuron.s = sum_of_elems + current_neuron.bias;
				current_neuron.x = sigma(current_neuron.s);

				net.layers[j].neurons[i] = current_neuron;
			}

		}
		else{ //hidden layers
			for(int k = 0; k<net.layers[j-1].size; k++){
				hidden_x.push_back(net.layers[j-1].neurons[k].x);
			}
			for(int i = 0; i<current_layer.size; i++){
				sum_of_elems = 0;
				current_neuron = current_layer.neurons[i];
				for(int l = 0; l<current_neuron.out_weights.size(); l++){
					sum_of_elems += current_neuron.out_weights[l] * hidden_x[l];
				}
				current_neuron.s = sum_of_elems + current_neuron.bias;

				current_neuron.x = sigma(current_neuron.s);

				net.layers[j].neurons[i] = current_neuron;
			}
		}
	}
}

void back_prop(vector<float> input, vector<float> target){
	
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
			for(int k = 0; k<net.layers[i+1].size; k++){
				net.layers[i].neurons[j].dx += net.layers[i+1].neurons[k].out_weights[j] * net.layers[i+1].neurons[k].ds;
			}
			net.layers[i].neurons[j].ds = net.layers[i].neurons[j].dx * dsigma(net.layers[i].neurons[j].s);
			net.layers[i].neurons[j].dbias += net.layers[i].neurons[j].ds;
			for(int k = 0; k<net.layers[i].neurons[j].dw.size(); k++){
				if(i != 0)
					net.layers[i].neurons[j].dw[k] += net.layers[i-1].neurons[k].x * net.layers[i].neurons[j].ds;
				else
					net.layers[i].neurons[j].dw[k] += input[k] * net.layers[i].neurons[j].ds;
			}
		}
	}
}

void update_weights(float rate){
	for(int i = 0; i < net.nb_layers; i++){
		for(int j = 0; j< net.layers[i].size; j++){
			for(int k = 0; k< net.layers[i].neurons[j].out_weights.size(); k++){
				net.layers[i].neurons[j].out_weights[k] = net.layers[i].neurons[j].out_weights[k] - rate * net.layers[i].neurons[j].dw[k];
			}
			net.layers[i].neurons[j].bias = net.layers[i].neurons[j].bias - rate * net.layers[i].neurons[j].dbias;			
		}
	}
}

void reset_gradients(){
	for(int i = 0; i < net.nb_layers; i++){
		for(int j = 0; j< net.layers[i].size; j++){
			for(int k = 0; k< net.layers[i].neurons[j].out_weights.size(); k++){
				net.layers[i].neurons[j].dw[k] = 0.;
			}
			net.layers[i].neurons[j].dbias = 0.;			
		}
	}
}

int main(void)
{	
	// Load MNIST dataset using external library (https://github.com/wichtounet/mnist)
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>();

	// Process input data and labels
	int desired_nb_images = 1000;

	vector<vector<u_int8_t>> train_input_full= dataset.training_images;
	vector<u_int8_t> train_target_full= dataset.training_labels;

	vector<vector<float>> train_input = process_images(train_input_full, desired_nb_images);
	vector<vector<float>> train_target = process_targets(train_target_full, desired_nb_images);

	int nb_images = train_input.size();
	int train_input_size = train_input[0].size();
	int train_target_size = train_target[0].size();

	for(int j = 0; j<train_target_size; j++){
		//printf("%f\n", train_target[50][j]);
	}

	// Training parameters
	int hidden1 = 50;
	float learning_rate = 0.1 / (float)nb_images;
	float acc_loss = 0.;

	// Net creation
	int nb_epochs = 50;
	int nb_layers = 2;
	vector<Layer> net_layers;
	vector<int> layer_size;
	layer_size.insert( layer_size.end(), { train_input_size, hidden1, train_target_size} );
	for(int i = 1; i< nb_layers+1;i++){
		net_layers.push_back(Layer(layer_size[i],layer_size[i-1]));
	}
	net.setLayers(net_layers, nb_layers);

	// Training
	for(int e = 0; e < nb_epochs; e++){
		
		reset_gradients();
		acc_loss = 0.;
		printf("EPOCH %d\n", e);

		for (int i = 0 ; i<nb_images; i++){
			
			float fraction = (float)i/(float)nb_images;
			//printf("\r%% %.2f", fraction*100);
			forward_prop(train_input[i]);
			back_prop(train_input[i], train_target[i]);
			
			vector<float> output_vector;
			for(int j = 0; j<net.layers[nb_layers-1].size; j++){
				output_vector.push_back(net.layers[nb_layers-1].neurons[j].x);
				printf("%f, %d \n", output_vector[j],j);
			}

			acc_loss += MSELoss(output_vector, train_target[i]);
			
		}

		update_weights(learning_rate);
		
		printf("\n%.6f \n", acc_loss);	
	}
	

	return 0;
}
