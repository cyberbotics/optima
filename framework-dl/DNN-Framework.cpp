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
				dw.push_back(normal(generator));
			}
			bias = 0.;
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

float relu(float x){	
	float output;
	
	output = fmax(x,0.);

	return output;
}

float drelu(float x){
	float output;

	if(x <= 0.)
		output = 0.;
	else
		output = 1.;
	

	return output;
}

float MSELoss(vector<float> input, vector<float> target){
	 
	float loss = 0;
	int i = 0;
	for(i = 0; i<input.size(); i++){
		loss += pow(input[i] - target[i],2);
	}
	return loss;
}

float MSEdLoss(float input, float target){
	float dloss;

	dloss = 2*input - 2*target;
	
	return dloss;
}

void forward_prop(vector<u_int8_t> input){
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
					sum_of_elems += current_neuron.out_weights[l] * (float)input[l];
				}
				current_neuron.s = sum_of_elems + current_neuron.bias;
				current_neuron.x = relu(current_neuron.s);

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
				current_neuron.x = relu(current_neuron.s);

				net.layers[j].neurons[i] = current_neuron;
			}
		}
	}
}

void back_prop(vector<u_int8_t> input, vector<float> target){
	
	for(int i=0;i<target.size();i++)
    {           
        net.layers[net.nb_layers-1].neurons[i].dx = MSEdLoss(net.layers[net.nb_layers-1].neurons[i].x, target[i]);
		net.layers[net.nb_layers-1].neurons[i].ds = drelu(net.layers[net.nb_layers-1].neurons[i].s)*net.layers[net.nb_layers-1].neurons[i].dx;
		net.layers[net.nb_layers-1].neurons[i].dbias += net.layers[net.nb_layers-1].neurons[i].ds;	 
		for(int j = 0; j<net.layers[net.nb_layers-1].neurons[i].dw.size(); j++){
			net.layers[net.nb_layers-1].neurons[i].dw[j] += net.layers[net.nb_layers-2].neurons[j].x * net.layers[net.nb_layers-1].neurons[i].ds;
		}
                      
    }

	for(int i = net.nb_layers-2; i>=0; i--){
		for(int j = 0; j<net.layers[i].size; j++){
			for(int k = 0; k<net.layers[i+1].size; k++){
				net.layers[i].neurons[j].dx += net.layers[i+1].neurons[k].out_weights[j] * net.layers[i+1].neurons[k].ds;
			}
			net.layers[i].neurons[j].ds = net.layers[i].neurons[j].dx * drelu(net.layers[i].neurons[j].s);
			net.layers[i].neurons[j].dbias += net.layers[i].neurons[j].ds;
			for(int k = 0; k<net.layers[i].neurons[j].dw.size(); k++){
				if(i != 0)
					net.layers[i].neurons[j].dw[k] += net.layers[i-1].neurons[k].x * net.layers[i].neurons[j].ds;
				else
					net.layers[i].neurons[j].dw[k] += (float)input[k] * net.layers[i].neurons[j].ds;
			}
		}
	}
}

int main(void)
{	
	// Load MNIST using external library (https://github.com/wichtounet/mnist)
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

	// Define input and target vector
	vector<vector<u_int8_t>> train_input= dataset.training_images;
	vector<u_int8_t> train_target= dataset.training_labels;
	// TODO -> translate labels to vectors. -> vector of vectors

	int nb_images = train_input.size();
	
	int train_input_size = train_input[0].size();
	int train_target_size = train_target[0].size();

	// Net creation
	int nb_epochs = 20;
	int nb_layers = 2;
	vector<Layer> net_layers;
	vector<int> layer_size;
	layer_size.insert( layer_size.end(), { train_input_size, 300, train_target_size} );
	for(int i = 1; i< nb_layers+1;i++){
		net_layers.push_back(Layer(layer_size[i],layer_size[i-1]));
	}
	net.setLayers(net_layers, nb_layers);

	for(int e = 0; e < nb_epochs; e++){
		//Training
		for (int i = 0 ; i<nb_images; i++){
			forward_prop(train_input[i]);
			back_prop(train_input[i], train_target[i]);
			//compute total loss
		}

		//update weights

		printf("%.20f \n", net.layers[1].neurons[1].dw[3]);	
	}
	

	return 0;
}
