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

#define CONV 0
#define POOL 1
#define LINEAR 2

const float INPUT_MEANS[] = {75.5759, 80.3124, 82.3455};
const float INPUT_STDS[] = {41.0884, 50.9617, 55.9314};
const float TARGET_MEANS[] = {6.3157e-04, 5.1197e+01};
const float TARGET_STDS[] = {0.0729, 16.7403};

using namespace std;
using PropResult = tuple<vector<vector<float>>, vector<vector<float>>>;

template<typename T>
vector<T> flatten(const vector<vector<T>> &orig)
{
    vector<T> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());
    return ret;
}

class Neuron{
	public:
		vector<float> out_weights;
		float bias;	

		int nb_weights;

		Neuron(){};

		Neuron(int input_size){
			for (int i = 0; i < input_size; i++) { 
				out_weights.push_back(0.);
			}
			bias = 0.;

			nb_weights = out_weights.size();
		};

		~Neuron(){};
};

class Kernel{
	public:
		int nb_weights;
		int k_size;
		int k_depth;

		vector<float> weights;
		float bias;

		Kernel(){};
		Kernel(int size, int depth){
			for (int i = 0; i < size*size*depth; i++) { 
				weights.push_back(0.);
			}
			bias = 0.;

			nb_weights = weights.size();
			k_size = size;
			k_depth = depth;	
		};

		~Kernel(){};
};


class LinearLayer{
	public:
		int nb_neurons;
		vector<Neuron> neurons;


		LinearLayer(){};

		LinearLayer(int s, int input_size){
			nb_neurons = s;
			for (int i = 0; i < s; i++) {
				neurons.push_back(Neuron(input_size));
			}
		};

		~LinearLayer(){};

};

class ConvLayer{
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
		ConvLayer(int k_size, int inChan, int inH, int inW, int outChan, int outH, int outW){
			
			in_channels = inChan;
			in_width = inW;
			in_height = inH;

			out_width = outW;
			out_height = outH;
			nb_kernels = outChan;
			kernel_size = k_size;
			for(int i = 0; i< outChan; i++){
				kernels.push_back(Kernel(k_size, inChan));
			}
		};

		~ConvLayer(){};
};

class MaxPoolLayer{
	public: 
		int filter_size;
		int in_channels;
		int in_width;
		int in_height;

		MaxPoolLayer(){};
		MaxPoolLayer(int f_size, int inChan, int inH, int inW){
			filter_size = f_size;
			in_channels = inChan;
			in_height = inH;
			in_width = inW;
		};

		~MaxPoolLayer(){};
};

class Network{
	public:	
		vector<ConvLayer> conv_layers;
		vector<MaxPoolLayer> pool_layers;
		vector<LinearLayer> linear_layers;

		vector<int> layer_types;
		vector<int> layers;

		int nb_layers;

		Network(){
			conv_layers.push_back(ConvLayer(3,3,80,320,16,78,318));
			conv_layers.push_back(ConvLayer(3,16,39,159,32,37,157));
			conv_layers.push_back(ConvLayer(3,32,18,78,64,16,76));

			pool_layers.push_back(MaxPoolLayer(2,16,78,318));
			pool_layers.push_back(MaxPoolLayer(2,32,37,157));
			pool_layers.push_back(MaxPoolLayer(2,64,16,76));

			linear_layers.push_back(LinearLayer(500,19456));
			linear_layers.push_back(LinearLayer(2,500));

			layer_types = {CONV,POOL,CONV,POOL,CONV,POOL,LINEAR,LINEAR};
			layers = {0,0,1,1,2,2,0,1};

			nb_layers = layers.size();
		};

		~Network(){};

};

Network net;

float elu(float x){	
	float output = 0.;
	
	//output = fmax(x,0.);
	output = (x > 0.0f) * x + (x <= 0.0f) * ((float)exp(x) - 1.0f);

	return output;
}

vector<float> forward_prop(vector<float> input){
	float sum = 0.;
	
	ConvLayer current_conv_layer;
	MaxPoolLayer current_pool_layer;
	LinearLayer current_lin_layer;
	Neuron current_neuron;

	vector<float> x;
	vector<float> prevx;

	for(int i = 0; i<net.nb_layers; i++){
		int layer_id = net.layers[i];
		int layer_type = net.layer_types[i];
		
		prevx.clear();
		if(i == 0)
			prevx = input;
		else
			prevx = x;
		x.clear();

		if(layer_type == CONV){
			current_conv_layer = net.conv_layers[layer_id];
			for (int nc = 0; nc < current_conv_layer.nb_kernels; nc++)
			{
				for (int nh = 0; nh < current_conv_layer.out_height; nh++)
				{
					for (int nw = 0; nw < current_conv_layer.out_width; nw++)
					{
						sum = 0;
						for (int kc = 0; kc < current_conv_layer.in_channels; kc++)
						{
							for (int kh = 0; kh < current_conv_layer.kernel_size; kh++)
							{
								for (int kw = 0; kw < current_conv_layer.kernel_size; kw++)
								{
									const int prevIdx = kc*current_conv_layer.in_height*current_conv_layer.in_width + (kh+nh)*current_conv_layer.in_width + (kw+nw);
									const int kernelIdx = kc*current_conv_layer.kernel_size*current_conv_layer.kernel_size + kh*current_conv_layer.kernel_size + kw;
									sum += prevx[prevIdx] * current_conv_layer.kernels[nc].weights[kernelIdx];
								}
							}
						}
						sum += current_conv_layer.kernels[nc].bias;
						
						x.push_back(elu(sum));
					}
				}
			}

		}

		else if(layer_type == POOL){		
			current_pool_layer = net.pool_layers[layer_id];

			int out_channels = current_pool_layer.in_channels;
			int out_height = floor(current_pool_layer.in_height/current_pool_layer.filter_size);
			int out_width = floor(current_pool_layer.in_width/current_pool_layer.filter_size);

			for (int nc = 0; nc < out_channels; nc++)
			{
				//printf("%d\n", nc);
				for (int nh = 0; nh<out_height; nh++)
				{	
					for (int nw = 0; nw < out_width; nw++)
					{
						//printf("%d\n", nw);
						float result = 0;

						for (int ph = 0; ph < current_pool_layer.filter_size; ph++)
						{
							for (int pw = 0; pw < current_pool_layer.filter_size; pw++)
							{
								const int inY = nh + ph;
								const int inX = nw + pw;
								if (inY >= 0 && inY<current_pool_layer.in_height && inX >= 0 && inX<current_pool_layer.in_width)
								{
									const int prevIdx = nc*current_conv_layer.in_height*current_conv_layer.in_width + inY*current_conv_layer.in_width + inX;

									if (result < prevx[prevIdx])
									{
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

		else if(layer_type == LINEAR){
			current_lin_layer = net.linear_layers[layer_id];

			for(int j = 0; j<current_lin_layer.nb_neurons; j++){
				sum = 0.;
				current_neuron = current_lin_layer.neurons[j];
				for(int k = 0; k<current_neuron.nb_weights; k++){
					sum += current_neuron.out_weights[k] * prevx[k];
				}

				x.push_back(elu(sum + current_neuron.bias));
			}
		}		
	}

	return x;
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
		if(net.layer_types[i] == CONV){
			int layer_id = net.layers[i];
			for(int j = 0; j< net.conv_layers[layer_id].nb_kernels; j++){
				for(int k = 0; k< net.conv_layers[layer_id].kernels[j].nb_weights; k++){
					getline(wfile, line);
					stringstream ss(line);
					ss >> tmp;
					net.conv_layers[layer_id].kernels[j].weights[k] = tmp;
				}
				getline(bfile,line);
				stringstream ss(line);
				ss >> tmp;
				net.conv_layers[layer_id].kernels[j].bias = tmp;
			}
		}

		else if(net.layer_types[i] == LINEAR){
			int layer_id = net.layers[i];
			for(int j = 0; j< net.linear_layers[layer_id].nb_neurons; j++){
				for(int k = 0; k< net.linear_layers[layer_id].neurons[j].nb_weights; k++){
					getline(wfile, line);
					stringstream ss(line);
					ss >> tmp;
					net.linear_layers[layer_id].neurons[j].out_weights[k] = tmp;
				}
				getline(bfile,line);
				stringstream ss(line);
				ss >> tmp;
				net.linear_layers[layer_id].neurons[j].bias = tmp;
			}
		}

		
	}

	bfile.close();
	wfile.close();
}

int main(void)
{		
	load_weights();
	printf("Weights and biases successfully loaded !\n");
	
	vector<float> input(3*80*320);
    float value = 255.;
    fill(input.begin(), input.end(), value);

	// Normalize input
	for(int i = 0; i < input.size()/3; i++){
		input[i] = (input[i] - INPUT_MEANS[0])/INPUT_STDS[0];
	}
	for(int i = input.size()/3; i < 2*input.size()/3; i++){
		input[i] = (input[i] - INPUT_MEANS[1])/INPUT_STDS[1];
	}
	for(int i = 2*input.size()/3; i < input.size(); i++){
		input[i] = (input[i] - INPUT_MEANS[2])/INPUT_STDS[2];
	}


	float acc_time = 0;
	struct timeval start;
	gettimeofday(&start, NULL);

	vector<float> result;
	result = forward_prop(input);

	struct timeval end;
	gettimeofday(&end, NULL);
	acc_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1e-6;
	printf("%f\n", acc_time);
	
	float steering = result[0] * TARGET_STDS[0] + TARGET_MEANS[0];
    float speed = result[1] * TARGET_STDS[1] + TARGET_MEANS[1];

	printf("%f\n", steering);
	printf("%f\n", speed);

	return 0;
}
