# OPTIMA
The goal of the [OPTIMA project](https://optima-hpc.eu/project/) is to take advantage of FPGA-based High Performance Computing (HPC) systems to optimize academic and industrial softwares and applications. It also aims at providing guidelines to ease future development of FPGA applications.

In the context of Cyberbotics, the goal is to show that running a Webots robot simulation which uses deep learning in its controllers on FPGA-based systems is much faster than on CPU or GPU. This repository summarizes the work performed on the Jumax machine to adapt a deep-learning robot simulation for FPGA.

## Cloning this repository

When cloning this repository, don't forget to initialize the submodules.
``` console
git clone https://github.com/cyberbotics/optima.git
git submodule update --init --recursive
```


## Webots on Jumax
To install Webots on Jumax, please refer to the 2 following pages: [get access to Jumax](https://github.com/cyberbotics/optima/wiki/Access-Jumax) and [compile Webots on Jumax](https://github.com/cyberbotics/optima/wiki/Compile-Webots-on-Jumax).

This page explains how to start the simulator: [start Webots](https://github.com/cyberbotics/optima/wiki/Start-Webots).

## Creating and Running FPGA Programs
To understand how DFEs (running the FPGAs) on Jumax work, please head to the following page: [basics of DFE applications](https://github.com/cyberbotics/optima/wiki/Basics-of-DFE-Applications).

To code your own applications on Jumax, these 2 pages explain the required workflow: [start MaxIDE](https://github.com/cyberbotics/optima/wiki/Start-MaxIDE) and [compile with MaxCompiler](https://github.com/cyberbotics/optima/wiki/Compile-with-MaxCompiler)


## Multilayers Perceptrons (MLP) framework
The first contribution consists in creating a framework which allows to create simple neural networks: multilayers perceptrons. The details of its implementation in C++ is given here: [Creation of a Deep Learning Frameowrk in C++](https://github.com/cyberbotics/optima/wiki/Creation-of-a-MLP-Deep-Learning-Framework-in-CPP). 

The source code of the implementation is located in the `MLP-train-framework` directory. You can compile it using the following command
``` console
cd MLP-train-framework
mkdir build
g++ -O3 -Imnist src/MLP-train-framework.cpp -o build/MLP-train-framework
```
Then you can run the framework using `./build/MLP-train-framework`. You can choose any structure you want and it will train your network on the MNIST dataset.


## Forward propagation performance
The `MLP-forward` directory contains the files for the comparison of the execution time of forward propagation between CPU and FPGA. 

Implementations are explained here: [MLP Forward Propagation on CPU: Tests and Results](https://github.com/cyberbotics/optima/wiki/MLP-Forward-Propagation-on-CPU:-Tests-and-Results) & [MLP Forward Propagation on DFE: Structure](https://github.com/cyberbotics/optima/wiki/MLP-Forward-Propagation-on-DFE:-Structure). Final results can be found here: [MLP Forward Propagation on DFE: Results and Optimization](https://github.com/cyberbotics/optima/wiki/MLP-Forward-Propagation-on-DFE:-Results-and-Optimization).

Respective source codes are located in `MLP-forward/CPU` and `MLP-forward/DFE`. Informations on how to run the DFE projects can be found in [Run DFE Applications with MaxIDE](https://github.com/cyberbotics/optima/wiki/Run-DFE-Applications-With-MaxIDE). The CPU codes can be compiled with the following commands.
```console
cd MLP-forward/CPU
mkdir build
g++ -O3 -fopenmp -Imnist src/fixed-forward-test-CPU.cpp -o build/fixed_forward_test-CPU
```

Note that the source codes are optimized for the Jumax machine, in particular concerning the multi-threading optimization. 

## Autonomous car simulation
The `CNN-autonomous-car` directory contains the Webots project of the self-driving car using a convolutional neural network. Its description can be found [here](https://github.com/cyberbotics/optima/wiki/Autonomous-Car-Simulation).

The training and testing worlds are both located in `CNN-autonomous-car/worlds`.

The different controller source codes are located in `CNN-autonomous-car/controllers`. The available controllers are the following.
* **CNN_autonomous_car_cpu_float**: this controller contains the naive version of the neural network with floating point represention to drive the car. 
* **CNN_autonomous_car_cpu_fixed**: this controller is the optimized CPU version of the neural network with fixed point representation and multithreading to drive the car.
* **CNN_autonomous_car_fpga**: this controller uses the _cnn_dfe_ library to run the neural network on a DFE. This controller has 3 different loop optimization modes. The mode must be passed through the controllerArgs field in `CNN-autonomous-car/worlds/autonomous_car_test.wbt`.

The DFE optimization of the neural network is compiled in a shared library: `CNN-autonomous-car/libraries/cnn_dfe`. The corresponding DFE kernels and manager are in: `CNN-autonomous-car/libraries/cnn_dfe/src`.

You can select the controller to drive the car by editing the controller field in `CNN-autonomous-car/worlds/autonomous_car_test.wbt` and then [run Webots](https://github.com/cyberbotics/optima/wiki/Start-Webots) on Jumax.
