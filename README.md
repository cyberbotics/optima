# optima
The goal of the [OPTIMA project](https://optima-hpc.eu/project/) is to take advantage of FPGA-based High Performance Computing (HPC) systems to optimize academic and industrial softwares and applications. It also aims at providing guidelines to ease future development of FPGA applications.

In the context of Cyberbotics, the goal is to show that running a Webots robot simulation which uses deep learning in its controllers on FPGA-based systems is much faster than on CPU or GPU. This repository summarizes the work performed on the Jumax machine to adapt a deep-learning robot simulation for FPGA.

## Cloning this repository

When cloning this repository, don't forget to initialize the submodules.
```
git clone https://github.com/cyberbotics/optima.git
git submodule update --init --recursive
```


## Webots on Jumax
To install Webots on Jumax, please refer to the 2 following pages: [get access to Jumax](https://github.com/cyberbotics/optima/wiki/Access-Jumax) and [compile Webots on Jumax](https://github.com/cyberbotics/optima/wiki/Compile-Webots-on-Jumax).

This page explains how to start the simulator: [start Webots](https://github.com/cyberbotics/optima/wiki/Start-Webots).

## Creating and Running FPGA Programs
To understand how DFEs (running the FPGAs) on Jumax work, please head to the following page: [basics of DFE applications](https://github.com/cyberbotics/optima/wiki/Basics-of-DFE-Applications).

To code your own applications on Jumax, these 2 pages explain the required workflow: [start MaxIDE](https://github.com/cyberbotics/optima/wiki/Start-MaxIDE) and [compile with MaxCompiler](https://github.com/cyberbotics/optima/wiki/Compile-with-MaxCompiler)

## Machine Learning Applications
To use deep learning in simulation, the first step is to create a framework which allows to create neural networks. The details of its implementation in C++ is given here: [Creation of a Deep Learning Frameowrk in C++](https://github.com/cyberbotics/optima/wiki/Creation-of-a-Deep-Learning-Framework-in-CPP). 

The source code of the implementation is located in the `framework-dl` directory. You can compile it using the following command
```
cd framework-dl
mkdir build
make
```
Then you can run the framework using `./build/DNN-framework`. You can choose any structure you want and it will train your network on the MNIST dataset.
