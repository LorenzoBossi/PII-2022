# Hardware acceleration of a convolutional neural network using *Vitis HLS*


## Project
This repository contains  first experience approaching to *High Level Synthesis* during this [project](https://pii.dei.polimi.it/accelerazione-hardware-di-una-rete-neurale-convoluzionale-mediante-sintesi-ad-alto-livello/), the project required us to create and synthesize a Convolutional Neural Network capable of recognising hand written digits using HLS.
During the last semester we:
* Learnt the structure and implemented a *Convolutional Neural Network* (CNN) in python;
* Trained said network;
* Implemented the CNN in c/c++ using the training data obtained from the python training;
* Revisited the code to make it more "HLS friendly";
* Developed the neural network with *Vitis HLS*;
* Applied different *Vitis* pragmas;
* Evalute pragma effect on performances

## Group
- ###   Federico Mandelli ([@FedericoMandelli](https://github.com/federico-mandelli))
- ###   Lorenzo Bossi ([@LorenzoBossi](https://github.com/LorenzoBossi))


## How to run this project
* To run the C implementation extract the folder, after you installed gcc compile the code with "gcc mainc.c -lm -o exe" and then run the code with "./exe";
* To run the Vitis code install vitis 2021.2, create a new project and add all the file to the project, after that you can test the code.

## Used tools and languages
* Python
* Tensorflow
* Keras
* C/C++
* Vitis HLS

