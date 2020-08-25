#pragma once
#include"Matrix.h"
#include<vector>
#include<iostream>
#include<cmath>
#include<string>
#include<functional>
#include<chrono>

#define E 2.718281828459045235f //Euler's number

//Activation functions IDs
#define SIGMOID 0
#define FASTSIGMOID 1
#define TANH 2
#define RELU 3

//Functor that defines each activation function. Used to avoid unnecessary switch statements
class ActivationFunctor
{
public:
	std::function<float(float)> m_activationFunction;

	inline float operator()(float val) const { return m_activationFunction(val); }
	static inline float sigmoid(float val) { return (1 / (1 + pow(E, -val))); }
	static inline float fastSigmoid(float val) { return 1 / (1 + abs(val)); }
	static inline float tanH(float val) { return std::tanh(val); }
	static inline float relU(float val) { return std::fmax(0.0f, val); }
};

class NeuralNetwork
{
private:
	std::vector<Matrix> m_layers;				 //holds all artificial neurons
	std::vector<Matrix> m_biases;				 //holds all biases
	std::vector<Matrix> m_weights;				 //holds all weights that interconnect the layers of neurons
	float m_LR;									 //learning rate
	std::vector<Matrix> m_errors;				 //holds all errors during back propagation
	std::vector<Matrix> m_errorDerivatives;		 //error derivatives (gradient descend)
	std::vector<unsigned int> m_activationTypes; //defines what activation function should be used at the layer of the given index
	std::vector<unsigned int> m_topology;		 //defines the number of neurons for each layer
	std::vector<Matrix> m_inputs;				 //all input data
	std::vector<Matrix> m_targets;				 //all target data
	Matrix m_currentTarget;						 //target used by current iteration

public:
	NeuralNetwork(std::vector<unsigned int>& topology, float LR, std::vector<Matrix>& inputs, std::vector<Matrix>& targets);
	void train(unsigned int epochs, bool printResults = false);
	void setInput(Matrix& input);
	void setTarget(Matrix& target);
	void activate(unsigned int layer);
	void printNetwork();
	void printInputs();
	void printOutputs();
	void feedForward();
	void updateWeights();
	void backPropagation();
	void fetchErrors();
	void updateBiases(unsigned int layer);
	Matrix deriveErrors(Matrix& errors);
	float sigmoidDerivative(float val);
	void setActivationFunctions(std::vector<unsigned int>& activationTypes);
};

