#pragma once
#include"AnnMaths.h"
#include<vector>
#include<iostream>
#include<math.h>

class NeuralNetwork
{
private:
	std::vector<int> topology;
	std::vector<std::vector<std::vector<double>>> layers;
	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> targets;
	std::vector<std::vector<std::vector<double>>> errors;
	double learnrate;

public:
	NeuralNetwork(std::vector<int>& topology, std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& targets, double lr);
	void setLayers(std::vector<std::vector<std::vector<double>>>& layers);
	void setWeights(std::vector<std::vector<std::vector<double>>>& weights);
	void printInput();
	void printOutput();
	void printErrors();
	void printNetwork();
	void feedForward();
	void backPropagation();
	void setErrors();
	void setInput(std::vector<std::vector<double>>& input);
	void setTargets(std::vector<std::vector<double>>& targets);
};

