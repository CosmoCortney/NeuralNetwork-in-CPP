#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int>& topology, std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& targets, double lr)
{
	this->topology = topology;
	this->layers.clear();
	this->weights.clear();
	this->targets.clear();
	this->targets = targets;
	this->layers.resize(topology.size());
	this->errors.resize(this->layers.size());
	this->weights.resize(topology.size()-1);
	this->learnrate = lr;

	for (int layer = 0; layer < topology.size(); ++layer)
	{
		if (layer == 0)
		{
			this->layers[layer].push_back(std::vector<double>());
			this->layers[layer] = input;
		}
		else
		{
			for (int neuron = 0; neuron < topology[layer]; ++neuron)
			{
				this->layers[layer].push_back(std::vector<double>());
				this->layers[layer][neuron].push_back(/*AnnMaths::randomVal()*/0.0);
			}
		}

		if (layer < (topology.size() - 1))
		{
			for (int row = 0; row < topology[layer+1]; ++row)
			{
				this->weights[layer].push_back(std::vector<double>());
				for (int col = 0; col < topology[layer]; ++col)
				{
					this->weights[layer][row].push_back(AnnMaths::randomVal());
				}
			}
		}
	}
	this->errors = this->layers;
}

void NeuralNetwork::setLayers(std::vector<std::vector<std::vector<double>>>& layers)
{
	this->layers.clear();
	this->layers = layers;
}

void NeuralNetwork::setWeights(std::vector<std::vector<std::vector<double>>>& weights)
{
	this->weights.clear();
	this->weights = weights;
}

void NeuralNetwork::printInput()
{
	std::cout << "Input layer:\n";
	for (int neuron = 0; neuron < layers[0].size(); ++neuron)
	{
		std::cout << layers[0][neuron][0] << "   ";
	}
	std::cout << "\n\n";
}

void NeuralNetwork::printOutput()
{
	std::cout << "Output layer:\n";
	for (int neuron = 0; neuron < layers[layers.size()-1].size(); ++neuron)
	{
		std::cout << layers[layers.size() - 1][neuron][0] << "   ";
	}
	std::cout << "\n\n";
}

void NeuralNetwork::printErrors()
{
	std::cout << "Errors:\n";
	for (int neuron = 0; neuron < errors.size(); ++neuron)
	{
		std::cout << errors.back()[neuron][0] << "   ";
	}
	std::cout << "\n\n";
}

void NeuralNetwork::printNetwork()
{
	for (int layer = 0; layer < layers.size(); ++layer)
	{
		if (layer == 0) { std::cout << "Input layer:\n"; }
		else if (layer == layers.size()-1) { std::cout << "Output layer:\n"; }
		else { std::cout << "Layer " << layer << ":\n";	}

		AnnMaths::printMatrix(layers[layer]);

		if (layer < layers.size() - 1)
		{
			if (layer == 0) { std::cout << "Weights Input layer to layer " << layer + 1 << ":\n"; }
			else if (layer == layers.size() - 2) { std::cout << "Weights layer " << layer << " to output layer:\n"; }
			else { std::cout << "Weights layer " << layer << " to layer " << layer + 1 << ":\n"; }

			AnnMaths::printMatrix(weights[layer]);
		}
	}

	//std::cout << "Output errors:\n";
	//AnnMaths::printMatrix(errors.back());
	std::cout << "\n\n";
}

void NeuralNetwork::feedForward()
{
	for (int layer = 0; layer < weights.size(); ++layer)
	{
		layers[layer + 1] = AnnMaths::multiply(weights[layer], layers[layer], true);
	}
}

void NeuralNetwork::setErrors()
{
	for (int layer = layers.size() - 1; layer >= 0; --layer)
	{
		if (layer == layers.size() - 1)
		{
			this->errors[layer] = AnnMaths::getOutputErrors(this->targets, layers[layer]);
		}
		else
		{
			std::vector<std::vector<double>> weightsT = AnnMaths::transpose(this->weights[layer]);
			std::vector<std::vector<double>> tmpErrors = AnnMaths::multiply(weightsT, this->errors[layer+1]);
			this->errors[layer] = tmpErrors;
		}
	}
}

void NeuralNetwork::setInput(std::vector<std::vector<double>>& input)
{
	this->layers[0] = input;
}

void NeuralNetwork::setTargets(std::vector<std::vector<double>>& target)
{
	this->targets = targets;
}

void NeuralNetwork::backPropagation()
{
	setErrors();

	for (int layer = layers.size() - 2; layer >= 0; --layer)
	{
		std::vector<std::vector<double>> prevOutputT = AnnMaths::transpose(layers[layer]);
		std::vector<std::vector<double>> appliedErrors = AnnMaths::applyErrors(this->errors[layer + 1], layers[layer+1]);
		std::vector<std::vector<double>> deltaWeights = AnnMaths::multiply(appliedErrors, prevOutputT);
		deltaWeights = AnnMaths::applyLearnrate(deltaWeights, learnrate);
	    weights[layer] = AnnMaths::add(deltaWeights, weights[layer]);
	}
}
