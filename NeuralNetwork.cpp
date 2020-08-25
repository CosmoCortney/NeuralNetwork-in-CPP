#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<unsigned int>& topology, float LR, std::vector<Matrix>& inputs, std::vector<Matrix>& targets)
{
	//Configure network
	m_topology = topology;
	m_layers.resize(topology.size());
	m_biases.resize(topology.size() - 1);			//input layer has no biases
	m_errors.resize(topology.size() - 1);			//input layer has no errors
	m_errorDerivatives.resize(topology.size() - 1);	//input layer has no errors
	m_weights.resize(topology.size() - 1);			
	m_activationTypes.resize(topology.size() - 1);	//input layer has no activation function
	m_LR = LR;
	m_inputs = inputs;
	m_targets = targets;

	for (int layer = 0; layer < topology.size(); ++layer)
	{
		m_layers[layer] = Matrix(topology[layer], 1, true); //initialize all neurons

		//initialize rest of network
		if (layer < (topology.size() - 1))
		{
			m_biases[layer] = Matrix(topology[layer+1], 1, true);
			m_weights[layer] = Matrix(topology[layer + 1], topology[layer], true);
			m_errors[layer] = Matrix(topology[layer+1], 1);
			m_errorDerivatives[layer] = Matrix(topology[layer+1], 1);
		}
	}
}

//computes and derives errors
void NeuralNetwork::fetchErrors()
{
	for (int layer = m_layers.size() - 2; layer >= 0; --layer)
	{
		if (layer == m_layers.size() - 2)	//errors of output layer
		{
			m_errors[layer] = m_currentTarget - m_layers.back();
			m_errorDerivatives[layer] = m_errors[layer] * deriveErrors(m_layers.back());
		}
		else								//errors of hidden layers
		{
			m_errors[layer] = m_weights[layer + 1].transpose().dot(m_errorDerivatives[layer+1]);
			m_errorDerivatives[layer] = deriveErrors(m_layers[layer + 1]) * m_errors[layer];
		}
	}
}

//Updates biases by adding the derived errors times the learning rate
void NeuralNetwork::updateBiases(unsigned int layer)
{
	for (int nBias = 0; nBias < m_biases[layer].get_numRows(); ++nBias)
	{
		m_biases[layer].m_data[nBias][0] += m_errorDerivatives[layer].sumCol(0) * m_LR;
	}
}

//derives errors using the Sigmoid loss function
Matrix NeuralNetwork::deriveErrors(Matrix& errors)
{
	Matrix results(errors.get_numRows(), 1, false);
	for (int i = 0; i < errors.get_numRows(); ++i)
	{
		results.m_data[i][0] = sigmoidDerivative(errors.m_data[i][0]);
	}
	return results;
}

float NeuralNetwork::sigmoidDerivative(float val)
{
	return (val * (1 - val));
}

//sets the activation types
void NeuralNetwork::setActivationFunctions(std::vector<unsigned int>& activationTypes)
{
	m_activationTypes = activationTypes;
}

//sets the input for the current iteration
void NeuralNetwork::setInput(Matrix& inputs)
{
	m_layers[0] = inputs;
}

//sets the target for the current iteration
void NeuralNetwork::setTarget(Matrix& target)
{
	m_currentTarget = target;
}

//activates neurons by applying the defined activation function
void NeuralNetwork::activate(unsigned int layer)
{
	ActivationFunctor activationFunction;

	switch (m_activationTypes[layer-1])
	{
	case FASTSIGMOID: activationFunction.m_activationFunction = ActivationFunctor::fastSigmoid; break;
	case TANH: activationFunction.m_activationFunction = ActivationFunctor::tanH; break;
	case RELU: activationFunction.m_activationFunction = ActivationFunctor::relU; break;
	default: activationFunction.m_activationFunction = ActivationFunctor::sigmoid;
	}

	for (int neuron = 0; neuron < m_layers[layer].get_numRows(); ++neuron)
	{
		m_layers[layer].m_data[neuron][0] = activationFunction(m_layers[layer].m_data[neuron][0] + m_biases[layer - 1].m_data[neuron][0]);
	}
}

//prints the entire network
void NeuralNetwork::printNetwork()
{
	for (int layer = 0; layer < m_topology.size(); ++layer)
	{
		std::string str = std::string("Layer ").append(std::to_string(layer)).append(" neurons:");
		std::cout << (layer == 0 ? "Input Layer:" : (layer == (m_topology.size() - 1) ? "Output Layer:" : str)) << "\n";
		m_layers[layer].print();
		std::cout << "\n";

		if (layer < m_topology.size() - 1)
		{
			std::cout << "Biases Layer: " << layer << "\n";
			m_biases[layer].print();
			std::cout << "\n";

			std::cout << "Weights Layer: " << layer << "\n";
			m_weights[layer].print();
			std::cout << "\n";
		}
	}
	std::cout << "---------\n\n";
}

void NeuralNetwork::printInputs()
{
	std::cout << "Input Layer:\n";
	for (int neuron = 0; neuron < m_layers[0].get_numRows(); ++neuron)
	{
		std::cout << m_layers[0].m_data[neuron][0] << "\n";
	}
}

void NeuralNetwork::printOutputs()
{
	std::cout << "Output Layer:\n";
	for (int neuron = 0; neuron < m_layers.back().get_numRows(); ++neuron)
	{
		std::cout << m_layers.back().m_data[neuron][0] << "\n";
	}
}

//updates the weights concernign to the errors
void NeuralNetwork::updateWeights()
{
	for (int layer = m_layers.size() - 2; layer >= 0; --layer)
	{
		Matrix deltaWeights = m_layers[layer].transpose();
		deltaWeights = m_errorDerivatives[layer].dot(deltaWeights) *m_LR;
		m_weights[layer] = m_weights[layer] + deltaWeights;
		updateBiases(layer);
	}
}

//process input values
void NeuralNetwork::feedForward()
{
	for (int layer = 0; layer < m_weights.size(); ++layer)
	{
		m_layers[layer + 1] = m_weights[layer].dot(m_layers[layer]);
		activate(layer + 1);
	}
}

//back-propagates the errors and updates the weights
void NeuralNetwork::backPropagation()
{
	fetchErrors();
	updateWeights();
}

//training routine
void NeuralNetwork::train(unsigned int epochs, bool printResults)
{
	auto start = std::chrono::steady_clock::now();
	for (struct { unsigned int dataSet = 0; int epoch = 0; } count; count.epoch < epochs; ++count.epoch)
	{
		if (count.dataSet == m_inputs.size())
		{
			count.dataSet = 0;
		}

		setInput(m_inputs[count.dataSet]);
		setTarget(m_targets[count.dataSet]);

		feedForward();

		backPropagation();

		++count.dataSet;

		if (count.epoch > epochs - m_inputs.size()-1)
		{
			//get time here to neglect printing time
			auto end = std::chrono::steady_clock::now();

			if (printResults)
			{ 
				printInputs();
				std::cout << "\n";
				printOutputs();
				std::cout << "- - - - - - -\n\n";

				if (count.epoch == epochs - 1)
				{
					std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms\n";
				}
			}
		}
	}
}