#include<iostream>
#include<vector>
#include<cmath>
#include<chrono>
#include"Matrix.h"
#include"NeuralNetwork.h"

//This example solves the XOR-Problem

int main()
{
	std::vector<Matrix> xorInputs;
	xorInputs.push_back(Matrix(2, 1));
	xorInputs.push_back(Matrix(2, 1));
	xorInputs.push_back(Matrix(2, 1));
	xorInputs.push_back(Matrix(2, 1));
	xorInputs[0].m_data[0][0] = 0.0f;	xorInputs[0].m_data[1][0] = 0.0f; //0, 0
	xorInputs[1].m_data[0][0] = 0.0f;	xorInputs[1].m_data[1][0] = 1.0f; //0, 1
	xorInputs[2].m_data[0][0] = 1.0f;	xorInputs[2].m_data[1][0] = 0.0f; //1, 0
	xorInputs[3].m_data[0][0] = 1.0f;	xorInputs[3].m_data[1][0] = 1.0f; //1, 1

	std::vector<Matrix> xorTargets;
	xorTargets.push_back(Matrix(1, 1));
	xorTargets.push_back(Matrix(1, 1));
	xorTargets.push_back(Matrix(1, 1));
	xorTargets.push_back(Matrix(1, 1));
	xorTargets[0].m_data[0][0] = 0.0f; //0
	xorTargets[1].m_data[0][0] = 1.0f; //1
	xorTargets[2].m_data[0][0] = 1.0f; //1
	xorTargets[3].m_data[0][0] = 0.0f; //0

	//defines topology. First index is input layer. Last index output layer. All layers between define all hidden layers
	std::vector<unsigned int> topology;
	topology.push_back(2);
	topology.push_back(2);
	topology.push_back(1);

	float LR = 0.3;		//learning rate

	//defines the used activation function
	std::vector<unsigned int> at;
	at.push_back(SIGMOID);
	at.push_back(SIGMOID);

	NeuralNetwork nn(topology, LR, xorInputs, xorTargets);  //create Network
	nn.setActivationFunctions(at);							//set activation types
	nn.printNetwork();										//prints network (optional)
	nn.train(10000, true);									//train network 10'000 times and print results
	return 0;
}