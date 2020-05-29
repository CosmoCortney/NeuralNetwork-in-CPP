#include"AnnMaths.h"
#include<iostream>
#include<vector>
#include"NeuralNetwork.h"

int main()
{
	std::vector<std::vector<std::vector<double>>> input;
	input.push_back(std::vector<std::vector<double>>());
	input.push_back(std::vector<std::vector<double>>());
	input.push_back(std::vector<std::vector<double>>());
	input.push_back(std::vector<std::vector<double>>());
	input[0].push_back(std::vector<double>());
	input[0].push_back(std::vector<double>());
	input[0][0].push_back(0.0);
	input[0][1].push_back(0.0);
	input[1].push_back(std::vector<double>());
	input[1].push_back(std::vector<double>());
	input[1][0].push_back(1.0);
	input[1][1].push_back(0.0);
	input[2].push_back(std::vector<double>());
	input[2].push_back(std::vector<double>());
	input[2][0].push_back(0.0);
	input[2][1].push_back(1.0);
	input[3].push_back(std::vector<double>());
	input[3].push_back(std::vector<double>());
	input[3][0].push_back(1.0);
	input[3][1].push_back(1.0);

	std::vector<std::vector<std::vector<double>>> targets;
	targets.push_back(std::vector<std::vector<double>>());
	targets.push_back(std::vector<std::vector<double>>());
	targets.push_back(std::vector<std::vector<double>>());
	targets.push_back(std::vector<std::vector<double>>());
	targets[0].push_back(std::vector<double>());
	targets[0][0].push_back(0.0);
	targets[1].push_back(std::vector<double>());
	targets[1][0].push_back(1.0);
	targets[2].push_back(std::vector<double>());
	targets[2][0].push_back(1.0);
	targets[3].push_back(std::vector<double>());
	targets[3][0].push_back(0.0);

	std::vector<int> topology;
	topology.push_back(input[0].size());
	topology.push_back(3);
	topology.push_back(3);
	topology.push_back(targets[0].size());

	NeuralNetwork nn(topology, input[0], targets[0], 0.0);
	
	int count = 0;
	for (int i = 0; i < 10000; ++i)
	{
		if (count == 4) { count = 0; }
		nn.setInput(input[count]);
		nn.setTargets(targets[count]);
		++count;

		nn.feedForward();
		nn.backPropagation();

		if (i % 51 == 0)
		{
			nn.printInput();
			nn.printOutput();
		}
	}
	return 0;
}